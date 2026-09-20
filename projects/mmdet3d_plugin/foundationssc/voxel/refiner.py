"""本地 VoxFormer/DFA3D 路径：候选交叉注意力、掩码先验、自注意力扩散。

参照 FoundationSSC 的单尺度/单左相机配置。保留默认 512×512 展平 self-attention
布局（它不是物理 BEV）；CUDA 使用本地移植的原 DFA3D 与 MMCV 算子，PyTorch 用于调试。
模块命名不同，不支持直接载入原完整 SSC checkpoint；立体骨干 checkpoint 不受影响。
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .geometry import project
from ..ops import prepare_attention, deform_attention_prepared, use_cuda


def sample_depth_weighted(value, depth, locations):
    """对隐式体 value[u,v]*depth[d,u,v] 做三线性插值，不显式构造巨大 3D 特征体。

    value[B,heads,c,H,W], depth[B,D,H,W], locations[B,Q,heads,P,3]。
    normalized [0,1]，align_corners=False；越界补零，与 deformable sampling 一致。
    注意不能把双线性 feature 与三线性 depth 分别采样后相乘，两者不等价。
    """
    b, heads, c, h, w = value.shape
    d = depth.shape[1]
    xyz = locations * locations.new_tensor([w, h, d]) - .5
    lo = xyz.floor().long()
    frac = xyz - lo
    values = value.permute(0, 1, 3, 4, 2).reshape(b, heads, h * w, c)
    probs = depth.reshape(b, d * h * w)
    bi = torch.arange(b, device=value.device)[:, None, None, None]
    hi = torch.arange(heads, device=value.device)[None, None, :, None]
    result = value.new_zeros((*locations.shape[:-1], c))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                x, y, z = (lo + lo.new_tensor([dx, dy, dz])).unbind(-1)
                valid = (x >= 0) & (x < w) & (y >= 0) & (y < h) & (z >= 0) & (z < d)
                xy = y.clamp(0, h - 1) * w + x.clamp(0, w - 1)
                p = probs[bi, z.clamp(0, d - 1) * h * w + xy]
                v = values[bi, hi, xy]
                wx = frac[..., 0] if dx else 1 - frac[..., 0]
                wy = frac[..., 1] if dy else 1 - frac[..., 1]
                wz = frac[..., 2] if dz else 1 - frac[..., 2]
                result = result + v * (p * wx * wy * wz * valid)[..., None]
    return result


def init_offsets(layer, heads, points, queues=1, depth=False):
    nn.init.zeros_(layer.weight)
    theta = torch.arange(heads) * (2 * math.pi / heads)
    if depth:
        direction = ((theta.cos() + theta.sin()) / 2)[:, None]
    else:
        direction = torch.stack((theta.cos(), theta.sin()), -1)
        direction = direction / direction.abs().amax(-1, keepdim=True)
    bias = direction[:, None, None] * torch.arange(1, points + 1)[None, None, :, None]
    with torch.no_grad():
        layer.bias.copy_(bias.expand(heads, queues, points, direction.shape[-1]).reshape(-1))


class CrossLayer(nn.Module):
    def __init__(self, channels, heads, points, ffn_channels, dropout, chunk, ops_backend='pytorch'):
        super().__init__()
        self.heads, self.points, self.chunk = heads, points, chunk
        self.ops_backend = ops_backend
        self.offset_uv = nn.Linear(channels, heads * points * 2)
        self.offset_d = nn.Linear(channels, heads * points)
        self.weights = nn.Linear(channels, heads * points)
        self.value = nn.Linear(channels, channels)
        self.output = nn.Linear(channels, channels)
        self.drop = nn.Dropout(dropout)
        self.norm1, self.norm2 = nn.LayerNorm(channels), nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, ffn_channels), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ffn_channels, channels), nn.Dropout(dropout))
        init_offsets(self.offset_uv, heads, points)
        init_offsets(self.offset_d, heads, points, depth=True)
        nn.init.zeros_(self.weights.weight); nn.init.zeros_(self.weights.bias)
        for layer in (self.value, self.output):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, query, context, depth, reference, visible):
        #* ================== 2.5.4.1 输入与图像 value 准备 ==================
        #* 候选交叉注意力：query 从左图 context 中采样信息，并结合深度概率约束采样贡献。
        # query=[B,Q,C]，context=[B,C,h,w]，depth=[B,D,h,w]；此处 Q 仅为候选数量。
        #* depth 是 depth_prob 深度概率分布，不是 stereo_depth 米制深度图，也不是 GT。
        # 默认 D=112，各通道对应一个米制深度采样位置；张量元素是概率，不是米数。
        # reference=[B,Q,3] 是归一化 (u,v,depth)，visible=[B,Q] 是投影视野 mask。
        b, c, h, w = context.shape
        value = self.value(context.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).reshape(b, self.heads, c // self.heads, h, w)
        cuda = use_cuda(self.ops_backend, value)
        #todo: 原先每块都 repeat 深度分布并由 autograd 保存；改为每层准备一次、各块共享。
        #* depth 作用①：与 value 一起整理为 CUDA 采样数据，各 query 块共享；memory 不是历史帧缓存。
        # 此处仅准备布局，不执行注意力汇聚，也不 detach；梯度仍可回传到 depth_prob 的预测网络。
        memory = prepare_attention(value, depth) if cuda else None
        chunks = []
        for start in range(0, query.shape[1], self.chunk):
            #* ================== 2.5.4.2 分块 query 与三维采样位置 ==================
            # 仅分块处理 query，共享完整图像 memory；不是将图像切成互不交互的小块。
            q = query[:, start:start + self.chunk]
            uv = self.offset_uv(q).reshape(b, -1, self.heads, self.points, 2)
            dd = self.offset_d(q).reshape(b, -1, self.heads, self.points, 1)
            #* depth 作用②：用深度通道数 D 将 dd 从深度网格单位转换为归一化偏移；这里只用 shape，不用概率值。
            # dd 由 query 预测，不是从 depth 中直接取出的深度；reference 的深度则来自体素中心的几何投影。
            offset = torch.cat((uv, dd), -1) / q.new_tensor([w, h, depth.shape[1]])
            #* 在几何投影参考点附近学习 (u,v,深度) 偏移，而不是对所有图像像素做全连接注意力。
            locations = reference[:, start:start + self.chunk, None, None] + offset
            # locations=[B,Q_chunk,heads,points,3]，末维为归一化 (u,v,depth)，不是 LiDAR 的 (x,y,z)。
            #* ================== 2.5.4.3 预测各采样点的注意力权重 ==================
            # 每个 query、每个 head 对 points 个采样点做 softmax；这是注意力权重，不是深度概率。
            weights = self.weights(q).reshape(b, -1, self.heads, self.points).softmax(-1)

            #* ================== 2.5.4.4 深度概率引导的图像采样与汇聚 ==================
            #* 对隐式的 图像 value×深度概率 进行三维插值采样，再按注意力权重汇聚。
            #* depth 作用③（核心）：F(u,v,d)=value(u,v)×P(d|u,v)，让同一像素在不同深度处具有不同特征贡献。
            # 在隐式 F 上按 locations 三线性插值，再用 weights 汇聚；并非分别插值 value/概率后简单相乘。
            # 例如某像素 10 米概率高、5 米概率低，则相应深度位置的图像特征贡献一强一弱。
            # 深度概率提供几何软权重；weights 是 query 学出的采样点权重，两者不是同一个分布。
            # 同一像素射线上的体素可具有不同深度，深度概率用于区分这些位置的特征贡献。
            if cuda:
                #* 调用原 FoundationSSC DFA3D：depth_score 采样 → weighted attention。
                #* depth 已由循环外的 prepare_attention(value, depth) 打包进 memory，所以这里不再单独传入。
                # memory=(values, distribution, shapes, starts)；distribution 就是按算子布局整理后的深度概率。
                # deform_attention_prepared 内部解包，将 values 和 distribution 一起传给 DFA3D 完成采样。
                # 各 query 块共享 memory，但 locations/weights 不同；准备 memory 不等于已完成概率加权或采样。
                # 算子返回 [B,Q_chunk,heads,C/heads]，flatten(-2) 合并各 head 得到 [B,Q_chunk,C]。
                update = deform_attention_prepared(memory, locations, weights).flatten(-2)
            else:
                # PyTorch 调试路径；当前配置 ops_backend='cuda' 时不执行。
                # 不使用打包的 memory，直接传 value/depth 进行采样，再沿采样点维加权求和。
                samples = sample_depth_weighted(value, depth, locations)
                update = (samples * weights[..., None]).sum(-2).flatten(-2)

            #* ================== 2.5.4.5 视野过滤、残差与 FFN 更新 ==================
            # update=[B,Q_chunk,C]；output 为通道投影，两个 norm 分别接在注意力残差和 FFN 残差之后。
            update = update * visible[:, start:start + self.chunk, None]  # 屏蔽视野外的采样更新；残差/投影偏置/FFN 仍会执行。
            x = self.norm1(q + self.drop(self.output(update)))
            chunks.append(self.norm2(x + self.ffn(x)))
        return torch.cat(chunks, 1)  # 按原候选顺序拼回 [B,Q,C]，作为下一层 CrossLayer 的 query。


class SelfLayer(nn.Module):
    def __init__(self, channels, heads, points, ffn_channels, dropout, chunk, ops_backend='pytorch'):
        super().__init__()
        self.heads, self.points, self.chunk = heads, points, chunk
        self.ops_backend = ops_backend
        #* 原 DeformSelfAttention 将同一 voxel 序列复制为两队列，再平均结果。
        self.offset = nn.Linear(2 * channels, heads * 2 * points * 2)
        self.weights = nn.Linear(2 * channels, heads * 2 * points)
        self.value = nn.Linear(channels, channels)
        self.output = nn.Linear(channels, channels)
        self.norm1, self.norm2 = nn.LayerNorm(channels), nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(channels, ffn_channels), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ffn_channels, channels), nn.Dropout(dropout))
        init_offsets(self.offset, heads, points, queues=2)
        nn.init.zeros_(self.weights.weight); nn.init.zeros_(self.weights.bias)
        for layer in (self.value, self.output):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, query, pos, layout):

        #* ================== 2.5.5.1 全体素特征与二维采样布局 ==================
        #* 全体素自注意力：候选和非候选位置一起更新，不再直接采样图像。
        # query=[B,V,C]，V=X*Y*Z；pos=[1,V,C]，layout 默认 512×512，仅为序列采样布局。
        #* 该二维布局不是物理 BEV，也不是把高度维压缩掉；512*512=128*128*16，体素数量不变。
        b, qn, c = query.shape
        h, w = layout # 512,512 序列采样布局
        #* 三维展平已在外层 coarse.flatten(2).transpose(1,2) 完成：[B,C,X,Y,Z] → [B,V,C]。
        # query 保留该顺序：Z 变化最快，体素 (ix,iy,iz) 的序号 id=(ix*Y+iy)*Z+iz。
        #* 下面 reshape(b,h,w,...) 再将序列按行排成二维表：row=id//w，col=id%w；不是 BEV 投影。
        # 默认 X/Y/Z=128/128/16、h/w=512/512：一行容纳 32 个 Y 位置×16 个高度，一个 X 位置对应 4 行。
        # 例：(0,0,0)→id0→(行0,列0)；(0,0,1)→id1→(行0,列1)；(0,32,0)→id512→(行1,列0)。
        # self.value 只变换通道，不改变体素顺序；后续 permute/reshape 只整理多头布局，没有池化或高度压缩。
        # 通道投影并拆分 head：[B,V,C] → [B*heads,C/heads,h,w]；h/w 是布局尺寸，不是图像尺寸。
        values = self.value(query).reshape(b, h, w, self.heads, c // self.heads).permute(0, 3, 4, 1, 2).reshape(b * self.heads, c // self.heads, h, w)
        cuda = use_cuda(self.ops_backend, values)

        #* ================== 2.5.5.2 共享双队列 memory 与参考坐标 ==================
        #* 双队列不是“做过/没做过交叉注意力”两组：每份 value 都包含全部候选与非候选体素。
        # 输入已由候选位置的交叉注意力特征、非候选位置的 prior(LSS) 特征合成，再整份复制为两队列。
        #* 两队列读取相同内容，但分别预测采样偏移和权重，最后平均更新结果；不保证学出不同分工。
        # 队列数 2 沿用原 DeformSelfAttention 的结构，不代表两个历史帧、两个尺度或两类体素。
        #todo: 双队列整份 value 及其连续布局必须放到分块循环外准备。
        # 旧版每块保存独立 256MiB value，128 块约 32GiB/层；现在共享同一份存储，保留梯度。
        memory = prepare_attention(values.reshape(b, self.heads, c // self.heads, h, w), batch_repeats=2) if cuda else None
        # memory 是四元组 (values, distribution, shapes, starts)，不是单个 Tensor，也不是历史帧缓存。
        # values：[2*B,V,heads,C/heads]，V=h*w；每个样本连续复制两份，供两个队列读取。
        # 默认 B=1、V=262144、heads=8、C=128 时，memory[0] 的 shape 为 [2,262144,8,16]。
        # distribution：None，此处无深度概率；shapes：int64 [1,2]，内容 [[512,512]]（默认布局）。
        # starts：int64 [1]，内容 [0]，表示唯一特征层在展平序列中的起始位置；张量均与 value 同设备。

        # PyTorch 后端下 memory=None，后续直接用上面的 values 做 grid_sample，不读取此四元组。
        # 没有传 depth，memory 的 distribution=None，因此后续调用二维可变形注意力，不走 DFA3D 深度采样。
        # batch_repeats=2 将同一份 value 复制成两队列；不是历史帧缓存，各 query 分块共享已准备的 memory。
        ids = torch.arange(qn, device=query.device)
        # 为展平布局中的每个位置生成归一化中心坐标，后续加可学习二维偏移。
        # ids=[V]；ids%w 是列号、ids//w 是行号，+0.5 取格子中心，再除以宽/高映射到 [0,1]。
        # refs=[V,2]，末维为 (布局横坐标,布局纵坐标)，不是图像像素或物理空间 X/Y；两队列共用这些参考点。
        refs = torch.stack(((ids % w + .5) / w, (ids // w + .5) / h), -1)
        outputs = []

        for start in range(0, qn, self.chunk):
            #* ================== 2.5.5.3 预测各队列的采样偏移和权重 ==================
            # 只分块处理 query，每块仍可从完整 value 布局采样；不是分块隔离体素间交互。
            q = query[:, start:start + self.chunk]  # [B,Q_chunk,C]，这里遍历全部 V 个体素，不只是 proposal 候选。
            inp = torch.cat((q, q + pos[:, start:start + self.chunk]), -1)  # [B,Q_chunk,2C]：内容与带位置编码的内容拼接。

            # offset=[B,Q_chunk,heads,2,P,2]：前一个 2 是队列数，最后一个 2 是布局横/纵方向偏移。
            #* offset：网络预测“往哪里偏”，每个 query/head/队列有 P 个 (Δ列,Δ行)，单位为二维布局格子。
            # 不是米制位移，也不是三维 (Δx,Δy,Δz)；P=self.points 为每个 head、每个队列的采样点数。
            offset = self.offset(inp).reshape(b, -1, self.heads, 2, self.points, 2)
            # refs=[V,2]；偏移除以 [w,h] 后加到参考坐标上，得到归一化 loc，非真实空间米制坐标。
            #* loc：实际“在哪里取特征” = query 自身在二维表中的参考位置 + 归一化偏移，与 offset 同 shape。
            # loc 是由 refs/offset 计算得到，不是另一个网络输出；可能超出 [0,1]，采样时越界部分补零。
            loc = refs[None, start:start + q.shape[1], None, None, None] + offset / q.new_tensor([w, h])
            # 每个 query/head/队列分别对 P 个采样点归一化；两队列共享 value，但可学习不同偏移和权重。
            #* weight：网络预测“各采样点贡献多少”，shape=[B,Q_chunk,heads,2,P]，每组 P 个权重之和为 1。
            # 每个队列的汇聚为 Σ_p weight[p]×在 loc[p] 处插值得到的特征，最后再对两个队列的结果取平均。
            weight = self.weights(inp).reshape(b, -1, self.heads, 2, self.points).softmax(-1)

            #* ================== 2.5.5.4 从体素特征自身采样并汇聚 ==================
            if cuda:
                #* 与原 DeformSelfAttention 一致：两队列合入 batch，各自计算后取平均。
                # 两队列来自同一份当前体素特征，不代表两个历史帧。
                queue_loc = loc.permute(0, 3, 1, 2, 4, 5).reshape(b*2, q.shape[1], self.heads, self.points, 2)
                queue_weight = weight.permute(0, 3, 1, 2, 4).reshape(b*2, q.shape[1], self.heads, self.points)
                # 输出 [B*2,Q_chunk,heads,C/heads]；每个队列先独立对采样点加权汇聚。
                update = deform_attention_prepared(memory, queue_loc, queue_weight)
                update = update.reshape(b, 2, q.shape[1], c).mean(1)  # 合并 head，并对两队列等权平均 → [B,Q_chunk,C]。
            else:
                # PyTorch 调试路径：用 grid_sample 实现采样；当前 CUDA 配置跳过。
                grid = loc.permute(0, 2, 1, 3, 4, 5).reshape(b * self.heads, -1, 2 * self.points, 2)
                sampled = F.grid_sample(values, grid * 2 - 1, align_corners=False)
                sampled = sampled.reshape(b, self.heads, c // self.heads, q.shape[1], 2, self.points).permute(0, 3, 1, 4, 5, 2)
                update = (sampled * weight[..., None]).sum(-2).mean(-2).flatten(-2)

            #* ================== 2.5.5.5 残差、归一化与 FFN 更新 ==================
            # 原 query + 注意力增量，再接 FFN 残差；没有 detach，梯度可回传到输入体素特征及本层参数。
            x = self.norm1(q + self.drop(self.output(update)))
            outputs.append(self.norm2(x + self.ffn(x)))
        return torch.cat(outputs, 1)  # 按原体素顺序拼回 [B,V,C]；三维 [B,C,X,Y,Z] 的恢复由外层 VoxelRefiner 完成。


class VoxelRefiner(nn.Module):
    def __init__(self, voxel_shape, channels=128, cross_layers=3, self_layers=2,
                 heads=8, points=8, ffn_channels=256, dropout=.1,
                 self_layout=(512, 512), query_chunk=2048, ops_backend='pytorch'):
        super().__init__()
        self.voxel_shape = tuple(voxel_shape)
        self.layout = tuple(self_layout)
        if math.prod(self_layout) != math.prod(voxel_shape):
            raise ValueError('self_layout 的面积必须等于 X*Y*Z；默认 512*512=128*128*16')
        if channels % heads or channels % 2 or query_chunk < 1 or min(cross_layers, self_layers) < 1:
            raise ValueError('通道需整除 head 数且为偶数，层数/chunk 必须为正')
        #* 每个体素具有可学习 query；prior 则从 LSS 粗特征生成完整网格的初始特征。
        self.query = nn.Embedding(math.prod(voxel_shape), channels)
        self.row_embed = nn.Embedding(self_layout[0], channels // 2)
        self.col_embed = nn.Embedding(self_layout[1], channels // 2)
        self.camera_embed = nn.Parameter(torch.randn(1, channels) * .02)
        self.level_embed = nn.Parameter(torch.randn(1, channels) * .02)
        self.prior = nn.Sequential(nn.Linear(channels, channels // 2), nn.LayerNorm(channels // 2), nn.LeakyReLU(), nn.Linear(channels // 2, channels))
        args = (channels, heads, points, ffn_channels, dropout, query_chunk, ops_backend)
        self.cross = nn.ModuleList(CrossLayer(*args) for _ in range(cross_layers))
        self.self_attention = nn.ModuleList(SelfLayer(*args) for _ in range(self_layers))

    def forward(self, context, depth_prob, coarse, proposal, geometry, cam_params, depth_bound):

        #* ================== 2.5.1 体素中心投影与采样参考位置 ==================
        # 默认 context=[B,1,128,48,160]、depth_prob=[B,112,48,160]。
        # coarse=[B,C,X,Y,Z]，proposal=[B,1,X,Y,Z]；以下 V=X*Y*Z=128*128*16。
        b, c = coarse.shape[:2]
        # V 是全部体素的数量，不是相机数：V=X*Y*Z，默认 128*128*16=262144。
        # geometry.centers 已在 geometry.py 中按 空间起点+(网格索引+0.5)*体素尺寸 换算为米制中心坐标。
        points = geometry.centers.reshape(1, 1, -1, 3).expand(b, 1, -1, -1)  # [B,1,V,3]：batch、左相机维、体素数、(x,y,z)。
        # project 先返回 [B,1,V,3]；[:,0] 取左相机并去掉相机维，不是取第一个体素。
        uvd = project(points, cam_params)[:, 0]  # [B,V,3]：每个体素中心投影后的 (u,v,相机 Z 深度)，u/v 为像素，Z 为米。
        # 将像素坐标和米制深度转换为交叉注意力使用的归一化参考位置 ref=[B,V,3]。
        ref = torch.stack((uvd[..., 0] / geometry.input_size[1], uvd[..., 1] / geometry.input_size[0],
                           (uvd[..., 2] - depth_bound[0]) / (depth_bound[1] - depth_bound[0])), -1)
        #* visible 只检查正深度和图像边界，不判断遮挡，也未限制深度必须落在 depth_bound 内。
        visible = (uvd[..., 2] > 1e-5) & (ref[..., :2] > 1e-5).all(-1) & (ref[..., :2] < 1 - 1e-5).all(-1)

        #* ================== 2.5.2 完整体素序列与位置编码 ==================
        #* 将 128×128×16 个三维体素展平，再按顺序排成 512×512 的二维特征表，供自注意力采样。
        #* 只是重新排列，未压缩/求和高度维；这张表不是 BEV 俯视图，行列也不是物理 X/Y 坐标。
        # 三维展开顺序为 Z 最快：体素 (0,0,0)/(0,0,1) 分别对应二维表的 (行0,列0)/(行0,列1)。
        # 即真实空间的高度变化在表中可能表现为列变化；表中邻近不保证等于三维空间最近邻。
        # row/col embedding 是可学习的表格行列位置编码，不是以米为单位的三维坐标。
        # 默认各编码 C/2=64 维；展开后 rows/cols 均为 [512,512,64]，拼接再展平得到 pos=[1,V,128]。
        rows = self.row_embed.weight[:, None].expand(-1, self.layout[1], -1)
        cols = self.col_embed.weight[None].expand(self.layout[0], -1, -1)
        pos = torch.cat((cols, rows), -1).reshape(1, -1, c)
        flattened = coarse.flatten(2).transpose(1, 2)  # [B,V,C]，Z 轴最快；与 proposal/centers 的展开顺序一致。
        results = []
        #* 各样本候选数不同，逐样本 refine，避免原实现把 batch 与 voxel 索引混在一起。
        for i in range(b):
            #* ================== 2.5.3 筛选候选并初始化整张体素网格 ==================
            chosen = proposal[i].flatten().bool()  # [V]：预测深度命中的候选，不等于 visible 视野 mask。
            idx = chosen.nonzero().flatten()  # [Q]：当前样本候选索引，Q 随样本变化。
            volume = self.prior(flattened[i:i+1])  # [1,V,C]：可学习 MLP 处理 LSS 特征，不是 GT 或固定常量。
            if idx.numel():
                #* ================== 2.5.4 候选体素与图像做交叉注意力 ==================
                #* 初始 query = 体素位置相关的可学习先验 + 当前图像/场景相关的 LSS 特征。
                # self.query.weight：每个体素位置独有、不同样本共享的可学习 embedding，训练更新，不由当前图像直接生成。
                # flattened：当前样本的 context 经深度概率加权、几何投影汇聚得到，随图像/场景变化。
                # idx 为当前样本的 Q 个候选体素索引；两项均为 [1,Q,C]，逐元素相加而非拼接。
                q = self.query.weight[None, idx] + flattened[i:i+1, idx]  # [1,Q,C]：后续作为交叉注意力 query，从图像继续采样信息。

                image = context[i:i+1, 0] + (self.camera_embed + self.level_embed)[..., None, None]
                # image=[1,C,h,w]，默认 [1,128,48,160]；取当前样本左图 context，加上广播到所有像素的相机/尺度 embedding。
                #* 此处是单尺度图像交互：前端虽融合过多尺度信息，但传入的 context 只有一个空间分辨率。
                # level_embed 只编码这一个尺度，不代表已有多尺度采样；多尺度交互需额外提供各尺度特征和深度概率。
                #* self.cross 是多层交叉注意力，不是多个尺度：各层依次更新 q，始终读取同一份 image/depth_prob。
                # 输入依次为：q=[1,Q,C] 候选特征；image=[1,C,h,w] 图像特征；depth_prob=[1,D,h,w] 深度概率；
                # ref=[1,Q,3] 候选体素的归一化 (u,v,depth) 参考位置；visible=[1,Q] 投影视野 mask。
                # 每层围绕参考位置学习采样偏移及权重，输出仍为 [1,Q,C]，供下一层继续细化。
                for layer in self.cross:
                    q = layer(q, image, depth_prob[i:i+1], ref[i:i+1, idx], visible[i:i+1, idx])

                volume = volume.index_copy(1, idx, q)  # 候选位置换成更新后的 q，非候选位置保留 prior 特征。
            #* ================== 2.5.5 全体素自注意力细化与形状恢复 ==================
            #* 空 proposal 不随机造点，也不将所有网格当作候选；用 LSS prior 继续扩散。
            for layer in self.self_attention:
                volume = layer(volume, pos, self.layout)

            results.append(volume.transpose(1, 2).reshape(1, c, *self.voxel_shape))  # [1,C,X,Y,Z]，恢复完整三维网格。
        # refined=[B,C,X,Y,Z]；第二项 [B] 是视野内体素中心数量，不是候选 Q，也不是 LSS 的 hits。
        return torch.cat(results), visible.sum(1)
