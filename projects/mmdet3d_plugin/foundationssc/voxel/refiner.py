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
        b, c, h, w = context.shape
        value = self.value(context.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).reshape(b, self.heads, c // self.heads, h, w)
        cuda = use_cuda(self.ops_backend, value)
        #todo: 原先每块都 repeat 深度分布并由 autograd 保存；改为每层准备一次、各块共享。
        memory = prepare_attention(value, depth) if cuda else None
        chunks = []
        for start in range(0, query.shape[1], self.chunk):
            q = query[:, start:start + self.chunk]
            uv = self.offset_uv(q).reshape(b, -1, self.heads, self.points, 2)
            dd = self.offset_d(q).reshape(b, -1, self.heads, self.points, 1)
            offset = torch.cat((uv, dd), -1) / q.new_tensor([w, h, depth.shape[1]])
            locations = reference[:, start:start + self.chunk, None, None] + offset
            weights = self.weights(q).reshape(b, -1, self.heads, self.points).softmax(-1)
            if cuda:
                #* 调用原 FoundationSSC DFA3D：depth_score 采样 → weighted attention。
                update = deform_attention_prepared(memory, locations, weights).flatten(-2)
            else:
                samples = sample_depth_weighted(value, depth, locations)
                update = (samples * weights[..., None]).sum(-2).flatten(-2)
            update = update * visible[:, start:start + self.chunk, None]
            x = self.norm1(q + self.drop(self.output(update)))
            chunks.append(self.norm2(x + self.ffn(x)))
        return torch.cat(chunks, 1)


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
        b, qn, c = query.shape
        h, w = layout
        values = self.value(query).reshape(b, h, w, self.heads, c // self.heads).permute(0, 3, 4, 1, 2).reshape(b * self.heads, c // self.heads, h, w)
        cuda = use_cuda(self.ops_backend, values)
        #todo: 双队列整份 value 及其连续布局必须放到分块循环外准备。
        # 旧版每块保存独立 256MiB value，128 块约 32GiB/层；现在共享同一份存储，保留梯度。
        memory = prepare_attention(values.reshape(b, self.heads, c // self.heads, h, w), batch_repeats=2) if cuda else None
        ids = torch.arange(qn, device=query.device)
        refs = torch.stack(((ids % w + .5) / w, (ids // w + .5) / h), -1)
        outputs = []
        for start in range(0, qn, self.chunk):
            q = query[:, start:start + self.chunk]
            inp = torch.cat((q, q + pos[:, start:start + self.chunk]), -1)
            offset = self.offset(inp).reshape(b, -1, self.heads, 2, self.points, 2)
            loc = refs[None, start:start + q.shape[1], None, None, None] + offset / q.new_tensor([w, h])
            weight = self.weights(inp).reshape(b, -1, self.heads, 2, self.points).softmax(-1)
            if cuda:
                #* 与原 DeformSelfAttention 一致：两队列合入 batch，各自计算后取平均。
                queue_loc = loc.permute(0, 3, 1, 2, 4, 5).reshape(b*2, q.shape[1], self.heads, self.points, 2)
                queue_weight = weight.permute(0, 3, 1, 2, 4).reshape(b*2, q.shape[1], self.heads, self.points)
                update = deform_attention_prepared(memory, queue_loc, queue_weight)
                update = update.reshape(b, 2, q.shape[1], c).mean(1)
            else:
                grid = loc.permute(0, 2, 1, 3, 4, 5).reshape(b * self.heads, -1, 2 * self.points, 2)
                sampled = F.grid_sample(values, grid * 2 - 1, align_corners=False)
                sampled = sampled.reshape(b, self.heads, c // self.heads, q.shape[1], 2, self.points).permute(0, 3, 1, 4, 5, 2)
                update = (sampled * weight[..., None]).sum(-2).mean(-2).flatten(-2)
            x = self.norm1(q + self.drop(self.output(update)))
            outputs.append(self.norm2(x + self.ffn(x)))
        return torch.cat(outputs, 1)


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
        b, c = coarse.shape[:2]
        points = geometry.centers.reshape(1, 1, -1, 3).expand(b, 1, -1, -1)
        uvd = project(points, cam_params)[:, 0]
        ref = torch.stack((uvd[..., 0] / geometry.input_size[1], uvd[..., 1] / geometry.input_size[0],
                           (uvd[..., 2] - depth_bound[0]) / (depth_bound[1] - depth_bound[0])), -1)
        visible = (uvd[..., 2] > 1e-5) & (ref[..., :2] > 1e-5).all(-1) & (ref[..., :2] < 1 - 1e-5).all(-1)
        rows = self.row_embed.weight[:, None].expand(-1, self.layout[1], -1)
        cols = self.col_embed.weight[None].expand(self.layout[0], -1, -1)
        pos = torch.cat((cols, rows), -1).reshape(1, -1, c)
        flattened = coarse.flatten(2).transpose(1, 2)
        results = []
        #* 各样本候选数不同，逐样本 refine，避免原实现把 batch 与 voxel 索引混在一起。
        for i in range(b):
            chosen = proposal[i].flatten().bool()
            idx = chosen.nonzero().flatten()
            volume = self.prior(flattened[i:i+1])
            if idx.numel():
                q = self.query.weight[None, idx] + flattened[i:i+1, idx]
                image = context[i:i+1, 0] + (self.camera_embed + self.level_embed)[..., None, None]
                for layer in self.cross:
                    q = layer(q, image, depth_prob[i:i+1], ref[i:i+1, idx], visible[i:i+1, idx])
                volume = volume.index_copy(1, idx, q)
            #* 空 proposal 不随机造点，也不将所有网格当作候选；用 LSS prior 继续扩散。
            for layer in self.self_attention:
                volume = layer(volume, pos, self.layout)
            results.append(volume.transpose(1, 2).reshape(1, c, *self.voxel_shape))
        return torch.cat(results), visible.sum(1)
