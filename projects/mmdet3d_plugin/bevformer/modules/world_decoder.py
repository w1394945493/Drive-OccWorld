import numpy as np
import math
import torch
import torch.nn as nn
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class WorldDecoder(TransformerLayerSequence):

    """
    Decoder of End-to-End prediction transformer.
    Attention with both self and cross.

    对应论文 3.2 的 World Decoder W_D：
    learnable BEV queries 依次经过 deformable self-attention、
    temporal cross-attention、action conditional cross-attention 和 FFN，
    输出未来帧 BEV embeddings。
    """

    def __init__(self, *args,
                 return_intermediate=False,
                 keep_idx=(2,),
                 **kwargs):

        super().__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.keep_idx = keep_idx
        # remove latent rendering in previous layers.
        #* 论文 Future Forecasting with World Decoder：
        #* 多层 decoder 逐层更新 future BEV query，return_intermediate=True 时返回每层结果做辅助监督。
        for lid, layer in enumerate(self.layers):
            if lid not in self.keep_idx:
                # if this is not the last layer, and remove operations in previous layers.
                if getattr(layer, 'latent_render', None):
                    del layer.latent_render
                    layer.operation_order = ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')

    @auto_fp16()
    def forward(self,
                bev_query,
                prev_feats,
                *args,
                tgt_points=None,
                ref_points=None,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): BEV queries with shape as [b, bev_h * bev_w, dims]
            prev_feats (Tensor): previous BEV features with shape as
                [b, num_frames, bev_h * bev_w, dims]
            bev_pos (Tensor): bev positional embedding with shape as
                [bs, bev_h * bev_w, dims]
            tgt_points: positions of points in deformable self-attention layers.
                positions of query points in reference frame coordinates with shape
                as [bs, tgt_bev_h * tgt_bev_w, 2]
            ref_points: positions of points in deformable cross-attention layers.
                positions of query points in previous frame coordinates with shape
                as [bs, ref_bev_h * ref_bev_w, 2]
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = bev_query
        intermediate = []

        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                prev_feats,
                *args,
                bev_pos=bev_pos,
                tgt_points=tgt_points,
                ref_points=ref_points,
                bev_h=bev_h,
                bev_w=bev_w,
                **kwargs)

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        else:
            return output.unsqueeze(0)


@TRANSFORMER_LAYER.register_module()
class PredictionTransformerLayer(MyCustomBaseTransformerLayer):
    """Implements decoder layer in End-to-End future point
    cloud prediction network.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 latent_render=None,
                 **kwargs):
        super().__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False

    def forward(self,
                query,
                prev_feats=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                tgt_points=None,
                ref_points=None,
                bev_h=None,
                bev_w=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [bs, num_queries, dims]
            prev_feats (Tensor): The key / value tensor in cross-attention
                with shape [bs, num_frames, num_keys, dims]
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: updated future BEV query with shape [bs, num_queries, embed_dims].
        """
        #* ================== 1. 输入含义与注意力 mask 准备 ==================
        #* 本层更新同一未来时刻的 BEV 特征，不是在此循环预测多个未来帧。
        # query=[B,Q,C]，Q=bev_h*bev_w；prev_feats=[B,T,Q,C] 为可读取的 memory。
        # T 是 memory 帧数，可包含历史/当前帧或此前预测帧，由外层递推维护。
        # tgt_points=[B,Q,2]：目标 BEV 自身的归一化坐标，用于 query 内部自注意力。
        # ref_points=[B,Q,T,2]：同一目标位置经位姿变换后在各 memory 中的采样参考点。
        # 此处不重新计算位姿，也不预先 warp 整张 memory；后续注意力按参考点采样。
        # bev_pos 是实际传入注意力的位置编码；签名中的 query_pos/key_pos 在此未直接使用。
        # 三个 index 分别索引 attention/norm/FFN 子模块；identity 用于残差连接。
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        #* ================== 2. 采样坐标与 memory 展平布局 ==================
        # 自注意力只读取目标 query 这一张 BEV：补 level 维，[B,Q,2] → [B,Q,1,2]。
        tgt_points = tgt_points.unsqueeze(2)
        # 交叉注意力通常已有 T 个参考点；单 memory 输入 [B,Q,2] 时补成 [B,Q,1,2]。
        if len(ref_points.shape) != 4:
            ref_points = ref_points.unsqueeze(2)
        # 各 memory 帧的网格尺寸必须一致，随后按帧依次拼成一条 token 序列。
        bs, num_frames, prev_token_num, prev_dims = prev_feats.shape
        assert prev_feats.shape[2] == bev_h * bev_w

        # 自注意力：1 个二维 level，大小 H×W，序列起始位置为 0。
        self_attn_spatial_shapes = torch.tensor(
            [[bev_h, bev_w]], device=query.device)
        self_attn_level_start_index = torch.tensor([0], device=query.device)

        # 交叉注意力：将 T 帧视为 T 个 level（这里不是图像金字塔尺度）。
        # spatial_shapes=[T,2]；各帧起始下标为 0,Q,2Q,...，用于定位展平 memory。
        cross_attn_spatial_shapes = torch.tensor(
            [[bev_h, bev_w] for i in range(num_frames)], device=query.device)
        cross_attn_level_start_index = torch.cat((cross_attn_spatial_shapes.new_zeros(
            (1,)), cross_attn_spatial_shapes.prod(1).cumsum(0)[:-1]))
        prev_feats = prev_feats.view(bs, num_frames * prev_token_num, prev_dims)

        #* ================== 3. 按配置顺序更新未来 query ==================
        # 常用顺序：self_attn → norm → cross_attn → norm → cross_attn_action → norm → ffn → norm。
        # 循环项是本层内的运算名，不是未来时间步；只有配置中的分支才会执行。
        for layer in self.operation_order:
            #* ================================================================
            #* 3.1 自注意力：目标 BEV query 之间交换信息，使用 tgt_points。
            #* 当前 PredictionMSDeformableAttention 是稀疏可变形自注意力，不是 Q×Q 全局密集注意力。
            # 每个 query/head 只读取 P 个学习采样点；偏移无固定窗口限制，可读取远处，但不保证覆盖全局。
            # key/value 传 None，由注意力模块使用 query 自身；在参考点附近学习采样偏移。
            if layer == 'self_attn':
                query = self.attentions[attn_index](
                    query,
                    None,
                    None,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=tgt_points,
                    spatial_shapes=self_attn_spatial_shapes,
                    level_start_index=self_attn_level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                # 按 operation_order 在对应位置归一化，保持 [B,Q,C] 不变。
                query = self.norms[norm_index](query)
                norm_index += 1

            #* ================================================================
            #* 3.2 跨帧注意力：未来 query 从展平后的 memory[B,T*Q,C] 读取场景特征。
            # key/value 均为 prev_feats；围绕已对齐的 ref_points 学习偏移并加权聚合。
            # 更新的是 query，不直接改写 prev_feats；未来各帧递推由外层负责。
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    prev_feats,
                    prev_feats,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    # use ref_points coordinates in previous frames.
                    reference_points=ref_points,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=cross_attn_spatial_shapes,
                    level_start_index=cross_attn_level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query

            #* ================================================================
            #* 3.3 动作条件交互：将上游编码的 action_condition 作为 key/value。
            # 条件具体包含哪些 can_bus/command/velocity/plan_traj 信息由上游配置决定。
            # [B,C] → [B,1,C]，作为条件 token；此处不读取原始 CAN bus，也不输出占据类别。
            elif layer == 'cross_attn_action':
                action_condition = kwargs['action_condition'].unsqueeze(1)
                query = self.attentions[attn_index](
                    query,
                    action_condition,
                    action_condition,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'latent_render':
                # 可选分支：恢复 [B,H,W,C] 做 latent_render，再还原 [B,Q,C]。
                # 常用 operation_order 未包含此项时不执行。
                bs, token_num, embed_dim = query.shape
                query = self.latent_render(query.view(bs, bev_h, bev_w, embed_dim))
                query = query.view(bs, token_num, embed_dim)

            elif layer == 'ffn':
                #* 3.4 FFN：逐 token 做通道变换与残差更新，不改变 BEV 网格数量。
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        #* ================== 4. 返回本层未来 BEV 特征 ==================
        # [B,Q,C]，交给下一层 WorldDecoder 或外层占据解码；不是 occupancy logits。
        return query


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class PlanDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default: `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(PlanDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    def forward(self,
                pose_queries,
                bev_feats,
                prev_pose=None,
                bev_pos=None,
                *args,
                **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(bs, num_query, embed_dims)`.
        Returns:
            Tensor: Results with shape [bs, num_query, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, bs, num_query, embed_dims].
        """
        output = pose_queries
        intermediate = []

        for lid, layer in enumerate(self.layers):
            output = layer(
                pose_queries,
                bev_feats,
                prev_pose=prev_pose,
                bev_pos=bev_pos,
                *args,
                **kwargs)

            pose_queries = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


@TRANSFORMER_LAYER.register_module()
class PlanTransformerLayer(MyCustomBaseTransformerLayer):
    """Implements decoder layer in End-to-End future plan prediction network.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super().__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False

    def forward(self,
                pose_queries,
                bev_feats,
                prev_pose=None,
                bev_pos=None,
                *args,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            pose_queries (Tensor): The input query with shape
                [bs, 1, dims]
            bev_feats (Tensor): The key / value tensor in cross-attention
                with shape [bs, bev_h * bev_w, dims]
            prev_pose (Tensor): pose feats of the (previous)+current frame with shape
                [bs, 2, dims]
            bev_pos (Tensor): The positional encoding for bev_feats.
                with shape [bs, bev_h * bev_w, dims]
        Returns:
            Tensor: forwarded results with shape [bs, 1, embed_dims].
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0

        query=pose_queries
        identity = query

        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':
                query = self.attentions[attn_index](
                    query,
                    prev_pose,
                    prev_pose,
                    identity if self.pre_norm else None,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # Spatial cross-attention for query features from current bev feats.
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    bev_feats,
                    bev_feats,
                    identity if self.pre_norm else None,
                    key_pos=bev_pos,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query


from mmcv.utils import ConfigDict, build_from_cfg, deprecated_api_warning, to_2tuple
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.cnn import xavier_init, constant_init
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, \
    MultiScaleDeformableAttnFunction_fp16
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
@ATTENTION.register_module()
class PredictionMSDeformableAttention(BaseModule):
    """An attention module used in Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        assert self.batch_first
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiScaleDeformableAttention')
    def forward(self,
                query,  # [B,Q,C]，以下按当前 batch_first=True 路径标注。
                key=None,  # [B,S,C] 或 None；本实现不使用 key 计算相似度。
                value=None,  # [B,S,C]；自注意力默认使用 query，此时 S=Q。
                identity=None,  # [B,Q,C]，残差；默认使用原始 query。
                query_pos=None,  # [B,Q,C] 或可广播到该形状的位置编码。
                key_padding_mask=None,  # [B,S]，True 对应的 value 置零。
                reference_points=None,  # [B,Q,L,2]；框参考模式为 [B,Q,L,4]。
                spatial_shapes=None,  # [L,2]，各 level 的 (H_l,W_l)。
                level_start_index=None,  # [L]，各 level 在 S 维中的起始下标。
                flag='decoder',
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key, embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes, with shape as
                [bs, num_query, 2]
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
             Tensor: [bs, num_query, embed_dims] in the WorldDecoder batch_first=True path.
        """

        #* ================== 1. 区分自注意力/跨帧注意力，准备残差 ==================
        #* 两种注意力复用本类，但由外层创建两个独立实例，参数不共享。
        # 自注意力：value=None → 读取 query 自身；reference_points 是 tgt_points。
        # 跨帧注意力：value=prev_feats[B,T*H*W,C]；reference_points 是对齐后的 ref_points。
        # 下文记 B=batch、Q=query数、S=value数、M=head数、L=level数、P=每level采样点数、C=通道数。
        #* 这里不是标准 QK^T 注意力：key 保留兼容接口，后续不参与打分。
        # 采样偏移和权重直接由 query 预测；实际被读取的内容来自 value。
        if value is None:
            value = query  # [B,Q,C]，自注意力 S=Q。
        if key is None:
            key = query  # [B,Q,C]，仅兼容接口。

        if identity is None:
            identity = query  # [B,Q,C]。
        # 位置编码仅加入生成偏移/权重的 query；上面保留的 value/identity 不随之相加。
        if query_pos is not None:
            query = query + query_pos  # [B,Q,C]，shape 不变。

        bs, num_query, _ = query.shape  # B,Q,C。
        bs, num_value, _ = value.shape  # B,S,C。
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        #* ================== 2. value 通道投影与多头拆分 ==================
        # S 必须等于各 level 的 H_l*W_l 之和；跨帧场景把各帧当作 level，而非图像尺度。
        value = self.value_proj(value)  # [B,S,C]，只变换通道，不改变 token 顺序。
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)  # mask=[B,S,1] → value=[B,S,C]。
        value = value.view(bs, num_value, self.num_heads, -1)  # [B,S,M,C/M]。

        #* ================== 3. query 预测采样偏移与注意力权重 ==================
        # sampling_offsets=[B,Q,M,L,P,2]：每个 query/head/level 的 P 个二维偏移。
        #* 自注意力读取目标 BEV 时，每个 head 仅采样 P 个位置，而非遍历全部 Q 个 token。
        # 点参考模式下偏移以对应 BEV 网格单元为单位；不是三维 XYZ，也不是米。
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)  # [B,Q,M*L*P*2] → [B,Q,M,L,P,2]。
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)  # [B,Q,M*L*P] → [B,Q,M,L*P]。
        #* 每个 query/head 联合对 L*P 个位置归一化，跨帧时也会分配不同帧的贡献。
        # 自注意力也只对采样点分配权重，不形成标准全局自注意力的 Q×Q 相关矩阵。
        # 不是每帧单独 softmax；权重来自 query，而非与每个 memory token 逐一计算相似度。
        attention_weights = attention_weights.softmax(-1)  # [B,Q,M,L*P]，最后一维概率和为 1。

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)  # [B,Q,M,L,P]。

        #* ================== 4. 几何参考点 + 学习偏移 → 实际采样位置 ==================
        #* 位姿对齐已在外层完成；本模块在参考点周围进一步学习从哪里读取信息。
        #* sampling_offsets 没有固定窗口/半径约束，可跨区域采样；不等于对全局所有位置做注意力。
        # reference + offset 的写法不保证偏移很小；越界按后续零填充采样规则处理。
        # reference_points=[B,Q,L,2]，sampling_locations=[B,Q,M,L,P,2]。
        # 坐标顺序为布局 (x,y)/(宽,高)；不是物理 LiDAR 三维坐标。
        if reference_points.shape[-1] == 2:
            # spatial_shapes 存 (H,W)，所以除数要换为 (W,H)，将网格偏移归一化。
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)  # [L,2]，每项为 (W_l,H_l)。
            # 广播：[B,Q,1,L,1,2] + [B,Q,M,L,P,2] / [1,1,1,L,1,2]。
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]  # [B,Q,M,L,P,2]。
        elif reference_points.shape[-1] == 4:
            # 通用框参考分支：(中心x,中心y,宽,高)，按框尺寸缩放偏移。
            # 当前 WorldDecoder 传二维点，通常不走此分支。
            # 中心/宽高均为 [B,Q,1,L,1,2]，与偏移 [B,Q,M,L,P,2] 广播。
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5  # [B,Q,M,L,P,2]。
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        #* ================== 5. 从 value 采样并加权汇聚 ==================
        #* 每个 query/head 在各 level 的 P 个位置做二维双线性采样，再按权重求和。
        # 因此无需构建 Q×S 的全连接注意力矩阵；学习后的位置可能越界，不会在此 clamp。
        # 越界区域按零填充规则采样；得到 [B,Q,C]，已合并各 head 的输出。
        # Input_shapes:
        #   * value: bs, num_value, num_heads, embed // num_heads.
        #       multi-level features are stacked at the {num_value} dimension
        #       and are split at this dimension during inference.
        #   * spatial_shapes: spatial shapes of each feature map.
        #       [-1, 2], where -1 means num_levels.
        #   * sampling_locations: The location of sampled points, with shape
        #       as [bs ,num_queries, num_heads, num_levels, num_points, 2]
        #       0-1 position at the value map scale.
        #   * attention_weights: The weight of sampled points with shape as
        #       [bs ,num_queries, num_heads, num_levels, num_points].
        if torch.cuda.is_available() and value.is_cuda:

            # GPU 张量走 CUDA 扩展；此处两个 dtype 分支均选择 fp32 包装器，保持原实现。
            # using fp16 deformable attention is unstable because it performs many sum operations
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,  # [B,S,M,C/M], [L,2], [L], [B,Q,M,L,P,2]。
                attention_weights, self.im2col_step)  # 权重 [B,Q,M,L,P]；输出 [B,Q,C]。
        else:
            # 非 CUDA 路径：用 PyTorch 采样实现相同的汇聚流程。
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)  # 输出 [B,Q,C]，与 CUDA 路径一致。

        #* ================== 6. 输出投影 + Dropout + 残差 ==================
        # 当前 WorldDecoder 使用 batch_first=True，输出与 identity 均为 [B,Q,C]。
        # 本模块不做 LayerNorm/FFN，二者由外层 PredictionTransformerLayer 继续执行。
        output = self.output_proj(output)  # [B,Q,C] → [B,Q,C]。

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)  # [B,Q,C] → [Q,B,C]；当前配置不走此分支。
        return self.dropout(output) + identity  # 当前 batch_first=True：返回 [B,Q,C]。
