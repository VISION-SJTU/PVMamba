import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat
from torch.nn.init import xavier_uniform_, constant_

import math
import copy
try:
    from utils import PatchMerging, PatchEmbeding, Mlp
except:
    from .utils import PatchMerging, PatchEmbeding, Mlp
from fvcore.nn import flop_count, parameter_count

class tTensor(torch.Tensor):
    @property
    def shape(self):
        shape = super().shape
        return tuple([int(s) for s in shape])


to_ttensor = lambda *args: tuple([tTensor(x) for x in args]) if len(args) > 1 else tTensor(args[0])


def spatial_adaptive_sampling_core_pytorch(value, value_spatial_shapes, sampling_locations):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    # sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode='bilinear', padding_mode='zeros',
                                          align_corners=False)
    #  sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    # feature_tree_sampling = (torch.stack(sampling_value_list, dim=-2).flatten(-2)).view(N_, M_ * D_, Lq_)
    feature_tree_sampling = sampling_value_l_.contiguous()

    return feature_tree_sampling


class spatial_adaptive_operator(nn.Module):
    def __init__(self, d_model, n_points=4, points_type=0):
        super().__init__()
        self.n_points = n_points
        self.im2col_step = 64

        self.n_heads = 1
        self.n_levels = 1
        self.points_type = points_type

        if self.points_type == 2:
            self.sampling_offsets = None
        else:
            self.sampling_offsets = nn.Linear(d_model, n_points * 2)

        self._reset_parameters()

    def _reset_parameters(self):
        if self.sampling_offsets is not None:
            constant_(self.sampling_offsets.weight.data, 0.)
            thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1,
                                                                                                                  self.n_levels,
                                                                                                                  self.n_points,
                                                                                                                  1)
            for i in range(self.n_points):
                grid_init[:, :, i, :] *= i + 1
            with torch.no_grad():
                self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, x, deltaA, H=None, W=None, points=None):
        B, N, C = x.shape
        spatial_shape = [(H, W)]

        spatial_shape = torch.as_tensor(spatial_shape, dtype=torch.long, device=x.device)
        valid_ratio = torch.ones(size=(B, 1, 2), dtype=torch.float, device=x.device)

        if self.points_type == 0:
            reference_points = self.get_reference_points(spatial_shape, valid_ratio, device=x.device)
            sampling_offsets = self.sampling_offsets(x).view(B, N, 1, 1, self.n_points, 2)
            offset_normalizer = torch.stack([spatial_shape[..., 1], spatial_shape[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]

        elif self.points_type == 1:
            reference_points = points
            sampling_offsets = self.sampling_offsets(x).view(B, N, 1, 1, self.n_points, 2)
            offset_normalizer = torch.stack([spatial_shape[..., 1], spatial_shape[..., 0]], -1)
            sampling_locations = reference_points + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        else:
            sampling_locations = points

        feature_in = x

        feature_weight_in = torch.cat([feature_in, deltaA], dim=-1)
        feature_weight_tree_sampling = spatial_adaptive_sampling_core_pytorch(feature_weight_in.unsqueeze(2).contiguous(), spatial_shape, sampling_locations)  # b, c, n, point
        feature_tree_sampling, weight_tree_sampling = feature_weight_tree_sampling[:,:C, ], feature_weight_tree_sampling[:, C:, ]
        feature_aggre_other = feature_tree_sampling * (weight_tree_sampling - deltaA.transpose(1, 2).unsqueeze(-1))  # b,c,n,points
        feature_in = feature_aggre_other.mean(-1).transpose(1, 2)

        out = feature_in.contiguous()
        return out, sampling_locations


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., **kwargs):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.inner_dim = inner_dim


    def forward(self, x, H=None, W=None, points=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), None


class mamba_op(nn.Module):
    def __init__(
        self,
        d_model,
        d_conv=3,
        conv_init=None,
        expand=2,
        headdim=64,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="silu",
        bias=False,
        conv_bias=True,
        chunk_size=256,
        use_mem_eff_path=False,
        layer_idx=None,
        device=None,
        dtype=None,
        d_state = 64,
        pattern='LoDSA',
        blk_idx=0,
        spatial_flag=0,
        **kwargs
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.headdim = headdim
        self.d_state = d_state
        if ngroups == -1:
            ngroups = self.d_inner // self.headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        # projection: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, int(d_in_proj), bias=bias, **factory_kwargs) #

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state

        self.conv2d = nn.Conv2d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            groups=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True


        # A D parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.d_inner, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # Norm layer
        self.norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.pattern = pattern
        if self.pattern == 'LoDSA':
            k_p = 3
            self.pool = torch.nn.AvgPool2d(kernel_size=k_p, stride=1, padding=3 // 2, count_include_pad=False)
        elif self.pattern == 'SaDSA':
            self.sa_op = spatial_adaptive_operator(d_model=self.d_inner, n_points=4, points_type=spatial_flag)
        self.kwargs = kwargs

    def ssm_op(self, x, dt, A, B, C, D, H=None, W=None, points=None):
        '''
        x: (B, L, H, D)
        dt: (B, L, nheads)
        A: (nheads) or (d_inner, d_state)
        B: (B, L, d_state)
        C: (B, L, d_state)
        D: (nheads)
        '''
        batch, seqlen, head, dim = x.shape
        dstate = B.shape[2]
        x_in = x.permute(0, 2, 1, 3)  # (B, H, L, D)
        dt = dt.permute(0, 2, 1)  # (B, H, L)
        dA = dt.unsqueeze(-1) * A.view(1, head, 1, dim).repeat(batch, 1, seqlen, 1)  # B H L 1

        B = B.view(batch, 1, seqlen, dstate)  # (B, 1, L, D)

        if self.pattern =='LoDSA':
            deltaA_x = dA * x_in
            deltaA_x_HW = deltaA_x.view(batch, head, H, -1, dim).permute(0, 1, 4, 2, 3).reshape(batch, -1, H, W)
            deltaA_x_pool = self.pool(deltaA_x_HW)
            deltaA_x = deltaA_x_pool.flatten(2).view(batch, head, dim, seqlen).permute(0, 1, 3, 2).contiguous()
        else:
            x_in_nh = x_in.permute(0, 1, 3, 2).reshape(batch, -1, seqlen).permute(0, 2, 1)
            dA_nh = dA.permute(0, 1, 3, 2).reshape(batch, -1, seqlen).permute(0, 2, 1)
            deltaA_x_nh, points = self.sa_op(x=x_in_nh, deltaA=dA_nh, H=H, W=W, points=points)
            deltaA_x = deltaA_x_nh.view(batch, -1, seqlen, dim).contiguous()


        if self.ngroups == 1:
            h = B.transpose(-2, -1) @ deltaA_x  # (B, H, dstate, D)
            C = C.view(batch, 1, seqlen, dstate)
            Ch = C @ h # (B, H, L, D)
            y = Ch + x_in * D.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)
            y = y.permute(0, 2, 1, 3)  # (B, L, H, D)
        else:
            assert head % self.ngroups == 0
            dstate = dstate // self.ngroups
            B = B.view(batch, 1, seqlen, self.ngroups, dstate).permute(0, 1, 3, 2, 4)  # (B, 1, g, L, dstate)
            deltaA_x = deltaA_x.view(batch, head//self.ngroups, self.ngroups, seqlen, dim)  # (B, H//g, g, L, D)
            C = C.view(batch, 1, seqlen, self.ngroups, dstate).permute(0, 1, 3, 2, 4)  # (B, 1, g, L, dstate)

            h = B.transpose(-2, -1) @ deltaA_x  # (B, H//g, g, dstate, D)
            Ch = C @ h  # (B, H//g, g, L, D)
            Dx = (x_in * D.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)).view(batch, head//self.ngroups, self.ngroups, seqlen, dim)  # (B, H//g, g, L, D)
            y = Ch + Dx  # (B, H//g, g, L, D)
            y = y.permute(0, 3, 1, 2, 4).flatten(2, 3).reshape(batch, seqlen, head, dim)  # (B, L, H, D)

        return y.contiguous(), points


    def forward(self, u, H, W, points):
        """
        u: (B,C,H,W)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)

        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
        )
        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
        assert self.activation in ["silu", "swish"]


        #2D Convolution
        xBC = xBC.view(batch, H, W, -1).permute(0, 3, 1, 2).contiguous()
        xBC = self.act(self.conv2d(xBC))
        xBC = xBC.permute(0, 2, 3, 1).view(batch, H*W, -1).contiguous()

        # Split into 3 main branches: X, B, C
        x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        x, dt, A, B, C = to_ttensor(x, dt, A, B, C)

        y, points = self.ssm_op(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt, A, B, C, self.D, H, W, points
            )

        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.norm(y)
        y = y*z
        out = self.out_proj(y)
        return out, points


class PVMAMBABlock(nn.Module):
    r"""
    Basic Block structure, including token mixer and MLP mixer
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, ssd_expansion=2, ssd_ngroups=1,
                 ssd_chunk_size=256, d_state = 64, blk_idx=0, spatial_flag=0,
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        if kwargs.get('token_mixer_types', 'attention') == 'attention':
            self.token_mixer = Attention(dim=dim, heads=num_heads, dim_head=dim // num_heads, dropout=drop)
        elif kwargs.get('token_mixer_types', 'mamba') == 'mamba':
            self.token_mixer = mamba_op(d_model=dim, expand=ssd_expansion, headdim= dim*ssd_expansion // num_heads,
                                ngroups=ssd_ngroups, chunk_size=ssd_chunk_size, d_state=d_state, **kwargs)


        if kwargs.get('token_mixer_types', 'LoDSA') == 'attention':
            self.token_mixer = Attention(dim=dim, heads=num_heads, dim_head=dim // num_heads, dropout=drop)
        elif kwargs.get('token_mixer_types', 'LoDSA') == 'LoDSA':
            self.token_mixer = mamba_op(d_model=dim, expand=ssd_expansion, headdim= dim*ssd_expansion // num_heads,
                                ngroups=ssd_ngroups, chunk_size=ssd_chunk_size, d_state=d_state, pattern='LoDSA', **kwargs)
        elif kwargs.get('token_mixer_types', 'SaDSA') == 'SaDSA':
            self.token_mixer = mamba_op(d_model=dim, expand=ssd_expansion, headdim= dim*ssd_expansion // num_heads,
                                ngroups=ssd_ngroups, chunk_size=ssd_chunk_size, d_state=d_state, pattern='SaDSA',
                                blk_idx=blk_idx, spatial_flag=spatial_flag,  **kwargs)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, H=None, W=None, points=None):
        B, L, C = x.shape
        if H & W is None:
            H, W = self.input_resolution
            assert L == H * W, "input feature has wrong size"

        x = x + self.cpe1(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        shortcut = x

        x = self.norm1(x)

        x, points = self.token_mixer(x, H, W, points)
        x = shortcut + self.drop_path(x)
        x = x + self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, points


class BasicLayer(nn.Module):
    """ A basic layer for one stage, referred from MLLA.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, mlp_ratio=4., qkv_bias=True, drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 ssd_expansion=2, ssd_ngroups=1, ssd_chunk_size=256, d_state=64,
                 spatial_flag=None, **kwargs):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            PVMAMBABlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer,
                      ssd_expansion=ssd_expansion, ssd_ngroups=ssd_ngroups, ssd_chunk_size=ssd_chunk_size, d_state=d_state,
                      blk_idx= i, spatial_flag= spatial_flag[i] if kwargs['token_mixer_types'] =='SaDSA' else None, **kwargs)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim)
        else:
            self.downsample = None

    def forward(self, x, H=None, W=None, points=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x, points = checkpoint.checkpoint(blk, x, H, W, points )
            else:
                x, points = blk(x, H, W, points)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
        return x, points

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class PVMAMBA(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=64, depths=[2, 4, 12, 4], num_heads=[2, 4, 8, 16],
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, use_checkpoint=False,
                 ssd_expansion=2, ssd_ngroups=1, ssd_chunk_size=256, d_state=64,
                 spatial_flag=[[], [],[0, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1],
                                       [0, 2, 2, 1, 2]],
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.token_mixer_types = kwargs.get('token_mixer_types', ['mamba', 'mamba', 'mamba', 'attention'])

        self.patch_embed = PatchEmbeding(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            kwargs['token_mixer_types'] = self.token_mixer_types[i_layer]
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               ssd_expansion=ssd_expansion,
                               ssd_ngroups=ssd_ngroups,
                               ssd_chunk_size=ssd_chunk_size,
                               d_state = d_state,
                               spatial_flag=spatial_flag[i_layer],
                               **kwargs)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.no_grad()
    def flops(self, shape=(3, 224, 224), verbose=True):
        # shape = self.__input_shape__[1:]
        supported_ops = {
            "aten::silu": None,  # as relu is in _IGNORED_OPS
            "aten::neg": None,  # as relu is in _IGNORED_OPS
            "aten::exp": None,  # as relu is in _IGNORED_OPS
            "aten::flip": None,  # as permute is in _IGNORED_OPS
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        try:
            Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)
        except Exception as e:
            print('get exception', e)
            print('Error in flop_count, set to default value 1e9')
            return 1e9
        del model, input

        return sum(Gflops.values()) * 1e9

    def forward_features(self, x):
        H, W = x.shape[-2:]
        x = self.patch_embed(x)
        H, W = H // 4, W // 4  # downsampled by patch_embed

        x = self.pos_drop(x)
        points = None
        for layer in self.layers:
            x, points = layer(x, H, W, points)
            H, W = H // 2, W // 2  # downsampled by layer

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class Backbone_PVMAMBA(PVMAMBA):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, **kwargs):
        super().__init__(**kwargs)
        norm_layer = nn.LayerNorm

        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.layers[i].dim)
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.head
        del self.norm
        del self.avgpool
        self.load_pretrained(pretrained,key=kwargs.get('key','model'))

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return

        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt} from {key}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")


    def forward(self, x):

        def layer_forward(l, x, H=None, W=None):
            for blk in l.blocks:
                x = blk(x, H, W)
            if l.downsample is not None:
                y = l.downsample(x, H, W)
            else:
                y = x
            return x, y

        H, W = x.shape[-2:]
        x = self.patch_embed(x)
        if self.simple_patch_embed:
            H, W = H//4, W//4
        else:
            H, W = int((H - 1) / 2) + 1, int((W - 1) / 2) + 1
            H, W = int((H - 1) / 2) + 1, int((W - 1) / 2) + 1
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x, H, W)  # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                B, L, C = out.shape
                out = out.view(B, H, W, C).permute(0, 3, 1, 2) # B, C, H, W
                outs.append(out.contiguous())
            #calculate H, W for next layer, with conv stride 3, stride 2 and padding 1
            H, W = int((H-1)/2)+1, int((W-1)/2)+1

        if len(self.out_indices) == 0:
            return x

        return outs



def build_pvmamba_tiny():
    model = PVMAMBA(
        image_size=224,
        patch_size=32,
        in_chans=3,
        embed_dim=64,
        depths=[2, 4, 8, 4],
        num_heads=[2, 4, 8, 16],  # [2, 2, 2, 2],#[2, 4, 8, 16],
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.2,
        simple_downsample=False,
        simple_patch_embed=False,
        ssd_expansion=2,
        ssd_ngroups=1,
        ssd_chunk_size=256,
        lepe=False,
        token_mixer_types=['LoDSA', 'LoDSA', 'SaDSA', 'attention'],
        bidirection=False,
        d_state=64,
        spatial_flag=[[], [], [0, 2, 2, 1, 2, 2, 1, 2], [0, 2, 2, 1]],

    )
    return model

def build_pvmamba_small():
    model = PVMAMBA(
        image_size=224,
        patch_size=32,
        in_chans=3,
        embed_dim=64,
        depths=[3, 4, 21, 5],
        num_heads=[2, 4, 8, 16],  # [2, 2, 2, 2],#[2, 4, 8, 16],
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.4,
        simple_downsample=False,
        simple_patch_embed=False,
        ssd_expansion=2,
        ssd_ngroups=1,
        ssd_chunk_size=256,
        lepe=False,
        token_mixer_types=['LoDSA', 'LoDSA', 'SaDSA', 'attention'],
        bidirection=False,
        d_state=64,
        spatial_flag=[[], [], [0, 2, 2, 1, 2, 2, 1, 2, 2, 1,
                               2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2],
                               [0, 2, 2, 1, 2]],    )
    return model

def build_pvmamba_base():
    model = PVMAMBA(
        image_size=224,
        patch_size=32,
        in_chans=3,
        embed_dim=96,
        depths=[3, 4, 21, 5],
        num_heads=[3, 6, 12, 24],  # [2, 2, 2, 2],#[2, 4, 8, 16],
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.6,
        simple_downsample=False,
        simple_patch_embed=False,
        ssd_expansion=2,
        ssd_ngroups=1,
        ssd_chunk_size=256,
        lepe=False,
        token_mixer_types=['LoDSA', 'LoDSA', 'SaDSA', 'attention'],
        bidirection=False,
        d_state=64,
        spatial_flag=[[], [], [0, 2, 2, 1, 2, 2, 1, 2, 2, 1,
                               2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2],
                               [0, 2, 2, 1, 2]],
    )
    return model


if __name__ == '__main__':
    print('Fixed inference speed evaluation and model scailing evaluation')
    import torch.backends.cudnn
    import torch.distributed as dist
    import random
    import numpy as np

    seed = 1001
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('set torch.backends.cudnn.deterministic/torch.backends.cudnn.benchmark/random.seed')

    pth = '/home/xyren/data/fei/save_pth/mae_hivit_base_1600ep.pth'

    """build model"""
    config_MODEL_NUM_CLASSES =1000
    config_MODEL_GROOTV_CHANNELS = 80
    config_MODEL_GROOTV_DEPTHS = [2, 2, 9, 2]
    config_MODEL_GROOTV_LAYER_SCALE = None
    config_MODEL_GROOTV_POST_NORM = False
    config_MODEL_GROOTV_MLP_RATIO = 4.0
    config_TRAIN_USE_CHECKPOINT = False
    config_MODEL_DROP_PATH_RATE = 0.1

    s_model = build_pvmamba_base()

    print('build model done')
    ff= s_model.flops()
    print('model self flops: '+str(ff))

    """load checkpoint"""
    # b3: [3, 4, 18, 3],
    # param = torch.load(pth)
    # missing_keys, unexpected_keys = s_model.load_state_dict(param, strict=False)
    print('No state dict of checkpoint is loaded ')

    n_parameters = sum(p.numel() for n, p in s_model.named_parameters())
    print('total params is :' + '%.2f' % (n_parameters / 1e6))

    """test settings"""
    fps_add = 0
    batch_size = 1
    num_video = 5
    Lz = 224 * 1 # 224  # 112 #128
    num_frame = 100
    inputz_test_fixed = torch.randn([batch_size, 3, Lz, Lz]).cuda()
    inputz_test = torch.randn([num_video, batch_size, 3, Lz, Lz]).cuda()
    input_test = torch.randn([num_video, num_frame, batch_size, 3, Lz, Lz]).cuda()
    print('length of z is ' + str(Lz))
    print('number of video is ' + str(num_video))
    print('number of frame in each video is ' + str(num_frame))
    s_model.eval().cuda()
    print('set model to eval mode and put it into cuda')

    """evaluation for model parameter and flops"""
    from thop import profile
    flops_tools, params = profile(s_model, inputs=([inputz_test_fixed]), custom_ops=None,
                                  verbose=False)

    print('thop lib tools: flops is :' + '%.2f' % (flops_tools / 1e9))
    print('thop lib tools: params is :' + '%.2f' % (params / 1e6))

    """inference speed"""
    import cv2
    import time

    print('torch.no_grad')
    with torch.no_grad():
        for video_index in range(num_video):
            start = time.time()
            # tic = cv2.getTickCount()
            for frame_index in range(num_frame):
                ouput = s_model( inputz_test[0,])  # inputz_test[video_index, ])
            # toc = cv2.getTickCount()

            torch.cuda.synchronize()
            end = time.time()
            avg_lat = (end - start) / num_frame
            fps = 1. / avg_lat
            print('For Video ' + str(video_index) + ", FPS using time tool: %.2f fps" % (fps))
            fps_add = fps + fps_add

    print('fps average is : ' + '%.2f' % (fps_add / num_video))

    a = 1
