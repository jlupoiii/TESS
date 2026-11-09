"""
U-Net architecture components for the diffusion model 
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class FourierEncoder(nn.Module):
    """
    Unchanged from unet.py.
    Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py#L183
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - t: (B, 1, 1, 1)
        Returns:
        - embeddings: (B, dim)
        """
        t = t.view(-1, 1)
        freqs = t * self.weights * 2 * math.pi
        sin_embed = torch.sin(freqs)
        cos_embed = torch.cos(freqs)
        return torch.cat([sin_embed, cos_embed], dim=-1) * 1.414  # (bs, dim)


def posemb_sincos_2d(h: int, w: int, dim: int, device) -> torch.Tensor:
    """
    Standard ViT-style 2D sin-cos positional embedding.
    """
    assert dim % 4 == 0, (
        f"posemb_sincos_2d requires channel dim divisible by 4, got {dim}. "
        "All stages listed in heads_at must have n_feat*mult divisible by 4."
    )
    y, x = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    quarter = dim // 4
    omega = torch.arange(quarter, device=device, dtype=torch.float32) / max(quarter - 1, 1)
    omega = 1.0 / (10000 ** omega)
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)  # (h*w, dim)


class AttentionFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim * 2)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        # x: (B, seq_len, dim)
        data, gate = self.linear1(x).chunk(2, dim=-1)
        x = data * self.gelu(gate)
        x = self.linear2(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.net = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)

    def forward(self, x):
        return self.net(x, x, x, need_weights=False)[0]


class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim=128, heads=4):
        super().__init__()
        self.dim = dim
        self.context_dim = context_dim
        self.heads = heads
        self.net = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, kdim=context_dim,
                                         vdim=context_dim, batch_first=True)

    def forward(self, x, context):
        return self.net(x, context, context, need_weights=False)[0]


class AttentionBlock(nn.Module):
    def __init__(self, dim, context_dim, mlp_dim, heads=4, use_cross_attn=True):
        super().__init__()
        self.dim = dim
        self.context_dim = context_dim
        self.mlp_dim = mlp_dim
        self.heads = heads
        self.use_cross_attn = use_cross_attn

        self.conv1 = nn.Conv2d(dim, dim, stride=1, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(dim, dim, stride=1, kernel_size=1, padding=0)

        self.layernorm1 = nn.LayerNorm(dim)
        self.layernorm3 = nn.LayerNorm(dim)
        # UPDATE: min(32, dim) instead of hard-coded 32, matching ResidualConvBlock,
        # so small-width configs don't crash.
        self.groupnorm = nn.GroupNorm(min(32, dim), dim)

        self.self_attn = SelfAttention(dim, heads)
        if use_cross_attn:
            self.layernorm2 = nn.LayerNorm(dim)
            self.cross_attn = CrossAttention(dim, context_dim, heads)
        self.ffn = AttentionFFN(dim, mlp_dim)

        # UPDATE: cache for the positional embedding, keyed by (H, W).
        # Plain dict (not a buffer): it is deterministic and cheap to rebuild,
        # so it should not enter the state_dict.
        self._pe_cache = {}

    def _pos_emb(self, H: int, W: int, device) -> torch.Tensor:
        key = (H, W, str(device))
        pe = self._pe_cache.get(key)
        if pe is None:
            pe = posemb_sincos_2d(H, W, self.dim, device)
            self._pe_cache[key] = pe
        return pe

    def forward(self, x, context=None):
        skip_x = x  # long skip connection
        x = self.groupnorm(x)
        x = self.conv1(x)  # shuffles features before heads
        B, C, H, W = x.shape

        # Self Attention Block B, C, H, W -> B, (H*W), C
        x = rearrange(x, 'b c h w -> b (h w) c')
        # UPDATE: inject explicit positional information before attention.
        # Cast to x.dtype so bf16 autocast tensors are not silently promoted to fp32.
        x = x + self._pos_emb(H, W, x.device).to(x.dtype).unsqueeze(0)

        attn_skip = x  # short skip connection
        x = self.layernorm1(x)
        x = self.self_attn(x)
        x = x + attn_skip

        # Cross Attention Block (UPDATE: optional)
        if self.use_cross_attn and context is not None:
            cross_skip = x
            x = self.layernorm2(x)
            x = self.cross_attn(x, context)
            x = x + cross_skip

        # GeGLU Block
        x = self.layernorm3(x)
        x = self.ffn(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x = self.conv2(x)
        x = x + skip_x

        return x


class ResidualConvBlock(nn.Module):
    """
    Main Unet Block
    """
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int) -> None:
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(min(32, in_channels), in_channels)
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)

        self.conv1 = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.skip_conv = (nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
                          if not self.same_channels else nn.Identity())

        self.film1 = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, 2 * in_channels))
        self.film2 = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, 2 * out_channels))
        nn.init.zeros_(self.film1[-1].weight)
        nn.init.zeros_(self.film1[-1].bias)
        nn.init.zeros_(self.film2[-1].weight)
        nn.init.zeros_(self.film2[-1].bias)

        # Start learning from identity
        nn.init.zeros_(self.conv2[-1].weight)
        nn.init.zeros_(self.conv2[-1].bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # x:   (B, in_channels, H, W)
        # emb: (B, emb_dim) — fused time + camera + context embedding
        scale1, shift1 = self.film1(emb).view(-1, 2 * self.in_channels, 1, 1).chunk(2, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale1) + shift1
        h = self.conv1(h)

        scale2, shift2 = self.film2(emb).view(-1, 2 * self.out_channels, 1, 1).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1 + scale2) + shift2
        h = self.conv2(h)

        return h + self.skip_conv(x)


class UnetDown(nn.Module):
    """
    Downsampling half of unet
    """
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int,
                 num_heads: int = None, context_dim: int = 256, num_res: int = 2,
                 use_cross_attn: bool = True):
        super().__init__()
        self.conv1 = ResidualConvBlock(in_channels, out_channels, emb_dim)
        self.attn1 = (AttentionBlock(out_channels, context_dim, mlp_dim=out_channels * 2,
                                     heads=num_heads, use_cross_attn=use_cross_attn)
                      if num_heads is not None else nn.Identity())

        self.rem_blocks = []
        for _ in range(num_res - 1):
            self.rem_blocks.append(ResidualConvBlock(out_channels, out_channels, emb_dim))
            if num_heads is not None:
                self.rem_blocks.append(AttentionBlock(out_channels, context_dim,
                                                      mlp_dim=out_channels * 2, heads=num_heads,
                                                      use_cross_attn=use_cross_attn))
        self.rem_blocks = nn.ModuleList(self.rem_blocks)
        self.down_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, c_attn, emb: torch.Tensor):
        """
        x:      (B, in_channels, H, W)
        c_attn: (B, num_tokens, context_dim) or None — context for cross-attention
        emb:    (B, emb_dim) — fused time + camera + context embedding
        Returns:
            x:    (B, out_channels, H/2, W/2)
            skip: (B, out_channels, H, W)
        """
        x = self.conv1(x, emb)
        x = self.attn1(x, c_attn) if isinstance(self.attn1, AttentionBlock) else self.attn1(x)

        for block in self.rem_blocks:
            if isinstance(block, ResidualConvBlock):
                x = block(x, emb)
            else:
                x = block(x, c_attn)

        skip = x
        x = self.down_conv(x)
        return x, skip


class UnetUp(nn.Module):
    """
    Upsampling block. Same conditioning updates as UnetDown.
    """
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int,
                 num_heads: int = None, context_dim: int = 256, num_res: int = 2,
                 use_cross_attn: bool = True):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.conv1 = ResidualConvBlock(2 * out_channels, out_channels, emb_dim)
        self.attn1 = (AttentionBlock(out_channels, context_dim, mlp_dim=out_channels * 2,
                                     heads=num_heads, use_cross_attn=use_cross_attn)
                      if num_heads is not None else nn.Identity())
        self.rem_blocks = []
        for _ in range(num_res - 1):
            self.rem_blocks.append(ResidualConvBlock(out_channels, out_channels, emb_dim))
            if num_heads is not None:
                self.rem_blocks.append(AttentionBlock(out_channels, context_dim,
                                                      mlp_dim=out_channels * 2, heads=num_heads,
                                                      use_cross_attn=use_cross_attn))
        self.rem_blocks = nn.ModuleList(self.rem_blocks)

    def forward(self, x: torch.Tensor, c_attn, emb: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        x:      (B, in_channels, H, W)
        c_attn: (B, num_tokens, context_dim) or None
        emb:    (B, emb_dim)
        skip:   (B, out_channels, H, W)
        Returns:
            x: (B, out_channels, H*2, W*2)
        """
        x = self.up_conv(x)
        x = torch.cat((x, skip), 1)
        x = self.conv1(x, emb)
        x = self.attn1(x, c_attn) if isinstance(self.attn1, AttentionBlock) else self.attn1(x)
        for block in self.rem_blocks:
            if isinstance(block, ResidualConvBlock):
                x = block(x, emb)
            else:
                x = block(x, c_attn)
        return x


class UNetBottleneck(nn.Module):
    """Same conditioning updates as UnetDown/UnetUp."""
    def __init__(self, in_channels: int, out_channels: int, context_dim: int,
                 emb_dim: int, use_cross_attn: bool = True):
        super().__init__()
        self.conv1 = ResidualConvBlock(in_channels, out_channels, emb_dim)
        self.attn1 = AttentionBlock(out_channels, context_dim, mlp_dim=out_channels * 2,
                                    heads=8, use_cross_attn=use_cross_attn)
        self.conv2 = ResidualConvBlock(out_channels, out_channels, emb_dim)

    def forward(self, x: torch.Tensor, c_attn, emb: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, emb)
        x = self.attn1(x, c_attn)
        x = self.conv2(x, emb)
        return x


class TokenizeContext(nn.Module):
    """
    Turns a context vector of shape (B, context_len) into (B, context_len,
    context_dim) tokens for cross-attention.
    """
    def __init__(self, context_len: int, context_dim: int):
        super().__init__()
        self.context_len = context_len
        self.context_dim = context_dim
        # Shared scalar -> vector lift 
        self.net = nn.Sequential(
            nn.Linear(1, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim),
        )
        self.param_embed = nn.Parameter(torch.randn(context_len, context_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, context_len)
        tokens = self.net(x.unsqueeze(-1))          # (B, context_len, context_dim)
        return tokens + self.param_embed.unsqueeze(0)


class EmbedFC(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ConditioningEmbedding(nn.Module):
    def __init__(self, time_dim: int, context_len: int, emb_dim: int, num_cameras: int):
        super().__init__()
        self.time_fourier = FourierEncoder(time_dim)
        self.time_mlp = EmbedFC(time_dim, emb_dim)
        self.context_mlp = EmbedFC(context_len, emb_dim)
        self.camera_embed = nn.Embedding(num_cameras, emb_dim)
        nn.init.normal_(self.camera_embed.weight, std=0.02)
        # A final nonlinearity+linear lets the network mix the three sources.
        self.fuse = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, emb_dim))

    def forward(self, t: torch.Tensor, c: torch.Tensor, cam=None) -> torch.Tensor:
        # t: (B, 1, 1, 1);  c: (B, context_len);  cam: (B,) long or None
        emb = self.time_mlp(self.time_fourier(t)) + self.context_mlp(c)
        if cam is not None:
            emb = emb + self.camera_embed(cam)
        return self.fuse(emb)


class ContextUnet(nn.Module):
    """
    Main Class
    """
    def __init__(self, in_feats, context_len,
                 n_feat=64, time_dim=128, context_dim=64, emb_dim=256,
                 channel_mults=(1, 2, 2, 4), heads_at=(2, 3), num_res=(2, 2, 2, 2),
                 num_cameras=4, use_cross_attn=True):
        super().__init__()
        self.in_feats = in_feats
        self.context_len = context_len
        self.n_feat = n_feat
        self.channel_mults = channel_mults
        self.num_res = num_res if isinstance(num_res, list) else [num_res] * len(channel_mults)
        self.init_conv = nn.Conv2d(in_feats, n_feat, 3, 1, 1)
        self.heads_at = heads_at
        self.use_cross_attn = use_cross_attn

        self.cond_embed = ConditioningEmbedding(time_dim, context_len, emb_dim, num_cameras)

        #learned null embedding
        self.null_context = nn.Parameter(torch.zeros(context_len))

        if use_cross_attn:
            self.attn_context_embed = TokenizeContext(context_len, context_dim)

        # Downs
        self.downs = nn.ModuleList()
        in_ch = n_feat * channel_mults[0]
        for i, mult in enumerate(channel_mults):
            out_ch = n_feat * mult
            num_heads = 8 if i in heads_at else None
            self.downs.append(UnetDown(in_ch, out_ch, emb_dim=emb_dim, num_heads=num_heads,
                                       context_dim=context_dim, num_res=self.num_res[i],
                                       use_cross_attn=use_cross_attn))
            in_ch = out_ch

        # Bottleneck
        bot_ch = n_feat * channel_mults[-1]
        self.bottleneck = UNetBottleneck(bot_ch, bot_ch, context_dim, emb_dim=emb_dim,
                                         use_cross_attn=use_cross_attn)

        # Ups
        self.ups = nn.ModuleList()
        prev_ch = bot_ch
        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = n_feat * mult
            stage_idx = len(channel_mults) - 1 - i
            num_heads = 8 if stage_idx in heads_at else None
            self.ups.append(UnetUp(prev_ch, out_ch, emb_dim=emb_dim, num_heads=num_heads,
                                   context_dim=context_dim, num_res=self.num_res[stage_idx],
                                   use_cross_attn=use_cross_attn))
            prev_ch = out_ch

        self.out = nn.Sequential(
            nn.GroupNorm(8, prev_ch),
            nn.SiLU(),
            nn.Conv2d(in_channels=prev_ch, out_channels=1, kernel_size=3, stride=1, padding=1)
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, c, t, context_mask, cam=None):
        x = self.init_conv(x)

        # Flatten context and mask to (B, context_len)
        c = c.reshape(c.shape[0], c.shape[2])
        context_mask = context_mask.reshape(c.shape[0], c.shape[1])

        c = c * (1 - context_mask) + self.null_context.view(1, -1) * context_mask

     
        emb = self.cond_embed(t, c, cam)

        c_attn = self.attn_context_embed(c) if self.use_cross_attn else None

        skips = []
        for down in self.downs:
            x, skip = down(x, c_attn=c_attn, emb=emb)
            skips.append(skip)

        x = self.bottleneck(x, c_attn=c_attn, emb=emb)

        for up in self.ups:
            x = up(x, c_attn=c_attn, emb=emb, skip=skips.pop())

        return self.out(x)
