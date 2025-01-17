import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial
from einops import rearrange, reduce
import math

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim_out, dim_in=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv1d( dim_out, dim_in, 3, padding=1)
    )

def Downsample(dim_in, dim_out=None):
    return nn.Conv1d(dim_in, dim_out or dim_in, 4, stride=2, padding=1)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups)
        self.block2 = Block(dim_out, dim_out, groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, t = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) t -> b h c t", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h t d -> b (h d) t")
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv1d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, t = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) t -> b h c t", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c t -> b (h c) t", h=self.heads)
        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)



class Unet1D(nn.Module):
    def __init__(
        self,
        dim,
        init_dim,
        dim_mults,
        channels,
        groups
    ):
        super().__init__()

        self.channels = channels
        
        input_channels = channels + 1  # +1 for anomaly score
        out_dim=channels
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock,groups=groups)

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in,time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim,time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        # First upsampling block
                        block_klass(dim_out+dim_in, dim_in+dim_out, time_emb_dim=time_dim),
            # Convolution layer to adjust channels after concatenation
                        #nn.Conv1d(dim_out + dim_in, dim_out, kernel_size=1),  # Adjust channels to dim_out
                        block_klass(dim_out+dim_in , dim_out, time_emb_dim=time_dim),
            # Second convolution layer to adjust channels after concatenation
           # nn.Conv1d(dim_out + dim_in, dim_out, kernel_size=1),  # Adjust channels to dim_out
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),

                        Upsample(dim_out, dim_in) if not is_last else nn.Conv1d(dim_out, dim_in, 3, padding=1),
        ]
                )
            )

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)
    def forward(self, x, t,anomaly_scores):
        #x=x.permute(0,2,1)
        #print(f"Initial input shape (x): {x.shape}")  # Shape of the input features
        #print(f"Anomaly scores shape: {anomaly_scores.shape}")  # Shape of the anomaly scores

    # Concatenating anomaly scores to input
        x = torch.cat([x, anomaly_scores], dim=1)
        #print(f"Shape of x after concatenating anomaly scores: {x.shape}")

    # Initial convolution
        x = self.init_conv(x)
        #print(f"Shape of x after init_conv: {x.shape}")

    # Saving residual for concatenation
        r = x.clone()
        #print(f"Shape of residual (r): {r.shape}")
       
        
    # Time embedding
        t = self.time_mlp(t)
        #print(f"Shape of time embeddings (t): {t.shape}")

    # Begin downsampling path
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            #print(f"Shape of x after block1 in downs: {x.shape}")

            x = block2(x, t)
            #print(f"Shape of x after block2 in downs: {x.shape}")

            x = attn(x)+x
            #print(f"Shape of x after attn in downs: {x.shape}")

            h.append(x)  # Save for skip connection

            x = downsample(x)
            #print(f"Shape of x after downsample: {x.shape}")

    # Mid-block
        x = self.mid_block1(x, t)
        #print(f"Shape of x after mid_block1: {x.shape}")

        x = self.mid_attn(x)
        #print(f"Shape of x after mid_attn: {x.shape}")

        x = self.mid_block2(x, t)
        #print(f"Shape of x after mid_block2: {x.shape}")

    # Begin upsampling path
        for block1, block2, attn, upsample in self.ups:
            h_pop = h.pop()

            #print(f"Shape of h.pop() in ups: {h_pop.shape}")
            #print(f"Shape of x before concat with h.pop(): {x.shape}")

        # Concatenating skip connection
            x = torch.cat((x, h_pop), dim=1)
            #print(f"Shape of x after concat in ups: {x.shape}")

        # Apply blocks and attention
            x = block1(x, t)
            #print(f"Shape of x after block1 in ups: {x.shape}")

            x = block2(x, t)
            #print(f"Shape of x after block2 in ups: {x.shape}")

            x = attn(x)+x
            #print(f"Shape of x after attn in ups: {x.shape}")

        # Upsample
            x = upsample(x)
            #print(f"Shape of x after upsample: {x.shape}")

    # Final residual block and convolution
        x = torch.cat((x, r), dim=1)
        #print(f"Shape of x after concatenating with residual (r): {x.shape}")

        x = self.final_res_block(x, t)
        #print(f"Shape of x after final_res_block: {x.shape}")

        x = self.final_conv(x)
        #print(f"Shape of x after final_conv (output): {x.shape}")

        return x
