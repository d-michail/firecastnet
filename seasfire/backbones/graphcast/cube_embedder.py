import torch
from torch import nn
# from vit_pytorch import ViT

from ..utae import LTAE2d

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        assert num_classes % num_patches == 0, 'Num of classes must be divisible by the number of patches.'

        self.mlp_head = nn.Linear(dim, num_classes//num_patches)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)

        x = self.mlp_head(x)

        x = rearrange(x, 'b p c -> b (p c)')

        return x



class CubeConv3dViT(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        grid_shape: tuple[int, int, int],
        kernel_size: tuple[int, int, int],
        patch_size: int = 72,
        dim: int = 64,
        depth: int = 1,
        heads: int = 1,
        mlp_dim: int = 64,
    ):
        super(CubeConv3dViT, self).__init__()

        self.conv_3d = torch.nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size[0], 1, 1),
            stride=(kernel_size[0], 1, 1),
        )
        self.output_dim = (
            grid_shape[0] // kernel_size[0],
            grid_shape[1] // kernel_size[1],
            grid_shape[2] // kernel_size[2],
        )
        self.out_channels = out_channels
        self.vit = ViT(
            channels=out_channels,
            image_size=(grid_shape[1], grid_shape[2]),
            patch_size=patch_size,
            num_classes=out_channels * self.output_dim[1] * self.output_dim[2],
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
        )
        self.layer_norm = nn.LayerNorm(out_channels*self.output_dim[1]*self.output_dim[2])

    def forward(
        self,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # [C0, T0, W0, H0] -> [C, T, W, H]
        o = self.conv_3d(x)
        # [C, T, W, H] -> [T, C, W, H]
        o = o.permute(1, 0, 2, 3)
        # [T, C, W, H] -> [T, C*W*H]
        o = self.vit(o)
        o = self.layer_norm(o)
        # [T, C*W*H] -> [T, C, W, H]
        o = o.reshape(self.output_dim[0], self.out_channels, self.output_dim[1], self.output_dim[2])
        # [T, C, W, H] -> [C, T, W, H]
        o = o.permute(1, 0, 2, 3)
        return o


class CubeConv3d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        origin_shape: tuple[int, int, int],
        kernel_size: tuple[int, int, int],
    ):
        super(CubeConv3d, self).__init__()

        self.conv_3d = torch.nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=kernel_size,
        )
        self.layer_norm = nn.LayerNorm(
            (
                out_channels,
                origin_shape[0] // kernel_size[0],
                origin_shape[1] // kernel_size[1],
                origin_shape[2] // kernel_size[2],
            )
        )

    def forward(
        self,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # [C, T, W, H]
        out = self.conv_3d(x)
        out = self.layer_norm(out)
        return out


class CubeConv2dLTAE(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dim,
        num_heads,
        d_k,
        out_channels,
        cube_width: int,
        cube_height: int,
    ):
        super(CubeConv2dLTAE, self).__init__()

        self.conv_2d = torch.nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=(cube_width, cube_height),
            stride=(cube_width, cube_height),
        )

        self.ltae_2d = LTAE2d(
            hidden_dim,
            n_head=num_heads,
            d_k=d_k,
            mlp=[hidden_dim, out_channels],
            dropout=0.1,
            d_model=None,
            T=1000,
            return_att=False,
            positional_encoding=True,
        )

    def forward(
        self,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # [C, T, W, H] -> [T, C, W, H]
        x = x.permute(1, 0, 2, 3)
        # [T, C, W, H] -> [T, C', W, H]
        out = self.conv_2d(x)
        # [T, C', W, H] -> [1, T, C', W, H]
        out = out.unsqueeze(0)

        batches, timesteps, _, _, _ = out.size()
        batch_positions = torch.arange(timesteps).unsqueeze(0).repeat(batches, 1).to(x)
        out = self.ltae_2d(out, batch_positions=batch_positions)

        # [1, C', W, H] -> [C, 1, W, H]
        out = out.permute(1, 0, 2, 3)

        return out
