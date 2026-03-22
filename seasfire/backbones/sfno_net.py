import logging

import torch.nn as nn
from torch import Tensor

from torch_harmonics.examples.models import SphericalFourierNeuralOperator
from torch_harmonics.examples.models._layers import LearnablePositionEmbedding

from .graphcast.cube_embedder import CubeConv2dLTAE, CubeConv3d, CubeConv3dViT

logger = logging.getLogger(__name__)


class SFNONet(nn.Module):
    """Spherical Fourier Neural Operator backbone with optional CubeEmbedder preprocessing.

    The CubeEmbedder (typically CubeConv3d) collapses the temporal dimension via 3-D
    convolution, producing a single-timestep spatial representation that is then
    processed by the SFNO on a regular lat-lon grid.  An optional PixelShuffle layer
    restores the full spatial resolution after the SFNO.
    """

    def __init__(
        self,
        grid_width: int = 720,
        grid_height: int = 1440,
        timeseries_len: int = 1,
        input_dim_grid_nodes: int = 11,
        output_dim_grid_nodes: int = 16,
        embed_cube: bool = True,
        embed_cube_width: int = 4,
        embed_cube_height: int = 4,
        embed_cube_time: int = 3,
        embed_cube_dim: int = 32,
        embed_cube_layer_norm: bool = False,
        embed_cube_vit_enable: bool = False,
        embed_cube_vit_patch_size: int = 36,
        embed_cube_vit_dim: int = 64,
        embed_cube_vit_depth: int = 1,
        embed_cube_vit_heads: int = 1,
        embed_cube_vit_mlp_dim: int = 64,
        embed_cube_ltae_enable: bool = False,
        embed_cube_ltae_num_heads: int = 4,
        embed_cube_ltae_d_k: int = 16,
        sfno_embed_dim: int = 128,
        sfno_num_layers: int = 4,
        sfno_scale_factor: int = 3,
        sfno_hard_thresholding_fraction: float = 1.0,
        sfno_normalization_layer: str = "instance_norm",
        sfno_use_mlp: bool = True,
        sfno_mlp_ratio: float = 2.0,
        sfno_pos_embed: str = "learnable latlon",
    ):
        super(SFNONet, self).__init__()

        if not embed_cube and timeseries_len > 1:
            raise ValueError(
                "SFNONet requires embed_cube=True when timeseries_len > 1. "
                "The CubeEmbedder handles temporal compression via 3-D convolution."
            )

        self._downsample = None
        self._upsample = None

        if embed_cube:
            if embed_cube_vit_enable:
                self._downsample = CubeConv3dViT(
                    input_dim_grid_nodes,
                    embed_cube_dim,
                    (timeseries_len, grid_width, grid_height),
                    (embed_cube_time, embed_cube_width, embed_cube_height),
                    patch_size=embed_cube_vit_patch_size,
                    dim=embed_cube_vit_dim,
                    depth=embed_cube_vit_depth,
                    heads=embed_cube_vit_heads,
                    mlp_dim=embed_cube_vit_mlp_dim,
                    use_layer_norm=embed_cube_layer_norm,
                )
            elif embed_cube_ltae_enable:
                self._downsample = CubeConv2dLTAE(
                    input_dim_grid_nodes,
                    embed_cube_dim,
                    (timeseries_len, grid_width, grid_height),
                    (embed_cube_time, embed_cube_width, embed_cube_height),
                    d_k=embed_cube_ltae_d_k,
                    num_heads=embed_cube_ltae_num_heads,
                    use_layer_norm=embed_cube_layer_norm,
                )
            else:
                self._downsample = CubeConv3d(
                    input_dim_grid_nodes,
                    embed_cube_dim,
                    (timeseries_len, grid_width, grid_height),
                    (embed_cube_time, embed_cube_width, embed_cube_height),
                    use_layer_norm=embed_cube_layer_norm,
                )

            if embed_cube_width == embed_cube_height:
                self._upsample = nn.PixelShuffle(embed_cube_width)
            else:
                self._upsample = nn.ConvTranspose3d(
                    output_dim_grid_nodes,
                    output_dim_grid_nodes,
                    kernel_size=(embed_cube_time, embed_cube_width, embed_cube_height),
                    stride=(embed_cube_time, embed_cube_width, embed_cube_height),
                )

            sfno_in_chans = embed_cube_dim
            sfno_nlat = grid_width // embed_cube_width
            sfno_nlon = grid_height // embed_cube_height
        else:
            sfno_in_chans = input_dim_grid_nodes
            sfno_nlat = grid_width
            sfno_nlon = grid_height

        self._sfno = SphericalFourierNeuralOperator(
            img_size=(sfno_nlat, sfno_nlon),
            grid="equiangular",
            scale_factor=sfno_scale_factor,
            in_chans=sfno_in_chans,
            out_chans=output_dim_grid_nodes,
            embed_dim=sfno_embed_dim,
            num_layers=sfno_num_layers,
            hard_thresholding_fraction=sfno_hard_thresholding_fraction,
            normalization_layer=sfno_normalization_layer,
            use_mlp=sfno_use_mlp,
            mlp_ratio=sfno_mlp_ratio,
            pos_embed=sfno_pos_embed,
        )

        # torch_harmonics initializes learnable positional embeddings at the
        # internally-downsampled size (img_size / scale_factor) but applies them
        # to x *before* any spatial downsampling (i.e. at full img_size).
        # Replace pos_embed with a correctly-sized one when needed.
        if sfno_pos_embed in ("learnable lat", "learnable latlon"):
            embed_type = "lat" if sfno_pos_embed == "learnable lat" else "latlon"
            self._sfno.pos_embed = LearnablePositionEmbedding(
                (sfno_nlat, sfno_nlon),
                num_chans=sfno_embed_dim,
                grid="legendre-gauss",
                embed_type=embed_type,
            )

        logger.info(
            "SFNO grid {}x{}, in_chans={}, out_chans={}, embed_dim={}, num_layers={}".format(
                sfno_nlat, sfno_nlon, sfno_in_chans, output_dim_grid_nodes,
                sfno_embed_dim, sfno_num_layers,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [C, T, W, H]
        if self._downsample is not None:
            x = self._downsample(x)
            # x: [C_embed, 1, W_small, H_small]

        # Collapse the (single) time dimension: [C, 1, W, H] → [C, W, H]
        x = x[:, 0, :, :]

        # Add batch dimension for SFNO: [C, W, H] → [1, C, W, H]
        x = x.unsqueeze(0)

        # SFNO forward: [1, C_in, nlat, nlon] → [1, C_out, nlat, nlon]
        x = self._sfno(x)

        if self._upsample is not None:
            # PixelShuffle: [1, C_out, nlat, nlon] → [1, 1, W_full, H_full]
            x = self._upsample(x)

        # Output: [1, C_out, W, H]  interpreted as [T=1, C_out, W, H]
        return x
