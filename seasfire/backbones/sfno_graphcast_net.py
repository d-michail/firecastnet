import logging

import torch
import torch.nn as nn
from torch import Tensor
from dgl import DGLGraph

from torch_harmonics.examples.models import SphericalFourierNeuralOperator
from torch_harmonics.examples.models._layers import LearnablePositionEmbedding

from .graphcast.graph_cast_net import GraphCastNet
from .graphcast.gnn_layers.mesh_graph_mlp import MeshGraphMLP

logger = logging.getLogger(__name__)


class TemporalConv1d(nn.Module):
    """Lightweight temporal model using 1-D convolution over the time dimension.

    Applied independently at each spatial location: the time axis is treated as
    the sequence dimension of a standard Conv1d, while (W, H) are absorbed into
    the batch dimension.

    Parameters
    ----------
    channels : int
        Number of feature channels (in == out, so the shape is unchanged).
    kernel_size : int
        Temporal kernel size.  Symmetric padding preserves sequence length.
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [T, C, W, H]
        T, C, W, H = x.shape

        # Rearrange so that time is the Conv1d sequence dimension
        x = x.permute(2, 3, 1, 0).reshape(W * H, C, T)  # [W*H, C, T]
        x = self.conv(x)                                  # [W*H, C, T]
        x = x.reshape(W, H, C, T).permute(3, 2, 0, 1)   # [T, C, W, H]
        return x


class SFNOGraphCastNet(nn.Module):
    """Backbone that cleanly separates spatial, temporal, and graph processing.

    Pipeline (all shapes shown for a single sample, no batch dimension):

        Input            [C, T, W, H]
        spatial_embed    [T, C_embed, W', H']   – 2-D conv, one timestep at a time
        SFNO             [T, C_embed, W', H']   – spherical FNO, T treated as batch
        temporal         [T, C_embed, W', H']   – Conv1d over time at each location
        last timestep    [C_embed, W', H']       – carries full temporal context
        flatten          [W'*H', C_embed]        – graph node features
        graph (residual) [W'*H', hidden_dim]     – GraphCast correction + skip
        final_head       [W'*H', pixel_dim]      – pixel_dim = out_dim * W_stride * H_stride
        reshape          [1, pixel_dim, W', H']
        upsample         [1, out_dim, W, H]      – PixelShuffle or ConvTranspose2d

    Parameters
    ----------
    grid_width, grid_height : int
        Spatial dimensions of the *input* (before downsampling), e.g. 720 × 1440.
    embed_width, embed_height : int
        Spatial stride used by the 2-D downsampling conv, e.g. 4 × 4.
    embed_dim : int
        Number of channels after spatial embedding (and throughout SFNO / temporal).
    sfno_* : various
        Hyper-parameters forwarded to SphericalFourierNeuralOperator.
    temporal_kernel_size : int
        Kernel size for TemporalConv1d.
    input_dim_mesh_nodes, input_dim_edges : int
        Dimensions of the pre-built mesh / edge features in the DGL graphs.
    output_dim_grid_nodes : int
        Number of output channels *before* PixelShuffle.  Set to
        ``true_output_dim * embed_width * embed_height`` so that PixelShuffle
        (or ConvTranspose2d) restores full spatial resolution.
    processor_layers, hidden_layers, hidden_dim : int
        GraphCastNet architecture parameters.
    """

    def __init__(
        self,
        mesh_graph: DGLGraph,
        g2m_graph: DGLGraph,
        m2g_graph: DGLGraph,
        grid_width: int = 720,
        grid_height: int = 1440,
        timeseries_len: int = 12,
        input_dim_grid_nodes: int = 14,
        input_dim_mesh_nodes: int = 3,
        input_dim_edges: int = 4,
        output_dim_grid_nodes: int = 16,
        embed_width: int = 4,
        embed_height: int = 4,
        embed_dim: int = 64,
        sfno_embed_dim: int = 64,
        sfno_num_layers: int = 4,
        sfno_scale_factor: int = 3,
        sfno_hard_thresholding_fraction: float = 1.0,
        sfno_normalization_layer: str = "instance_norm",
        sfno_use_mlp: bool = True,
        sfno_mlp_ratio: float = 2.0,
        sfno_pos_embed: str = "learnable latlon",
        temporal_kernel_size: int = 3,
        processor_layers: int = 8,
        hidden_layers: int = 1,
        hidden_dim: int = 64,
        aggregation: str = "sum",
        norm_type: str = "LayerNorm",
        do_concat_trick: bool = False,
    ):
        super().__init__()

        W_small = grid_width // embed_width
        H_small = grid_height // embed_height

        # ------------------------------------------------------------------
        # 1. Spatial 2-D downsampler — no temporal mixing
        # ------------------------------------------------------------------
        # Input [T, C, W, H] → [T, embed_dim, W', H']
        self._spatial_embed = nn.Conv2d(
            input_dim_grid_nodes,
            embed_dim,
            kernel_size=(embed_width, embed_height),
            stride=(embed_width, embed_height),
        )

        # ------------------------------------------------------------------
        # 2. SFNO spatial encoder — applied per timestep (T as batch)
        # ------------------------------------------------------------------
        # in_chans == out_chans == embed_dim so features stay the same width
        self._sfno = SphericalFourierNeuralOperator(
            img_size=(W_small, H_small),
            grid="equiangular",
            scale_factor=sfno_scale_factor,
            in_chans=embed_dim,
            out_chans=embed_dim,
            embed_dim=sfno_embed_dim,
            num_layers=sfno_num_layers,
            hard_thresholding_fraction=sfno_hard_thresholding_fraction,
            normalization_layer=sfno_normalization_layer,
            use_mlp=sfno_use_mlp,
            mlp_ratio=sfno_mlp_ratio,
            pos_embed=sfno_pos_embed,
        )
        # torch_harmonics initialises the positional embedding at the
        # internally-downsampled size but applies it at full img_size.
        # Replace with a correctly-sized embedding (same fix as SFNONet).
        if sfno_pos_embed in ("learnable lat", "learnable latlon"):
            embed_type = "lat" if sfno_pos_embed == "learnable lat" else "latlon"
            self._sfno.pos_embed = LearnablePositionEmbedding(
                (W_small, H_small),
                num_chans=sfno_embed_dim,
                grid="legendre-gauss",
                embed_type=embed_type,
            )

        # ------------------------------------------------------------------
        # 3. Temporal model — Conv1d over the time dimension
        # ------------------------------------------------------------------
        self._temporal = TemporalConv1d(embed_dim, temporal_kernel_size)

        # ------------------------------------------------------------------
        # 4. GraphCast block
        #    output_dim = hidden_dim so we can add a residual skip connection
        # ------------------------------------------------------------------
        self._graph = GraphCastNet(
            mesh_graph=mesh_graph,
            g2m_graph=g2m_graph,
            m2g_graph=m2g_graph,
            input_dim_grid_nodes=embed_dim,
            input_dim_mesh_nodes=input_dim_mesh_nodes,
            input_dim_edges=input_dim_edges,
            output_dim_grid_nodes=hidden_dim,   # same as hidden_dim for residual
            processor_layers=processor_layers,
            hidden_layers=hidden_layers,
            hidden_dim=hidden_dim,
            aggregation=aggregation,
            norm_type=norm_type,
            do_concat_trick=do_concat_trick,
            has_time_dim=False,
        )

        # ------------------------------------------------------------------
        # 5. Residual projection  embed_dim → hidden_dim
        # ------------------------------------------------------------------
        self._residual_proj = nn.Linear(embed_dim, hidden_dim)

        # ------------------------------------------------------------------
        # 6. Final prediction head  hidden_dim → pixel_dim
        #    pixel_dim = output_dim * embed_width * embed_height
        #    so that the upsampler restores full spatial resolution
        # ------------------------------------------------------------------
        pixel_dim = output_dim_grid_nodes * embed_width * embed_height
        self._final_head = MeshGraphMLP(
            input_dim=hidden_dim,
            output_dim=pixel_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=nn.SiLU(),
            norm_type=None,
        )

        # ------------------------------------------------------------------
        # 7. Spatial upsampler — restores full W × H resolution
        # ------------------------------------------------------------------
        if embed_width == embed_height:
            # PixelShuffle: [1, pixel_dim, W', H'] → [1, output_dim, W, H]
            self._upsample: nn.Module = nn.PixelShuffle(embed_width)
        else:
            # ConvTranspose2d for asymmetric strides
            self._upsample = nn.ConvTranspose2d(
                pixel_dim,
                output_dim_grid_nodes,
                kernel_size=(embed_width, embed_height),
                stride=(embed_width, embed_height),
            )

        logger.info(
            "SFNOGraphCastNet: grid %dx%d → embed %dx%d, embed_dim=%d, "
            "sfno_layers=%d, temporal_kernel=%d, hidden_dim=%d, "
            "processor_layers=%d, pixel_dim=%d",
            grid_width, grid_height,
            W_small, H_small,
            embed_dim, sfno_num_layers,
            temporal_kernel_size, hidden_dim,
            processor_layers, pixel_dim,
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [C, T, W, H]
        C, T, W, H = x.shape

        # ------------------------------------------------------------------
        # Step 1: Spatial downsampling per timestep (no temporal mixing)
        # ------------------------------------------------------------------
        x = x.permute(1, 0, 2, 3)          # [T, C, W, H]
        x = self._spatial_embed(x)          # [T, C_embed, W', H']
        _, C_embed, W_small, H_small = x.shape

        # ------------------------------------------------------------------
        # Step 2: SFNO applied per timestep — T is treated as batch
        # ------------------------------------------------------------------
        x = self._sfno(x)                   # [T, C_embed, W', H']

        # ------------------------------------------------------------------
        # Step 3: Temporal model
        # ------------------------------------------------------------------
        x = self._temporal(x)               # [T, C_embed, W', H']

        # ------------------------------------------------------------------
        # Step 4: Take last timestep — it now carries full temporal context
        # ------------------------------------------------------------------
        x_spatial = x[-1]                   # [C_embed, W', H']

        # ------------------------------------------------------------------
        # Step 5: Flatten spatial dims to graph node features
        # ------------------------------------------------------------------
        # [C_embed, W', H'] → [W'*H', C_embed]
        x_flat = x_spatial.view(C_embed, -1).permute(1, 0)

        # ------------------------------------------------------------------
        # Step 6: GraphCast block with residual correction
        # ------------------------------------------------------------------
        graph_out = self._graph(x_flat)     # [W'*H', hidden_dim]
        x_proj    = self._residual_proj(x_flat)  # [W'*H', hidden_dim]
        x_corr    = x_proj + graph_out      # [W'*H', hidden_dim]  — residual

        # ------------------------------------------------------------------
        # Step 7: Final prediction head
        # ------------------------------------------------------------------
        logits = self._final_head(x_corr)   # [W'*H', pixel_dim]

        # ------------------------------------------------------------------
        # Step 8: Reshape back to spatial grid
        # ------------------------------------------------------------------
        # [W'*H', pixel_dim] → [pixel_dim, W', H'] → [1, pixel_dim, W', H']
        logits = logits.view(W_small, H_small, -1).permute(2, 0, 1)
        logits = logits.unsqueeze(0)

        # ------------------------------------------------------------------
        # Step 9: Upsample to original spatial resolution
        # ------------------------------------------------------------------
        # [1, pixel_dim, W', H'] → [1, output_dim, W, H]
        logits = self._upsample(logits)

        return logits

    def dglTo(self, device):
        """Move all DGL graphs to *device*."""
        self._graph.dglTo(device)
