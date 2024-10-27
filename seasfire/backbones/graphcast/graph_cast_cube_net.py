#!/usr/bin/env python

import torch
import logging

from torch import Tensor
import torch.nn as nn
from dgl import DGLGraph

from .graph_cast_net import GraphCastNet
from .cube_embedder import CubeConv3dViT, CubeConv3d, CubeConv2dLTAE

logger = logging.getLogger(__name__)


class GraphCastCubeNet(torch.nn.Module):
    def __init__(
        self,
        mesh_graph: DGLGraph,
        g2m_graph: DGLGraph,
        m2g_graph: DGLGraph,
        grid_width: int = 180,
        grid_height: int = 360,
        timeseries_len: int = 1,
        input_dim_grid_nodes: int = 10,
        input_dim_mesh_nodes: int = 3,
        input_dim_edges: int = 4,
        output_dim_grid_nodes: int = 1,
        processor_layers: int = 4,
        hidden_layers: int = 1,
        hidden_dim: int = 512,
        aggregation: str = "sum",
        norm_type: str = "LayerNorm",
        do_concat_trick: bool = False,
        embed_cube: bool = False,
        embed_cube_width: int = 4,
        embed_cube_height: int = 4,
        embed_cube_time: int = 1,
        embed_cube_dim: int = 128,
        embed_cube_layer_norm: bool = True,
        embed_cube_vit_enable: bool = False,
        embed_cube_vit_patch_size: int = 36,
        embed_cube_vit_dim: int = 64,
        embed_cube_vit_depth: int = 1,
        embed_cube_vit_heads: int = 1,
        embed_cube_vit_mlp_dim: int = 64,
        embed_cube_ltae_enable: bool = False,
        embed_cube_ltae_num_heads: int = 4,
        embed_cube_ltae_d_k: int = 16
    ):
        super(GraphCastCubeNet, self).__init__()

        # find out whether we have a time dimension
        self._has_time_dim = timeseries_len > 1 and (
            embed_cube is not True or embed_cube_time < timeseries_len
        )
        logger.info("Graph model with time dimension = {}".format(self._has_time_dim))

        self._net = GraphCastNet(
            mesh_graph,
            g2m_graph,
            m2g_graph,
            embed_cube_dim if embed_cube else input_dim_grid_nodes,
            input_dim_mesh_nodes,
            input_dim_edges,
            output_dim_grid_nodes,
            processor_layers,
            hidden_layers,
            hidden_dim,
            aggregation,
            norm_type,
            do_concat_trick,
            self._has_time_dim,
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
                upscale_factor = embed_cube_height
                # Cout = Cin / upscale_factor^2
                # hout = hin * upscale_factor
                # wout = win * upscale_factor
                self._upsample = nn.PixelShuffle(upscale_factor)
            else:
                self._upsample = nn.ConvTranspose3d(
                    output_dim_grid_nodes,
                    output_dim_grid_nodes,
                    kernel_size=(embed_cube_time, embed_cube_width, embed_cube_height),
                    stride=(embed_cube_time, embed_cube_width, embed_cube_height),
                )

    def forward(self, x: Tensor):
        channels, timesteps, width, height = x.size()

        if self._downsample is not None:
            x = self._downsample(x)

            # reread after downsample
            channels, timesteps, width, height = x.size()

        if not self._has_time_dim:
            if timesteps > 1:
                raise ValueError("Found timesteps > 1 while timeseries is disabled")

            # [C, 1, W, H] -> [W * H, C]
            x = x[:, 0, :, :].view(x.size(0), -1).permute(1, 0)

            logits = self._net(x)

            # [W * H, C] -> [W, H, C]
            logits = logits.view(width, height, -1)
            # [W, H, C] -> [C, W, H]
            logits = logits.permute(2, 0, 1)
            # [C, W, H] -> [1, C, W, H]
            logits = logits.unsqueeze(0)
        else:
            # [C, T, W, H] -> [T, W * H, C]
            x = x.view(channels, timesteps, -1).permute(1, 2, 0)

            logits = self._net(x)

            # [T, W * H, C] -> [T, W, H, C]
            logits = logits.reshape(timesteps, width, height, -1)

            # [T, W, H, C] -> [C, T, W, H]
            logits = logits.permute(3, 0, 1, 2)

        if self._upsample is not None:
            logits = self._upsample(logits)

        return logits

    def dglTo(self, device):
        self._net.dglTo(device)
