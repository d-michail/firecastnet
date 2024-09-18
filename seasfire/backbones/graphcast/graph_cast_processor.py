# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
from dgl import DGLGraph
from torch import Tensor

from .gnn_layers.mesh_edge_block import MeshEdgeBlock
from .gnn_layers.mesh_node_block import MeshNodeBlock


class GraphCastProcessor(nn.Module):
    """Processor block used in GraphCast operating on a latent space
    represented by hierarchy of icosahedral meshes.

    Parameters
    ----------
    aggregation : str, optional
        message passing aggregation method ("sum", "mean"), by default "sum"
    processor_layers : int, optional
        number of processor layers, by default 16
    input_dim_nodes : int, optional
        input dimensionality of the node features, by default 512
    input_dim_edges : int, optional
        input dimensionality of the edge features, by default 512
    hidden_dim : int, optional
        number of neurons in each hidden layer, by default 512
    hidden_layers : int, optional
        number of hiddel layers, by default 1
    activation_fn : nn.Module, optional
        type of activation function, by default nn.SiLU()
    norm_type : str, optional
        normalization type, by default "LayerNorm"
    do_conat_trick: : bool, default=False
        whether to replace concat+MLP with MLP+idx+sum
    """

    def __init__(
        self,
        aggregation: str = "sum",
        processor_layers: int = 16,
        input_dim_nodes: int = 512,
        input_dim_edges: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        do_concat_trick: bool = False,
    ):
        super().__init__()

        edge_block_invars = (
            input_dim_nodes,
            input_dim_edges,
            input_dim_edges,
            hidden_dim,
            hidden_layers,
            activation_fn,
            norm_type,
            do_concat_trick,
        )
        node_block_invars = (
            aggregation,
            input_dim_nodes,
            input_dim_edges,
            input_dim_nodes,
            hidden_dim,
            hidden_layers,
            activation_fn,
            norm_type,
        )

        layers = []
        for _ in range(processor_layers):
            layers.append(MeshEdgeBlock(*edge_block_invars))
            layers.append(MeshNodeBlock(*node_block_invars))

        self.processor_layers = nn.ModuleList(layers)
        self.num_processor_layers = len(self.processor_layers)

    def forward(
        self,
        efeat: Tensor,
        nfeat: Tensor,
        graph: DGLGraph,
    ) -> Tensor:
        for layer in self.processor_layers:
            efeat, nfeat = layer(efeat, nfeat, graph)

        return efeat, nfeat
