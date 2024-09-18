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

from typing import Tuple

import torch
import torch.nn as nn
from dgl import DGLGraph
from torch import Tensor

from .utils import aggregate_and_concat, concat_efeat

from .mesh_graph_mlp import MeshGraphMLP


class MeshProcessorBlock(nn.Module):
    """Process block operating on a latent space represented by a mesh.

    Parameters
    ----------
    aggregation : str, optional
        Aggregation method (sum, mean) , by default "sum"
    input_dim_nodes : int, optional
        Input dimensionality of the node features, by default 512
    input_dim_edges : int, optional
        Input dimensionality of the edge features, by default 512
    output_dim_nodes : int, optional
        Output dimensionality of the node features, by default 512
    output_dim_edges : int, optional
        Output dimensionality of the edge features, by default 512
    hidden_dim : int, optional
        Number of neurons in each hidden layer, by default 512
    hidden_layers : int, optional
        Number of neurons in each hidden layer, by default 1
    activation_fn : nn.Module, optional
       Type of activation function, by default nn.SiLU()
    norm_type : str, optional
        Normalization type, by default "LayerNorm"
    """

    def __init__(
        self,
        aggregation: str = "sum",
        input_dim_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim_nodes: int = 512,
        output_dim_edges: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
    ):
        super().__init__()
        self.aggregation = aggregation

        self.input_dim_nodes = input_dim_nodes
        self.output_dim_nodes = output_dim_nodes

        self.input_dim_edges = input_dim_edges
        self.output_dim_edges = output_dim_edges

        self.edge_mlp = MeshGraphMLP(
            input_dim=input_dim_nodes + input_dim_nodes + input_dim_edges,
            output_dim=output_dim_edges,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

        self.node_mlp = MeshGraphMLP(
            input_dim=input_dim_nodes + output_dim_edges,
            output_dim=output_dim_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

    @torch.jit.ignore()
    def forward(
        self,
        efeat: Tensor,
        nfeat: Tensor,
        graph: DGLGraph,
    ) -> Tuple[Tensor, Tensor]:
        # First concatenate edge src and dst nodes with edge features.
        # Pass result from edge MLP.
        cat_efeat = concat_efeat(efeat, nfeat, graph)
        efeat_new = self.edge_mlp(cat_efeat)
        if self.input_dim_edges == self.output_dim_edges:
            efeat_new = efeat_new + efeat

        # Aggregate edge features and concat with destination node
        cat_nfeat = aggregate_and_concat(efeat_new, nfeat, graph, self.aggregation)
        # update node features + residual connection
        nfeat_new = self.node_mlp(cat_nfeat)
        if self.input_dim_nodes == self.output_dim_nodes:
            nfeat_new = nfeat_new + nfeat

        return efeat_new, nfeat_new
