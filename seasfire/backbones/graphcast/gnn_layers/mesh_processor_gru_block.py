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
import math

try:
    from apex.normalization import FusedLayerNorm

    apex_imported = True
except ImportError:
    apex_imported = False

from .utils import aggregate_and_concat, concat_efeat
import logging

logger = logging.getLogger(__name__)


class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation_fn=torch.nn.Tanh(), bias=True):
        super(GRUCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation_fn = activation_fn
        self.bias = bias

        self.x2h = nn.Linear(input_dim, 3 * hidden_dim, bias=bias)
        self.h2h = nn.Linear(hidden_dim, 3 * hidden_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim) if self.hidden_dim > 0 else 0
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):

        x = x.view(-1, x.size(1))

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_z, i_n = gate_x.chunk(3, 1)
        h_r, h_z, h_n = gate_h.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        updategate = torch.sigmoid(i_z + h_z)
        newgate = self.activation_fn(i_n + (resetgate * h_n))

        h = (1 - updategate) * newgate + updategate * hidden

        return h


class MeshProcessorGruBlock(nn.Module):
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
    norm_type : str, optional
        Normalization type, by default "LayerNorm"
    """

    def __init__(
        self,
        aggregation: str = "sum",
        input_dim_nodes: int = 512,
        input_dim_edges: int = 512,
        hidden_dim: int = 512,
        activation_fn=torch.nn.Tanh(),
        norm_type: str = "LayerNorm",
    ):
        super().__init__()
        self.aggregation = aggregation

        self.input_dim_nodes = input_dim_nodes
        self.input_dim_edges = input_dim_edges
        self.hidden_dim = hidden_dim

        self.e_gru_cell = GRUCell(
            input_dim_edges + 2 * input_dim_nodes,
            3 * hidden_dim,
            activation_fn=activation_fn,
        )
        self.e_mlp = nn.Linear(3 * hidden_dim, hidden_dim)
        self.n_gru_cell = GRUCell(
            input_dim_nodes + hidden_dim, 2 * hidden_dim, activation_fn=activation_fn
        )
        self.n_mlp = nn.Linear(2 * hidden_dim, hidden_dim)

        self.norm_type = norm_type
        self.e_norm_layer = None
        self.n_norm_layer = None

        if norm_type is not None:
            if norm_type not in [
                "LayerNorm",
                "GraphNorm",
                "InstanceNorm",
                "BatchNorm",
                "MessageNorm",
            ]:
                raise ValueError(norm_type)
            if norm_type == "LayerNorm" and apex_imported:
                norm_layer = FusedLayerNorm
                logger.info("Found apex, using FusedLayerNorm")
            else:
                norm_layer = getattr(nn, norm_type)
            self.e_norm_layer = norm_layer(hidden_dim)
            self.n_norm_layer = norm_layer(hidden_dim)

    def forward(
        self,
        efeat: Tensor,
        nfeat: Tensor,
        ehidden: Tensor,
        nhidden: Tensor,
        graph: DGLGraph,
    ) -> Tuple[Tensor, Tensor]:

        # First concatenate edge src and dst nodes with edge features.
        cat_efeat = concat_efeat(efeat, nfeat, graph)
        cat_ehidden = concat_efeat(ehidden, nhidden, graph)
        ehidden_new = self.e_gru_cell(cat_efeat, cat_ehidden)
        ehidden_new = self.e_mlp(ehidden_new)

        # # Aggregate edge features and concat with destination node
        cat_nfeat = aggregate_and_concat(ehidden_new, nfeat, graph, self.aggregation)
        cat_nhidden = aggregate_and_concat(
            ehidden_new, nhidden, graph, self.aggregation
        )
        nhidden_new = self.n_gru_cell(cat_nfeat, cat_nhidden)
        nhidden_new = self.n_mlp(nhidden_new)

        # And normalize layers
        if self.e_norm_layer is not None:
            ehidden_new = self.e_norm_layer(ehidden_new)

        if self.n_norm_layer is not None:
            nhidden_new = self.n_norm_layer(nhidden_new)

        return ehidden_new, nhidden_new
