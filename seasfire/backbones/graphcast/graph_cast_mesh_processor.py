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
import torch

from .gnn_layers.mesh_processor_block import MeshProcessorBlock
from .gnn_layers.mesh_processor_gru_block import MeshProcessorGruBlock


class GraphCastMeshProcessor(nn.Module):
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
        has_time_dim: bool = False,
    ):
        super().__init__()

        self.has_time_dim = has_time_dim
        self.hidden_dim = hidden_dim

        MPB = MeshProcessorGruBlock if has_time_dim else MeshProcessorBlock
        MPB_params = (
            (
                aggregation,
                input_dim_nodes,
                input_dim_edges,
                hidden_dim,
                nn.Tanh(),
                norm_type,
            )
            if has_time_dim
            else (
                aggregation,
                input_dim_nodes,
                input_dim_edges,
                input_dim_nodes,
                input_dim_edges,
                hidden_dim,
                hidden_layers,
                activation_fn,
                norm_type,
            )
        )

        layers = []
        for _ in range(processor_layers):
            layers.append(MPB(*MPB_params))

        self.processor_layers = nn.ModuleList(layers)
        self.num_processor_layers = len(self.processor_layers)

    def forward(
        self,
        efeat: Tensor,
        nfeat: Tensor,
        graph: DGLGraph,
    ) -> Tensor:
        if self.has_time_dim and len(nfeat.shape) != 3:
            raise ValueError(
                "Expected node features input shape of [Time, Nodes, Channels]"
            )

        if len(nfeat.shape) == 3:
            # in case efeat does not contain a time dimension, repeat it
            # this handles the use of this processor as a processor_encoder and a processor
            timesteps = nfeat.size(0)
            if len(efeat.shape) == 3:
                efeat_t = efeat
            else:
                efeat_t = efeat.unsqueeze(0).repeat(timesteps, 1, 1)

            z_ehidden = torch.zeros((graph.num_edges(), self.hidden_dim)).to(efeat)
            z_nhidden = torch.zeros((graph.num_nodes(), self.hidden_dim)).to(efeat)

            for layer in self.processor_layers:
                ehidden = []
                nhidden = []

                for i in range(timesteps):
                    if i == 0:
                        ehidden_new, nhidden_new = layer(
                            efeat_t[i], nfeat[i], z_ehidden, z_nhidden, graph
                        )
                    else:
                        ehidden_new, nhidden_new = layer(
                            efeat_t[i], nfeat[i], ehidden[-1], nhidden[-1], graph
                        )
                    ehidden.append(ehidden_new)
                    nhidden.append(nhidden_new)

            return torch.stack(ehidden), torch.stack(nhidden)
        else:
            for layer in self.processor_layers:
                efeat, nfeat = layer(efeat, nfeat, graph)
            return efeat, nfeat
