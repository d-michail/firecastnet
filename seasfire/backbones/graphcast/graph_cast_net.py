#!/usr/bin/env python

import torch
import logging

from torch import Tensor
from dgl import DGLGraph

from .gnn_layers.embedder import GraphCastDecoderEmbedder, GraphCastEncoderEmbedder
from .gnn_layers.mesh_graph_encoder import MeshGraphEncoder
from .gnn_layers.mesh_graph_decoder import MeshGraphDecoder
from .gnn_layers.mesh_graph_mlp import MeshGraphMLP
from .graph_cast_mesh_processor import GraphCastMeshProcessor

logger = logging.getLogger(__name__)


class GraphCastNet(torch.nn.Module):
    def __init__(
        self,
        mesh_graph: DGLGraph,
        g2m_graph: DGLGraph,
        m2g_graph: DGLGraph,
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
        has_time_dim: bool = False,
    ):
        super(GraphCastNet, self).__init__()

        self.input_dim_grid_nodes = input_dim_grid_nodes

        self.mesh_graph = mesh_graph

        self.g2m_graph = g2m_graph
        self.m2g_graph = m2g_graph

        self.g2m_edata = self.g2m_graph.edata["x"]
        self.m2g_edata = self.m2g_graph.edata["x"]
        self.mesh_edata = self.mesh_graph.edata["x"]
        self.mesh_ndata = self.mesh_graph.ndata["x"]

        activation_fn = torch.nn.SiLU()

        # initial feature embedder
        self.encoder_embedder = GraphCastEncoderEmbedder(
            input_dim_grid_nodes=input_dim_grid_nodes,
            input_dim_mesh_nodes=input_dim_mesh_nodes,
            input_dim_edges=input_dim_edges,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

        # grid2mesh encoder
        self.encoder = MeshGraphEncoder(
            aggregation=aggregation,
            input_dim_src_nodes=hidden_dim,
            input_dim_dst_nodes=hidden_dim,
            input_dim_edges=hidden_dim,
            output_dim_src_nodes=hidden_dim,
            output_dim_dst_nodes=hidden_dim,
            output_dim_edges=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            do_concat_trick=do_concat_trick,
        )

        # icosahedron processor
        if processor_layers <= 2:
            raise ValueError("Expected at least 3 processor layers")
        self.processor_encoder = GraphCastMeshProcessor(
            aggregation=aggregation,
            processor_layers=1,
            input_dim_nodes=hidden_dim,
            input_dim_edges=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            has_time_dim=has_time_dim,
        )
        self.processor = GraphCastMeshProcessor(
            aggregation=aggregation,
            processor_layers=processor_layers - 2,
            input_dim_nodes=hidden_dim,
            input_dim_edges=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            has_time_dim=has_time_dim,
        )
        self.processor_decoder = GraphCastMeshProcessor(
            aggregation=aggregation,
            processor_layers=1,
            input_dim_nodes=hidden_dim,
            input_dim_edges=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            has_time_dim=has_time_dim,
        )

        self.decoder_embedder = GraphCastDecoderEmbedder(
            input_dim_edges=input_dim_edges,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

        # mesh2grid decoder
        self.decoder = MeshGraphDecoder(
            aggregation=aggregation,
            input_dim_src_nodes=hidden_dim,
            input_dim_dst_nodes=hidden_dim,
            input_dim_edges=hidden_dim,
            output_dim_dst_nodes=hidden_dim,
            output_dim_edges=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            do_concat_trick=do_concat_trick,
        )

        # final MLP
        self.finale = MeshGraphMLP(
            input_dim=hidden_dim,
            output_dim=output_dim_grid_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=None,
        )

    def forward(self, grid_nfeat: Tensor):

        # make sure aux data is on the same device
        g2m_edata = self.g2m_edata.to(grid_nfeat)
        mesh_ndata = self.mesh_ndata.to(grid_nfeat)
        mesh_edata = self.mesh_edata.to(grid_nfeat)

        # embed graph features
        (
            grid_nfeat_embedded,
            mesh_nfeat_embedded,
            g2m_efeat_embedded,
            mesh_efeat_embedded,
        ) = self.encoder_embedder(
            grid_nfeat,
            mesh_ndata,
            g2m_edata,
            mesh_edata,
        )

        # encode lat/lon to multimesh
        grid_nfeat_encoded, mesh_nfeat_encoded = self.encoder(
            g2m_efeat_embedded,
            grid_nfeat_embedded,
            mesh_nfeat_embedded,
            self.g2m_graph,
        )

        # process multimesh graph
        mesh_efeat_processed, mesh_nfeat_processed = self.processor_encoder(
            mesh_efeat_embedded,
            mesh_nfeat_encoded,
            self.mesh_graph,
        )

        # process multimesh graph
        mesh_efeat_processed, mesh_nfeat_processed = self.processor(
            mesh_efeat_processed,
            mesh_nfeat_processed,
            self.mesh_graph,
        )

        # process multimesh graph
        _, mesh_nfeat_processed = self.processor_decoder(
            mesh_efeat_processed,
            mesh_nfeat_processed,
            self.mesh_graph,
        )

        # make sure aux data is on the same device
        m2g_edata = self.m2g_edata.to(grid_nfeat)

        m2g_efeat_embedded = self.decoder_embedder(m2g_edata)

        # decode multimesh to lat/lon
        grid_nfeat_decoded = self.decoder(
            m2g_efeat_embedded, grid_nfeat_encoded, mesh_nfeat_processed, self.m2g_graph
        )

        # map to the target output dimension
        grid_nfeat_finale = self.finale(
            grid_nfeat_decoded,
        )

        return grid_nfeat_finale

    def dglTo(self, device):
        logger.info(f"Calling .to({device}) on all DGL graphs")
        self.mesh_graph = self.mesh_graph.to(device)
        self.g2m_graph = self.g2m_graph.to(device)
        self.m2g_graph = self.m2g_graph.to(device)
