# SPDX-FileCopyrightText: 2022 German Aerospace Center (DLR)
# SPDX-FileContributor: Tobias Hecking <tobias.hecking@dlr.de>
# SPDX-FileContributor: Alexander Weinert <alexander.weinert@dlr.de>
#
# SPDX-License-Identifier: MIT

import torch
from torch_geometric.nn.models import GCN
from torch_geometric.nn.models import GAT
from torch_geometric.nn.models import GraphSAGE

class _ParityGameNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index):
        x = self.core(x, edge_index)
        row, col = edge_index
        edge_rep = torch.cat([x[row], x[col]], dim=1)

        return (self.node_classifier(x), self.edge_classifier(edge_rep))

class ParityGameGATNetwork(_ParityGameNetwork):
    def __init__(self, hidden_channels_nodes, hidden_channels_edges, core_iterations):
        super().__init__()
        self.core = GAT(3, hidden_channels_nodes, core_iterations, jk='lstm', flow='target_to_source')
        self.node_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels_nodes, hidden_channels_nodes),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(hidden_channels_edges, 2),
            torch.nn.Softmax(dim=1)
        )

        self.edge_classifier = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_channels_nodes, hidden_channels_edges),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(hidden_channels_edges, 2),
            torch.nn.Softmax(dim=1)
        )



class ParityGameGCNNetwork(_ParityGameNetwork):
    def __init__(self, hidden_channels_nodes, hidden_channels_edges, core_iterations):
        super().__init__()
        self.core = GCN(3, hidden_channels_nodes, core_iterations, jk='lstm', flow='target_to_source')
        self.node_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels_nodes, hidden_channels_nodes),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(hidden_channels_edges, 2),
            torch.nn.Softmax(dim=1)
        )
        
        self.edge_classifier = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_channels_nodes, hidden_channels_edges),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(hidden_channels_edges, 2),
            torch.nn.Softmax(dim=1)
        )
        
class ParityGameGraphSAGENetwork(_ParityGameNetwork):
    def __init__(self, hidden_channels_nodes, hidden_channels_edges, core_iterations):
        super().__init__()
        self.core = GraphSAGE(3, hidden_channels_nodes, core_iterations, jk='lstm', flow='target_to_source')
        self.node_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels_nodes, hidden_channels_nodes),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(hidden_channels_edges, 2),
            torch.nn.Softmax(dim=1)
        )
        
        self.edge_classifier = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_channels_nodes, hidden_channels_edges),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(hidden_channels_edges, 2),
            torch.nn.Softmax(dim=1)
        )
        
        
        