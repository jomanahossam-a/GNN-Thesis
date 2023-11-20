# SPDX-FileCopyrightText: 2022 German Aerospace Center (DLR)
# SPDX-FileContributor: Tobias Hecking <tobias.hecking@dlr.de>
# SPDX-FileContributor: Alexander Weinert <alexander.weinert@dlr.de>
#
# SPDX-License-Identifier: MIT
 
import torch
from torch_geometric.data import InMemoryDataset, Data
import numpy as np
import pg_parser as parser
import os

from typing import Tuple, Iterable
    
class InvalidDataException(Exception):
    """ Exception if solutions do not match games"""
    
class ParityGameDataset(InMemoryDataset):

    def __init__(self, root, games, solutions, transform=None, pre_transform=None, pre_filter=None):
        self._games = games
        self._solutions = solutions
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def make_graph(self, game, solution):
        nodes, edges = parser.parse_game_file(game)
        regions_0, strategy_0, regions_1, strategy_1 = parser.parse_solution(solution)
        
        y_nodes = torch.zeros(nodes.shape[0], dtype=torch.long)
        y_nodes[regions_1] = 1
        
        y_edges = torch.zeros(edges.shape[0], dtype=torch.long)
        index_0 = [np.where((edges == s).all(axis=1))[0][0] for s in strategy_0]
        index_1 = [np.where((edges == s).all(axis=1))[0][0] for s in strategy_1]
        y_edges[index_0] = 1
        y_edges[index_1] = 1
        
        return Data(x=torch.tensor(nodes, dtype=torch.float), edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(), y_nodes=y_nodes, y_edges=y_edges)

    def process(self):
        # Read data into huge `Data` list.
        data_list = [self.make_graph(game, solution) for (game, solution) in zip(self._games, self._solutions)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
