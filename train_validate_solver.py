#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2022 German Aerospace Center (DLR)
# SPDX-FileContributor: Tobias Hecking <tobias.hecking@dlr.de>
# SPDX-FileContributor: Alexander Weinert <alexander.weinert@dlr.de>
#
# SPDX-License-Identifier: MIT

import argparse #used to parse command line arguments
from torch_geometric.loader import DataLoader #used to load and batch data for DL
from parity_game_dataset import ParityGameDataset
from parity_game_network import ParityGameGCNNetwork, ParityGameGATNetwork, ParityGameGraphSAGENetwork
import pg_parser
import torch
import numpy as np
import sys
import os #for os specific functions
import parity_game_network as pn #alias
import csv
import time



class Training:
    
    
    

    def __init__(self):
        self.val_loader = None  # Define val_loader as an instance variable
        self.criterion_nodes = torch.nn.CrossEntropyLoss()
        self.criterion_edges = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9]))
        self.val_losses = []
        

    def train(self, games, solutions, val_games, val_solutions):
        data = ParityGameDataset('pg_data_20220708', games, solutions)
        val_data = ParityGameDataset('pg_data_validate', val_games, val_solutions)
        train_loader = DataLoader(data, batch_size=5, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=5, shuffle=False)  # Set the val_loader instance variable
        num_batches = len(train_loader)
        val_num_batches = len(self.val_loader)  # Access val_loader through self
        print(f"Number of training batches: {num_batches}")
        print(f"Number of validation batches: {val_num_batches}")

       

        #self._network.load_state_dict(torch.load('GATweights.pth'))  # Load pretrained weights

        self._train_internal(train_loader)

        torch.save(self._network.state_dict(), self._output_stream)

    def _train_internal(self, loader):
        running_loss = 0.
        i = 0

        optimizer = torch.optim.Adam(self._network.parameters(), lr=0.001)
        #optimizer = torch.optim.SGD(self._network.parameters(), lr=0.01, momentum=0.9)

        #scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=5, step_size_down=10, mode='triangular')
        #optimizer=torch.optim.RMSprop(self._network.parameters(), lr=0.001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False, foreach=None, maximize=False, differentiable=False)
        criterion_nodes = torch.nn.CrossEntropyLoss()
        criterion_edges = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9]))

        self._network.train()
        
        
        for data in loader:  # Iterate in batches over the training dataset.
            i += 1
            optimizer.zero_grad()  # Clear gradients.
            out_nodes, out_edges = self._network(data.x, data.edge_index)  # Perform a single forward pass.

            # Most edges do not belong to a winning strategy and thus the data is extemely imbalanced. The model will probably learn that predicting always "non-winning" for each edge
            # yields reasonable performance. To avoid this, approximately as many non-winning strategy edges are sampled as there are winning edges.
            # edge_selection = (torch.rand(data.y_edges.shape[0]) > 0.7) | (data.y_edges == 1)
            loss = criterion_nodes(out_nodes, data.y_nodes) + criterion_edges(out_edges,
                                                                              data.y_edges)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            #scheduler.step()
            running_loss += loss.item()
            if i % 50 == 0:
                last_loss = running_loss / 50  # loss per batch
                running_loss = 0.
                
            print("batch: ", i,"loss: ", loss.item())
            
            
        for data in self.val_loader:
            validation_loss = self._validate(self.val_loader)
            self.val_losses.append(validation_loss)
            print(f"Validation Loss : {validation_loss:.4f}")
            
        total_per=self._test(self.val_loader)
        print(f"Total Accuracy: {total_per:.4f}")

    
    def _validate(self, loader):
        self._network.eval()
        val_losses = []  # Create a list to store validation losses for each batch

        with torch.no_grad():
            for i, data in enumerate(loader):
                out_nodes, out_edges = self._network(data.x, data.edge_index)
                loss = self.criterion_nodes(out_nodes, data.y_nodes) + self.criterion_edges(out_edges, data.y_edges)
                val_losses.append(loss.item())  # Append loss for this batch
                print(f"Validation Batch {i + 1}/{len(loader)} - Loss: {loss.item():.4f}")

        return val_losses  # Return a list of validation losses for each batch

            
    
        
        
    def _test(self, loader):
        self._network.eval()

        correct_vertices = 0

        for data in loader:  # Iterate in batches over the training/test dataset.
            out_nodes, _ = self._network(data.x, data.edge_index)
            pred_nodes = out_nodes.argmax(dim=1)  # Use the class with highest probability.
            correct_vertices += (pred_nodes == data.y_nodes).sum() / len(
                pred_nodes)  # Check against ground-truth labels.
        return correct_vertices / len(loader)  # Derive ratio of correct predictions.

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, value):
        self._network = value

    @property
    def output(self):
        return self._output_stream

    @output.setter
    def output(self, value):
        self._output_stream = value

def train(args):
    training = Training()
    #training.weights_file= 'GAT_pg_solver_20220914.pth'

    if args.network == "GAT":
        training.network = ParityGameGATNetwork(256, 256, 10)
    elif args.network == "GCN":
        training.network = ParityGameGCNNetwork(256, 256, 20)
    elif args.network== "GraphSAGE":
        training.network = ParityGameGraphSAGENetwork(35, 35, 10)

    training.output = args.output

    if not len(args.files) % 2 == 0:
        raise ValueError("Invalid list of files given. List of files must comprise an even number of file names, alternating between games and solutions, starting with games")

    games, solutions = [], []
    for i in range(len(args.files)):
        if i % 2 == 0:
            games.append(np.loadtxt(args.files[i], dtype=str, skiprows=1))
        else:
            with open(args.files[i], 'r') as f:
                solutions.append(f.readlines())
                
    val_games, val_solutions = [], []
    
    for i in range(2100, 2999):
        game_number = f"{i:04}"  # Formats the number with leading zeros to make it 4 digits
        game_file = f'/path/to/games/game_{game_number}.txt'
        solution_file = f'/path/to/solutions/game_{game_number}_sol.txt'
        val_games.append(np.loadtxt(game_file, dtype=str, skiprows=1))


    
        with open(solution_file, 'r') as f:
            val_solutions.append(f.readlines())

   
    training.train(games, solutions,val_games, val_solutions)
    
    
class Predictor:

    def predict(self, games):
        self._network.load_state_dict(torch.load(self._weights), strict=True)
    
        if isinstance(self._output, str):
            with open(self._output, 'w') as f, open('prediction_times.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Game Path', 'Elapsed Time (s)'])
                for input in games:
                    print("Predicting path:", input)
                    start_time = time.perf_counter()
                    self._predict_single(np.loadtxt(input, dtype=str, skiprows=1), f, input)
                    elapsed_time = time.perf_counter() - start_time
                    writer.writerow([input, elapsed_time])
        else:
            with open('prediction_times.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Game Path', 'Elapsed Time (s)'])
                for input in games:
                    print("Predicting path:", input)
                    start_time = time.perf_counter()
                    self._predict_single(np.loadtxt(input, dtype=str, skiprows=1), self._output, input)
                    elapsed_time = time.perf_counter() - start_time
                    writer.writerow([input, elapsed_time])
 


    def _predict_single(self, input, output, identifier=None):
        nodes, edges = pg_parser.parse_game_file(input)

        out_nodes, _ = self._network(torch.tensor(nodes, dtype=torch.float),
                                     torch.tensor(edges, dtype=torch.long).t().contiguous())

        winning_region = [str(node) for node in range(len(out_nodes)) if out_nodes[node][0].item() > out_nodes[node][1].item()]
        output_line = (identifier if not identifier is None else "") + " " + (" ".join(winning_region) + "\n")
        output.write(output_line)


    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, value):
        self._network = value

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = value

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, value):
        self._output = value

def predict(args):
    predictor = Predictor()

    if args.network == "GAT":
       predictor.network = ParityGameGATNetwork(256, 256, 10)
        
    elif args.network == "GCN":
        predictor.network = ParityGameGCNNetwork(256, 256, 10)
        
    elif args.network == "GraphSAGE":
        predictor.network = ParityGameGraphSAGENetwork(35, 35, 10)

    predictor.weights = args.weights

    if args.output is None or args.output == '-':
        predictor.output = sys.stdout
    else:
        predictor.output = args.output
        
    
    predictor.predict(args.files)

        
class Evaluator:

    def __init__(self):
        self._statistics = []
        self.histogram = False

    def evaluate(self, inputs):
        if isinstance(self._output, str):
            with open(self._output, 'w') as f:
                self._evaluate(inputs, f)
        else:
            self._evaluate(inputs, sys.stdout)

    def _evaluate(self, inputs, output):
        for line in inputs:
            split_line = line.rstrip().split(' ')
            self._evaluate_single(split_line[0], split_line[-1], split_line[1:-1], output)
        if self.histogram:
            self._print_histogram(output)

    def _evaluate_single(self, game, reference, prediction, output):
        pred_region_0 = [int(node) for node in prediction]

        act_region_0, _, act_region_1, _ = pg_parser.get_solution(reference)

        correct_in_region_0, wrong_in_region_0 = compare_regions(pred_region_0, act_region_0)

        num_nodes = len(act_region_0) + len(act_region_1)
        self._statistics.append((num_nodes, correct_in_region_0, wrong_in_region_0))

        if not len(act_region_0) == 0:
            percentage_wrong = wrong_in_region_0 * 1.0 / len(act_region_0)
            
        else:
            percentage_wrong = 0.0
            
        
        output_line = " ".join([game, str(len(act_region_0)), str(correct_in_region_0), str(wrong_in_region_0), str(percentage_wrong)]) + "\n"
        output.write(output_line)

    def _print_histogram(self, output):
        max_wrong = max(map(lambda x: x[2], self._statistics))
        for num_wrong in range(0, max_wrong + 1):
            output.write(f"{num_wrong} {len(list(filter(lambda x: x[2] == num_wrong, self._statistics)))}\n")

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, value):
        self._output = value


    @property
    def histogram(self):
        return self._histogram

    @histogram.setter
    def histogram(self, value):
        self._histogram = value

def evaluate(args):
    evaluator = Evaluator()
    evaluator.output = args.output
    evaluator.histogram = args.histogram
    evaluator.evaluate(sys.stdin)

def compare_regions(pred, act):
    correct, wrong = 0, 0
    for pred_vertex in pred:
        if pred_vertex in act:
            correct += 1
        else:
            wrong += 1

    return correct, wrong

def main():

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help', required=True)
    train_parser = subparsers.add_parser('train', help='create a weights-file by training the solver on a set of solved games')
    train_parser.add_argument('-n', '--network', type=str, choices=['GCN','GAT'], required=True, help="The GNN architecture to train")
    train_parser.add_argument('-o', '--output', type=str, required=True, help="Where to write the output weights-file")
    train_parser.add_argument('files', nargs='*', help="A list of files containing games and solutions for training. Must alternate between games and solutions, starting with games.")
    train_parser.add_argument('files_val', nargs='*')
    train_parser.set_defaults(func=train)

    predict_parser = subparsers.add_parser('predict', help='predict winning regions of one or more parity games using a previously created weights-file')
    predict_parser.add_argument('-n', '--network', type=str, choices=['GCN','GAT'], required=True, help="The GNN architecture to train")
    predict_parser.add_argument('-w', '--weights', type=str, required=True, help="A weights-file produced by the train sub-command")
    predict_parser.add_argument('-o', '--output', type=str, help="Where to write the output file. If - or omitted, output is written to stdout")
    predict_parser.add_argument('files', nargs='*', help="A list of files containing games for which to predict winning regions.")
    predict_parser.set_defaults(func=predict)

    evaluate_parser = subparsers.add_parser('evaluate', help='evaluate the results of a prediction against a reference solution')
    evaluate_parser.add_argument('-o', '--output', type=str, help="Where to write the output file. If - or omitted, output is written to stdout")
    evaluate_parser.add_argument('--histogram', action='store_true', help="If set, histogram of wrongly predicted vertices will be printed after evaluation")
    evaluate_parser.set_defaults(func=evaluate)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    

 files = os.listdir('path/to/TrainingSet')
 files.sort()
 file_paths = [os.path.join('path/to/TrainingSet', file) for file in files]
                            
 files_val=os.listdir('path/to/ValidationSet')
 files_val.sort()
 files_val_paths=[os.path.join('path/to/ValidationSet', file) for file in files]
 
 
 args = argparse.Namespace(func = 'train', network  = 'GraphSAGE', weights = 'dummy.pth' , output = 'GNN_weights.pth' , files= file_paths)
 train(args)
 
 
 #args = argparse.Namespace(func = 'predict', network  = 'GraphSAGE', weights = 'GNN_weight.pth' , output = 'game_prediction.csv' , files= file_paths)
 #predict(args)
 
 
 #args = args = argparse.Namespace(func = 'evaluate', output='prediction_evaluation.csv', histogram= None )
 #evaluate(args)
    

