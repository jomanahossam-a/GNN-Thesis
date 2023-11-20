# SPDX-FileCopyrightText: 2022 German Aerospace Center (DLR)
# SPDX-FileContributor: Tobias Hecking <tobias.hecking@dlr.de>
#
# SPDX-License-Identifier: MIT

import random
import subprocess
import os
import argparse
from math import ceil

def create_graphs(num_graphs, min_n, max_n, min_rod, max_rod, out_dir, pgsolver_base):
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    for i in range(0, num_graphs):
        n = random.choice(range(min_n, max_n))
        min_out = ceil(min_rod * n)
        max_out = ceil(max_rod * n)
        print(pgsolver_base + '/bin/pgsolver')
        game_txt = subprocess.run([pgsolver_base + '/bin/randomgame', str(n), str(n), str(min_out), str(max_out)], capture_output=True, text=True)
        
        with open(out_dir + '/' + 'game_' + str(i) + '.txt', 'w') as f:
            f.write(game_txt.stdout)
        
def create_solutions(in_dir, out_dir, pgsolver_base):
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    for game_file in os.listdir(in_dir):
        print('Solve ' + game_file)
        solution = subprocess.run([pgsolver_base + '/bin/pgsolver', in_dir + '/' + game_file, '-global', 'recursive'], capture_output=True, text=True)
        
        with open(out_dir + '/solution_' + game_file, 'w') as f:
            f.write(solution.stdout)
            
def create_games_and_solutions(num_graphs, min_n, max_n, min_rod, max_rod, games_dir, solutions_dir, pgsolver_base):
    print("Create graphs ...")
    create_graphs(num_graphs, min_n, max_n, min_rod, max_rod, games_dir, pgsolver_base)
    print("Create solutions ...")
    create_solutions(games_dir, solutions_dir, pgsolver_base)
    
    
def main():
    parser = argparse.ArgumentParser(prog ='Parity Game Generator',
                                     description ='Tool for generation parity games.')
  
    parser.add_argument('-n', '--num_games', type = int, help ='Categories to parse.', required=True)
    
    parser.add_argument('-gdir', '--games_dir', type = str, help ='Directory to store games.', required=True)
    
    parser.add_argument('-sdir', '--solutions_dir', type = str, help ='Directory to store solutions.', required=True)
    
    parser.add_argument('-pg', '--pgsolver_base', type = str, help ='Home directory of pg_solver.', required=True)

    parser.add_argument('-minn', '--min_nodes', type = int,
                        help ='Minimum number of nodes', default=20)

    parser.add_argument('-maxn', '--max_nodes', type = int, 
                        help ='Maximum number of nodes.', default=200)
                        
    parser.add_argument('-minrod', '--min_rod', type = float,
                        help ='Minimum relative outdegree.', default=0.1)

    parser.add_argument('-maxrod', '--max_rod', type = float, 
                        help ='Maximum relative outdegree.', default=0.5)

    args = parser.parse_args()

    create_games_and_solutions(args.num_games, args.min_nodes, args.max_nodes, args.min_rod, args.max_rod, args.games_dir, args.solutions_dir, args.pgsolver_base)

if __name__ == '__main__':
    main()
    
