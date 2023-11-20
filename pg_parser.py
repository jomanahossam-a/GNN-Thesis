# SPDX-FileCopyrightText: 2022 German Aerospace Center (DLR)
#
# SPDX-License-Identifier: MIT

import os
import re
import numpy as np
    
def parse_edges(node_line):
    return np.array(list(np.broadcast([node_line[0]], node_line[3].split(',')))).astype(int)

def parse_game_file(node_lines):
    return (
        np.append(
            np.expand_dims(node_lines[:,1].astype(float) / np.max(node_lines[:,1].astype(float)), axis=1),
            [[1, 0] if node[2] == '0' else [0, 1] for node in node_lines], axis=1
        ), 
        np.concatenate([parse_edges(line) for line in node_lines])
    )

def get_nodes_and_edges(game_file):
    return parse_game_file(
        np.loadtxt(game_file, dtype=str, skiprows=1)
    )

def parse_regions(region_str):
    try:
        return np.array(re.match(r'\{(.*)\}', region_str).group(1).split(',')).astype(int)
    except:
        return np.empty(0, dtype=int
                    )
        
def parse_strategy(strategy_str):
    try:
        return np.array([x.split('->') for x in re.match(r'\[(.*)\]', strategy_str).group(1).split(',')]).astype(int)
    except:
        return np.empty((0,2), dtype=int)
    
def parse_solution(solution_lines, found_regions_p0=None, found_strategies_p0=None, found_regions_p1=None, found_strategies_p1=None):

    if len(solution_lines) == 0:
        return (found_regions_p0, found_strategies_p0, found_regions_p1, found_strategies_p1)
    elif solution_lines[0].strip() == 'Player 0 wins from nodes:':
        return parse_solution(solution_lines[2:], parse_regions(solution_lines[1].strip()), found_strategies_p0, found_regions_p1, found_strategies_p1)
    elif solution_lines[0].strip() == 'Player 1 wins from nodes:':
        return parse_solution(solution_lines[2:], found_regions_p0, found_strategies_p0, parse_regions(solution_lines[1].strip()), found_strategies_p1)
    elif solution_lines[0].strip() == 'with strategy':
        if found_strategies_p0 is None:
            return parse_solution(solution_lines[2:], found_regions_p0, parse_strategy(solution_lines[1].strip()), 
                found_regions_p1, found_strategies_p1)
        else:
            return parse_solution(solution_lines[2:], found_regions_p0, found_strategies_p0, 
                found_regions_p1, parse_strategy(solution_lines[1].strip()))
    else:
        return parse_solution(solution_lines[1:], found_regions_p0, found_strategies_p0, found_regions_p1, found_strategies_p1)

def get_solution(solution_file):
    with open(solution_file) as f:
        return parse_solution(f.readlines())
    
def get_solution_short_form(solution_file):
    with open(solution_file) as f:
        return parse_solution_short_form(f.readlines()[1:], [], []) # Skip the first line, as it is a fairly useless header
    
def parse_solution_short_form(lines, region_0_acc, region_1_acc):
    if len(lines) == 0:
        return region_0_acc, region_1_acc
    
    parts = lines[0].split(" ")
    node, winner = parts[0], 0 if parts[1] == "0" else 1
    if winner == 0:
        region_0_acc.append(int(node))
    else:
        region_1_acc.append(int(node))
    
    return parse_solution_short_form(lines[1:], region_0_acc, region_1_acc)