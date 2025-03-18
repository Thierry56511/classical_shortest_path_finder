#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Find the shortest path between 2 points in a graph with Dijkstra or A*.
"""

import argparse
import os
from my_research.utils.grid_dijkstra import (dijkstra_stepwise, astar_stepwise, heuristic, get_neighbors_diagonal, load_graph)

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--input", type=str, required=True, help="Name of the file JSON containing the graph (ex: graph.json")
    p.add_argument("--shortestpath", choices=['Dijkstra', 'A*'], default='Dijkstra', help="shortest path algorithm selection ('Dijkstra' or 'A*')")
    p.add_argument("--start", type=str, required=True, help="starting node (ex: '3,4')") 
    p.add_argument("--target", type=str, required=True, help="ending node (ex: '7,8')")
    p.add_argument("--diagonal_mode", choices=['diagonal', 'nondiagonal'], default='nondiagonal', help="Choose if the algorithm can travel in diagonal or not ('diagonal' or 'nondiagonal')")
    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    start = tuple(map(int, args.start.split(',')))   
    target = tuple(map(int, args.target.split(',')))
    
    G = load_graph(args.input)
    
    if args.shortestpath == "Dijkstra":
        evaluated_nodes, path_history = dijkstra_stepwise(G, start, target, args.diagonal_mode)
    elif args.shortestpath == "A*":
        evaluated_nodes, path_history = astar_stepwise(G, start, target, args.diagonal_mode)
    
    print("shortest path :")
    print(path_history[len(path_history)-1])
    print("path history :")
    print(path_history)

if __name__ == "__main__":
    main()
