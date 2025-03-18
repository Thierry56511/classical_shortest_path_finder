#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Find the shortest path between 2 points in a graph with Dijkstra or A*.
"""

import argparse
import os
from my_research.utils.grid_dijkstra import (dijkstra_stepwise, astar_stepwise, heuristic, load_graph)

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--input", type=str, required=True, help="Name of the file JSON containing the graph")
    p.add_argument("--shortestpath", choices=['Dijkstra', 'A*'], default='Dijkstra', help="shortest path algorithm selection ('Dijkstra' or 'A*')")
    p.add_argument("--output", type=str, required=True, help="Name of the file (ex: graph.json)")
    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    
    G = load_graph(args.input)
    save_graphe(G, args.output)
    # Afficher la grille générée et les noeuds du graph
    print("Grille générée :")
    print(grid)
    print("Graph nodes:", list(G.nodes))

if __name__ == "__main__":
    main()
