#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate random grids/graphs with obstacles.
"""

import argparse
import os
from my_research.utils.grid_dijkstra import (generer_grille, save_graph)

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--size', type=int, default=10, help="size of the grid (use a int)")
    p.add_argument('--obstacle_mode', choices=['ratio', 'number'], default='ratio', help="Obstacle mode selection ('ratio' or 'number')")
    p.add_argument('--obstacle_ratio', type=float, default=0.2, help="Obstacle ratio (use if 'obstacle_mode' is 'ratio')")
    p.add_argument('--obstacle_number', type=int, default=20, help="Number of obstacles (use if 'obstacle_mode' is 'number')")
    p.add_argument("--output", type=str, required=True, help="Name of the file (ex: graph.json)")
    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    grid, G = generer_grille(args.size, args.obstacle_mode, args.obstacle_ratio, args.obstacle_number)
    save_graph(G, args.output)
    print(f"✅ Graphe sauvegardé dans '{args.output}'.")
    # Afficher la grille générée et les noeuds du graph
    print("Grille générée :")
    print(grid)
    print("Graph nodes:", list(G.nodes))

if __name__ == "__main__":
    main()
