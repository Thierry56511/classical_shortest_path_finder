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
    p.add_argument('--obstacles', type=str, default='ratio:0.2', 
               help="Obstacle settings: 'ratio:<value>' or 'number:<value>'")
    p.add_argument("--output", type=str, required=True, help="Name of the file and number of graphs saved: 'graph.json;<number>'")
    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    mode, value = args.obstacles.split(':')
    value = float(value) if mode == 'ratio' else int(value)
    
    if mode == 'ratio' and value > 1:
        raise IOError('The obstacle ratio should not higher than 1.0'.format(args.obstacles))

    file, number = args.output.split(';')
    number = int(number)
    file = str(file)

    grid, G = generer_grille(args.size, mode, value, value)
    save_graph(G, file, number)
    print(f"✅ Graphe sauvegardé dans '{args.output}'.")
    # Afficher la grille générée et les noeuds du graph
    print("Grille générée :")
    print(grid)
    print("Graph nodes:", list(G.nodes))

if __name__ == "__main__":
    main()
