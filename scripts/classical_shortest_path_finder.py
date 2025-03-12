!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate random grids/graphs with obstacles and find its shortest path. The classical shortest path algotithms
are Dijkstra and A*.
"""

import argparse
import os

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--obstacle_mode', choices=['ratio', 'number'], default='ratio', help="Obstacle mode selection ('ratio' or 'number')")
    p.add_argument('--obstacle_ratio', type=float, default=0.2, help="Obstacle ratio (use if 'obstacle_mode' is 'ratio')")
    p.add_argument('--obstacle_number', type=int, default=20, help="Number of obstacles (use if 'obstacle_mode' is 'number')")
    return p
