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
    p.add_argument('radius', type=float, metavar='RADIUS',
                   help='Radius of the circle.')
    p.add_argument('--mode', choices=['area', 'perimeter'], required=True,
                   help="Choose between 'area' to calculate the area or 'perimeter' to calculate the perimeter.")
    return p
