#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculate the area or the perimeter of a circle
"""

import argparse
import os
from my_research.utils.Circle import circle_area, circle_perimeter

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('radius', type=float, metavar='RADIUS',
                   help='Radius of the circle.')
    p.add_argument('--mode', choices=['area', 'perimeter'], required=True,
                   help="Choose between 'area' to calculate the area or 'perimeter' to calculate the perimeter.")
    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.mode == 'area':
        result = circle_area(args.radius)
        print(f"L'aire du cercle de rayon {args.radius} est : {result:.2f}")
    elif args.mode == 'perimeter':
        result = circle_perimeter(args.radius)
        print(f"Le périmètre du cercle de rayon {args.radius} est : {result:.2f}")

if __name__ == "__main__":
    main()
