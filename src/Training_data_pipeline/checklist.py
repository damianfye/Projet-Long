#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Check if the PDB ID list contains multimeric complexs.

To properly train the neural network, you need to correctly encode the output of
the network.
To do so, every protein of the complex need to be in the PDB file used to calculate
RASA. But the Dockground database contain only binary informations.
So in order to not waste time to try to correctly use them for the network,
I decided to completly remove them. Not the best, but It works (I guess...).
"""

import sys
import os
import argparse

import numpy as np


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--pdblist", help="Path to the PDB ID lists.", type=str, required=True)
    args = parser.parse_args()

    return args.pdblist

def main():
    pdb_list = args()

    pdb_list = np.loadtxt(pdb_list, dtype="str")

    #Check for multimeric complexs
    multi = {}

    for pdb_id in pdb_list:
        if pdb_id[:4] not in multi:
             multi[pdb_id[:4]] = 1
        else:
            multi[pdb_id[:4]] += 1

    for pdb in multi:
        if multi[pdb] = 1:
            print(pdb)


if __name__ == "__main__":
    main()
