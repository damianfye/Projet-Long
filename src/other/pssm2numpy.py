"""
Ne fonctionne pas, a faire
"""

import sys
import os
import argparse

import numpy as np
from progress.bar import IncrementalBar #pip install progress


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--pdblist", help="Path to the PDB ID lists.", type=str, required=True)
    parser.add_argument("-p", "--pssm", help="Path to the PSSM directory.", type=str, required=True)
    parser.add_argument("-o", "--output", help="Path to the NumPy directory.", type=str, required=True)
    args = parser.parse_args()

    return args.pdblist, args.pssm, args.output


def pssm_ascii2numpy(pssm_ascii):
    pssm = []

    with open(pssm_ascii, "r") as f_pssm:
        for i, line in enumerate(f_pssm):
            if i > 2:
                if line == "\n":
                    break
                else:
                    pssm.append(line[91:-11].split())

    pssm = np.array(pssm, dtype=np.int) / 100
    
    return pssm

def main():
    pssm_ascii2numpy("test_pssm")


if __name__ == '__main__':
    main()