import sys
import os
import subprocess
import argparse

import numpy as np
import pbxplore
from progress.bar import IncrementalBar #pip install progress


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--pdblist", help="Path to the PDB ID lists.", type=str, required=True)
    parser.add_argument("-x", "--x", help="Path to the X files directory.", type=str, required=True)
    parser.add_argument("-w", "--window", type=int, help="Nombre impaire", required=True)
    parser.add_argument("-o", "--output", help="Path to the merged X files directory.", type=str, required=True)
    args = parser.parse_args()

    #Vérifier que w est impaire

    return args.pdblist, args.x, args.window, args.output



def main():
    pdb_list, x, window, output = args()

    pdb_list = np.loadtxt(pdb_list, dtype="str")

    bar = IncrementalBar("Generating X files array...", max=len(pdb_list))

    for pdb_id in pdb_list:

        arr_x = np.load(os.path.join(x, f"{pdb_id}.npy"))
        all_merged = []
        
        padding = window // 2

        for j in range(arr_x.shape[0]):
            aa_merged = np.zeros((window, arr_x.shape[1]+1))
            c = 0

            for k in range(j-padding, j+padding+1):
                if k < 0 or k >= arr_x.shape[0]:
                    row = np.append(np.zeros(arr_x.shape[1]), 1)
                    aa_merged[c] = row
                else:
                    row = np.append(arr_x[k], 0)
                    aa_merged[c] = row
                c += 1

            all_merged.append(aa_merged)

        all_merged = np.array(all_merged)
        np.save(os.path.join(output, f"{pdb_id}_merged.npy"), all_merged)
        bar.next()
    bar.finish()


if __name__ == '__main__':
    main()