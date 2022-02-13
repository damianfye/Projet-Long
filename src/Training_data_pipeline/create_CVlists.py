import os
import sys
import argparse

import numpy as np
import h5py
from progress.bar import IncrementalBar
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, GroupKFold



def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--pdblist", help="Path to the PDB ID lists.", type=str, required=True)
    parser.add_argument("-f", "--folds", help="Number of CV folds.", type=int, required=False, default=5)
    parser.add_argument("-o", "--output", help="Output folder.", type=str, required=True)
    args = parser.parse_args()

    return args.pdblist, args.output, args.folds



def main():
    pdb_list, output, folds = args()
    
    pdb_list = np.loadtxt(pdb_list, dtype="str")

    kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    for i, (train, test) in enumerate(kfold.split(np.zeros((pdb_list.shape[0], 1)), np.zeros((pdb_list.shape[0], 1)))):
        
        print(f"\nFold {i}")
        
        path_CV = os.path.join(output, f"CV_{i}")
        os.makedirs(path_CV)

        np.savetxt(os.path.join(path_CV, "test_pdb_id.txt"), pdb_list[test], fmt="%s")

        learnID_list, valID_list = train_test_split(pdb_list[train], test_size=0.1)     #test_size --> Number of proteins of the validation file.
        np.savetxt(os.path.join(path_CV, "learn_pdb_id.txt"), learnID_list, fmt="%s")
        np.savetxt(os.path.join(path_CV, "val_pdb_id.txt"), valID_list, fmt="%s")


if __name__ == '__main__':
    main()