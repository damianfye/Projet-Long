import os
import sys
import argparse

import numpy as np
import h5py
from progress.bar import IncrementalBar
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cv", help="Path to the CV list directory.", type=str, required=True)
    parser.add_argument("-x", "--x", help="Path to the X window files directory.", type=str, required=True)
    parser.add_argument("-y", "--y", help="Path to the Y ASA files directory.", type=str, required=True)
    parser.add_argument("-f", "--folds", help="Number of CV folds.", type=int, required=False, default=1)
    parser.add_argument("-d", "--dim", metavar="D", type=int, nargs=2, help="ROWS COLS", required=True)
    args = parser.parse_args()

    ROWS, COLS = args.dim

    return args.cv, args.x, args.y, args.folds, ROWS, COLS


def merge_asa(pdb_list, y):
    bar = IncrementalBar("Generating Y array...", max=len(pdb_list))
    arr_asa_all = np.array([])

    for pdb_id in pdb_list:
        arr_asa = np.load(os.path.join(y, f"{pdb_id}.npy"))
        arr_asa_all = np.append(arr_asa_all, arr_asa)
        bar.next()
    bar.finish()

    return arr_asa_all



def main():
    cv_dir, x, y, nbfolds, ROWS, COLS = args()
    

    for i in range(nbfolds):

        print(f"\nFold {i}")
        path_CV = os.path.join(cv_dir, f"CV_{i}")

            
        pdb_list = np.loadtxt(os.path.join(path_CV, f"val_pdb_id.txt"), dtype="str")
        
        arr_asa_all = merge_asa(pdb_list, y)
        N = len(arr_asa_all)

        Y = to_categorical(arr_asa_all >= 1, 2)

        with h5py.File(os.path.join(path_CV, f"Y_val.h5"), "w") as f_Y:
            dset = f_Y.create_dataset(f"y_val", data = Y)


        #Create X files
        bar = IncrementalBar("Generating X files", max = len(pdb_list))

        with h5py.File(os.path.join(path_CV, f"X_val.h5"), "w") as f_X:
            dset = f_X.create_dataset(f"x_val", (N, ROWS, COLS), dtype = np.float32)

            count = 0
            for pdb_ID in pdb_list:
                X = np.load(os.path.join(x, f"{pdb_ID}_merged.npy"))
                dset[count:count+X.shape[0]] = X
                count += X.shape[0]
                bar.next()
        bar.finish()




        pdb_list_learn = np.loadtxt(os.path.join(path_CV, f"learn_pdb_id.txt"), dtype="str")
        print(len(pdb_list_learn))
        pdb_list_test = np.loadtxt(os.path.join(path_CV, f"test_pdb_id.txt"), dtype="str")
        print(len(pdb_list_test))

        pdb_list = np.append(pdb_list_learn, pdb_list_test)
        print(len(pdb_list))

        arr_asa_all = merge_asa(pdb_list, y)
        N = len(arr_asa_all)

        Y = to_categorical(arr_asa_all >= 1, 2)

        with h5py.File(os.path.join(path_CV, f"Y_learn_test.h5"), "w") as f_Y:
            dset = f_Y.create_dataset(f"y_learn", data = Y)


        #Create X files
        bar = IncrementalBar("Generating X files", max = len(pdb_list))

        with h5py.File(os.path.join(path_CV, f"X_learn_test.h5"), "w") as f_X:
            dset = f_X.create_dataset(f"x_learn", (N, ROWS, COLS), dtype = np.float32)

            count = 0 
            for pdb_ID in pdb_list:
                X = np.load(os.path.join(x, f"{pdb_ID}_merged.npy"))
                dset[count:count+X.shape[0]] = X 
                count += X.shape[0]
                bar.next()
        bar.finish()





if __name__ == '__main__':
    main()
