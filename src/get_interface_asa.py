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
    parser.add_argument("-d", "--pdbdir", help="Path to the PDB directory.", type=str, required=True)
    parser.add_argument("-o", "--output", help="Path to the ASA directory.", type=str, required=True)
    args = parser.parse_args()

    return args.pdblist, args.pdbdir, args.output


def pdb2asa(pdb_file):

    sp = subprocess.run(["stride", pdb_file], capture_output=True, text=True).stdout

    arr_ASA = []

    for line in sp.split("\n"):
        if line.startswith("ASG"):
            ASA = float(line[60:69].strip())
            arr_ASA.append(ASA)

    return(np.array(arr_ASA))


def mergepdb(pdb_file1, pdb_file2):

    pdb_dir = os.path.dirname(pdb_file1)

    with open(os.path.join(pdb_dir, ".temp_mergedPDB.pdb"), "w") as f_pdb:
        with open(pdb_file1, "r") as f_pdb1:
            for line1 in f_pdb1:
                if line1.startswith("ATOM"):
                    f_pdb.write(line1)
        f_pdb.write("TER\n")
        with open(pdb_file2, "r") as f_pdb2:
            for line2 in f_pdb2:
                if line2.startswith("ATOM"):
                    f_pdb.write(line2)
        f_pdb.write("TER\n")


def interface_res(pdb_dir, pdb_id):
    pass

def main():
    pdb_list, pdb_dir, output_dir = args()

    pdb_list = np.loadtxt(pdb_list, dtype="str")

    bar = IncrementalBar("Generating ASA array...", max=len(pdb_list))

    for pdb_id in pdb_list:
        
        pdb_file1 = os.path.join(pdb_dir, f"{pdb_id}_1.pdb")
        pdb_file2 = os.path.join(pdb_dir, f"{pdb_id}_2.pdb")


        asa_1 = pdb2asa(pdb_file1)
        asa_2 = pdb2asa(pdb_file2)

        mergepdb(pdb_file1, pdb_file2)
        asa_merge = pdb2asa(os.path.join(pdb_dir, ".temp_mergedPDB.pdb"))

        if len(asa_1) + len(asa_2) == len(asa_merge):
            asa_diff1 = asa_1 - asa_merge[:len(asa_1)]
            np.save(os.path.join(output_dir, f"{pdb_id}_1.npy"), asa_diff1)
            
            asa_diff2 = asa_2 - asa_merge[len(asa_1):]
            np.save(os.path.join(output_dir, f"{pdb_id}_2.npy"), asa_diff2)
        bar.next()
    bar.finish()

if __name__ == '__main__':
    main()