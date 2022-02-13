import sys
import os
import argparse

import numpy as np
from progress.bar import IncrementalBar #pip install progress


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--pdblist", help="Path to the PDB ID lists.", type=str, required=True)
    parser.add_argument("-d", "--pdbdir", help="Path to the PDB directory.", type=str, required=True)
    parser.add_argument("-o", "--output", help="Path to the fastaA directory.", type=str, required=True)
    args = parser.parse_args()

    return args.pdblist, args.pdbdir, args.output


def fill(text, width=80):
    """Split text with a line return to respect fasta format"""
    return os.linesep.join(text[i:i+width] for i in range(0, len(text), width))


def pdb2fasta(pdb_file):
    letters = {"ALA":"A",
               "ARG":"R",
               "ASN":"N",
               "ASP":"D",
               "CYS":"C",
               "GLU":"E",
               "GLN":"Q",
               "GLY":"G",
               "HIS":"H",
               "ILE":"I",
               "LEU":"L",
               "LYS":"K",
               "MET":"M",
               "PHE":"F",
               "PRO":"P",
               "SER":"S",
               "THR":"T",
               "TRP":"W",
               "TYR":"Y",
               "VAL":"V"}

    fasta_seq = ""

    residue_nb = 0
    with open(pdb_file, "r") as f_pdb:
        for line in f_pdb:
            residue_nb2 = line[22:26].strip()
            if line[12:16].strip() == "CA" and residue_nb != residue_nb2:
                fasta_seq += letters[line[17:20].strip()]
                residue_nb = residue_nb2
    return(fasta_seq)



def main():
    pdb_list, pdb_dir, output_dir = args()
    
    pdb_list = np.loadtxt(pdb_list, dtype="str")

    bar = IncrementalBar("Generating multifasta file...", max=len(pdb_list))


    with open(os.path.join(output_dir, "multifasta.fasta"), "w") as f_out:

        for pdb_id in pdb_list:
            #1st protein of the complex
            pdb_file = os.path.join(pdb_dir, f"{pdb_id}_1.pdb")

            try:
                fasta_seq = pdb2fasta(pdb_file)
                f_out.write(f">{pdb_id}_1\n")
                f_out.write(fill(fasta_seq, width=80) + "\n")
            except:
                print(f"{pdb_id}_1")

            #2nd protein of the complex
            pdb_file = os.path.join(pdb_dir, f"{pdb_id}_2.pdb")

            try:
                fasta_seq = pdb2fasta(pdb_file)
                f_out.write(f">{pdb_id}_2\n")
                f_out.write(fill(fasta_seq, width=80) + "\n")
            except:
                print(f"{pdb_id}_2")


            bar.next()
    bar.finish()


if __name__ == "__main__":
    main()

