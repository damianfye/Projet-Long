import sys
import os

if len(sys.argv) <= 1:
    print("usage: python pdb2fasta.py file.pdb > file.fasta")
    exit()
 
input_file = sys.argv[1]

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

residue_nb = 0

with open(input_file, "r") as f_pdb:
    sys.stdout.write(f">{os.path.basename(input_file)[:-4]}\n")
    for line in f_pdb:
        residue_nb2 = line[22:26].strip()
        if line[12:16].strip() == "CA" and residue_nb != residue_nb2:
            sys.stdout.write(letters[line[17:20].strip()])
            residue_nb = residue_nb2
    sys.stdout.write("\n")
