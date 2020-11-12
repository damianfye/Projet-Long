import sys
import os
import subprocess


if len(sys.argv) <= 1:
    print("usage: python pdb2fasta.py file.pdb > file.fasta")
    exit()
 
input_file = sys.argv[1]

maxRSA = {"ALA": 129.0,
          "ARG": 274.0,
          "ASN": 195.0,
          "ASP": 193.0,
          "CYS": 167.0,
          "GLU": 223.0,
          "GLN": 225.0,
          "GLY": 104.0,
          "HIS": 224.0,
          "ILE": 197.0,
          "LEU": 201.0,
          "LYS": 236.0,
          "MET": 224.0,
          "PHE": 240.0,
          "PRO": 159.0,
          "SER": 155.0,
          "THR": 172.0,
          "TRP": 285.0,
          "TYR": 263.0,
          "VAL": 174.0}
 

sp = subprocess.run(["stride", input_file], capture_output=True, text=True).stdout

for line in sp.split("\n"):
    if line.startswith("ASG"):
        ASA = float(line[60:69].strip())
        aa = line[4:8].strip()
        sys.stdout.write(f"{ASA / maxRSA[aa]:.3f}\n")
