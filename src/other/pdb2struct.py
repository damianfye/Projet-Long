import sys
import os
import subprocess


if len(sys.argv) <= 1:
    print("usage: python pdb2fasta.py file.pdb > file.fasta")
    exit()
 
input_file = sys.argv[1]

onehot = {"H":"1 0 0 0 0 0 0",
          "G":"0 1 0 0 0 0 0",
          "I":"0 0 1 0 0 0 0",
          "E":"0 0 0 1 0 0 0",
          "B":"0 0 0 0 1 0 0",
          "b":"0 0 0 0 1 0 0",
          "T":"0 0 0 0 0 1 0",
          "C":"0 0 0 0 0 0 1"}


sp = subprocess.run(["stride", input_file], capture_output=True, text=True).stdout

for line in sp.split("\n"):
    if line.startswith("ASG"):
        sys.stdout.write(f"{onehot[line[24]]}\n")
