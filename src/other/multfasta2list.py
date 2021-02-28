import sys
import os

if len(sys.argv) <= 1:
    print("usage: python multifasta2list.py multi.fasta > pdbID_list.txt")
    exit()
 
input_file = sys.argv[1]


with open(input_file, "r") as f_fasta:
    for line in f_fasta:
        if line.startswith(">"):
            sys.stdout.write(f"{line[1:].strip()}\n")
