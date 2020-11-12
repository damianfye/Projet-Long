import sys
import os

if len(sys.argv) <= 1:
    print("usage: python list2list.py file.txt > file_i.txt")
    exit()
 
input_file = sys.argv[1]


with open(input_file, "r") as f_in:
    for line in f_in:
        sys.stdout.write(f"{line.strip()}_1\n")
        sys.stdout.write(f"{line.strip()}_2\n")
