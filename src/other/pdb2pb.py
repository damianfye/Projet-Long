import sys
import os
import pbxplore


if len(sys.argv) <= 1:
    print("usage: python pdb2fasta.py file.pdb > file.fasta")
    exit()
 
input_file = sys.argv[1]


pb = {"a":"1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
      "b":"0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
      "c":"0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
      "d":"0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0",
      "e":"0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0",
      "f":"0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0",
      "g":"0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0",
      "h":"0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0",
      "i":"0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0",
      "j":"0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0",
      "k":"0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0",
      "l":"0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0",
      "m":"0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0",
      "n":"0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0",
      "o":"0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0",
      "p":"0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0",
      "Z":"0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1"}

structure_reader = pbxplore.chains_from_files([input_file])
chain_name, chain = next(structure_reader)
dihedrals = chain.get_phi_psi_angles()
pb_seq = pbxplore.assign(dihedrals)

for aa in pb_seq:
    sys.stdout.write(f"{pb[aa]}\n")