# Projet-Long

This script will first be predicting the interface residues. In addition, to this
binary prediction, it will compute various usefull mesurements helping to characterise
and identify interfaces.
In prediction mode, this script takes as an input the 3D structure and the PSSM of the protein you want
to predict, and will output a .tsv file containing the predictions and various usefull mesurements.

Input:
- protein1: Path to the protein to predict in PDB format.
- PSSM: Path to the PSI-BLAST PSSM of protein1.
- protein2 (test mode): Path to the protein binded to protein1 in PDB format. If selected, the program will be in test mode.
- output: Path to the output directory.
- color (prediction, probability, neighbors or cluster): Type of coloring used to replace the alpha-carbon B-factor column in the PDB file.
- neighbors: Radius max between predicted interface residue to be considered neighbors (default = 5A).
- cluster: Distance max between predicted interface residue to be considered in the same cluster (default = 10A).
- model: Path to the network model.

Output:
- Output directory:
    - {PDB_ID}_pred.csv: .tsv file containing the prediction and usefull mesurments.
    - {PDB_ID}_pred.csv (test mode): .tsv file containing the prediction and usefull mesurments.
    - {PDB_ID}_colored.pdb: PDB file in which the alpha-carbon B-factor is replaced with a selected mesurement.

Requirements:
- Conda environement

Usage:
    $ python main.py [-h] -p1 PROTEIN1 -p PSSM [-p2 PROTEIN2] -o OUTPUT [-c {prediction, probability, neighbors, cluster}] [-nd NEIGHBORS] [-cd CLUSTER] [-m MODEL]
