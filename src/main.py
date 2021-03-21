"""
This script will first be predicting the interface residues. In addition, to this
binary prediction, it will compute various usefull mesurements helping to characterise
and identify interfaces.

In prediction mode, this script takes as an input the 3D structure and the PSSM of the protein you want
to predict, and will output a .tsv file containing the predictions and various usefull mesurements.

Input:
======
    protein1: Path to the protein to predict in PDB format.
    PSSM: Path to the PSI-BLAST PSSM of protein1.
    protein2 (test mode): Path to the protein binded to protein1 in PDB format. If selected, the program will be in test mode.
    output: Path to the output directory.
    color (prediction, probability, neighbors or cluster): Type of coloring used to replace the alpha-carbon B-factor column in the PDB file.
    neighbors: Radius max between predicted interface residue to be considered neighbors (default = 5A).
    cluster: Distance max between predicted interface residue to be considered in the same cluster (default = 10A).
    model: Path to the network model.

Output:
=======
    Output directory:
        {PDB_ID}_pred.csv: .tsv file containing the prediction and usefull mesurments.
        {PDB_ID}_pred.csv (test mode): .tsv file containing the prediction and usefull mesurments.
        {PDB_ID}_colored.pdb: PDB file in which the alpha-carbon B-factor is replaced with a selected mesurement.

Usage:
======
    $ python main.py [-h] -p1 PROTEIN1 -p PSSM [-p2 PROTEIN2] -o OUTPUT [-c {prediction, probability, neighbors, cluster}] [-nd NEIGHBORS] [-cd CLUSTER] [-m MODEL]
"""


__author__ = "Vander Meersche Yann"
__license__ = "M2-BI"
__version__ = "1.0.0"
__date__ = "19-03-2021"
__email__ = "yann-vm@hotmail.fr"
__copyright__ = "Copyright 2021, The Long Project Inc"


# MODULES ######################################################################
import os
import sys
import subprocess
import argparse

import pandas as pd 
import numpy as np 
import pbxplore
from scipy.spatial import distance_matrix
from sklearn.cluster import DBSCAN
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
################################################################################


# CONSTANTS ####################################################################
window = 15    #Window size
################################################################################


def args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-p1", "--protein1", help="Path to the protein to predict in PDB format.", type=isfile, required=True)
    parser.add_argument("-p", "--pssm", help="Path to the PSI-BLAST PSSM of protein1.", type=isfile, required=True)
    parser.add_argument("-p2", "--protein2", help="Path to the protein binded to protein1 in PDB format. If selected, the program will be in test mode.", type=isfile, required=False)
    parser.add_argument("-o", "--output", help="Path to the output directory.", type=str, required=True)
    parser.add_argument("-c", "--color", help="Type of coloring used to replace the alpha-carbon B-factor column in the PDB file.", type=str, default="prediction", choices=["prediction", "probability", "neighbors", "cluster"], required=False)
    parser.add_argument("-nd", "--neighbors", help="Radius max between predicted interface residue to be considered neighbors (default = 5A).", type=float, default=5, required=False)
    parser.add_argument("-cd", "--cluster", help="Distance max between predicted interface residue to be considered in the same cluster (default = 10A).", type=float, default=10, required=False)
    parser.add_argument("-m", "--model", help="Path to the network model.", type=isfile, required=False, default="../result/model/model.h5")
    args = parser.parse_args()

    return args.protein1, args.pssm, args.protein2, args.model, args.output, args.neighbors, args.cluster, args.color


def isfile(path):
    """
    Check if path is an existing file.
    
    Arguments
    ---------
        path: Path to the file
    Returns
    -------
        path: Path to the file
    """
    if not os.path.isfile(path):
        if os.path.isdir(path):
            msg = f"{path} is a directory."
        else:
            msg = f"{path} does not exist."
        raise argparse.ArgumentTypeError(msg)

    return path


def pssm_ascii2numpy(pssm_ascii):
    """
    Transform the PSI-BLAST PSSM file into a normalized Numpy array.
    
    PSSM computation:
        psiblast -query {PDB_ID}.fasta -db uniref50 -out {PDB_ID}.txt -out_ascii_pssm {PDB_ID}.pssm
    
    Arguments
    ---------
        psssm_ascii: Path to the PSI-BLAST PSSM file

    Returns
    -------
        pssm: PSSM Numpy array
    """
    pssm = []

    with open(pssm_ascii, "r") as f_pssm:
        for i, line in enumerate(f_pssm):
            if i > 2:
                if line == "\n":
                    break
                else:
                    pssm.append(line.split()[22:42])

    return np.array(pssm, dtype=np.int) / 100    #Normalize between 0 and 1 for efficient computation.


def parse_pdb(pdb_file):
    """
    Reads a PDB file and turns it into a Pandas dataframe.

    Arguments
    ---------
    pdb_file (string): Path to the PDB file

    Returns
    -------
    df_pdb (Pandas dataframe): Information of each atom of the PDB file
    """
    # List of list containing information about atoms from the PDB file
    rows = []

    with open(pdb_file, "r") as f_in:
        #Go through the file 
        for line in f_in:
            #If requiered take the first NMR structure
            if line.startswith("ENDMDL"):
               break
            #Extracts informations from the PDB
            if line.startswith("ATOM"):
                atom_num = int(line[6:11])
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain_id = line[21:22]
                res_num = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                #Appends these informations into a list
                rows.append([atom_num, atom_name, res_name, chain_id, res_num, x, y, z])

    #Create a Pandas dataframe containing atoms informations
    return pd.DataFrame(rows, columns=["atom_num", "atom_name", "res_name", "chain_id", "res_num", "x", "y", "z"])


def color_pdb(pdb_file, output, bfactor):
    """
    Reads a input PDB file and returns an output PDB file where the alpha-carbons of the B-factor column are replaced with an user selected value.

    Arguments
    ---------
    pdb_file (string): Path to the PDB file
    output: Path to the output directory
    bfactor: Numpy array of value you want to 

    Returns
    -------
    None
    """
    #Extract the PDB ID name from the input PDB file ({PDB_ID}.pdb)
    pdb_ID = os.path.basename(pdb_file).split(".")[0]

    with open(pdb_file, "r") as f_in, open(os.path.join(output, f"{pdb_ID}_colored.pdb"), "w") as f_out:
        i = 0
        # Go through the file 
        for line in f_in:
            # If requiered take the first NMR structure
            if line.startswith("ENDMDL"):
               break
            # Extracts informations from the PDB
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                line = line[:60] + f"{bfactor[i]:6.2f}" + line[66:]
                i += 1
            f_out.write(line)


def pdb2fasta(pdb_file):
    """
    Reads a PDB file and extract its sequence.

    Arguments
    ---------
    pdb_file (string): Path to the PDB file

    Returns
    -------
    fasta_seq (string): One letter sequence of the protein
    """
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
            if line[12:16].strip() == "CA":
                fasta_seq += letters[line[17:20].strip()]

    return fasta_seq


def fasta2aaindex(fasta_seq):
    """
    Convert an amino-acid sequence to a Numpy array of the 58 most informative AAindexes.

    Arguments
    ---------
    fasta_seq (string): One letter sequence of the protein

    Returns
    -------
    onehot_aaindex (Numpy array): Numpy array of the 58 most informative AAindexes
    """
    aaindex = {"A":[0.504, 0.249, 0.510, 0.112, 0.118, 0.000, 0.000, 0.904, 0.346, 0.248, 0.393, 0.109, 0.552, 0.124, 0.141, 0.578, 0.182, 0.000, 0.000, 0.000, 0.000, 0.837, 0.395, 0.676, 0.125, 0.815, 0.964, 0.500, 0.418, 0.700, 0.207, 0.999, 0.456, 0.106, 0.294, 0.000, 0.467, 0.406, 0.347, 0.217, 0.441, 0.147, 0.564, 0.360, 1.000, 0.980, 0.404, 0.405, 0.007, 0.210, 0.266, 0.000, 0.295, 0.471, 0.221, 0.397, 0.439, 0.514],
               "R":[0.110, 0.940, 0.667, 0.711, 0.461, 0.000, 1.000, 0.436, 0.421, 0.495, 0.244, 0.767, 0.719, 0.759, 1.000, 0.578, 0.916, 1.000, 0.750, 1.000, 0.000, 0.756, 0.691, 0.028, 0.775, 0.630, 0.465, 1.000, 0.779, 0.000, 1.000, 0.995, 0.105, 1.000, 0.529, 0.327, 0.299, 0.333, 0.162, 0.377, 0.134, 0.614, 0.308, 1.000, 0.316, 0.000, 0.525, 0.302, 0.082, 0.049, 0.455, 0.354, 0.759, 0.321, 0.506, 0.130, 0.346, 0.257],
               "N":[0.046, 0.675, 0.745, 0.328, 0.765, 1.000, 1.000, 0.106, 0.391, 0.908, 0.125, 0.442, 0.604, 0.365, 0.437, 0.578, 0.589, 0.500, 0.750, 0.000, 0.000, 0.640, 0.827, 0.268, 0.550, 0.859, 0.490, 0.500, 0.729, 0.111, 0.532, 0.963, 0.110, 0.483, 0.235, 0.140, 0.065, 0.157, 0.047, 0.449, 0.276, 0.250, 0.282, 0.773, 0.218, 0.459, 0.727, 0.149, 0.088, 0.389, 0.746, 0.000, 0.029, 0.164, 0.431, 0.082, 0.343, 0.314],
               "D":[0.000, 0.867, 0.745, 0.257, 0.608, 1.000, 0.000, 0.468, 0.128, 0.963, 0.034, 0.449, 0.615, 0.344, 0.465, 0.578, 0.486, 0.250, 1.000, 0.000, 1.000, 1.000, 1.000, 0.225, 0.562, 0.663, 0.495, 0.000, 0.696, 0.111, 0.535, 0.993, 0.186, 0.472, 0.235, 0.140, 0.000, 0.006, 0.040, 0.646, 0.150, 0.216, 0.256, 0.675, 0.241, 0.402, 0.828, 0.202, 1.000, 0.401, 0.775, 0.000, 1.000, 0.021, 0.330, 0.114, 0.294, 0.057],
               "C":[0.387, 0.205, 0.608, 0.313, 0.588, 0.000, 1.000, 0.138, 0.617, 0.450, 0.647, 0.357, 0.688, 0.301, 0.418, 0.578, 0.421, 0.000, 0.000, 0.000, 0.000, 0.645, 0.074, 1.000, 0.000, 0.207, 0.234, 0.500, 0.000, 0.778, 0.371, 0.932, 0.132, 0.356, 0.559, 0.140, 1.000, 0.906, 0.967, 0.029, 1.000, 0.429, 1.000, 0.174, 0.796, 0.837, 0.010, 0.124, 0.027, 0.586, 0.057, 0.000, 0.656, 0.760, 0.308, 0.897, 1.000, 1.000],
               "E":[0.032, 0.811, 0.667, 0.369, 0.245, 1.000, 0.000, 1.000, 0.000, 0.440, 0.000, 0.558, 0.750, 0.468, 0.679, 0.578, 0.404, 0.250, 1.000, 0.000, 1.000, 0.963, 0.914, 0.183, 0.625, 0.565, 0.539, 0.000, 0.717, 0.111, 0.707, 0.969, 0.172, 0.661, 0.529, 0.140, 0.131, 0.055, 0.091, 0.571, 0.161, 0.315, 0.256, 0.704, 0.172, 0.436, 0.960, 0.182, 0.068, 0.031, 0.689, 0.183, 0.046, 0.092, 0.368, 0.065, 0.242, 0.086],
               "Q":[0.131, 0.795, 0.667, 0.440, 0.275, 0.000, 1.000, 0.574, 0.549, 0.450, 0.014, 0.550, 0.740, 0.489, 0.703, 0.578, 0.442, 0.500, 0.750, 0.000, 0.000, 0.798, 0.691, 0.183, 0.638, 0.641, 0.352, 0.500, 0.711, 0.111, 0.694, 0.967, 0.094, 0.650, 0.529, 0.140, 0.192, 0.145, 0.112, 0.991, 0.177, 0.348, 0.256, 0.752, 0.179, 0.472, 0.672, 0.099, 0.075, 0.222, 1.000, 0.180, 0.603, 0.178, 0.338, 0.130, 0.316, 0.143],
               "G":[0.170, 1.000, 0.000, 0.000, 0.912, 1.000, 0.000, 0.000, 0.286, 1.000, 0.115, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.663, 0.506, 0.690, 0.063, 0.272, 1.000, 0.500, 0.704, 0.456, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.364, 0.262, 0.283, 0.351, 0.303, 0.000, 0.513, 0.454, 0.637, 1.000, 0.641, 0.000, 0.054, 1.000, 0.627, 0.000, 0.040, 0.275, 0.000, 0.049, 0.378, 0.429],
               "H":[0.053, 0.112, 0.686, 0.562, 0.471, 0.000, 1.000, 0.457, 0.376, 0.440, 0.475, 0.620, 0.667, 0.577, 0.550, 0.578, 0.815, 0.250, 0.250, 1.000, 0.000, 0.499, 0.679, 0.437, 0.362, 0.717, 0.092, 0.500, 0.635, 0.144, 0.742, 0.969, 0.233, 0.667, 0.529, 0.140, 0.426, 0.559, 0.478, 0.131, 0.657, 0.575, 0.667, 0.607, 0.548, 0.433, 0.232, 0.492, 0.075, 0.167, 0.865, 0.209, 0.192, 0.326, 0.478, 0.283, 0.588, 0.571],
               "I":[0.543, 0.671, 1.000, 0.455, 0.069, 0.000, 0.000, 0.543, 0.925, 0.000, 1.000, 0.434, 1.000, 0.495, 0.497, 1.000, 0.435, 0.000, 0.000, 0.000, 0.000, 0.845, 0.037, 0.887, 0.100, 0.848, 0.485, 0.500, 0.094, 1.000, 0.492, 0.975, 0.214, 0.544, 0.824, 0.308, 0.797, 1.000, 0.930, 0.049, 0.980, 0.588, 0.923, 0.031, 0.766, 0.989, 0.116, 0.711, 0.014, 0.074, 0.000, 0.000, 0.000, 1.000, 0.787, 0.772, 0.701, 0.857],
               "L":[0.989, 0.281, 0.961, 0.455, 0.098, 0.000, 0.000, 0.681, 0.699, 0.028, 0.773, 0.434, 0.958, 0.495, 0.497, 0.578, 0.603, 0.000, 0.000, 0.000, 0.000, 0.842, 0.000, 0.803, 0.137, 0.315, 0.793, 0.500, 0.355, 0.922, 0.559, 0.968, 0.096, 0.533, 0.824, 0.308, 0.742, 0.942, 0.773, 0.031, 0.941, 0.626, 0.846, 0.031, 0.778, 0.995, 0.157, 0.702, 0.000, 0.000, 0.377, 0.000, 0.000, 0.734, 0.814, 0.799, 0.677, 0.600],
               "K":[0.004, 0.687, 0.667, 0.535, 0.392, 0.000, 1.000, 0.628, 0.278, 0.661, 0.247, 0.551, 0.812, 0.590, 0.839, 0.578, 0.677, 0.500, 0.250, 1.000, 0.000, 0.750, 0.790, 0.000, 1.000, 0.511, 0.783, 1.000, 1.000, 0.067, 0.790, 1.000, 0.147, 0.833, 0.529, 0.327, 0.027, 0.000, 0.000, 1.000, 0.000, 0.325, 0.000, 0.799, 0.000, 0.466, 1.000, 0.434, 0.075, 0.123, 0.434, 0.530, 0.294, 0.000, 0.364, 0.000, 0.207, 0.000],
               "M":[1.000, 0.000, 0.765, 0.540, 0.000, 0.000, 1.000, 0.936, 0.511, 0.119, 0.725, 0.574, 0.802, 0.548, 0.747, 0.578, 0.664, 0.000, 0.000, 0.000, 0.000, 0.747, 0.099, 0.690, 0.188, 0.739, 0.054, 0.500, 0.115, 0.711, 0.629, 0.927, 0.080, 0.678, 0.765, 0.308, 0.615, 0.788, 0.827, 0.000, 0.965, 0.680, 0.846, 0.220, 0.842, 0.827, 0.242, 0.492, 0.014, 0.179, 0.025, 0.000, 0.652, 0.603, 0.650, 0.522, 0.442, 0.600],
               "F":[0.670, 0.076, 0.686, 0.709, 0.118, 0.000, 1.000, 0.596, 0.759, 0.174, 0.844, 0.698, 0.740, 0.729, 0.444, 0.578, 0.878, 0.000, 0.000, 0.000, 0.000, 0.757, 0.037, 0.775, 0.063, 0.283, 0.336, 0.500, 0.139, 0.811, 0.798, 0.969, 0.078, 0.733, 0.853, 0.682, 0.553, 0.968, 0.616, 0.009, 0.965, 0.811, 0.923, 0.000, 0.635, 0.859, 0.056, 0.950, 0.000, 0.228, 0.041, 0.000, 0.749, 0.665, 0.838, 0.967, 0.782, 0.743],
               "P":[0.220, 0.859, 0.353, 0.320, 1.000, 0.000, 0.000, 0.000, 0.135, 1.000, 0.356, 0.310, 0.000, 0.337, 0.356, 0.578, 0.579, 0.000, 0.000, 0.000, 0.000, 0.000, 0.383, 0.310, 0.500, 0.359, 0.439, 0.500, 0.882, 0.322, 0.382, 0.671, 0.400, 0.372, 0.588, 0.271, 0.127, 0.118, 0.091, 0.469, 0.465, 0.186, 0.308, 0.209, 0.601, 0.728, 0.732, 0.620, 0.088, 0.235, 0.316, 0.000, 0.157, 0.012, 0.354, 0.207, 0.000, 0.171],
               "S":[0.238, 0.851, 0.520, 0.152, 0.735, 0.000, 0.000, 0.213, 0.286, 0.881, 0.058, 0.232, 0.573, 0.198, 0.332, 0.578, 0.297, 0.250, 0.500, 0.000, 0.000, 0.673, 0.531, 0.451, 0.338, 1.000, 0.812, 0.500, 0.588, 0.411, 0.344, 0.966, 0.349, 0.278, 0.206, 0.000, 0.158, 0.137, 0.108, 0.346, 0.268, 0.140, 0.359, 0.468, 0.475, 0.666, 0.717, 0.107, 0.041, 0.327, 0.910, 0.000, 0.656, 0.155, 0.227, 0.114, 0.258, 0.257],
               "T":[0.273, 0.598, 0.490, 0.264, 0.480, 0.000, 0.000, 0.277, 0.617, 0.468, 0.125, 0.341, 0.656, 0.322, 0.356, 0.811, 0.379, 0.250, 0.500, 0.000, 0.000, 0.680, 0.457, 0.380, 0.338, 0.891, 0.643, 0.500, 0.554, 0.422, 0.384, 0.960, 0.298, 0.367, 0.235, 0.140, 0.265, 0.305, 0.199, 0.309, 0.358, 0.270, 0.462, 0.396, 0.461, 0.674, 0.470, 0.285, 0.027, 0.475, 0.029, 0.000, 0.745, 0.256, 0.354, 0.228, 0.352, 0.343],
               "W":[0.333, 0.040, 0.686, 1.000, 0.167, 0.000, 1.000, 0.543, 0.752, 0.119, 0.966, 1.000, 0.875, 1.000, 0.976, 0.578, 0.857, 0.250, 0.000, 0.000, 0.000, 0.835, 0.062, 0.648, 0.150, 0.000, 0.000, 0.500, 0.170, 0.400, 0.962, 0.970, 0.078, 0.906, 1.000, 1.000, 0.698, 0.961, 0.504, 0.100, 0.811, 1.000, 0.846, 0.053, 0.651, 0.629, 0.000, 1.000, 0.041, 0.364, 0.258, 1.000, 0.434, 0.681, 1.000, 1.000, 0.810, 0.914],
               "Y":[0.191, 0.502, 0.686, 0.729, 0.471, 0.000, 1.000, 0.128, 0.827, 0.615, 0.763, 0.822, 0.740, 0.801, 0.464, 0.578, 1.000, 0.250, 0.500, 0.000, 0.000, 0.756, 0.160, 0.296, 0.450, 0.272, 0.317, 0.500, 0.299, 0.356, 0.903, 0.902, 0.000, 0.861, 0.853, 0.682, 0.591, 0.649, 0.583, 0.377, 0.594, 0.710, 0.615, 0.592, 0.443, 0.619, 0.126, 0.711, 0.041, 0.265, 0.115, 0.730, 0.409, 0.591, 0.761, 0.728, 0.675, 0.600],
               "V":[0.355, 0.365, 0.745, 0.342, 0.039, 0.000, 0.000, 0.521, 1.000, 0.110, 0.769, 0.326, 0.927, 0.371, 0.356, 1.000, 0.379, 0.000, 0.000, 0.000, 0.000, 0.854, 0.123, 0.859, 0.113, 0.793, 0.747, 0.500, 0.255, 0.967, 0.401, 0.998, 0.237, 0.394, 0.647, 0.234, 0.814, 0.884, 1.000, 0.054, 0.894, 0.483, 0.872, 0.149, 0.794, 0.982, 0.136, 0.632, 0.007, 0.198, 0.029, 0.000, 0.045, 0.859, 0.601, 0.728, 0.602, 0.657]}

    onehot_aaindex = []
    for aa in fasta_seq:
        onehot_aaindex.append(aaindex[aa])

    return np.array(onehot_aaindex)


def fasta2onehot(fasta_seq):
    """
    One-hot encode an amino-acid sequence.

    Arguments
    ---------
    fasta_seq (string): One letter sequence of the protein

    Returns
    -------
    onehot_seq (Numpy array): Numpy array of the sequence in one-hot encoding
    """
    onehot = {"A":[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              "R":[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              "N":[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              "D":[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              "C":[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              "E":[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              "Q":[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              "G":[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              "H":[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              "I":[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              "L":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              "K":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              "M":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
              "F":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              "P":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              "S":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
              "T":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
              "W":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
              "Y":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
              "V":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}

    onehot_seq = []
    for aa in fasta_seq:
        onehot_seq.append(onehot[aa])

    return np.array(onehot_seq)


def pdb2struct(pdb_file):
    """
    Transform a PDB file into a Numpy array of the Stride secondary structure assignation.

    Arguments
    ---------
    pdb_file (string): Path to the PDB file

    Returns
    -------
    onehot_struct (Numpy array): Numpy array of the Stride secondary structure assignation in one-hot encoding
    """
    onehot = {"H":[1, 0, 0, 0, 0, 0, 0],
              "G":[0, 1, 0, 0, 0, 0, 0],
              "I":[0, 0, 1, 0, 0, 0, 0],
              "E":[0, 0, 0, 1, 0, 0, 0],
              "B":[0, 0, 0, 0, 1, 0, 0],
              "b":[0, 0, 0, 0, 1, 0, 0],
              "T":[0, 0, 0, 0, 0, 1, 0],
              "C":[0, 0, 0, 0, 0, 0, 1]}

    sp = subprocess.run(["stride", pdb_file], capture_output=True, text=True).stdout

    onehot_struct = []
    for line in sp.split("\n"):
        if line.startswith("ASG"):
            onehot_struct.append(onehot[line[24]])

    return np.array(onehot_struct)


def pdb2rsa(pdb_file):
    """
    Transform a PDB file into a Numpy array of the relative solvent accessibility calculated by Stride.

    Arguments
    ---------
    pdb_file (string): Path to the PDB file

    Returns
    -------
    onehot_struct (Numpy array): Numpy array of the Stride secondary structure assignation in one-hot encoding
    """
    #Tien et al. 2013 (theor.)
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
 
    sp = subprocess.run(["stride", pdb_file], capture_output=True, text=True).stdout

    RSA = []
    for line in sp.split("\n"):
        if line.startswith("ASG"):
            ASA = float(line[60:69].strip())
            aa = line[4:8].strip()
            RSA.append(round(ASA / maxRSA[aa], 3))

    RSA = np.array(RSA)

    return RSA.reshape(RSA.shape[0], -1)



def pdb2pb(pdb_file):
    """
    Transform a PDB file into a Numpy array of the protein blocks assignation in one-hot calculated by pbxplore.

    Arguments
    ---------
    pdb_file (string): Path to the PDB file

    Returns
    -------
    onehot_pb (Numpy array): Numpy array of the secondary structure assignation in one-hot encoding
    """
    pb = {"a":[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          "b":[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          "c":[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          "d":[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          "e":[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          "f":[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          "g":[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          "h":[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          "i":[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          "j":[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
          "k":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
          "l":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
          "m":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
          "n":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
          "o":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
          "p":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
          "Z":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}

    structure_reader = pbxplore.chains_from_files([pdb_file])
    chain_name, chain = next(structure_reader)
    dihedrals = chain.get_phi_psi_angles()
    pb_seq = pbxplore.assign(dihedrals)

    onehot_pb = []
    for aa in pb_seq:
        onehot_pb.append(pb[aa])

    return np.array(onehot_pb)




def pdb2asa(pdb_file):
    """
    Transform a PDB file into a Numpy array of the solvent accessibility calculated by Stride.

    Arguments
    ---------
    pdb_file (string): Path to the PDB file

    Returns
    -------
    arr_ASA (Numpy array): Numpy array of the solvent accessibility
    """
    sp = subprocess.run(["stride", pdb_file], capture_output=True, text=True).stdout

    arr_ASA = []

    for line in sp.split("\n"):
        if line.startswith("ASG"):
            ASA = float(line[60:69].strip())
            arr_ASA.append(ASA)

    return(np.array(arr_ASA))


def mergepdb(pdb_file1, pdb_file2):
    """
    Concatenate two PDB files containing a single chain into one containing the complex.

    Arguments
    ---------
    pdb_file1 (string): Path to the PDB file 1
    pdb_file2 (string): Path to the PDB file 2

    Returns
    -------
    None
    """
    pdb_dir = os.path.dirname(pdb_file1)

    #Create an hidden temporary PDB file
    with open(".temp_mergedPDB.pdb", "w") as f_pdb:
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


def get_interface_asa(pdb_file1, pdb_file2):
    """
    Calculate the difference of solvent accessibility of pdb_file1 after complexation with pdb_file2

    Arguments
    ---------
    pdb_file1 (string): Path to the PDB file 1
    pdb_file2 (string): Path to the PDB file 2

    Returns
    -------
    assa_diff (Numpy array): Numpy array of the solvent accessibility difference of pdb_file1 after complexation with pdb_file2
    """
    asa = pdb2asa(pdb_file1)

    mergepdb(pdb_file1, pdb_file2)
    asa_merge = pdb2asa(".temp_mergedPDB.pdb")
    asa_diff = asa - asa_merge[:len(asa)]

    return asa_diff


def res_type(pdb_file1):
    """
    Calculate the the type of the residues (burried or surface).

    Arguments
    ---------
    pdb_file1 (string): Path to the PDB file 1

    Returns
    -------
    arr_res_type (Numpy array): Numpy array of the type of the residues (burried or surface)
    """
    asa = pdb2asa(pdb_file1)

    arr_res_type = np.full((len(asa)), "b")    #burried
    arr_res_type[(asa > 0)] = "s"    #surface

    return arr_res_type


def res_type_test(pdb_file1, pdb_file2):
    """
    Calculate the the type of the residues in test mode (burried, surface or interface).

    Arguments
    ---------
    pdb_file1 (string): Path to the PDB file 1

    Returns
    -------
    arr_res_type (Numpy array): Numpy array of the type of the residues (burried, surface or interface)
    """
    asa_1 = pdb2asa(pdb_file1)

    mergepdb(pdb_file1, pdb_file2)
    asa_merge = pdb2asa(".temp_mergedPDB.pdb")

    arr_res_type = np.full((len(asa_1)), "b")    #burried
    asa_diff1 = asa_1 - asa_merge[:len(asa_1)]

    arr_res_type[(asa_1 > 0) & (asa_diff1 >= 1)] = "i"    #interface
    arr_res_type[(asa_1 > 0) & (asa_diff1 >= 0) & (asa_diff1 < 1)] = "s"    #surface

    return arr_res_type


def encode_x(pdb_file, pssm_ascii):
    """
    Use a PDB file and the associated PSSM file to compute the descriptors to encode the sequence with.

    Arguments
    ---------
    pdb_file (string): Path to the PDB file
    pssm_ascii (string): Path to the PSI-BLAST PSSM file

    Returns
    -------
    arr_x (Numpy array): Numpy array with all the features required to encode the sequence.
    """
    fasta_seq = pdb2fasta(pdb_file)
    arr_pssm = pssm_ascii2numpy(pssm_ascii)
    
    arr_aaindex = fasta2aaindex(fasta_seq)
    arr_onehot = fasta2onehot(fasta_seq)
    arr_struct = pdb2struct(pdb_file)
    arr_rsa = pdb2rsa(pdb_file)
    arr_pb = pdb2pb(pdb_file)

    return np.concatenate([arr_pssm, arr_aaindex, arr_onehot, arr_struct, arr_rsa, arr_pb], axis=1)


def window_X(arr_x, window):
    """
    Encode the amino acids with their neighbors in a given size window.

    Arguments
    ---------
    arr_x (Numpy array): Numpy array with all the features required to encode the sequence.
    window (int): Window size

    Returns
    -------
    arr_x_window (Numpy array): Path to the PDB file
    """
    padding = window // 2

    all_merged = []
    for j in range(arr_x.shape[0]):
        aa_merged = np.zeros((window, arr_x.shape[1]+1))
        c = 0

        for k in range(j-padding, j+padding+1):
            if k < 0 or k >= arr_x.shape[0]:
                row = np.append(np.zeros(arr_x.shape[1]), 1)
                aa_merged[c] = row
            else:
                row = np.append(arr_x[k], 0)
                aa_merged[c] = row
            c += 1
        all_merged.append(aa_merged)

    return np.array(all_merged)


def del_TER(pdb_file):
    """
    Delete the "TER" lines of gaps in the PDB file as it create problems with pbxplore

    Arguments
    ---------
    pdb_file (string): Path to the PDB file

    Returns
    -------
    None
    """
    with open(pdb_file, "r") as f:
        lines = f.readlines()
    with open(pdb_file, "w") as f:
        for line in lines:
            if not line.startswith("TER"):
                f.write(line)


def main():
    #Get the commande line arguments
    protein1, pssm, protein2, modelh5, OUTPUT, dist_neighbors, dist_cluster, color = args()

    #Create the output directory if required
    try:
        os.makedirs(OUTPUT)
    except OSError:
        print(f"The directory {OUTPUT} already exists.")
    else:
        print(f"Successfully created the directory {OUTPUT}.")

    #Get the PDB ID from the name of the PDB file
    pdb_ID = os.path.basename(protein1).split(".")[0]

    # Load model
    model = load_model(modelh5, compile=False)

    model.compile(loss="BinaryCrossentropy",
                  optimizer="Adam",
                  metrics=["accuracy"],
                  weighted_metrics=["accuracy"])

    #Delete the "TER" lines of gaps in the PDB file as it create problems with pbxplore
    del_TER(protein1)

    #Create X
    arr_merge = encode_x(protein1, pssm)
    x_test = window_X(arr_merge, window)

    # Predict on test dataset
    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    prob_y_pred = y_pred[:, 1]

    #Get the fasta sequence
    fasta_seq = pdb2fasta(protein1)
    seq_AA = list(fasta_seq)

    #Number of neighbors
    df_pdb = parse_pdb(protein1)
    arr_coors = df_pdb[df_pdb["atom_name"] == "CA"][["x", "y", "z"]].to_numpy()
    
    dist_mat = distance_matrix(arr_coors, arr_coors[y_pred_classes == 1])

    arr_nb_neighbors = np.zeros((dist_mat.shape))
    arr_nb_neighbors[dist_mat <= dist_neighbors] = 1
    arr_nb_neighbors = arr_nb_neighbors.sum(axis=1).astype(int)
    
    #Cluster
    arr_coors = arr_coors[y_pred_classes == 1]
    clustering = DBSCAN(eps=dist_cluster, min_samples=1).fit(arr_coors)
    labels = clustering.labels_

    arr_cluster = np.zeros((y_pred_classes.shape[0]))
    labels[labels == -1] = -2
    labels = labels+1
    arr_cluster[y_pred_classes == 1] = labels
    arr_cluster = arr_cluster.astype(int)

    #Solvant accessibility (ASA)
    asa = pdb2asa(protein1)
    
    #Select the metric to replace the alpha-carbon B-factor with
    if color == "prediction":
        color_class = y_pred_classes
    elif color == "probability":
        color_class = prob_y_pred
    elif color == "neighbors":
        color_class = arr_nb_neighbors
    elif color == "cluster":
        color_class = arr_cluster

    color_pdb(protein1, OUTPUT, color_class)


    if not protein2:
        #Res type
        arr_res_type = res_type(protein1)

        #Create the output dataframe
        zippedList =  list(zip(seq_AA, y_pred_classes, prob_y_pred, asa, arr_res_type, arr_nb_neighbors, arr_cluster))
        df = pd.DataFrame(zippedList, columns = ["res" , f"Ypred", "P_Ypred", "ASA", "res_type", "nb_neighbors", "cluster"])
        df.to_csv(os.path.join(OUTPUT, f"{pdb_ID}_pred.csv"), index=False, float_format="%.2f", sep="\t")

    elif protein2:
        #Res type
        arr_res_type = res_type_test(protein1, protein2)

        #Load Y
        asa_diff = get_interface_asa(protein1, protein2)
        bin_asa = (asa_diff >= 1).astype(int)

        #Create the test mode output dataframe
        zippedList =  list(zip(seq_AA, y_pred_classes, prob_y_pred, asa, bin_asa, arr_res_type, arr_nb_neighbors, arr_cluster))
        df = pd.DataFrame(zippedList, columns = ["res" , f"Ypred", "P_Ypred", "ASA", "Ytrue", "res_type", "nb_neighbors", "cluster"])
        df.to_csv(os.path.join(OUTPUT, f"{pdb_ID}_pred_test.csv"), index=False, float_format="%.2f", sep="\t")


if __name__ == "__main__":
    main()
