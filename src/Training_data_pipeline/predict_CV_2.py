

import os
import sys
import subprocess
import argparse 
import pandas as pd 
import numpy as np 
import pbxplore
from tabulate import tabulate
from progress.bar import IncrementalBar #pip install progress
from sklearn.metrics import matthews_corrcoef
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


#Constantes
window = 15
ROWS = 15
COLS = 124


def args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="Path to the h5 CV directory.", type=str, required=True)
    parser.add_argument("-t", help="Path to the 25 test PDB ID list.", type=str, required=False)
    parser.add_argument("-m", help="Path to the models directory.", type=str, required=True)
    parser.add_argument("-x", "--x", help="Path to the PDB directory.", type=str, required=True)
    parser.add_argument("-p", "--pssm", help="Path to the PSSM directory.", type=str, required=True)
    parser.add_argument("-o", "--output", help="Path to the output directory.", type=str, required=True)
    parser.add_argument("-n", "--neighbors", help="Radius max between predicted interface residue to be considered neighbors.", type=float, default=5)
    parser.add_argument("-c", "--cluster", help="Distance max between predicted interface residue to be considered in the same cluster. (aggregative clustering)", type=float, default=10)
    args = parser.parse_args()

    return args.i, args.m, args.x, args.pssm, args.output, args.neighbors, args.cluster, args.t



import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import backend as K
 
#import tensorflow_io as tfio

from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, Flatten, Activation
from tensorflow.keras.layers import MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D
from tensorflow.keras.layers import Reshape, BatchNormalization
from tensorflow.keras.layers import add, concatenate, multiply
 
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, TensorBoard, EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical 
 

from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report
from sklearn.utils import class_weight 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import label_binarize 
from sklearn.multiclass import OneVsRestClassifier 
 
#import tensorflow_addons
from tensorflow.keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives, AUC
from sklearn.metrics import confusion_matrix


def pssm_ascii2numpy(pssm_ascii):
    pssm = []

    with open(pssm_ascii, "r") as f_pssm:
        for i, line in enumerate(f_pssm):
            if i > 2:
                if line == "\n":
                    break
                else:
                    pssm.append(line.split()[22:42])

    pssm = np.array(pssm, dtype=np.int) / 100    #Normalize between 0 and 1 for efficient computation.
    
    return pssm


def parse_pdb(pdb_file):
    """
    Reads a PDB file and returns a list of list (PDB file lines) and a NumPy array (PDB file coordinates).

    Arguments
    ---------
    pdb_file (string): Path to the PDB file

    Returns
    -------
    arr_coors (NumPy array): Coordinates of each atom of the PDB file
    rows (list): All sequence information
    """
    # List of list containing information about atoms from the PDB file
    rows = []

    with open(pdb_file, "r") as f_in:
        # Go through the file 
        for line in f_in:
            # If requiered take the first NMR structure
            if line.startswith("ENDMDL"):
               break
            # Extracts informations from the PDB
            if line.startswith("ATOM"):
                atom_num = int(line[6:11])
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain_id = line[21:22]
                res_num = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                # Appends these informations into a list
                rows.append([atom_num, atom_name, res_name, chain_id, res_num, x, y, z])

    # Create a NumPy array containing atoms coordinates
    arr_coors = pd.DataFrame(rows, columns=["atom_num", "atom_name", "res_name", "chain_id", "res_num", "x", "y", "z"])

    return arr_coors


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




def fasta2aaindex(fasta_seq):
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

    return(np.array(onehot_aaindex))


def fasta2onehot(fasta_seq):
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

    return(np.array(onehot_seq))


def pdb2struct(pdb_file):
    onehot = {"H":[1, 0, 0, 0, 0, 0, 0],
              "G":[0, 1, 0, 0, 0, 0, 0],
              "I":[0, 0, 1, 0, 0, 0, 0],
              "E":[0, 0, 0, 1, 0, 0, 0],
              "B":[0, 0, 0, 0, 1, 0, 0],
              "b":[0, 0, 0, 0, 1, 0, 0],
              "T":[0, 0, 0, 0, 0, 1, 0],
              "C":[0, 0, 0, 0, 0, 0, 1]}

    onehot_struct = []

    sp = subprocess.run(["stride", pdb_file], capture_output=True, text=True).stdout

    for line in sp.split("\n"):
        if line.startswith("ASG"):
            onehot_struct.append(onehot[line[24]])

    return(np.array(onehot_struct))


def pdb2rsa(pdb_file):
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
 
    RSA = []

    sp = subprocess.run(["stride", pdb_file], capture_output=True, text=True).stdout

    for line in sp.split("\n"):
        if line.startswith("ASG"):
            ASA = float(line[60:69].strip())
            aa = line[4:8].strip()
            RSA.append(round(ASA / maxRSA[aa], 3))

    RSA = np.array(RSA)

    return(RSA.reshape(RSA.shape[0], -1))



def pdb2pb(pdb_file):
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

    onehot_pb = []

    structure_reader = pbxplore.chains_from_files([pdb_file])
    chain_name, chain = next(structure_reader)
    dihedrals = chain.get_phi_psi_angles()
    pb_seq = pbxplore.assign(dihedrals)

    for aa in pb_seq:
        onehot_pb.append(pb[aa])

    return(np.array(onehot_pb))


def encode_x(pdb_id, pdb_dir, pssm_dir):

    pdb_file = os.path.join(pdb_dir, f"{pdb_id}.pdb")
    pssm_ascii = os.path.join(pssm_dir, f"{pdb_id}.pssm")

    fasta_seq = pdb2fasta(pdb_file)
    arr_pssm = pssm_ascii2numpy(pssm_ascii)
    arr_aaindex = fasta2aaindex(fasta_seq)
    arr_onehot = fasta2onehot(fasta_seq)
    arr_struct = pdb2struct(pdb_file)
    arr_rsa = pdb2rsa(pdb_file)
    arr_pb = pdb2pb(pdb_file)

    return np.concatenate([arr_pssm, arr_aaindex, arr_onehot, arr_struct, arr_rsa, arr_pb], axis=1), fasta_seq


def window_X(arr_x, window):
    all_merged = []
        
    padding = window // 2

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


def pdb2asa(pdb_file):

    sp = subprocess.run(["stride", pdb_file], capture_output=True, text=True).stdout

    arr_ASA = []

    for line in sp.split("\n"):
        if line.startswith("ASG"):
            ASA = float(line[60:69].strip())
            arr_ASA.append(ASA)

    return(np.array(arr_ASA))


def color_pdb(pdb_file, output, bfactor):
    """
    Reads a PDB file and returns a list of list (PDB file lines) and a NumPy array (PDB file coordinates).

    Arguments
    ---------
    pdb_file (string): Path to the PDB file

    Returns
    -------
    arr_coors (NumPy array): Coordinates of each atom of the PDB file
    rows (list): All sequence information
    """
    # List of list containing information about atoms from the PDB file
    pdb_ID = os.path.basename(protein1).split(".")[0]

    with open(pdb_file, "r") as f_in, open(os.path.join(output, f"{pdb_ID}_colored.pdb"), "w") as f_out:
        # Go through the file 
        i = 0
        for line in f_in:
            # If requiered take the first NMR structure
            if line.startswith("ENDMDL"):
               break
            # Extracts informations from the PDB
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                line = line[:60] + f"{bfactor[i]:6.2f}" + line[66:]
                i += 1

            f_out.write(line)


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


def get_interface_asa(pdb_id, pdb_dir):

    pdb_file1 = os.path.join(pdb_dir, f"{pdb_id[:-2]}_1.pdb")
    pdb_file2 = os.path.join(pdb_dir, f"{pdb_id[:-2]}_2.pdb")

    asa_1 = pdb2asa(pdb_file1)
    asa_2 = pdb2asa(pdb_file2)

    mergepdb(pdb_file1, pdb_file2)
    asa_merge = pdb2asa(os.path.join(pdb_dir, ".temp_mergedPDB.pdb"))

    if len(asa_1) + len(asa_2) == len(asa_merge):
        asa_diff1 = asa_1 - asa_merge[:len(asa_1)]
        asa_diff2 = asa_2 - asa_merge[len(asa_1):]

    if pdb_id[-1] == "1":
        return asa_diff1, asa_1
    if pdb_id[-1] == "2":
        return asa_diff2, asa_2


def res_type(pdb_id, pdb_dir):
    pdb_file1 = os.path.join(pdb_dir, f"{pdb_id[:-2]}_1.pdb")
    pdb_file2 = os.path.join(pdb_dir, f"{pdb_id[:-2]}_2.pdb")

    asa_1 = pdb2asa(pdb_file1)
    asa_2 = pdb2asa(pdb_file2)

    mergepdb(pdb_file1, pdb_file2)
    asa_merge = pdb2asa(os.path.join(pdb_dir, ".temp_mergedPDB.pdb"))

    if len(asa_1) + len(asa_2) == len(asa_merge):
        if pdb_id[-1] == "1":
            arr_res_type = np.full((len(asa_1)), "b")    #burried
            asa_diff1 = asa_1 - asa_merge[:len(asa_1)]

            arr_res_type[(asa_1 > 0) & (asa_diff1 >= 1)] = "i"    #interface
            arr_res_type[(asa_1 > 0) & (asa_diff1 >= 0) & (asa_diff1 < 1)] = "s"    #surface

            return arr_res_type

        if pdb_id[-1] == "2":
            arr_res_type = np.full((len(asa_2)), "b")    #burried
            asa_diff2 = asa_2 - asa_merge[len(asa_1):]

            arr_res_type[(asa_2 > 0) & (asa_diff2 >= 1)] = "i"    #interface
            arr_res_type[(asa_2 > 0) & (asa_diff2 >= 0) & (asa_diff2 < 1)] = "s"    #surface

            return arr_res_type



if __name__ == "__main__":

    entry_dir, model_dir, x, pssm_dir, OUTPUT, dist_neighbors, dist_cluster, list25 = args()

    try:
        os.makedirs(OUTPUT)
    except OSError:
        print(f"The directory {OUTPUT} already exists.")
    else:
        print(f"Successfully created the directory {OUTPUT}.")

    row_lists = []
    clusters_dict = {}  
    cluster_pdb = []
    clusters_dict_interface = {}
    clusters_dict_interface_new = {}
    list25 = np.loadtxt(list25, dtype="str")

    for i in range(10):

        #Get entry files path
        ENTRY = os.path.join(entry_dir, f"CV_{i}")
        MODEL = os.path.join(model_dir, f"CV_{i}")

        PATH_TEST_LIST = os.path.join(ENTRY, "test_pdb_id.txt")
        PATH_MODEL_JSON = os.path.join(MODEL, "model.json")
        PATH_MODEL_H5 = os.path.join(MODEL, f"weights.h5")


        ### LOAD MODEL
        # Load JSON arch
        with open(PATH_MODEL_JSON, "r") as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        print(f"Loaded model architecture from: {PATH_MODEL_JSON}")
        # Load weights from H5
        model.load_weights(PATH_MODEL_H5)
        print(f"Loaded weights from: {PATH_MODEL_H5}")



        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                      optimizer="Adam",
                      metrics=["accuracy",
                              AUC(name="auc")],
                      weighted_metrics=["accuracy"])


        pdb_ID_list = np.loadtxt(PATH_TEST_LIST, dtype="str")


        bar = IncrementalBar("Generating array...", max=len(pdb_ID_list))

        for pdb_ID in pdb_ID_list:

            if pdb_ID in list25:

                #Load X
                arr_merge, fasta_seq = encode_x(pdb_ID, x, pssm_dir)
                x_test = window_X(arr_merge, window)

                #Load Y
                asa_diff, ASA = get_interface_asa(pdb_ID, x)
                bin_asa = (asa_diff >= 1).astype(int)
                y_test = to_categorical(y=bin_asa, num_classes=2)

                #Load fasta file
                seq_AA = list(fasta_seq)

                # Predict on test dataset
                y_pred = model.predict(x_test, verbose=0)
                y_pred_classes = np.argmax(y_pred, axis=1)
                prob_y_pred = y_pred[:, 1]
                #prob_y_pred = np.amax(y_pred, axis=1)

                #Res type
                arr_res_type = res_type(pdb_ID, x)

                # Number of neighbors
                arr_coors = parse_pdb(os.path.join(x, f"{pdb_ID}.pdb"))
                arr_coors = arr_coors[arr_coors["atom_name"] == "CA"][["x", "y", "z"]].to_numpy()
                
                dist_mat = distance_matrix(arr_coors, arr_coors[y_pred_classes == 1])

                arr_nb_neighbors = np.zeros((dist_mat.shape))
                arr_nb_neighbors[dist_mat <= dist_neighbors] = 1
                arr_nb_neighbors = arr_nb_neighbors.sum(axis=1).astype(int)
                
                # Cluster
                arr_coors = arr_coors[y_pred_classes == 1]
                clustering = DBSCAN(eps=dist_cluster, min_samples=1).fit(arr_coors)
                labels = clustering.labels_

                arr_cluster = np.zeros((y_pred_classes.shape[0]))
                labels[labels == -1] = -2
                labels = labels+1
                arr_cluster[y_pred_classes == 1] = labels
                arr_cluster = arr_cluster.astype(int)


                zippedList =  list(zip(seq_AA, y_pred_classes, prob_y_pred, bin_asa, arr_res_type, arr_nb_neighbors, arr_cluster, ASA))
                # Create a dataframe from zipped list
                df = pd.DataFrame(zippedList, columns = ["res" , f"Ypred", "P_Ypred", "Ytrue", "res_type", "nb_neighbors", "cluster", "ASA"])
                df.to_csv(os.path.join(OUTPUT, f"{pdb_ID}.csv"), index=False, float_format="%.2f", sep="\t")
                bar.next()





                #Chercher une protéine avec plusieurs interfaces possibles : max(arr_cluster) > 1, sum(ASA) cluster > 1000A² 
                if max(arr_cluster) not in clusters_dict:
                    clusters_dict[max(arr_cluster)] = 1
                else:
                    clusters_dict[max(arr_cluster)] += 1





                nb_cluster = 0

                for cluster in range(1, max(arr_cluster)+1):
                    if sum(df[df["cluster"]==cluster]["ASA"]) > 1000 and len(df[df["cluster"]==cluster]) >= 15:
                        nb_cluster += 1

                if nb_cluster not in clusters_dict_interface:
                    clusters_dict_interface[nb_cluster] = 1
                else:
                    clusters_dict_interface[nb_cluster] += 1



                nb_cluster = 0

                for cluster in range(1, max(arr_cluster)+1):
                    if sum(df[df["cluster"]==cluster]["ASA"]) > 1000 and len(df[df["cluster"]==cluster]) == len(df[(df["cluster"]==cluster) & (df["Ytrue"]==0)]) and len(df[df["cluster"]==cluster]) >= 15:
                        cluster_pdb.append(pdb_ID)

                        nb_cluster += 1

                if nb_cluster not in clusters_dict_interface_new:
                    clusters_dict_interface_new[nb_cluster] = 1
                else:
                    clusters_dict_interface_new[nb_cluster] += 1




                protein1 = os.path.join(x, f"{pdb_ID}.pdb")
                color_pdb(protein1, OUTPUT, arr_cluster)



                """
                fig = plt.figure()
                ax = plt.axes(projection = "3d")

                # Creating plot
                ax.scatter3D(arr_coors[:, 0], arr_coors[:, 1], arr_coors[:, 2], c=labels, marker="o")
                #plt.title("simple 3D scatter plot")
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                # show plot
                plt.show()

                # Model evaluation
                y_tmp = np.argmax(y_test, axis=1)   # Convert one-hot to index
                class_weights_test = class_weight.compute_class_weight("balanced", classes=np.unique(y_tmp), y = y_tmp)
                class_weights_test = dict(enumerate(class_weights_test))
                sample_weights_test = class_weight.compute_sample_weight(class_weights_test, y_tmp)
                
                scores = model.evaluate(x=x_test, y=y_test, sample_weight=sample_weights_test, verbose=0)
                row = [pdb_ID] + scores + [matthews_corrcoef(y_tmp, y_pred_classes)] + [matthews_corrcoef(y_tmp, y_pred_classes, sample_weight=sample_weights_test)]
                row_lists.append(row)
                """

        print("dict_cluster", clusters_dict)
        print("\n")
        print(cluster_pdb)

        bar.finish()


    #df_scores = pd.DataFrame(row_lists, columns = ["PDBid", "loss", "accuracy", "auc", "accuracy_1", "MCC_sk", "w_MCC_sk"])
    #df_scores.to_csv(f"evaluate.csv", index=False, float_format="%.3f", sep="\t")
    print("\nALL:")
    print("dict_cluster", clusters_dict)
    
    import collections
    od = dict(collections.OrderedDict(sorted(clusters_dict.items())))
    print(od)

    od_i = dict(collections.OrderedDict(sorted(clusters_dict_interface.items())))
    print(od_i)

    od_i_n = dict(collections.OrderedDict(sorted(clusters_dict_interface_new.items())))
    print(od_i_n)


    print("\n")
    print(cluster_pdb)
    sys.exit()







"""
#ASA > 1000, res = 15
ALL:
dict_cluster = {1: 1190, 2: 619, 3: 393, 4: 310, 5: 192, 6: 124, 7: 63, 8: 50, 9: 32, 10: 23, 11: 10, 12: 12, 13: 2, 14: 3, 15: 1, 16: 1, 17: 2, 20: 1, 21: 1}
{0: 95, 1: 2358, 2: 468, 3: 78, 4: 21, 5: 7, 6: 2}
{0: 2548, 1: 420, 2: 47, 3: 12, 4: 2}
new_clusters = ['1d7f0A0B_1', '1dos0A0B_1', '1ds60A0B_2', '1i8l0A0C_1', '1itv0A0B_1', '1jl00A0B_2', '1ki10A0B_2', '1l7d0A0B_2', '1m7r0A0B_1', '1m7r0A0B_1', '1ojw0A0B_1', '1pfq0A0B_1', '1q3e1A1B_1', '1qjs0A0B_1', '1sqj0A0B_2', '1u7u1A2A_1', '1uw40A0B_2', '1we30F0T_2', '1x9m0A0B_1', '1x9m0A0B_1', '2b5u0A0B_1', '2dzd1A1B_1', '2f6j0B0C_1', '2fjr0A0B_1', '2iw50A0B_1', '2pf41A1E_2', '2qg70A0E_2', '2qjw0A0B_1', '2qr41A1B_1', '2r5w1B1A_1', '2v7b0A0B_1', '2w4m1A2A_1', '2wv01A1B_1', '2y9y1A1B_1', '2y9y1A1B_2', '2y9y1A1B_2', '2y9y1A1B_2', '2zzv1A1B_1', '3c2x1A1B_1', '3dal1A2B_1', '3f701A1B_2', '3fey1A1C_2', '3ffl1A1C_1', '3h841A1B_1', '3k5n1A1B_1', '3k5n1A1B_1', '3kbp1A2A_1', '3kcp1A1B_1', '3kdg1A1B_1', '3l0o1A1B_1', '3l0o1A1B_1', '3lc61A1B_1', '3mag1A2A_1', '3oc31A1C_1', '3oc31A1C_1', '3oc31A1C_1', '3ohm1A1B_2', '3q411A1B_1', '3q411A1B_1', '3q9t1A2A_1', '3rgc1A1B_2', '3rhg1A2A_1', '3rhg1A2A_1', '3syl1A1B_1', '7mdh0A0B_2', '1aro0P0L_1', '1aro0P0L_1', '1gxl0A0B_1', '1jeq0A0B_1', '1l0w0A0B_1', '1mq80A0B_1', '1nva0A0B_2', '1olz0A0B_1', '1oya1A2A_1', '1t3c0A0B_1', '1t3c0A0B_1', '1t6b0X0Y_1', '1w360C0D_2', '1xb40A0B_1', '2acv0A0B_1', '2b4c0G0C_2', '2bs30A0D_1', '2bs30A0D_1', '2bs30A0D_1', '2d2x0A0B_1', '2fgy0A0B_1', '2j9q0A0B_1', '2rd51B1C_1', '2v620A0B_2', '2vro1A1B_2', '2wg41A1B_2', '2wjd1A2A_1', '2xkj1E2E_1', '2z9o1A1B_1', '2zci1A1C_1', '3ciy1A1B_2', '3cqz1A1E_2', '3d5o1F3B_1', '3fey1A1C_1', '3fey1A1C_1', '3g2o1A1B_1', '3gjx1B1A_2', '3gjx1B1A_2', '3hl31A2A_1', '3mkq1A1B_1', '3mkq1A1B_1', '3mkq1A1B_1', '3nai1A2A_1', '3nek1A1B_2', '3nt71C2C_1', '3o2q1A1B_1', '3p7l1A2A_1', '3pnd1A1B_1', '3q011A1B_1', '3rau1A1B_1', '3rau1A1B_1', '3rgf1A1B_1', '3tac1A1B_2', '3tku1A1B_1', '4a271A1B_1', '1bo10A0B_1', '1bo10A0B_1', '1dto1A2A_1', '1g600A0B_1', '1h7s0A0B_1', '1r7a0A0B_1', '1t080A0B_1', '1t080A0B_1', '1uq51A2A_1', '1wwl0A0B_2', '1x3z0A0B_1', '1xdp1A1B_1', '1xg20A0B_1', '1y1u0A0B_1', '1y4j0A0B_2', '1ztv0A0B_1', '2bcj0A0Q_1', '2bcj0A0Q_1', '2bcj0A0Q_1', '2dvw0A0B_1', '2e0a1A1B_1', '2e4g0A0B_1', '2j040A0B_1', '2nrk1A2A_1', '2pzm1A1B_1', '2r151A1B_1', '2w0x1A2A_1', '2wcu1A1B_1', '2xpi1A1D_1', '2xpi1A1D_1', '3bu81A1B_2', '3bxw1B1A_1', '3ccm1F1M_2', '3cqc1A1B_2', '3cw71A1D_1', '3dl21A1B_1', '3drw1A1B_1', '3efo1A1B_2', '3efo1A1B_2', '3efo1A1B_2', '3gqb1A1B_2', '3hhm1A1B_1', '3hhm1A1B_1', '3hhm1A1B_1', '3i6s1A1B_1', '3iko1B1C_1', '3iko1B1C_2', '3iko1B1C_2', '3kvc1A1B_1', '3lb61A1C_2', '3mmy1A1B_1', '3nvo1A1B_1', '3pol1A2A_1', '3s4w1A1B_2', '3s4w1A1B_2', '3s971A1C_1', '3su81A1X_2', '3su81A1X_2', '1b430A0B_1', '1cq10A0B_2', '1d2z0A0B_2', '1jmv0A0B_1', '1kcf0A0B_1', '1kfu0L0S_1', '1kfu0L0S_1', '1koa1A2A_1', '1np60A0B_2', '1obb0A0B_1', '1qup0A0B_1', '1qvi0A0Y_1', '1qxm0A0B_1', '1sxj0C0F_1', '1ta30A0B_1', '1tll0A0B_1', '1ult0A0B_1', '1xfz0A0O_1', '1zjc1A2A_1', '2aa40A0B_1', '2e4u0A0B_2', '2gtd0A0B_1', '2h4m0A0B_1', '2h4m0A0B_1', '2h630A0C_1', '2jjf1A2A_1', '2pok0A0B_1', '2rgr1A2A_1', '2v9w0A0B_1', '2vdw1A1B_1', '2vz81A1B_2', '2vz81A1B_2', '2vz81A1B_2', '2vz81A1B_2', '2wxy1C2C_1', '2xot1A1B_2', '2xpl1A1B_1', '2xwb1F1J_1', '2xwu1A1B_2', '2xwu1A1B_2', '2zt51A2A_1', '3cw21A1C_2', '3e331A1B_2', '3e331A1B_2', '3ee61A1B_1', '3fmo1A1B_1', '3fpk1A1B_1', '3ggn1A1B_2', '3igh1X2X_1', '3ll81A1B_1', '3lxr1A1F_2', '3nh41A2A_1', '3opb1A1B_2', '3pih1A2A_1', '3pih1A2A_1', '3pss1A1B_1', '3rep1A1B_2', '3tho1A1B_2', '1aq00A0B_1', '1bg51A2A_1', '1bqu0A0B_2', '1dov1A2A_1', '1fp30A0B_1', '1gud0A0B_1', '1m2o0A0C_1', '1noy0A0B_1', '1p8j0A0B_1', '1pfo1A2A_1', '1qx40A0B_1', '1vq00A0B_1', '1xg20A0B_2', '1y0z0A0B_2', '1y8q0A0B_2', '1ypt0A0B_2', '1ypt0A0B_2', '1zd31A2A_1', '1zd31A2A_1', '1zm10A0B_1', '2a1s0A0D_2', '2a1s0A0D_2', '2c5s1A2A_1', '2d4c0A0B_2', '2e310A0B_1', '2e7y1A1B_1', '2o1u1A1B_1', '2ooe1A2A_1', '2owy1A1B_1', '2vw81A2A_1', '2wjv1A1D_1', '2wjv1A1D_1', '2y5s1A1B_2', '3add1A1B_1', '3cka1A1B_2', '3fhq1A1B_1', '3i9f1A1B_2', '3ipo1A1B_2', '3kz11A1F_1', '3l6p1A2A_1', '3o9x1A1B_1', '3opy1A1E_1', '3opy1A1E_1', '3pij1A1B_2', '3qbx1A1B_2', '3qdk1A1B_1', '3r381A2A_1', '3rfw1A2A_1', '3s9l1A1C_2', '1bcc0A0B_1', '1cm50A0B_1', '1ern0A0B_2', '1ete0A0B_1', '1g8s1A2A_1', '1h5y0A0B_2', '1ixs0A0B_2', '1kfq0A0B_1', '1odt0C0H_1', '1qh40A0B_1', '1qop1B2B_1', '1rqg1A2A_1', '1sg90A0C_1', '1t6g0A0C_1', '1tht0A0B_1', '1we30F0T_1', '1xzp0A0B_1', '1y3t0A0B_1', '1zt20A0B_1', '1zt20A0B_1', '1zt20A0B_1', '2dfj0A0B_1', '2eul0A0B_1', '2eul0A0B_1', '2fq10A0B_1', '2hz61A2A_1', '2hz61A2A_1', '2id30A0B_1', '2ix30A0B_1', '2ix30A0B_1', '2pm90A0B_1', '2pr81A1B_1', '2q391A1B_1', '2qv21A2A_1', '2w8b1A1H_1', '2wus1A1R_1', '2x0s1A2A_1', '2x0s1A2A_1', '2xna1A1C_1', '2zb91A1B_1', '3ab81A1B_1', '3bny1A1D_2', '3cw91A1B_1', '3dax1A1B_1', '3edv1A1B_1', '3edv1A1B_1', '3ego1A2B_1', '3epg1A2A_1', '3err1A1B_1', '3err1A1B_1', '3f6c1A2B_1', '3gw71A1B_1', '3h3a1A1B_2', '3hju1A1B_1', '3ibp1A2A_1', '3iuc1A1C_2', '3kg81A1B_1', '3lfv1A1B_1', '3mj71A1B_2', '3mml1A1C_2', '3mve1A1B_1', '3njp1A1B_1', '3os61A1B_2', '3q2w1A2A_1', '3qv73D3A_1', '3r8c1A1B_2', '1bih0A0B_1', '1ct90A0D_1', '1d0n0A0B_1', '1e8u0A0B_2', '1ecj0A0B_1', '1nj11A2A_1', '1nj11A2A_1', '1o4z0A0B_2', '1otj0A0D_1', '1ql00A0B_1', '1u5e0A0B_1', '1yla0A0B_1', '2ey40A0C_1', '2ges1A2A_1', '2gjx0A0B_1', '2gw10A0B_1', '2ipc0A0B_1', '2ipc0A0B_1', '2ipc0A0B_1', '2itm0A0B_1', '2j0w1A2A_1', '2j3l0A0B_2', '2j5v0A0B_2', '2o7p0A0B_1', '2oq21A1B_2', '2owd1A1B_1', '2qj21A1B_2', '2quq1A2A_1', '2quq1A2A_1', '2quq1A2A_1', '2uy10A0B_1', '2uy10A0B_1', '2v8s0E0V_2', '2vqe1E1H_2', '2vso1A1E_1', '2w271A1B_1', '2xa71B1M_1', '2xec1A1D_1', '2xms1A2A_1', '3axy1A1C_2', '3bg91A1B_1', '3bg91A1B_1', '3cb21A1B_1', '3csv1A2A_1', '3cw21A1C_1', '3cwn1A2B_1', '3dlb1A1B_1', '3e201A1B_2', '3ejj1A1X_2', '3gty1X1S_1', '3hr01A1B_1', '3lj11A2A_1', '3lj11A2A_1', '3ln61A2A_1', '3ln61A2A_1', '3tc11A1B_2', '1a5t1A2A_1', '1bg61A2A_1', '1bvn0P0T_1', '1gl40A0B_1', '1gqp0A0B_2', '1hdh0A0B_1', '1jkw1A2A_1', '1pxv0A0C_1', '1r6w1A2A_1', '1s3s0F0G_1', '1t0f0A0C_1', '1w360C0D_1', '1w360C0D_1', '1w360C0D_1', '1w360C0D_1', '1yz30A0B_2', '1zt20A0B_2', '2c261A2A_1', '2e2e1A1B_1', '2grr0A0B_2', '2hqy0A0B_1', '2ijd1121_1', '2qsh1A1X_1', '2qsh1A1X_1', '2r421A2A_1', '2rez1A2A_1', '2x401A2A_1', '2xp11A2A_1', '2y8n1A1C_1', '2yv91A1B_1', '2z861A1B_2', '2z861A1B_2', '3al91A1B_1', '3d0g1A1E_1', '3d851C1D_2', '3dp71A1B_1', '3ix71A1B_1', '3k6s1A1B_1', '3k6s1A1B_1', '3k6s1A1B_2', '3l3p1A2A_1', '3l461A1B_2', '3mwp1A1B_2', '3nnf1A2A_1', '3o4h1A1B_1', '3qqa1A2A_1', '3s4u1A2A_1', '3zrl1A1B_2', '1a6d1B7B_1', '1am21A2A_1', '1aoz0A0B_1', '1c4k1A2A_1', '1c4k1A2A_1', '1eku0A0B_2', '1ev70A0B_1', '1gv40A0B_1', '1gvm0C0D_1', '1hrn0A0B_1', '1iyb0A0B_1', '1j7n0A0B_2', '1lpa0A0B_2', '1mkz0A0B_2', '1n4o0A0B_2', '1nh81A2A_1', '1t6z0A0B_2', '1w3b0A0B_1', '1wvu0A0B_2', '1xdt1T1R_1', '2be50B0D_2', '2be50B0D_2', '2fpr0A0B_2', '2if70A0B_2', '2vav1A1I_1', '2veo0A0B_1', '2wm91A2A_1', '2yin1B1D_1', '2z3z1A2A_1', '3a6z1A1C_2', '3aql1A1B_1', '3c251A1B_1', '3dem1A1B_1', '3e7j1A1B_1', '3e7p1A2A_1', '3eqx1A1B_1', '3h2u1A1B_2', '3hzz1C1D_1', '3hzz1C1D_1', '3lnc1A1B_1', '3lq71A1B_1', '3maj1A2A_1', '3maj1A2A_1', '3n081A1B_2', '3nk61A1B_1', '3o8c1A1B_1', '3qit1A1B_2', '3rlf1F1B_1', '3zss1A1B_1', '1ava0A0C_1', '1b3q0A0B_2', '1div1A2A_1', '1gww0A0B_1', '1i3q0A0H_1', '1i3q0A0H_1', '1i3q0A0H_1', '1ixr1B1C_1', '1jkx0A0B_1', '1k200A0B_1', '1p0w0A0B_2', '1pem1A2A_1', '1r891A2A_1', '1ryz0C0E_1', '1tdq0A0B_1', '1uiv0A0B_1', '1uiv0A0B_1', '1un80A0B_1', '2bmi0A0B_2', '2cw60A0B_1', '2ek51A1B_1', '2fi70A0B_1', '2h1l0C0O_1', '2ie40A0C_1', '2rag1A1B_1', '2rc81A1B_2', '2wss1A1S_2', '2xbb1A1C_1', '2z8v1A1D_1', '3apt1A1B_1', '3bni1A1B_1', '3bof1A1B_1', '3cio1A1D_1', '3d3k1A1B_1', '3da41A2B_1', '3do71A1B_1', '3do71A1B_1', '3dzy1A1D_2', '3ei41A1B_1', '3ei41A1B_1', '3ei41A1B_1', '3ghg1C1F_2', '3hme1A1B_2', '3i831A1B_1', '3irm1A1B_1', '3kjh1A2A_1', '3lsr1A2A_1', '3lvj1B1D_1', '3nc91A1B_2', '3nf21A2A_1', '3qel1B1D_1', '3r8s1T1Y_1', '3s4w1A1B_1', '3tj11A1B_1']

#ASA > 1000, res = 10
ALL:
{1: 1190, 2: 619, 3: 393, 4: 310, 5: 192, 6: 124, 7: 63, 8: 50, 9: 32, 10: 23, 11: 10, 12: 12, 13: 2, 14: 3, 15: 1, 16: 1, 17: 2, 20: 1, 21: 1}
{0: 63, 1: 2236, 2: 562, 3: 120, 4: 31, 5: 12, 6: 3, 7: 2}
{0: 2441, 1: 494, 2: 70, 3: 18, 4: 5, 6: 1}
new_clusters = ['1d7f0A0B_1', '1dos0A0B_1', '1ds60A0B_2', '1ezv1D1H_1', '1i8l0A0C_1', '1ii51A2A_1', '1itv0A0B_1', '1jl00A0B_2', '1ki10A0B_2', '1l7d0A0B_2', '1m7r0A0B_1', '1m7r0A0B_1', '1ojw0A0B_1', '1pfq0A0B_1', '1q3e1A1B_1', '1q670A0B_2', '1qha0A0B_1', '1qjs0A0B_1', '1sqj0A0B_2', '1u7u1A2A_1', '1uw40A0B_2', '1we30F0T_2', '1x9m0A0B_1', '1x9m0A0B_1', '1x9m0A0B_1', '2b5u0A0B_1', '2dzd1A1B_1', '2f6j0B0C_1', '2fjr0A0B_1', '2h0h1A2A_1', '2iw50A0B_1', '2omk0A0B_1', '2pf41A1E_2', '2qg70A0E_2', '2qg70A0E_2', '2qjw0A0B_1', '2qr41A1B_1', '2r5w1B1A_1', '2v7b0A0B_1', '2vvw1A1B_1', '2w4m1A2A_1', '2wv01A1B_1', '2y9y1A1B_1', '2y9y1A1B_2', '2y9y1A1B_2', '2y9y1A1B_2', '2y9y1A1B_2', '2zzv1A1B_1', '3c2x1A1B_1', '3dal1A2B_1', '3f701A1B_2', '3fey1A1C_2', '3fey1A1C_2', '3ffe1A1B_2', '3ffl1A1C_1', '3h841A1B_1', '3k5n1A1B_1', '3k5n1A1B_1', '3kbp1A2A_1', '3kcp1A1B_1', '3kdg1A1B_1', '3kgk1A1B_2', '3l0o1A1B_1', '3l0o1A1B_1', '3lc61A1B_1', '3mag1A2A_1', '3oc31A1C_1', '3oc31A1C_1', '3oc31A1C_1', '3ohm1A1B_2', '3q411A1B_1', '3q411A1B_1', '3q9t1A2A_1', '3rgc1A1B_2', '3rhg1A2A_1', '3rhg1A2A_1', '3syl1A1B_1', '7mdh0A0B_2', '1aro0P0L_1', '1aro0P0L_1', '1aro0P0L_1', '1dpg0A0B_1', '1gxl0A0B_1', '1hz41A2A_1', '1jeq0A0B_1', '1l0w0A0B_1', '1mq80A0B_1', '1mzg0A0B_1', '1nva0A0B_2', '1olz0A0B_1', '1oya1A2A_1', '1t3c0A0B_1', '1t3c0A0B_1', '1t3c0A0B_1', '1t3c0A0B_1', '1t6b0X0Y_1', '1w360C0D_2', '1xb40A0B_1', '1yvb0A0I_1', '1zk80A0B_1', '2acv0A0B_1', '2b4c0G0C_2', '2bs30A0D_1', '2bs30A0D_1', '2bs30A0D_1', '2d1g0A0B_1', '2d2x0A0B_1', '2etx0A0B_2', '2ex30A0B_1', '2fgy0A0B_1', '2h2w1A2A_1', '2j9q0A0B_1', '2rd51B1C_1', '2v620A0B_2', '2vro1A1B_2', '2vro1A1B_2', '2wg41A1B_2', '2wjd1A2A_1', '2x1k1A2A_1', '2xkj1E2E_1', '2z9o1A1B_1', '2zci1A1C_1', '3anw1A1B_2', '3ciy1A1B_2', '3cqz1A1E_2', '3d401A2A_1', '3d5o1F3B_1', '3f411A1B_1', '3f411A1B_1', '3fey1A1C_1', '3fey1A1C_1', '3g2o1A1B_1', '3gjx1B1A_2', '3gjx1B1A_2', '3hl31A2A_1', '3k0w1A2A_1', '3kt31A1B_1', '3mca1A1B_2', '3mkq1A1B_1', '3mkq1A1B_1', '3mkq1A1B_1', '3mw61A1B_1', '3n051A1B_1', '3nai1A2A_1', '3nek1A1B_2', '3nt71C2C_1', '3o2q1A1B_1', '3p7l1A2A_1', '3pnd1A1B_1', '3q011A1B_1', '3qne1A2A_1', '3rau1A1B_1', '3rau1A1B_1', '3rau1A1B_1', '3rgf1A1B_1', '3rv01A1B_2', '3tac1A1B_2', '3tku1A1B_1', '4a271A1B_1', '1bo10A0B_1', '1bo10A0B_1', '1dto1A2A_1', '1g600A0B_1', '1h7s0A0B_1', '1r7a0A0B_1', '1rve0A0B_1', '1t080A0B_1', '1t080A0B_1', '1uq51A2A_1', '1wwl0A0B_2', '1x3z0A0B_1', '1xdp1A1B_1', '1xg20A0B_1', '1y1u0A0B_1', '1y4j0A0B_2', '1ztv0A0B_1', '2bcj0A0Q_1', '2bcj0A0Q_1', '2bcj0A0Q_1', '2bcj0A0Q_1', '2dvw0A0B_1', '2e0a1A1B_1', '2e4g0A0B_1', '2hrz1A2A_1', '2iuw1A2A_1', '2j040A0B_1', '2nrk1A2A_1', '2otn1A1B_1', '2oxv1A2A_1', '2pzm1A1B_1', '2r151A1B_1', '2w0x1A2A_1', '2w0x1A2A_1', '2wcu1A1B_1', '2wyo1C1D_1', '2xpi1A1D_1', '2xpi1A1D_1', '2y2m1A2A_1', '3bt11A1U_1', '3bu81A1B_2', '3bxw1B1A_1', '3ccm1F1M_2', '3cqc1A1B_2', '3cw71A1D_1', '3dl21A1B_1', '3drw1A1B_1', '3drw1A1B_1', '3efo1A1B_2', '3efo1A1B_2', '3efo1A1B_2', '3gqb1A1B_2', '3hhm1A1B_1', '3hhm1A1B_1', '3hhm1A1B_1', '3i6s1A1B_1', '3iko1B1C_1', '3iko1B1C_2', '3iko1B1C_2', '3kvc1A1B_1', '3lb61A1C_2', '3mmy1A1B_1', '3nvo1A1B_1', '3pid1A2A_1', '3pol1A2A_1', '3pol1A2A_1', '3s4w1A1B_2', '3s4w1A1B_2', '3s971A1C_1', '3su81A1X_2', '3su81A1X_2', '1b430A0B_1', '1b430A0B_1', '1cq10A0B_2', '1d2z0A0B_2', '1ef00A0B_1', '1gla1F1G_1', '1jmv0A0B_1', '1kcf0A0B_1', '1kfu0L0S_1', '1kfu0L0S_1', '1koa1A2A_1', '1mi81A2A_1', '1np60A0B_2', '1obb0A0B_1', '1obb0A0B_1', '1qup0A0B_1', '1qvi0A0Y_1', '1qxm0A0B_1', '1qxm0A0B_1', '1sxj0C0F_1', '1sxj0C0F_1', '1ta30A0B_1', '1tll0A0B_1', '1tll0A0B_1', '1ult0A0B_1', '1xfz0A0O_1', '1y881A2A_1', '1zjc1A2A_1', '2aa40A0B_1', '2e4u0A0B_2', '2gtd0A0B_1', '2h4m0A0B_1', '2h4m0A0B_1', '2h630A0C_1', '2hzg0A0B_1', '2jjf1A2A_1', '2pml1X2X_1', '2pok0A0B_1', '2r371A2A_1', '2rgr1A2A_1', '2v9w0A0B_1', '2vdw1A1B_1', '2vz81A1B_2', '2vz81A1B_2', '2vz81A1B_2', '2vz81A1B_2', '2wxy1C2C_1', '2x4g1A2A_1', '2xhe1A1B_1', '2xhe1A1B_1', '2xot1A1B_2', '2xpl1A1B_1', '2xpl1A1B_1', '2xwb1F1J_1', '2xwu1A1B_2', '2xwu1A1B_2', '2zt51A2A_1', '3cw21A1C_2', '3e331A1B_2', '3e331A1B_2', '3ee61A1B_1', '3fmo1A1B_1', '3fpk1A1B_1', '3ggn1A1B_2', '3igh1X2X_1', '3ll81A1B_1', '3lxr1A1F_2', '3nh41A2A_1', '3opb1A1B_2', '3pih1A2A_1', '3pih1A2A_1', '3pss1A1B_1', '3rep1A1B_2', '3tho1A1B_2', '3tot1A1B_1', '1aq00A0B_1', '1bg51A2A_1', '1bqu0A0B_2', '1dov1A2A_1', '1fp30A0B_1', '1g6w0A0B_2', '1gud0A0B_1', '1ik90A0B_1', '1m2o0A0C_1', '1noy0A0B_1', '1p8j0A0B_1', '1pfo1A2A_1', '1qx40A0B_1', '1swv0A0B_1', '1vq00A0B_1', '1xg20A0B_2', '1y0z0A0B_2', '1y8q0A0B_2', '1ypt0A0B_2', '1ypt0A0B_2', '1zd31A2A_1', '1zd31A2A_1', '1zm10A0B_1', '2a1s0A0D_2', '2a1s0A0D_2', '2bll1A2A_1', '2c5s1A2A_1', '2d4c0A0B_2', '2e310A0B_1', '2e7y1A1B_1', '2fe30A0B_2', '2o1u1A1B_1', '2ooe1A2A_1', '2owy1A1B_1', '2vw81A2A_1', '2wjv1A1D_1', '2wjv1A1D_1', '2wjv1A1D_1', '2xrh1A2A_1', '2y4i1B1C_2', '2y5s1A1B_2', '3add1A1B_1', '3cka1A1B_2', '3euj1A1B_1', '3euj1A1B_1', '3fhq1A1B_1', '3i9f1A1B_2', '3ipo1A1B_2', '3ku41A1B_2', '3kz11A1F_1', '3l6p1A2A_1', '3lkb1A1B_1', '3men1A1B_2', '3mkk1A1B_1', '3o9x1A1B_1', '3opy1A1E_1', '3opy1A1E_1', '3pij1A1B_2', '3qbx1A1B_2', '3qdk1A1B_1', '3qml1A1C_2', '3r381A2A_1', '3rfw1A2A_1', '3s9l1A1C_2', '1bcc0A0B_1', '1cm50A0B_1', '1dp40A0C_2', '1epf0A0B_2', '1ern0A0B_2', '1ete0A0B_1', '1g8s1A2A_1', '1h5y0A0B_2', '1is31A2A_1', '1ixs0A0B_2', '1ixs0A0B_2', '1ixs0A0B_2', '1jhf0A0B_1', '1jhf0A0B_1', '1kfq0A0B_1', '1odt0C0H_1', '1qh40A0B_1', '1qop1B2B_1', '1rqg1A2A_1', '1sg90A0C_1', '1t6g0A0C_1', '1tht0A0B_1', '1we30F0T_1', '1xzp0A0B_1', '1y3t0A0B_1', '1zak0A0B_1', '1zt20A0B_1', '1zt20A0B_1', '1zt20A0B_1', '2dfj0A0B_1', '2eul0A0B_1', '2eul0A0B_1', '2fd51A2A_1', '2fq10A0B_1', '2hz61A2A_1', '2hz61A2A_1', '2id30A0B_1', '2ix30A0B_1', '2ix30A0B_1', '2ix30A0B_1', '2pm90A0B_1', '2pr81A1B_1', '2q391A1B_1', '2qv21A2A_1', '2r001A1B_1', '2w8b1A1H_1', '2wus1A1R_1', '2wus1A1R_1', '2x0s1A2A_1', '2x0s1A2A_1', '2x0s1A2A_1', '2xna1A1C_1', '2zb91A1B_1', '2zvy1A1B_1', '3ab81A1B_1', '3bny1A1D_2', '3cw91A1B_1', '3cw91A1B_1', '3dax1A1B_1', '3dax1A1B_1', '3edv1A1B_1', '3edv1A1B_1', '3ego1A2B_1', '3epg1A2A_1', '3err1A1B_1', '3err1A1B_1', '3f6c1A2B_1', '3gw71A1B_1', '3h3a1A1B_2', '3h3a1A1B_2', '3hju1A1B_1', '3ibp1A2A_1', '3iuc1A1C_2', '3k1i1A1D_2', '3kg81A1B_1', '3lfv1A1B_1', '3mj71A1B_2', '3mml1A1C_2', '3mml1A1C_2', '3mve1A1B_1', '3njp1A1B_1', '3os61A1B_2', '3q2w1A2A_1', '3qv73D3A_1', '3r8c1A1B_2', '3sxu1A1B_1', '3tr51C1D_2', '1bih0A0B_1', '1ct90A0D_1', '1d0n0A0B_1', '1e8u0A0B_2', '1ecj0A0B_1', '1ecj0A0B_1', '1nj11A2A_1', '1nj11A2A_1', '1o4z0A0B_2', '1otj0A0D_1', '1ql00A0B_1', '1u5e0A0B_1', '1yla0A0B_1', '2ey40A0C_1', '2ges1A2A_1', '2gjx0A0B_1', '2gw10A0B_1', '2ipc0A0B_1', '2ipc0A0B_1', '2ipc0A0B_1', '2itm0A0B_1', '2j0w1A2A_1', '2j3l0A0B_2', '2j5v0A0B_2', '2o7p0A0B_1', '2oq21A1B_2', '2owd1A1B_1', '2qj21A1B_2', '2quq1A2A_1', '2quq1A2A_1', '2quq1A2A_1', '2uy10A0B_1', '2uy10A0B_1', '2v8s0E0V_2', '2vqe1E1H_2', '2vso1A1E_1', '2w271A1B_1', '2xa71B1M_1', '2xec1A1D_1', '2xms1A2A_1', '3axy1A1C_2', '3bg91A1B_1', '3bg91A1B_1', '3cb21A1B_1', '3csv1A2A_1', '3cw21A1C_1', '3cwn1A2B_1', '3dlb1A1B_1', '3e201A1B_2', '3e571A1B_2', '3ejj1A1X_2', '3gty1X1S_1', '3hr01A1B_1', '3jsl1A1B_1', '3lj11A2A_1', '3lj11A2A_1', '3ln61A2A_1', '3ln61A2A_1', '3rbn1A1B_1', '3tc11A1B_2', '3tfz1A1B_1', '1a5t1A2A_1', '1b490A0C_1', '1bg61A2A_1', '1bvn0P0T_1', '1gl40A0B_1', '1gqp0A0B_2', '1hdh0A0B_1', '1jkw1A2A_1', '1pxv0A0C_1', '1r6w1A2A_1', '1s3s0F0G_1', '1t0f0A0C_1', '1w360C0D_1', '1w360C0D_1', '1w360C0D_1', '1w360C0D_1', '1w360C0D_1', '1w360C0D_1', '1xi30A0B_1', '1yz30A0B_2', '1zt20A0B_2', '2a190A0B_1', '2c261A2A_1', '2e2e1A1B_1', '2grr0A0B_2', '2hqy0A0B_1', '2ijd1121_1', '2ijd1121_1', '2j4r0A0B_1', '2qsh1A1X_1', '2qsh1A1X_1', '2qy61A2A_1', '2r421A2A_1', '2rez1A2A_1', '2vvm1A1B_1', '2x401A2A_1', '2xp11A2A_1', '2y8n1A1C_1', '2yv91A1B_1', '2z861A1B_2', '2z861A1B_2', '2z861A1B_2', '2zgq1A1B_1', '3al91A1B_1', '3d0g1A1E_1', '3d0g1A1E_1', '3d851C1D_2', '3dp71A1B_1', '3isq1A2A_1', '3ix71A1B_1', '3k6s1A1B_1', '3k6s1A1B_1', '3k6s1A1B_2', '3l3p1A2A_1', '3l461A1B_2', '3mwp1A1B_2', '3nnf1A2A_1', '3o4h1A1B_1', '3qqa1A2A_1', '3r2t1A2B_1', '3r2t1A2B_1', '3rt01A1C_1', '3s4u1A2A_1', '3zrl1A1B_2', '1a6d1B7B_1', '1am21A2A_1', '1aoz0A0B_1', '1aoz0A0B_1', '1c4k1A2A_1', '1c4k1A2A_1', '1c8k1A2A_1', '1eku0A0B_2', '1ev70A0B_1', '1gv40A0B_1', '1gv40A0B_1', '1gvm0C0D_1', '1hrn0A0B_1', '1ici0A0B_1', '1iyb0A0B_1', '1j7n0A0B_2', '1lpa0A0B_2', '1ma90A0B_1', '1mkz0A0B_2', '1n4o0A0B_2', '1n6m0A0B_1', '1nh81A2A_1', '1t6z0A0B_2', '1w3b0A0B_1', '1wvu0A0B_2', '1xdt1T1R_1', '1yea1A2A_1', '2be50B0D_2', '2be50B0D_2', '2be50B0D_2', '2be50B0D_2', '2bh71A2A_1', '2fpr0A0B_2', '2gs61A2A_1', '2if70A0B_2', '2vav1A1I_1', '2veo0A0B_1', '2wm91A2A_1', '2yin1B1D_1', '2z3z1A2A_1', '3a6z1A1C_2', '3aql1A1B_1', '3aql1A1B_1', '3bxf1A1B_1', '3c251A1B_1', '3dem1A1B_1', '3e7j1A1B_1', '3e7p1A2A_1', '3eqx1A1B_1', '3h2u1A1B_2', '3hu61A1B_1', '3hu61A1B_1', '3hzz1C1D_1', '3hzz1C1D_1', '3lnc1A1B_1', '3lq71A1B_1', '3maj1A2A_1', '3maj1A2A_1', '3n081A1B_2', '3nk61A1B_1', '3o8c1A1B_1', '3qit1A1B_2', '3rlf1F1B_1', '3zss1A1B_1', '1ava0A0C_1', '1b3q0A0B_2', '1div1A2A_1', '1fx41A2A_1', '1gww0A0B_1', '1i3q0A0H_1', '1i3q0A0H_1', '1i3q0A0H_1', '1ixr1B1C_1', '1jkx0A0B_1', '1k200A0B_1', '1mjh0A0B_2', '1p0w0A0B_2', '1pea1A2A_1', '1pem1A2A_1', '1r891A2A_1', '1ryz0C0E_1', '1tdq0A0B_1', '1uiv0A0B_1', '1uiv0A0B_1', '1un80A0B_1', '1v0d1A2A_1', '2bmi0A0B_2', '2cw60A0B_1', '2e670A0B_1', '2ek51A1B_1', '2fi70A0B_1', '2h1l0C0O_1', '2ie40A0C_1', '2ie40A0C_1', '2rag1A1B_1', '2rc81A1B_2', '2wss1A1S_2', '2wvq1A1B_1', '2xbb1A1C_1', '2z8v1A1D_1', '3apt1A1B_1', '3bni1A1B_1', '3bof1A1B_1', '3cio1A1D_1', '3cq41A1B_1', '3d3k1A1B_1', '3da41A2B_1', '3do71A1B_1', '3do71A1B_1', '3dzy1A1D_2', '3ei41A1B_1', '3ei41A1B_1', '3ei41A1B_1', '3ghg1C1F_2', '3h2t1A1B_1', '3hme1A1B_2', '3i831A1B_1', '3irm1A1B_1', '3kjh1A2A_1', '3lsr1A2A_1', '3lvj1B1D_1', '3m7k1A2A_1', '3n9i1A1B_1', '3nc91A1B_2', '3nc91A1B_2', '3nf21A2A_1', '3ojy1A1B_2', '3qel1B1D_1', '3qel1B1D_1', '3r8s1T1Y_1', '3s4w1A1B_1', '3szr1A2A_1', '3tj11A1B_1']




"""