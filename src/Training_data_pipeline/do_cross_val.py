"""
This script performs the cross validation.

Inputs:
=======
    entry_dir: Path to the .h5 CV directory.
    output_dir: Path to the output directory.
    data: Type of entry data.
    ROWS, COLS: Dimensions of the data.

Output:
========
    CV_{i}: Cross validation directories:
        model.json: Model architecture.
        weights_{data}.h5: Model weights.
        CP_{data}.h5: Best weight checkpoint.
        history_{data}: History of training.
    logs: Tensorboard logs

Usage:
======
    $ python do_cross_val.py [-h] -i I -o O (-S | -NS | -3 | -5) -d D D
"""


# Modules ######################################################################
import os
import sys
import math
import argparse
import datetime

import h5py
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow import keras 
from tensorflow.keras import backend as K 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import plot_model
import tensorflow_addons
from tensorflow.keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives, AUC
import pickle
import tensorflow_io as tfio
import functools
from itertools import product

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
################################################################################


def args():
    """Parse the command-line arguments.

    Arguments
    ---------
    python command-line

    Returns
    -------
    entry_dir: Path to the .h5 CV directory.
    output_dir: Path to the output directory.
    data: Type of entry data.
    nb: Number of output classes.
    ROWS, COLS: Dimensions of the data.
    """
    #Declaration of expexted arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="Path to the .h5 CV directory.", type=str, required=True)
    parser.add_argument("-o", help="Path to the output directory.", type=str, required=True)
    parser.add_argument("-d", "--dim", metavar="D", type=int, nargs=2, help="ROWS COLS", required=True)
    parser.add_argument("-c", "--classes", type=int, help="Number of classes", default=2)
    parser.add_argument("-f", "--folds", help="Number of CV folds.", type=int, required=False, default=5)
    args = parser.parse_args()

    entry_dir = args.i
    output_dir = args.o
    ROWS, COLS = args.dim
    nb = args.classes
    folds = args.folds

    return entry_dir, output_dir, ROWS, COLS, nb, folds



if __name__ == "__main__":

    #Command-line arguments
    entry_dir, output_dir, ROWS, COLS, nb, folds = args()

    #Fixing seed for reproductibility
    np.random.seed(42)
    tf.random.set_seed(42)
    

    #Loop over the 10 cross validation folds
    for i in range(folds):

        print(f"K = {i+1}/{folds}")

        #Get entry files path
        ENTRY = os.path.join(entry_dir, f"CV_{i}")

        X_learn_file = os.path.join(ENTRY, "X_learn.h5")
        Y_learn_file = os.path.join(ENTRY, f"Y_learn.h5")
        X_val_file = os.path.join(ENTRY, "X_val.h5")
        Y_val_file = os.path.join(ENTRY, f"Y_val.h5")

        #Create output directory
        OUTPUT = os.path.join(output_dir, f"CV_{i}")
        os.makedirs(OUTPUT, exist_ok=True)

        #Network parameters
        BATCH_SIZE = 1024 #######################################################
        nb_filters = 128

        #Learning data
        #Load Y file
        with h5py.File(Y_learn_file, "r") as h5f:
            Y_learn = h5f["y_learn"][:]

        #Calculate class weights
        Y_tmp = np.argmax(Y_learn, axis=1)
        class_weights_learn = class_weight.compute_class_weight("balanced", np.unique(Y_tmp), Y_tmp)
        class_weights_learn = dict(enumerate(class_weights_learn))
        sample_weights_learn = class_weight.compute_sample_weight(class_weights_learn, Y_tmp)
        # Transform numpy array into tf.data.Dataset
        sample_weights_learn = tf.data.Dataset.from_tensor_slices((sample_weights_learn))

        X_learn = tfio.IODataset.from_hdf5(X_learn_file, dataset="/x_learn")
        Y_learn = tfio.IODataset.from_hdf5(Y_learn_file, dataset="/y_learn")


        #Validation data
        #Load Y file
        with h5py.File(Y_val_file, "r") as h5f:
            Y_val = h5f["y_val"][:]

        #Calculate class weights
        Y_tmp = np.argmax(Y_val, axis=1)
        class_weights_val = class_weight.compute_class_weight("balanced", np.unique(Y_tmp), Y_tmp)
        class_weights_val = dict(enumerate(class_weights_val))
        sample_weights_val = class_weight.compute_sample_weight(class_weights_val, Y_tmp)
        # Transform numpy array into tf.data.Dataset
        sample_weights_val = tf.data.Dataset.from_tensor_slices((sample_weights_val))


        X_val = tfio.IODataset.from_hdf5(X_val_file, dataset="/x_val")
        Y_val = tfio.IODataset.from_hdf5(Y_val_file, dataset="/y_val")


        # Transform HDF5 -> tensorflow.data.Dataset
        # Load by batches of BATCH_SIZE
        # Ask tensorflow to guess the best number of batches to prefetch in memory
        learn = tf.data.Dataset.zip((X_learn, Y_learn, sample_weights_learn)).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
        val = tf.data.Dataset.zip((X_val, Y_val, sample_weights_val)).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)


        
        #Create model
        model = Sequential()
        model.add(Conv1D(filters=nb_filters, kernel_size=1, input_shape = (ROWS, COLS), padding="same", activation = "relu"))
        model.add(Dropout(0.4))
        model.add(Conv1D(filters=nb_filters, kernel_size=3, padding="same", activation = "relu"))
        model.add(Dropout(0.4))
        model.add(Conv1D(filters=nb_filters, kernel_size=5, padding="same", activation = "relu"))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(nb, activation="softmax"))

        model.summary()

        log_dir = os.path.join(output_dir, f"logs/CV_{i}")

        callbacks_list = [
        ModelCheckpoint(filepath=os.path.join(OUTPUT, f"CP.h5"),
                        monitor="val_weighted_accuracy", 
                        mode="max",
                        save_best_only=True),
        EarlyStopping(monitor="val_weighted_accuracy", 
                      patience=15,
                      restore_best_weights=True),
        TensorBoard(log_dir=log_dir, histogram_freq=1),
        ReduceLROnPlateau(factor=0.1, cooldown=0, patience=10, min_lr=1e-6)
        ]

        loss_fun = tf.keras.losses.BinaryCrossentropy()
        #loss_fun = tf.keras.losses.CategoricalCrossentropy()
        
        #opt = tf.keras.optimizers.Adam()
        opt = tensorflow_addons.optimizers.RectifiedAdam(lr=1e-4) #Start (5e-4 then 5e-5 then 1e-5)
        opt = tensorflow_addons.optimizers.Lookahead(opt)

        model.compile(loss=loss_fun,
                      optimizer=opt,
                      metrics=["accuracy"],
                      weighted_metrics=["accuracy"])

        H = model.fit(learn,
                  epochs=500,
                  callbacks=callbacks_list, 
                  validation_data=val,
                  shuffle=True,
                  verbose=1)

        #Save the last model  
        model.save_weights(os.path.join(OUTPUT, f"weights.h5"))
        model_json = model.to_json(indent=4)
        with open(os.path.join(OUTPUT, f"model.json"), "w") as json_file:
            json_file.write(model_json)

        #Save the history of training
        with open(os.path.join(OUTPUT, f"history"), "wb") as file_pi:
            pickle.dump(H.history, file_pi)

