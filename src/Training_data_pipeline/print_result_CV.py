
import os 
import sys 
import math 
import datetime 
import argparse 

import h5py
import pandas as pd 
import numpy as np 
from tabulate import tabulate
import functools
from itertools import product


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", help="Path to the h5 CV directory.", type=str, required=True)
parser.add_argument("-m", help="Path to the models directory.", type=str, required=True)
parser.add_argument("-c", "--classes", help='The classes you want to predict, as a string. Ex: "abcdefghijklmnop"', required=True)
args = parser.parse_args()

entry_dir = args.i
model_dir = args.m


CLASSES = args.classes
# Transform classes from string to list
class_names = list(CLASSES)   #CLASSES is a command line argmuent



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow_io as tfio

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
import tensorflow_addons
from tensorflow.keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives, AUC
 
from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report
from sklearn.utils import class_weight 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import label_binarize 
from sklearn.multiclass import OneVsRestClassifier 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef

# This snippet is necessary !
# Otherwise we get this error: Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)



def matthews_correlation(y_true, y_pred):
    """
    Compute the Matthews Correlation Coefficient for 2 class prediction.
    """
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


def all_statistics(y_true, y_pred):

    ### Requier class_names variable ###

    #Variable initialization and creation of dictionnary
    top_class  = []
    dict_top_class =  {}

    for cla in class_names:
        dict_top_class[cla] = []

    for i in range(len(y_pred)):
        index = y_true[i]

        # Sort of class index by predicted probability
        top_rev = np.argsort(y_pred[i])
        # Array top reverse order (max to minus)
        top     = top_rev[::-1]
        # Determine rank of the true 
        rank = 1
        for j in top:
            if j == y_true[i]:
                top_class.append(rank)
                cla = class_names[index]
                dict_top_class[cla].append(rank)
            rank += 1


    #Create a list of dataframe rows
    rows_list = []

    ######
    top1_all = 0
    top2_all = 0
    top3_all = 0
    top4_all = 0
    top5_all = 0

    top1_avg = 0
    top2_avg = 0
    top3_avg = 0
    top4_avg = 0
    top5_avg = 0

    num_all = 0

    for cla in class_names:
        top_class = dict_top_class[cla]

        top1 = 0
        top2 = 0
        top3 = 0
        top4 = 0
        top5 = 0

        for i in top_class:
            if i == 1:
                top1 += 1
                top2 += 1
                top3 += 1
                top4 += 1
                top5 += 1
            elif i <= 2:
                top2 += 1
                top3 += 1
                top4 += 1
                top5 += 1
            elif i <= 3:
                top3 += 1
                top4 += 1
                top5 += 1
            elif i <= 4:
                top4 += 1
                top5 += 1
            elif i <= 5:
                top5 += 1

        top1_all += top1
        top2_all += top2
        top3_all += top3
        top4_all += top4
        top5_all += top5
        num_all += len(top_class)

        top1_avg += top1 / len(top_class)
        top2_avg += top2 / len(top_class)
        top3_avg += top3 / len(top_class)
        top4_avg += top4 / len(top_class)
        top5_avg += top5 / len(top_class)
               
        rows_list.append([top1/len(top_class)*100, top2/len(top_class)*100, top3/len(top_class)*100, top4/len(top_class)*100, top5/len(top_class)*100])

    rows_list.append([np.nan, np.nan, np.nan, np.nan, np.nan])
    rows_list.append([top1_all/num_all*100, top2_all/num_all*100, top3_all/num_all*100, top4_all/num_all*100, top5_all/num_all*100])
    rows_list.append([top1_avg/len(class_names)*100, top2_avg/len(class_names)*100, top3_avg/len(class_names)*100, top4_avg/len(class_names)*100, top5_avg/len(class_names)*100])

    rows_list = np.array(rows_list)
    header = ["", "TOP 1", "TOP 2", "TOP 3", "TOP 4", "TOP 5"]
    rows = class_names + ["", "ALL", "Weighted"]

    return rows_list, header, rows


def freq_probability(y_true, y_pred_classes):

    ### Requier class_names variable ###

    #Variable ititialization
    dict_tp_class_prob = {}
    dict_tn_class_prob = {}
    dict_fp_class_prob = {}
    dict_fn_class_prob = {}

    for cla in class_names:
        dict_tp_class_prob[cla] = []
        dict_tn_class_prob[cla] = []
        dict_fp_class_prob[cla] = []
        dict_fn_class_prob[cla] = []


    for i in range(len(y_true)):

        index_pred = y_pred_classes[i]
        index_true = y_true[i]

        true_class = class_names[index_true]
        pred_class = class_names[index_pred]

        # Various computation
        if y_pred_classes[i] == y_true[i]:
            dict_tp_class_prob[true_class].append(y_pred[i][index_true])

            for idx in range(len(class_names)):
                cla=class_names[idx]
                if idx != index_true:
                    dict_tn_class_prob[cla].append(y_pred[i][idx])

        else:    
            dict_fp_class_prob[pred_class].append(y_pred[i][index_pred])
            dict_fn_class_prob[true_class].append(y_pred[i][index_true])


    #Create a list of dataframe rows
    rows_list = []
    rows = []

    #### PROBABILITIES CLASS ######
    prob1    = 0
    prob2    = 0
    prob3    = 0
    prob4    = 0
    prob5    = 0
    prob6    = 0
    prob7    = 0
    prob8    = 0
    prob9    = 0
    prob10   = 0
    prob_all = 0

    for cla in class_names:

        prob1_tp  = 0
        prob2_tp  = 0
        prob3_tp  = 0
        prob4_tp  = 0
        prob5_tp  = 0
        prob6_tp  = 0
        prob7_tp  = 0
        prob8_tp  = 0
        prob9_tp  = 0
        prob10_tp = 0

        prob1_tn  = 0
        prob2_tn  = 0
        prob3_tn  = 0
        prob4_tn  = 0
        prob5_tn  = 0
        prob6_tn  = 0
        prob7_tn  = 0
        prob8_tn  = 0
        prob9_tn  = 0
        prob10_tn = 0

        prob1_fp  = 0
        prob2_fp  = 0
        prob3_fp  = 0
        prob4_fp  = 0
        prob5_fp  = 0
        prob6_fp  = 0
        prob7_fp  = 0
        prob8_fp  = 0
        prob9_fp  = 0
        prob10_fp = 0

        prob1_fn  = 0
        prob2_fn  = 0
        prob3_fn  = 0
        prob4_fn  = 0
        prob5_fn  = 0
        prob6_fn  = 0
        prob7_fn  = 0
        prob8_fn  = 0
        prob9_fn  = 0
        prob10_fn = 0

        prob_class_tp = dict_tp_class_prob[cla]

        for i in prob_class_tp:
            if i >= 0.9:
                prob1_tp += 1
            elif i >= 0.8:
                prob2_tp += 1
            elif i >= 0.7:
                prob3_tp += 1
            elif i >= 0.6:
                prob4_tp += 1
            elif i >= 0.5:
                prob5_tp += 1
            elif i >= 0.4:
                prob6_tp += 1
            elif i >= 0.3:
                prob7_tp += 1
            elif i >= 0.2:
                prob8_tp += 1
            elif i >= 0.1:
                prob9_tp += 1
            elif i >= 0.00:
                prob10_tp += 1
        rows_list.append([prob1_tp, prob2_tp, prob3_tp, prob4_tp, prob5_tp, prob6_tp, prob7_tp, prob8_tp, prob9_tp, prob10_tp])
        rows.append(f"TP  {cla}")

        prob_class_fn = dict_fn_class_prob[cla]
        for i in prob_class_fn:
            if i >= 0.9:
                prob1_fn += 1
            elif i >= 0.8:
                prob2_fn += 1
            elif i >= 0.7:
                prob3_fn += 1
            elif i >= 0.6:
                prob4_fn += 1
            elif i >= 0.5:
                prob5_fn += 1
            elif i >= 0.4:
                prob6_fn += 1
            elif i >= 0.3:
                prob7_fn += 1
            elif i >= 0.2:
                prob8_fn += 1
            elif i >= 0.1:
                prob9_fn += 1
            elif i >= 0.00:
                prob10_fn += +1

        rows_list.append([prob1_fn, prob2_fn, prob3_fn, prob4_fn, prob5_fn, prob6_fn, prob7_fn, prob8_fn, prob9_fn, prob10_fn])
        rows.append(f"FN  {cla}")

        prob_class_fp = dict_fp_class_prob[cla]

        for i in prob_class_fp:
            if i >= 0.9:
                prob1_fp += 1
            elif i >= 0.8:
                prob2_fp += 1
            elif i >= 0.7:
                prob3_fp += 1
            elif i >= 0.6:
                prob4_fp += 1
            elif i >= 0.5:
                prob5_fp += 1
            elif i >= 0.4:
                prob6_fp += 1
            elif i >= 0.3:
                prob7_fp += 1
            elif i >= 0.2:
                prob8_fp += 1
            elif i >= 0.1:
                prob9_fp += 1
            elif i >= 0.00:
                prob10_fp += +1

        rows_list.append([prob1_fp, prob2_fp, prob3_fp, prob4_fp, prob5_fp, prob6_fp, prob7_fp, prob8_fp, prob9_fp, prob10_fp])
        rows.append(f"FP  {cla}")

        prob_class_tn = dict_tn_class_prob[cla]

        for i in prob_class_tn:
            if i >= 0.9:
                prob1_tn += 1
            elif i >= 0.8:
                prob2_tn += 1
            elif i >= 0.7:
                prob3_tn += 1
            elif i >= 0.6:
                prob4_tn += 1
            elif i >= 0.5:
                prob5_tn += 1
            elif i >= 0.4:
                prob6_tn += 1
            elif i >= 0.3:
                prob7_tn += 1
            elif i >= 0.2:
                prob8_tn += 1
            elif i >= 0.1:
                prob9_tn += 1
            elif i >= 0.00:
                prob10_tn += +1

        rows_list.append([prob1_tn, prob2_tn, prob3_tn, prob4_tn, prob5_tn, prob6_tn, prob7_tn, prob8_tn, prob9_tn, prob10_tn])
        rows.append(f"TN  {cla}")

    #Create the pretty table with tabulate
    rows_list = np.array(rows_list)
    header = ["", "0.90", "0.80", "0.70", "0.60", "0.50", "0.40", "0.30", "0.20", "0.10", "0.00"]
    
    return rows_list, header, rows


def report2arr(cr):
    # Parse rows (messy but it works...)
    rows_list = []

    cr = cr.split("\n")

    for i in range(2, len(cr)-5):
        rows_list.append(cr[i].split()[1:])
        print(rows_list)
    rows_list.append([np.nan, np.nan] + cr[len(cr)-4].split()[1:])
    rows_list.append(cr[len(cr)-3].split()[2:])
    rows_list.append(cr[len(cr)-2].split()[2:])

    rows_list = np.array(rows_list, dtype=float)
    header = ["", "Precision", "Recall", "F1-Score", "Support"]
    rows = class_names + ["Accuracy", "Macro Avg", "Weighted Avg"]

    return rows_list, header, rows


def all_results(y_true, y_pred_classes):

    ### Requier class_names variable ###

    #Variable initialization
    nb_classes = len(class_names)

    dict_tp_class = {}
    dict_tn_class = {}
    dict_fp_class = {}
    dict_fn_class = {}

    for cla in class_names:
        dict_tp_class[cla] = 0
        dict_tn_class[cla] = 0 
        dict_fp_class[cla] = 0
        dict_fn_class[cla] = 0

    #Save data
    for i in range(len(y_pred_classes)):
        index_true = y_true[i]
        index_pred = y_pred_classes[i]

        true_class = class_names[index_true]
        pred_class = class_names[index_pred]

        # Various computation
        if y_pred_classes[i] == y_true[i]:
            dict_tp_class[true_class] = dict_tp_class[true_class] + 1

            for cla in class_names:
                if cla != true_class:
                    dict_tn_class[cla] = dict_tn_class[cla] + 1

        else:
            dict_fp_class[pred_class] = dict_fp_class[pred_class] + 1
            dict_fn_class[true_class] = dict_fn_class[true_class] + 1


    #Create a list of dataframe rows
    rows_list = []

    #Calculate the differents metrics
    sum_mcc                = 0
    sum_tpr_sens_recall    = 0
    sum_fnr_missrate       = 0
    sum_fpr_fallout        = 0
    sum_ppv_precision      = 0
    sum_tnr_specificity    = 0
    sum_balanced_accurracy = 0
    sum_acc                = 0
    sum_F1_score           = 0
    tot_support            = 0

    sum_tp = 0
    sum_fp = 0
    sum_tn = 0
    sum_fn = 0

    for cla in class_names:
        tp = dict_tp_class[cla]
        tn = dict_tn_class[cla]
        fp = dict_fp_class[cla]
        fn = dict_fn_class[cla]


        mcc                = (tp * tn - fp * fn) / (math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        tpr_sens_recall    = tp / (tp + fn)
        fnr_missrate       = fn / (fn + tp) 
        fpr_fallout        = fp / (fp + tn)
        ppv_precision      = tp / (tp + fp)
        tnr_specificity    = tn / (tn + fp)
        balanced_accurracy = (tpr_sens_recall + tnr_specificity) / 2
        acc                = (tp + tn) / (tp + tn + fp + fn)
        F1_score           = 2 * (ppv_precision * tpr_sens_recall) / (ppv_precision + tpr_sens_recall)
        support            = tp + fn
            
        #Save the metrics in a list
        rows_list.append([mcc, tpr_sens_recall, fnr_missrate, fpr_fallout, ppv_precision, tnr_specificity, balanced_accurracy, acc, F1_score, support])

        #Calculate the sum of the metrics
        sum_mcc                += mcc
        sum_tpr_sens_recall    += tpr_sens_recall
        sum_fnr_missrate       += fnr_missrate
        sum_fpr_fallout        += fpr_fallout
        sum_ppv_precision      += ppv_precision
        sum_tnr_specificity    += tnr_specificity
        sum_balanced_accurracy += balanced_accurracy
        sum_acc                += acc
        sum_F1_score           += F1_score
        tot_support            += support

        sum_tp += tp
        sum_fp += fp
        sum_tn += tn
        sum_fn += fn

    #Save the metrics in a list
    rows_list.append([sum_mcc/nb_classes, sum_tpr_sens_recall/nb_classes, sum_fnr_missrate/nb_classes, sum_fpr_fallout/nb_classes, sum_ppv_precision/nb_classes, sum_tnr_specificity/nb_classes, sum_balanced_accurracy/nb_classes, sum_acc/nb_classes, sum_F1_score/nb_classes, tot_support])

    macro_mcc                = (sum_tp * sum_tn - sum_fp * sum_fn) / (math.sqrt((sum_tp + sum_fp) * (sum_tp + sum_fn) * (sum_tn + sum_fp) * (sum_tn + sum_fn)))
    macro_tpr_sens_recall    = sum_tp / (sum_tp + sum_fn)
    macro_fnr_missrate       = sum_fn / (sum_fn + sum_tp) 
    macro_fpr_fallout        = sum_fp / (sum_fp + sum_tn)
    macro_ppv_precision      = sum_tp / (sum_tp + sum_fp)
    macro_tnr_specificity    = sum_tn / (sum_tn + sum_fp)
    macro_balanced_accurracy = (macro_tpr_sens_recall + macro_tnr_specificity) / 2
    macro_acc                = (sum_tp + sum_tn) / (sum_tp + sum_tn + sum_fp + sum_fn)
    macro_F1_score           = 2 * (macro_ppv_precision * macro_tpr_sens_recall) / (macro_ppv_precision + macro_tpr_sens_recall)
    macro_support            = sum_tp + sum_fn

    #Save the metrics in a list
    rows_list.append([macro_mcc, macro_tpr_sens_recall, macro_fnr_missrate, macro_fpr_fallout, macro_ppv_precision, macro_tnr_specificity, macro_balanced_accurracy, macro_acc, macro_F1_score, macro_support])

    rows_list = np.array(rows_list)
    header = ["MCC", "Sens\nTPR", "FNR\nMissRate", "FPR\nFall Out", "PRE\nPPV", "Specif\nTNR", "Bal. ACC", "ACC", "F1-Score", "N"]
    rows = class_names + ["Mic Avg", "Mac Avg"]

    return rows_list, header, rows


def conf_mat(y_true, y_pred_classes):

    ### Requier class_names variable ###

    cm = confusion_matrix(y_true, y_pred_classes)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    header = ["   Pred\nTrue   "] + class_names
    rows = class_names

    return cm, cm_norm, header, rows



if __name__ == "__main__":
    
    #Save metrics for CV
    loss_cv = []
    accuracy_cv = []
    auc_cv = []
    accuracy_1_cv = []
    matthews_corr_cv = []
    skMCC_cv = []
    w_skMCC_cv = []

    all_stats_cv = []
    freq_prob_cv = []
    classif_report_cv = []
    all_result_cv = []
    cm_cv = []
    cm_norm_cv = []

    for i in range(10):

        #Get entry files path
        ENTRY = os.path.join(entry_dir, f"CV_{i}")
        MODEL = os.path.join(model_dir, f"CV_{i}")


        PATH_HDF5_X_TEST = os.path.join(ENTRY, f"X_test_evaluate25.h5")
        PATH_HDF5_Y_TEST = os.path.join(ENTRY, f"Y_test_evaluate25.h5")
 

        ### LOAD DATA
        with h5py.File(PATH_HDF5_X_TEST, "r") as h5f:
            x_test = h5f["x_test"][:]
        with h5py.File(PATH_HDF5_Y_TEST, "r") as h5f:
            y_test = h5f["y_test"][:]
        


        ### BUILD THE TFIO TEST DATASET
        y_tmp = np.argmax(y_test, axis=1)   # Convert one-hot to index 
        class_weights_test = class_weight.compute_class_weight("balanced", np.unique(y_tmp), y_tmp)
        class_weights_test = dict(enumerate(class_weights_test))
        sample_weights_test = class_weight.compute_sample_weight(class_weights_test, y_tmp)   # For classification_report()
        """
        sample_weights_test_tf = tf.data.Dataset.from_tensor_slices((sample_weights_test))   # Transform numpy array into tf.data.Dataset
        # Use Tensorflow-io to load data into GPU more efficiently by reading HDF5 directly and loading data from file by batches instead of all at once.
        x_test_tfio = tfio.IODataset.from_hdf5(PATH_HDF5_X_TEST, dataset="/x_test")
        y_test_tfio = tfio.IODataset.from_hdf5(PATH_HDF5_Y_TEST, dataset="/y_test")
        BATCH_SIZE = 1024 # (only used to accelerate computations) 
        test = tf.data.Dataset.zip((x_test_tfio, y_test_tfio, sample_weights_test_tf)).batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE)
         - .zip      : Transform HDF5 into tensorflow.data.Dataset
         - .batch    : Load by batches of BATCH_SIZE 
         - .prefetch : Ask tensorflow to guess the best number of batches to prefetch in memory
         - drop_remainder is set to False because we want to use the whole dataset as batch has no impact on evaluation/prediction
        """

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

        opt = tensorflow_addons.optimizers.RectifiedAdam(lr=1e-4)
        opt = tensorflow_addons.optimizers.Lookahead(opt)

        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                      optimizer=opt,
                      metrics=["accuracy", matthews_correlation,
                              AUC(name="auc")],
                      weighted_metrics=["accuracy"])


        # Evaluate the model on test dataset 
        print("Evaluating on test dataset...")
        scores = model.evaluate(x=x_test, y=y_test, batch_size=1024, sample_weight=sample_weights_test, verbose=0)

        # Predict on test dataset
        print("Predicting on test dataset...")
        y_pred = model.predict(x_test, verbose=0) 

        # Get predicted classes as one-hot vector
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)



        ############################################################################



        print("\n"*3)
        
        print(f"K = {i+1}/10")

        print("\n"*2)
        
        print("########################################## MODEL EVALUATION #########################################")
        print("\n")

        print(f"{model.metrics_names[0]}: {scores[0]:.4f}")
        print(f"{model.metrics_names[1]}: {scores[1]:.4f}")
        print(f"{model.metrics_names[2]}: {scores[2]:.4f}")
        print(f"{model.metrics_names[3]}: {scores[3]:.4f}")
        print(f"{model.metrics_names[4]}: {scores[4]:.4f}")
        print(f"sklearn MCC: {matthews_corrcoef(y_true, y_pred_classes):.4f}")
        print(f"Weighted sklearn MCC: {matthews_corrcoef(y_true, y_pred_classes, sample_weights_test):.4f}")


        loss_cv.append(scores[0])
        accuracy_cv.append(scores[1])
        auc_cv.append(scores[3])
        accuracy_1_cv.append(scores[4])
        matthews_corr_cv.append(scores[2])
        skMCC_cv.append(matthews_corrcoef(y_true, y_pred_classes))
        w_skMCC_cv.append(matthews_corrcoef(y_true, y_pred_classes, sample_weights_test))

        print("\n")

        print("########################################## ALL STATISTICS ###########################################")

        print("\n")

        print("#TOP X True Positive rate (Sensitivity/Recall):\n")
        all_stats, header_stats, rows_stats = all_statistics(y_true, y_pred)
        all_stats_cv.append(all_stats)
        #Create the pretty table with tabulate
        all_stats_table = tabulate(pd.DataFrame(np.round(all_stats, 3)).replace(np.nan, ""), headers = header_stats, showindex=rows_stats, floatfmt=".3f", tablefmt="rst")
        print(all_stats_table)


        print("\n")


        print("#Frequency of probabilities of TP, FP, TN, FN:\n")
        freq_prob, header_freq_prob, rows_freq_prob = freq_probability(y_true, y_pred_classes)
        freq_prob_cv.append(freq_prob)
        freq_prob_table = tabulate(freq_prob, headers = header_freq_prob, showindex=rows_freq_prob, floatfmt=".0f", tablefmt="rst")
        print(freq_prob_table)


        print("\n")

        """
        print("Summary:\n")
        classif_report, header_classif_report, rows_classif_report = report2arr(classification_report(y_true, y_pred_classes, target_names=class_names, sample_weight=sample_weights_test, digits = 9))
        classif_report_cv.append(classif_report)
        classif_report_table = tabulate(pd.DataFrame(np.round(classif_report, 3)).replace(np.nan, ""), headers=header_classif_report, showindex=rows_classif_report, tablefmt="rst", floatfmt=(".3f", ".3f", ".3f", ".3f", ".0f"))
        print(classif_report_table)
        """

        print("\n"*2)

        print("############################################ ALL RESULTS ############################################")

        print("\n"*2)


        all_res, header_all_res, rows_all_res = all_results(y_true, y_pred_classes)
        all_result_cv.append(all_res)
        #Create the pretty table with tabulate
        all_results_table = tabulate(all_res, headers=header_all_res, showindex=rows_all_res, floatfmt=(".3f", ".3f", ".3f", ".3f", ".3f", ".3f", ".3f", ".3f", ".3f", ".3f", ".0f"), tablefmt="rst")
        print(all_results_table)


        print("\n")


        cm , cm_norm, header_cm, rows_cm = conf_mat(y_true, y_pred_classes)
        print("Confusion matrix:\n")
        cm_cv.append(cm)
        cm_table = tabulate(cm, headers=header_cm, showindex=rows_cm, tablefmt="rst")
        print(cm_table)

        print("\n")

        print("Normalized confusion matrix:\n")
        cm_norm_cv.append(cm_norm)
        cm_norm_table = tabulate(cm_norm, headers=header_cm, showindex=rows_cm, tablefmt="rst", floatfmt=".3f")
        print(cm_norm_table)

        print("\n"*3)



    ############################################################################



    print("\n"*2)
    print("############################################ FINAL RESULTS ##########################################")

    print("\n"*2)
    
    print("########################################### MODEL EVALUATION ########################################")

    print("\n")

    print(f"{model.metrics_names[0]}: {np.mean(loss_cv):.4f} ± {np.std(loss_cv):.4f}")
    print(f"{model.metrics_names[1]}: {np.mean(accuracy_cv):.4f} ± {np.std(accuracy_cv):.4f}")
    print(f"{model.metrics_names[3]}: {np.mean(auc_cv):.4f} ± {np.std(auc_cv):.4f}")
    print(f"{model.metrics_names[4]}: {np.mean(accuracy_1_cv):.4f} ± {np.std(accuracy_1_cv):.4f}")
    print(f"{model.metrics_names[2]}: {np.mean(matthews_corr_cv):.4f} ± {np.std(matthews_corr_cv):.4f}")
    print(f"sklearn MCC: {np.mean(skMCC_cv):.4f} ± {np.std(skMCC_cv):.4f}")
    print(f"Weighted sklearn MCC: {np.mean(w_skMCC_cv):.4f} ± {np.std(w_skMCC_cv):.4f}")


    print("\n")

    print("########################################### ALL STATISTICS ##########################################")


    print("\n")

    print("#TOP X True Positive rate(Sensitivity/Recall):\n")

    print("Mean:")
    all_stats_table = tabulate(pd.DataFrame(np.round(np.mean(all_stats_cv, axis=0), 3)).replace(np.nan, ""), headers = header_stats, showindex=rows_stats, floatfmt=".3f", tablefmt="rst")
    print(all_stats_table)

    print("\nStandard deviation:")
    all_stats_table = tabulate(pd.DataFrame(np.round(np.std(all_stats_cv, axis=0), 3)).replace(np.nan, ""), headers = header_stats, showindex=rows_stats, floatfmt=".3f", tablefmt="rst")
    print(all_stats_table)

    print("\n")

    print("# Frequency of probabilities of TP, FP, TN, FN:\n")

    print("Mean:")
    freq_prob_table = tabulate(np.mean(freq_prob_cv, axis=0), headers = header_freq_prob, showindex=rows_freq_prob, floatfmt=".0f", tablefmt="rst")
    print(freq_prob_table)

    print("\nStandard deviation:")
    freq_prob_table = tabulate(np.std(freq_prob_cv, axis=0), headers = header_freq_prob, showindex=rows_freq_prob, floatfmt=".0f", tablefmt="rst")
    print(freq_prob_table)

    print("\n")

    print("Summary:\n")
    """
    print("Mean:")
    classif_report_table = tabulate(pd.DataFrame(np.round(np.mean(classif_report_cv, axis=0), 3)).replace(np.nan, ""), headers=header_classif_report, showindex=rows_classif_report, tablefmt="rst", floatfmt=(".3f", ".3f", ".3f", ".3f", ".0f"))
    print(classif_report_table)

    print("\nStandard deviation:")
    classif_report_table = tabulate(pd.DataFrame(np.round(np.std(classif_report_cv, axis=0), 3)).replace(np.nan, ""), headers=header_classif_report, showindex=rows_classif_report, tablefmt="rst", floatfmt=(".3f", ".3f", ".3f", ".3f", ".0f"))
    print(classif_report_table)
    """
    print("\n"*2)

    print("############################################# ALL RESULTS ###########################################")

    print("\n"*2)

    print("Mean:")
    all_results_table = tabulate(np.mean(all_result_cv, axis=0), headers=header_all_res, showindex=rows_all_res, floatfmt=(".3f", ".3f", ".3f", ".3f", ".3f", ".3f", ".3f", ".3f", ".3f", ".3f", ".0f"), tablefmt="rst")
    print(all_results_table)

    print("\nStandard deviation:")
    all_results_table = tabulate(np.std(all_result_cv, axis=0), headers=header_all_res, showindex=rows_all_res, floatfmt=(".3f", ".3f", ".3f", ".3f", ".3f", ".3f", ".3f", ".3f", ".3f", ".3f", ".0f"), tablefmt="rst")
    print(all_results_table)


    print("\n")

    print("Confusion matrix:\n")
    print("Mean:")
    cm_table = tabulate(np.round(np.mean(cm_cv, axis=0), 0), headers=header_cm, showindex=rows_cm, tablefmt="rst")
    print(cm_table)

    print("\nStandard deviation:")
    cm_table = tabulate(np.round(np.std(cm_cv, axis=0), 0), headers=header_cm, showindex=rows_cm, tablefmt="rst")
    print(cm_table)

    print("\n")

    print("Normalized confusion matrix:\n")
    print("Mean:")
    cm_table = tabulate(np.round(np.mean(cm_norm_cv, axis=0), 3), headers=header_cm, showindex=rows_cm, tablefmt="rst")
    print(cm_table)

    print("\nStandard deviation:")
    cm_table = tabulate(np.round(np.std(cm_norm_cv, axis=0), 3), headers=header_cm, showindex=rows_cm, tablefmt="rst")
    print(cm_table)
