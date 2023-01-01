import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label,sum_labels
from scipy.signal import savgol_filter
import os, pickle
import pandas as pd
import seaborn as sns

def convert_bool_to_binary(bool_array):
    return copy.copy(np.array(bool_array,"int"))

def column_dilation(array):
    zeors = np.zeros((array.shape[0],array.shape[1]*2))
    zeors[:,list(range(0,zeors.shape[1],2))] = array
    return zeors

def interpolate_data(array,undetected_bool,period=None):
    array = copy.copy(array)
    for col in range(array.shape[1]):
        fp = array[:,col]
        xp = np.arange(len(fp))
        cells_to_interpolate = undetected_bool[:,col]
        fp_nonan = fp[np.invert(cells_to_interpolate)]
        xp_nonan = xp[np.invert(cells_to_interpolate)]
        if period is not None:
            array[:,col] = np.interp(xp, xp_nonan, fp_nonan, period=period)
        else: array[:,col] = np.interp(xp, xp_nonan, fp_nonan)
    return array

def find_cells_to_interpolate(array):
    undetected_springs_bool = np.isnan(array)
    undetected_springs_for_long_time = filter_continuity(convert_bool_to_binary(undetected_springs_bool),min_size=8)
    cells_to_interpolate = copy.copy(undetected_springs_bool)
    cells_to_interpolate[undetected_springs_for_long_time] = False
    return cells_to_interpolate

def filter_continuity(binary_array,min_size=0,max_size=np.inf):
    binary_array_dilated = column_dilation(binary_array)
    labeled,labels = label(binary_array_dilated)
    labels = np.arange(labels+1)[1:]
    labeled_summed = sum_labels(binary_array_dilated,labeled,index=labels).astype(int)
    labels_to_remove = labels[np.invert((labeled_summed>min_size)&(labeled_summed<max_size))]
    labeled[np.isin(labeled,labels_to_remove)]=0
    labeled = labeled[:,list(range(0,labeled.shape[1],2))]
    return labeled>=1