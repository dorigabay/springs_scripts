import copy
import numpy as np
from scipy.ndimage import label,sum_labels
import os

def convert_bool_to_binary(bool_array):
    return copy.copy(np.array(bool_array,"int"))

def column_dilation(array):
    zeors = np.zeros((array.shape[0],array.shape[1]*2))
    zeors[:,list(range(0,zeors.shape[1],2))] = array
    return zeors

def difference(array,spacing=1,axis=0):
    # calculate the difference between each element and the element at the given spacing, without a loop
    if axis==0:
        zeors = np.zeros((spacing,array.shape[1]))
        return np.concatenate((zeors, array[spacing:] - array[:-spacing]))
    elif axis==1:
        zeors = np.zeros((array.shape[0],spacing))
        return np.concatenate((zeors, array[:spacing] - array[-spacing:]))

def calc_angular_velocity(angles,diff_spacing=1):
    THERSHOLD = 5.5
    diff = difference(angles,spacing=diff_spacing,axis=0)
    diff[(diff>THERSHOLD)] = diff[diff>THERSHOLD]-2*np.pi
    diff[(diff<-THERSHOLD)] = diff[diff<-THERSHOLD]+2*np.pi
    return diff

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

def iter_folder(path):
    to_analyze = {}
    directories_to_search = [path]
    while directories_to_search:
        dir = directories_to_search.pop()
        found_dirs = [folder_name for folder_name in [x for x in os.walk(dir)][0][1] if "_force" in folder_name]
        for found_dir in found_dirs:
            videos_names = [x for x in os.walk(os.path.join(dir,found_dir))][0][1]
            if found_dir not in to_analyze:
                to_analyze[found_dir] = [os.path.normpath(os.path.join(dir,found_dir,x))+"\\" for x in videos_names]
            else:
                to_analyze[found_dir] += [os.path.normpath(os.path.join(dir,found_dir,x))+"\\" for x in videos_names]
        else:
            for subdir in [x for x in os.walk(dir)][0][1]:
                directories_to_search.append(os.path.join(dir, subdir))
    return to_analyze