import copy
import glob

import numpy as np
from scipy.ndimage import label,sum_labels
import os
from scipy.optimize import curve_fit

def calc_angle_matrix(a, b, c):
    """
    Calculate the angle between vectors a->b and b->c.
    a, b, and c are all 3D arrays of the same shape, where last dimension is the x, y coordinates.
    :param a:
    :param b:
    :param c:
    :return:
    """
    ba = a - b
    bc = c - b
    ba_y = ba[:,:,0]
    ba_x = ba[:,:,1]
    dot = ba_y*bc[:,:,0] + ba_x*bc[:,:,1]
    det = ba_y*bc[:,:,1] - ba_x*bc[:,:,0]
    angles = np.arctan2(det, dot)
    return angles


def deduce_bias_equation(x,y):
    def bias_equation(x, a, b,c):
        return a*np.cos(x+b)+c
    params, _ = curve_fit(bias_equation, x, y)
    return lambda x: bias_equation(x, *params)


def normalize(y, x, bias_equation):
    normalized_y = y - bias_equation(x)
    return normalized_y

def convert_bool_to_binary(bool_array):
    return copy.copy(np.array(bool_array,"int"))

def column_dilation(array):
    zeors = np.zeros((array.shape[0],array.shape[1]*2))
    zeors[:,list(range(0,zeors.shape[1],2))] = array
    return zeors

def column_erosion(array):
    # remove all the columns that are all zeros
    return array[:,np.any(array,axis=0)]

def difference(array,spacing=1):
    # calculate the difference between each element and the element at the given spacing, without a loop
    if spacing == 1:
        diff_array = np.diff(array,axis=0)
        return np.concatenate((np.zeros((1,array.shape[1])).reshape(1,array.shape[1]),diff_array))
    elif (spacing != 1) & (spacing % 2 != 0):
        raise ValueError("diff_spacing must be an even number")
    else:
        zeros = np.zeros((int(spacing/2),array.shape[1]))
        diff_array = np.concatenate((zeros, array[spacing:] - array[:-spacing]))
        diff_array = np.concatenate((diff_array, zeros))
        return diff_array
    # elif axis==1:
    #     zeros = np.zeros((array.shape[0],spacing))
    #     diff_array = np.concatenate((zeros, array[:,spacing:] - array[:,:-spacing]),axis=1)
    #     diff_array = np.concatenate((diff_array, zeros),axis=1)
    #     return diff_array


def calc_angular_velocity(angles,diff_spacing=1):
    # calculate the angular velocity of the angles
    THERSHOLD = 5.5
    diff = difference(angles,spacing=diff_spacing)
    diff[(diff>THERSHOLD)] = diff[diff>THERSHOLD]-2*np.pi
    diff[(diff<-THERSHOLD)] = diff[diff<-THERSHOLD]+2*np.pi
    return diff

# def sum_n_lines(array,n_lines=10):
#     summed_array = np.abs(array.copy()[list(range(0,array.shape[0],n_lines)),:])
#     for n in range(1,n_lines):
#         add = np.abs(array[list(range(n,array.shape[0],n_lines)),:])
#         summed_array[0:add.shape[0],:] += add
#     return summed_array

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

def find_dirs(path):
    to_analyze = {}
    directories_to_search = [path]
    while directories_to_search:
        dir = directories_to_search.pop()
        found_dirs = [folder_name for folder_name in [x for x in os.walk(dir)][0][1] if "_force" in folder_name]
        for found_dir in found_dirs:
            videos_names = [x for x in os.walk(os.path.join(dir,found_dir))][0][1]
            videos_paths = [os.path.normpath(os.path.join(dir,found_dir,x))+"\\" for x in videos_names]
            videos_paths_filtered = [x for x in videos_paths if len(glob.glob(x+'\\*.csv'))>0]
            if found_dir not in to_analyze:
                to_analyze[found_dir] = videos_paths_filtered
            else:
                to_analyze[found_dir] += videos_paths_filtered
        else:
            for subdir in [x for x in os.walk(dir)][0][1]:
                directories_to_search.append(os.path.join(dir, subdir))
    return to_analyze

def get_outliers(array, threshold=1.1):
    # get the outliers from an array
    # outliers are defined as values that are more than percent_threshold away from the median
    median = np.nanmedian(array)
    outliers = np.where(np.abs(array - median) > threshold * median)[0]
    return outliers

