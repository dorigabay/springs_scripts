import copy
import numpy as np
from scipy.ndimage import label, sum_labels
from scipy.signal import savgol_filter
import os
import cv2
import time


def projection_on_axis(X, Y, axis=0):
    X,Y = copy.copy(X).astype(float),copy.copy(Y).astype(float)
    if not axis in [0,1]:
        raise ValueError("axis must be 0 or 1")
    second_axis = 1-axis
    X_a = np.sqrt(X[:,axis]**2+X[:,second_axis]**2)
    X_angles = np.arctan2(X[:,second_axis],X[:,axis])
    Y_a = np.sqrt(Y[:,axis]**2+Y[:,second_axis]**2)
    Y_angles = np.arctan2(Y[:,second_axis],Y[:,axis])
    X_Y_angles = X_angles - Y_angles
    X[:,second_axis] = 0
    X[:,axis] = X_a
    Y[:,axis] = np.cos(X_Y_angles)*Y_a
    Y[:,second_axis] = np.sin(X_Y_angles)*Y_a
    return X,Y


def calc_pulling_angle_matrix(a, b, c):
    a, b, c = copy.copy(a).astype(float), copy.copy(b).astype(float), copy.copy(c).astype(float)
    a, b, c = a-b, b-b, c-b
    for col in range(c.shape[1]):
        a[:, col], c[:, col] = projection_on_axis(a[:, col], c[:, col], axis=1)
    ba = a - b
    bc = c - b
    ba_y = ba[:, :, 0]
    ba_x = ba[:, :, 1]
    dot = ba_y*bc[:, :, 0] + ba_x*bc[:, :, 1]
    det = ba_y*bc[:, :, 1] - ba_x*bc[:, :, 0]
    angles = np.arctan2(det, dot)
    return angles


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


def normalize(y, x, bias_equation):
    normalized_y = y - bias_equation(x)
    return normalized_y


def convert_bool_to_binary(bool_array):
    return copy.copy(np.array(bool_array,"int"))


def column_dilation(array):
    zeors = np.zeros((array.shape[0],array.shape[1]*2))
    zeors[:,list(range(0,zeors.shape[1],2))] = array
    return zeors


# def column_erosion(array):
#     # remove all the columns that are all zeros
#     return array[:,np.any(array,axis=0)]


def difference(array, spacing=1):
    if spacing == 1:
        diff_array = np.diff(array, axis=0)
        zeros_array = np.zeros_like(array[:1])
        return np.concatenate((zeros_array, diff_array))
    elif (spacing != 1) & (spacing % 2 != 0):
        raise ValueError("diff_spacing must be an even number")
    else:
        zeros = np.zeros_like(array[:int(spacing / 2)])
        diff_array = np.concatenate((zeros, array[spacing:] - array[:-spacing]))
        diff_array = np.concatenate((diff_array, zeros))
        return diff_array


def calc_translation_velocity(coordinates, spacing=1):
    coordinates = coordinates.copy()
    horizontal_component = difference(coordinates[:, 0], spacing=spacing)
    vertical_component = difference(coordinates[:, 1], spacing=spacing)
    movement_direction = np.arctan2(vertical_component, horizontal_component)
    movement_magnitude = np.sqrt(horizontal_component ** 2 + vertical_component ** 2)
    return movement_direction, movement_magnitude


def calc_angular_velocity(angles,diff_spacing=1):
    # calculate the angular velocity of the angles
    THERSHOLD = 5.5
    diff = difference(angles, spacing=diff_spacing)
    diff[(diff>THERSHOLD)] = diff[diff > THERSHOLD]-2*np.pi
    diff[(diff<-THERSHOLD)] = diff[diff <- THERSHOLD]+2*np.pi
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


def find_cells_to_interpolate(array, min_size=8):
    undetected_springs_bool = np.isnan(array)
    undetected_springs_for_long_time = filter_continuity(convert_bool_to_binary(undetected_springs_bool),min_size=min_size)
    cells_to_interpolate = copy.copy(undetected_springs_bool)
    cells_to_interpolate[undetected_springs_for_long_time] = False
    return cells_to_interpolate


def filter_continuity(binary_array, min_size=0, max_size=np.inf):
    binary_array_dilated = column_dilation(binary_array)
    labeled, labels = label(binary_array_dilated)
    labels = np.arange(labels+1)[1:]
    labeled_summed = sum_labels(binary_array_dilated, labeled,index=labels).astype(int)
    labels_to_remove = labels[np.invert((labeled_summed > min_size) & (labeled_summed < max_size))]
    labeled[np.isin(labeled, labels_to_remove)] = 0
    labeled = labeled[:, list(range(0,labeled.shape[1], 2))]
    return labeled >= 1


def sine_function(x, amplitude, frequency, phase, offset):
    return amplitude * np.sin(frequency * x + phase) + offset


def norm_values(explained, explaining, models):
    explaining = np.expand_dims(explaining, axis=2)
    X = np.concatenate((np.sin(explaining), np.cos(explaining)), axis=2)
    prediction_matrix = np.zeros(explained.shape)
    for col in range(explained.shape[1]):
        not_nan_idx = ~(np.isnan(explained[:, col]) + np.isnan(X[:, col]).any(axis=1))
        model = models[col]
        if len(explained.shape) == 2:
            prediction_matrix[not_nan_idx, col] = model.predict(X[not_nan_idx, col, :]).flatten()
    return prediction_matrix


def create_projective_transform_matrix(dst, dst_quality=None, quality_threshold=0.02, src_dimensions=np.array([3840,1920])):
    perfect_squares = dst_quality < quality_threshold if dst_quality is not None else np.full(dst.shape[0], True)
    PTMs = np.full((dst.shape[0], 3, 3), np.nan)
    w, h = src_dimensions
    for count, perfect_square in enumerate(perfect_squares):
        if np.all(perfect_square) and not np.any(np.isnan(dst[count])):
            sdst = dst[count].astype(np.float32)
            src = np.array([[0, 0], [0, h], [w, 0], [w, h]]).astype(np.float32)
            # src = np.array([zero_point, zero_point+np.array([0, h]), zero_point+np.array([w, 0]), zero_point+np.array([w, h])]).astype(np.float32)
            PTM, _ = cv2.findHomography(src, sdst, 0)
            PTMs[count] = PTM
    return PTMs


def apply_projective_transform(coordinates, projective_transformation_matrices):
    original_shape = coordinates.shape
    # print("coordinates before", coordinates.shape)
    if len(original_shape) == 2:
        coordinates = np.expand_dims(np.expand_dims(coordinates, axis=1), axis=2)
    elif len(original_shape) == 3:
        coordinates = np.expand_dims(coordinates, axis=2)
    transformed_coordinates = np.full(coordinates.shape, np.nan)
    nan_count = 0
    # print("coordinates_after", coordinates.shape)
    # print("coordinates_after", coordinates[count])
    for count, PTM in enumerate(projective_transformation_matrices):
        if not np.any(np.isnan(PTM)):
            PTM_inv = np.linalg.inv(PTM)
            transformed_coordinates[count] = cv2.perspectiveTransform(coordinates[count], PTM_inv)
        else:
            nan_count += 1
    transformed_coordinates[np.isnan(coordinates)] = np.nan
    transformed_coordinates = transformed_coordinates.reshape(original_shape)
    return transformed_coordinates


def wait_for_existance(path, file_name):
    existing_attempts = 0
    while not os.path.exists(os.path.join(path, file_name)):
        print(f"\rWaiting for external program to finish... (waited for {existing_attempts * 10} seconds already)")
        time.sleep(10)
        existing_attempts += 1
        if existing_attempts > 10080:  # 3 hours
            raise ValueError("matlab is stuck, please check")


def project_plane_perspective(coordinates, params):
    height, x_center, y_center = params
    # x_center, y_center = 3840 * 0.57, 1920 * 0.82
    # height = 0.015
    plane_to_center_distance = coordinates - np.array([x_center, y_center])
    plane_displacement_distance = plane_to_center_distance * (height / (1 - height))
    plane_correct_coordinates = coordinates - plane_displacement_distance
    return plane_correct_coordinates


def smooth_columns(array):
    for col in range(array.shape[1]):
        array[:, col] = np.abs(np.round(savgol_filter(array[:, col], 31, 2)))
    return array


# import pandas as pd
#
#
# def force_calculations(force_magnitude, force_direction, angle_to_nest, object_center_coordinates):
#     # self.force_magnitude[~np.isnan(self.force_magnitude)*self.rest_bool] -= np.nanmean(self.force_magnitude[~np.isnan(self.force_magnitude)*self.rest_bool])
#     # self.force_direction[~np.isnan(self.force_direction)*self.rest_bool] -= np.nanmean(self.force_direction[~np.isnan(self.force_direction)*self.rest_bool])
#     net_force_direction, net_force_magnitude, net_tangential_force = calc_net_force(force_magnitude, force_direction, angle_to_nest)
#     angular_velocity = calc_angular_velocity(angle_to_nest, diff_spacing=20) / 20
#     angular_velocity = np.where(np.isnan(angular_velocity).all(axis=1), np.nan, np.nanmedian(angular_velocity, axis=1))
#     momentum_direction, momentum_magnitude = calc_translation_velocity(object_center_coordinates, spacing=40)
#     net_force_direction = np.array(pd.Series(net_force_direction).rolling(window=40, center=True).median())
#     net_force_magnitude = np.array(pd.Series(net_force_magnitude).rolling(window=40, center=True).median())
#     net_tangential_force = np.array(pd.Series(net_tangential_force).rolling(window=5, center=True).median())
#     return net_force_direction, net_force_magnitude, net_tangential_force, angular_velocity, momentum_direction, momentum_magnitude
#
#
# def calc_net_force(force_magnitude, force_direction, angle_to_nest):
#     horizontal_component = force_magnitude * np.cos(force_direction + angle_to_nest)
#     vertical_component = force_magnitude * np.sin(force_direction + angle_to_nest)
#     net_force_direction = np.arctan2(np.nansum(vertical_component, axis=1), np.nansum(horizontal_component, axis=1))
#     net_force_magnitude = np.sqrt(np.nansum(horizontal_component, axis=1) ** 2 + np.nansum(vertical_component, axis=1) ** 2)
#     tangential_force = np.sin(force_direction) * force_magnitude
#     net_tangential_force = np.where(np.isnan(tangential_force).all(axis=1), np.nan, np.nansum(tangential_force, axis=1))
#     return net_force_direction, net_force_magnitude, net_tangential_force


