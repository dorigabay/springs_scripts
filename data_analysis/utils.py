import copy
import numpy as np
from scipy.ndimage import label, sum_labels
from scipy.signal import savgol_filter
import os
import cv2
import time
import apytl
from matplotlib import pyplot as plt
from scipy.stats import f_oneway, ttest_ind
import itertools


def projection_on_axis(X, Y, axis=0):
    X, Y = copy.copy(X).astype(float),copy.copy(Y).astype(float)
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
    zeros = np.zeros((array.shape[0], array.shape[1]*2))
    zeros[:, list(range(0, zeros.shape[1], 2))] = array
    return zeros


def row_dilation(array):
    zeros = np.zeros((array.shape[0]*2, array.shape[1]))
    zeros[list(range(0, zeros.shape[0], 2)), :] = array
    return zeros


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
    horizontal_component = difference(coordinates[:, 0], spacing=spacing) / spacing
    vertical_component = difference(coordinates[:, 1], spacing=spacing) / spacing
    movement_direction = np.arctan2(vertical_component, horizontal_component)
    movement_magnitude = np.sqrt(horizontal_component ** 2 + vertical_component ** 2)
    return movement_direction, movement_magnitude


def calc_angular_velocity(angles, diff_spacing=1):
    # calculate the angular velocity of the angles
    THERSHOLD = 5.5
    diff = difference(angles, spacing=diff_spacing)
    diff[(diff > THERSHOLD)] = diff[diff > THERSHOLD]-2*np.pi
    diff[(diff < -THERSHOLD)] = diff[diff < -THERSHOLD]+2*np.pi
    return diff


def interpolate_rows(data, interpolation_boolean=None, period=None):
    data = copy.copy(data)
    original_shape = data.shape
    data = data.reshape(-1, 1) if len(original_shape) == 1 else data
    if interpolation_boolean is not None:
        interpolation_boolean = interpolation_boolean.reshape(-1, 1) if len(original_shape) == 1 else interpolation_boolean
    else:
        interpolation_boolean = np.isnan(data)
    for row in range(data.shape[0]):
        fp = data[row, :]
        xp = np.arange(len(fp))
        cells_to_interpolate = interpolation_boolean[row, :]
        fp_nonan = fp[np.invert(cells_to_interpolate)]
        xp_nonan = xp[np.invert(cells_to_interpolate)]
        if len(fp_nonan) != 0:
            if period is not None:
                data[row, :] = np.interp(xp, xp_nonan, fp_nonan, period=period)
            else:
                data[row, :] = np.interp(xp, xp_nonan, fp_nonan)
    return data.reshape(original_shape)


def interpolate_columns(data, interpolation_boolean=None, period=None):
    data = copy.copy(data)
    original_shape = data.shape
    data = data.reshape(-1, 1) if len(original_shape) == 1 else data
    if interpolation_boolean is not None:
        interpolation_boolean = interpolation_boolean.reshape(-1, 1) if len(original_shape) == 1 else interpolation_boolean
    else:
        interpolation_boolean = np.isnan(data)
    for col in range(data.shape[1]):
        fp = data[:, col]
        xp = np.arange(len(fp))
        cells_to_interpolate = interpolation_boolean[:, col]
        fp_nonan = fp[np.invert(cells_to_interpolate)]
        xp_nonan = xp[np.invert(cells_to_interpolate)]
        if not len(fp_nonan) == 0 or len(fp_nonan) == len(fp):
            if period is not None:
                data[:, col] = np.interp(xp, xp_nonan, fp_nonan, period=period)
            else:
                data[:, col] = np.interp(xp, xp_nonan, fp_nonan)
    data = data.reshape(original_shape)
    return data


def find_cells_to_interpolate(array, min_size=8):
    undetected_springs_bool = np.isnan(array)
    undetected_springs_for_long_time = filter_continuity(undetected_springs_bool, min_size=min_size)
    cells_to_interpolate = copy.copy(undetected_springs_bool)
    cells_to_interpolate[undetected_springs_for_long_time] = False
    return cells_to_interpolate


def filter_continuity_vector(bool_vector, min_size=0, max_size=np.inf):
    # binary_array_missing = vector == 0
    binary_vector = bool_vector.astype(int)
    labeled, labels = label(binary_vector)
    labels = np.arange(labels + 1)[1:]
    labeled_summed = sum_labels(binary_vector, labeled, index=labels).astype(int)
    labels_to_remove = labels[np.invert((labeled_summed > min_size) & (labeled_summed < max_size))]
    labeled[np.isin(labeled, labels_to_remove)] = 0
    return labeled >= 1


def filter_continuity(bool_array, min_size=0, max_size=np.inf):
    binary_array = bool_array.astype(int)
    binary_array_dilated = column_dilation(binary_array)
    labeled, labels = label(binary_array_dilated)
    labels = np.arange(labels+1)[1:]
    labeled_summed = sum_labels(binary_array_dilated, labeled, index=labels).astype(int)
    labels_to_remove = labels[np.invert((labeled_summed > min_size) & (labeled_summed < max_size))]
    labeled[np.isin(labeled, labels_to_remove)] = 0
    labeled = labeled[:, list(range(0, labeled.shape[1], 2))]
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


def test_if_on_boundaries(coordinates, frame_size):
    if frame_size[1] * 0.1 < coordinates[0] < frame_size[0] - frame_size[1] * 0.1 \
            and frame_size[1] * 0.1 < coordinates[1] < frame_size[1] * 0.9:
        return True
    else:
        return False


def interpolate_assigned_ants(ants_assigned_to_springs, num_of_frames):
    arranged_frames = np.arange(num_of_frames)
    not_empty_columns = np.where(np.sum(ants_assigned_to_springs, axis=0) != 0)[0]
    for count, ant in enumerate(not_empty_columns):
        apytl.Bar().drawbar(count, len(not_empty_columns), fill='*')
        vector = ants_assigned_to_springs[:, ant]
        zeros = vector == 0
        small_chunks = filter_continuity_vector(zeros, max_size=5)
        xp = arranged_frames[~zeros]
        fp = vector[xp]
        x = arranged_frames[small_chunks]
        vector[x] = np.round(np.interp(x, xp, fp))
        small_chunks_springs = filter_continuity_vector(vector != 0, max_size=15)
        vector[small_chunks_springs + zeros] = 0
        if len(np.unique(vector)) > 1:
            small_chunks = filter_continuity_vector(vector == 0, max_size=10)
            xp = arranged_frames[vector != 0]
            fp = vector[xp]
            x = arranged_frames[small_chunks]
            vector[x] = np.round(np.interp(x, xp, fp))
        small_chunks = np.full(len(vector), False)
        for spring in np.unique(vector):
            if spring != 0:
                small_chunks = small_chunks + filter_continuity_vector(vector == spring, max_size=10)
        vector[small_chunks] = 0
        small_chunks = filter_continuity_vector(vector == 0, max_size=50)
        labeled, num_features = label(small_chunks)
        for i in range(1, num_features + 1):
            idx = np.where(labeled == i)[0]
            start, end = idx[0], idx[-1]
            if end != len(vector) - 1 and vector[start - 1] == vector[end + 1]:
                vector[idx] = vector[start - 1]
        ants_assigned_to_springs[:, ant] = vector
    return ants_assigned_to_springs


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
        print(f"\rWaiting for external program to finish... (waited for {existing_attempts * 10} seconds already)", end="")
        time.sleep(10)
        existing_attempts += 1
        if existing_attempts > 10080:  # 3 hours
            raise ValueError("matlab is stuck, please check")
    if os.path.exists(os.path.join(path, file_name)) and existing_attempts > 0:
        time.sleep(300)


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


def kuramoto_order_parameter(phases):
    """
    Calculate the Kuramoto order parameter for an array of phase values.

    Parameters:
    - phases: A 1D NumPy array or list containing the phase values of oscillators.

    Returns:
    - R: Magnitude of the order parameter.
    - Phi: Phase of the order parameter.
    """
    # Ensure phases are in the range [0, 2*pi)
    phases = np.mod(phases, 2 * np.pi)
    # Calculate the complex sum of unit vectors corresponding to each phase
    complex_sum = np.sum(np.exp(1j * phases))
    # Calculate the order parameter magnitude and phase
    R = np.abs(complex_sum) / len(phases)
    Phi = np.angle(complex_sum)
    return R, Phi


def compare_two_pairs(pair1, pair2):
    group1, group2, group3, group4 = pair1[0], pair1[1], pair2[0], pair2[1]
    f_statistic, p_value_anova = f_oneway(group1, group2, group3, group4)
    alpha_anova = 0.05
    if p_value_anova < alpha_anova:
        print("There are significant differences between at least two groups.")
        t_statistic_pair1, p_value_pair1 = ttest_ind(*pair1)
        t_statistic_pair2, p_value_pair2 = ttest_ind(*pair2)
        alpha_ttest = 0.05
        if p_value_pair1 < alpha_ttest and p_value_pair2 < alpha_ttest and abs(t_statistic_pair1) > abs(t_statistic_pair2):
            print("The difference between pair1 is larger than the difference between pair2.")
            pair1_difference_is_larger_than_pair2_differences = True
        else:
            print("The difference between pair1 is not larger than the difference between pair2.")
            pair1_difference_is_larger_than_pair2_differences = False
    else:
        pair1_difference_is_larger_than_pair2_differences = False
        print("There are no significant differences between groups.")
    return pair1_difference_is_larger_than_pair2_differences


def draw_significant_stars(df, combinations=None, axs=None, y_range=None):
    if combinations is None:
        combinations = np.array(list(itertools.combinations(np.arange(len(df.columns)), 2)))
    else:
        combinations = np.array(combinations)
    combinations = combinations[np.argsort([i[1]-i[0] for i in combinations])]
    for col_1_idx, col_2_idx in combinations:
        array1 = df.iloc[:, col_1_idx]
        array2 = df.iloc[:, col_2_idx]
        nan_idx = np.logical_or(np.isnan(array1), np.isnan(array2))
        t_stat, p_value = ttest_ind(array1[~nan_idx], array2[~nan_idx])
        if p_value < 0.05:
            print("There is a significant difference between the two groups.")
            if axs is None:
                top = plt.gca().get_ylim()[1]
                bottom = plt.gca().get_ylim()[0]
            else:
                top = axs.get_ylim()[1]
                bottom = axs.get_ylim()[0]
            y_range = top - bottom
            bar_height = (y_range * 0.02) + top
            bar_tips = bar_height - (y_range * 0.02)
            if axs is None:
                plt.plot(
                    [col_1_idx, col_1_idx, col_2_idx, col_2_idx],
                    [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
                plt.text(np.abs(col_2_idx + col_1_idx) / 2, bar_height, "***", ha='center', va='center', fontsize=12)
            else:
                axs.plot(
                    [col_1_idx, col_1_idx, col_2_idx, col_2_idx],
                    [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
                axs.text(np.abs(col_2_idx + col_1_idx) / 2, bar_height, "***", ha='center', va='center', fontsize=12)
        else:
            print(f"There is no significant difference between {col_1_idx} and {col_2_idx}.")


def discretize_angular_velocity(angular_velocity, sets_frames):
    discrete_velocity_vectors = []
    threshold_stop = np.nanpercentile(np.abs(angular_velocity), 50)
    threshold_very_slow = np.nanpercentile(np.abs(angular_velocity), 67.5)
    threshold_slow = np.nanpercentile(np.abs(angular_velocity), 75)
    threshold_medium = np.nanpercentile(np.abs(angular_velocity), 87.5)

    for count, set_idx in enumerate(sets_frames):
        s, e = set_idx[0][0], set_idx[-1][1]+1
        set_angular_velocity = angular_velocity[s:e]
        set_angular_velocity = interpolate_columns(set_angular_velocity)
        stop = np.abs(set_angular_velocity) < threshold_stop
        very_slow = np.all([threshold_stop <= np.abs(set_angular_velocity), np.abs(set_angular_velocity) < threshold_very_slow], axis=0)
        slow = np.all([threshold_very_slow <= np.abs(set_angular_velocity), np.abs(set_angular_velocity) < threshold_slow], axis=0)
        medium = np.all([threshold_slow <= np.abs(set_angular_velocity), np.abs(set_angular_velocity) < threshold_medium], axis=0)
        fast = np.abs(set_angular_velocity) >= threshold_medium
        signed_angular_velocity = set_angular_velocity.copy()
        signed_angular_velocity[stop] = 0
        signed_angular_velocity[very_slow] = np.sign(signed_angular_velocity[very_slow]) * 0.25
        signed_angular_velocity[slow] = np.sign(signed_angular_velocity[slow]) * 0.5
        signed_angular_velocity[medium] = np.sign(signed_angular_velocity[medium]) * 0.75
        signed_angular_velocity[fast] = np.sign(signed_angular_velocity[fast]) * 1
        velocity_labels = [1, 0.75, 0.5, 0.25, 0]
        for i in velocity_labels:
            small_parts = filter_continuity_vector(np.abs(signed_angular_velocity) == i, max_size=5)
            signed_angular_velocity = interpolate_columns(signed_angular_velocity, small_parts)
        small_parts = filter_continuity_vector(~np.isin(np.abs(signed_angular_velocity), [1, 0.75, 0.5, 0.25, 0]), max_size=5)
        signed_angular_velocity = interpolate_columns(signed_angular_velocity, small_parts)
        for velocity_count, velocity in enumerate(velocity_labels[1:]):
            out_of_label = (signed_angular_velocity > velocity) * (signed_angular_velocity < velocity_labels[velocity_count])
            signed_angular_velocity[out_of_label] = velocity
            out_of_label_minus = (signed_angular_velocity < -velocity) * (signed_angular_velocity > -velocity_labels[velocity_count])
            signed_angular_velocity[out_of_label_minus] = -velocity
        for velocity_count, velocity in enumerate(velocity_labels[:-1]):
            small_parts_plus = filter_continuity_vector(signed_angular_velocity == velocity, max_size=5)
            signed_angular_velocity[small_parts_plus] = velocity_labels[velocity_count+1]
            small_parts_minus = filter_continuity_vector(signed_angular_velocity == -velocity, max_size=5)
            signed_angular_velocity[small_parts_minus] = -velocity_labels[velocity_count+1]
        discrete_velocity_vectors.append(signed_angular_velocity)
    discrete_velocity = np.concatenate(discrete_velocity_vectors)
    change = np.abs(np.diff(discrete_velocity)) > 0.5
    change = np.concatenate([[False], change])
    return discrete_velocity * -1, change


def create_color_space(hue_data, around_zero=False):
    x = np.linspace(0, 1, 100)
    blue = (0, 0, 1)
    white = (1, 1, 1)
    red = (1, 0, 0)
    colors = np.empty((100, 3))
    if around_zero:
        for i in range(3):
            colors[:, i] = np.interp(x, [0, 0.5, 1], [blue[i], white[i], red[i]])
    else:  # range color from white to red
        for i in range(3):
            colors[:, i] = np.interp(x, [0, 1], [white[i], blue[i]])
    color_range = (colors * 255).astype(int)
    flatten = hue_data.flatten()#[0:5000]
    flatten = flatten[~np.isnan(flatten)]
    median_biggest = np.median(np.sort(flatten)[-100:])
    median_smallest = np.median(np.sort(flatten)[:100])
    color_range_bins = np.linspace(median_smallest, median_biggest, 100)
    if around_zero:
        color_range_bins = np.linspace(-np.median(np.sort(np.abs(flatten))[-100:]), np.median(np.sort(np.abs(flatten))[-100:]), 100)
    return color_range, color_range_bins


