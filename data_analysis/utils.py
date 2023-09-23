import copy
import glob
import numpy as np
from scipy.ndimage import label,sum_labels
import os
from scipy.optimize import curve_fit
import cv2

def bound_angle(angle):
    above_2pi = angle > 2 * np.pi
    below_0 = angle < 0
    angle[above_2pi] = angle[above_2pi] - 2 * np.pi
    angle[below_0] = angle[below_0] + 2 * np.pi
    return angle


def load_first_frame(video_path):
    video = cv2.VideoCapture(video_path)
    ret, frame = video.read()
    return frame


# def projection_on_axis(X, Y, axis=0):
#     X,Y = copy.copy(X).astype(float),copy.copy(Y).astype(float)
#     if not axis in [0,1]:
#         raise ValueError("axis must be 0 or 1")
#     second_axis = 1-axis
#     X_a = X[:,axis]+1j*X[:,second_axis]
#     Y_a = Y[:,axis]+1j*Y[:,second_axis]
#     X_angles = np.angle(X_a)
#     # print(X_angles)
#     # big_angles = (X_angles > np.pi/2)
#     # small_angles = (X_angles < -np.pi/2)
#     # X_angles[big_angles] = 0
#     # X_angles[small_angles] = np.pi
#     # X_angles -= X_angles
#     # print(X_angles)
#     X_angles = np.zeros_like(X_angles)
#     X_ex = np.exp(1j*(-X_angles))
#     X_a = X_a*X_ex
#     Y_a = Y_a*X_ex
#     X[:,axis] = X_a.real
#     Y[:,axis] = Y_a.real
#     X[:,second_axis] = X_a.imag
#     Y[:,second_axis] = Y_a.imag
#     return X,Y

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
    a,b,c  = copy.copy(a).astype(float),copy.copy(b).astype(float),copy.copy(c).astype(float)
    a,b,c = a-b,b-b,c-b
    for col in range(c.shape[1]):
        a[:,col],c[:,col] = projection_on_axis(a[:,col],c[:,col],axis=1)
    ba = a - b
    bc = c - b
    ba_y = ba[:,:,0]
    ba_x = ba[:,:,1]
    dot = ba_y*bc[:,:,0] + ba_x*bc[:,:,1]
    det = ba_y*bc[:,:,1] - ba_x*bc[:,:,0]
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
    # relative_horizontal_component = difference(relative_coordinates[:, :, 0], spacing=spacing)
    # relative_vertical_component = difference(relative_coordinates[:, :, 1], spacing=spacing)
    # horizontal_component = horizontal_component - np.nanmedian(relative_horizontal_component, axis=1)
    # vertical_component = vertical_component - np.nanmedian(relative_vertical_component, axis=1)
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


def norm_values(explained, explaining, models):
    explaining = np.expand_dims(explaining, axis=2)
    X = np.concatenate((np.sin(explaining), np.cos(explaining)), axis=2)
    not_nan_idx = ~(np.isnan(explained) + np.isnan(X).any(axis=2))
    prediction_matrix = np.zeros(explained.shape)
    for col in range(explained.shape[1]):
        model = models[col]
        if len(explained.shape) == 2:
            prediction_matrix[not_nan_idx[:, col], col] = model.predict(X[not_nan_idx[:, col], col, :])
    return prediction_matrix


def create_projective_transform_matrix(dst, dst_quality=None, quality_threshold=0.02, src_dimensions=(3840, 2160)):
    """
    Creates projective transformation matrix
    :param dst: destination points coordinates
    :param src_distances: source points distances, x-axis and y-axis, default is 4K
    :param dst_quality: destination points quality
    :param quality_threshold: quality threshold
    :return:
    """
    perfect_squares = dst_quality < quality_threshold if dst_quality is not None else np.full(dst.shape[0], True)
    PTMs = np.full((dst.shape[0], 3, 3), np.nan)
    w, h = src_dimensions
    for count, perfect_square in enumerate(perfect_squares):
        if perfect_square and not np.any(np.isnan(dst[count])):
            sdst = dst[count].astype(np.float32)
            src = np.array([[0, 0], [0, h], [w, 0], [w, h]]).astype(np.float32)
            PTM, _ = cv2.findHomography(src, sdst, 0)
            PTMs[count] = PTM
    return PTMs


def apply_projective_transform(coordinates, projective_transformation_matrices):
    original_shape = coordinates.shape
    if len(original_shape) == 2:
        coordinates = np.expand_dims(np.expand_dims(coordinates, axis=1), axis=2)
    elif len(original_shape) == 3:
        coordinates = np.expand_dims(coordinates, axis=2)
    transformed_coordinates = np.full(coordinates.shape, np.nan)
    nan_count = 0
    for count, PTM in enumerate(projective_transformation_matrices):
        if not np.any(np.isnan(PTM)):
            PTM_inv = np.linalg.inv(PTM)
            transformed_coordinates[count] = cv2.perspectiveTransform(coordinates[count], PTM_inv)
        else:
            nan_count += 1
    transformed_coordinates[np.isnan(coordinates)] = np.nan
    transformed_coordinates = transformed_coordinates.reshape(original_shape)
    return transformed_coordinates


# def create_PTM(dst, src_distances=(3840, 2160), dst_quality=None, quality_threshold=0.02):
#     """
#     Creates projective transformation matrix
#     :param dst: destination points coordinates
#     :param src_distances: source points distances, x-axis and y-axis, default is 4K
#     :param dst_quality: destination points quality
#     :param quality_threshold: quality threshold
#     :return: PTM or None
#     """
#     perfect_squares = np.all(dst_quality < quality_threshold, axis=1) if dst_quality is not None else np.full(dst.shape[0], True)
#     PTMs = []
#     distance = np.array(src_distances)
#
#     for count, perfect_square in enumerate(perfect_squares):
#         if perfect_square:
#             sdst = dst[count].astype(np.float32)
#             src = np.array(([0, 0], [distance[0], 0], [distance[0], distance[1]], [0, distance[1]])).astype(np.float32)
#
#             try:
#                 PTM = cv2.getPerspectiveTransform(src, sdst)
#                 PTMs.append(PTM)
#             except cv2.error:
#                 PTMs.append(None)
#
#     return PTMs
#
#
# def apply_PTM(coordinates, PTMs):
#     original_shape = coordinates.shape
#     if len(original_shape) == 3:
#         coordinates = coordinates.reshape((-1, 2))
#
#     transformed_coordinates = []
#
#     for count, M in enumerate(PTMs):
#         if M is not None:
#             homogeneous_coords = np.column_stack((coordinates, np.ones(len(coordinates))))
#             transformed = np.dot(homogeneous_coords, M.T)
#             transformed = transformed[:, :2] / transformed[:, 2, np.newaxis]
#             transformed_coordinates.append(transformed)
#         else:
#             transformed_coordinates.append(np.full_like(coordinates, np.nan))
#
#     transformed_coordinates = np.stack(transformed_coordinates)
#     return transformed_coordinates.reshape(original_shape)

# def create_bias_correction_models(class_object):
#     from sklearn.pipeline import make_pipeline
#     from sklearn.preprocessing import PolynomialFeatures
#     from sklearn.linear_model import LinearRegression
#     y_length = np.linalg.norm(class_object.free_ends_coordinates - class_object.fixed_ends_coordinates, axis=2)
#     y_angle = calc_pulling_angle_matrix(class_object.fixed_ends_coordinates, class_object.object_center_repeated, class_object.free_ends_coordinates)
#     angles_to_nest = np.expand_dims(class_object.fixed_end_angle_to_nest, axis=2)
#     X = np.concatenate((np.sin(angles_to_nest), np.cos(angles_to_nest)), axis=2)
#     idx = class_object.rest_bool
#     class_object.models_lengths = []
#     class_object.models_angles = []
#     not_nan_idx = ~(np.isnan(y_angle) + np.isnan(X).any(axis=2)) * idx
#     for col in range(y_length.shape[1]):
#         X_fit = X[not_nan_idx[:, col], col]
#         y_length_fit = y_length[not_nan_idx[:, col], col]
#         y_angle_fit = y_angle[not_nan_idx[:, col], col]
#         model_length = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
#         model_angle = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
#         class_object.models_lengths.append(model_length.fit(X_fit, y_length_fit))
#         class_object.models_angles.append(model_angle.fit(X_fit, y_angle_fit))
#     return class_object


# def remove_outliers_rows(matrix, percent_threshold=.1):
#     # remove outliers from a matrix
#     # outliers are defined as values that are more than percent_threshold away from the median
#     percentile = np.nanpercentile(np.abs(matrix), 100-percent_threshold)
#     matrix[np.any(np.abs(matrix) > percentile, axis=1)] = np.nan
#     return matrix