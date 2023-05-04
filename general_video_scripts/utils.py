import copy

import cv2
import numpy as np
from skimage.morphology import skeletonize, thin
import os
import scipy.cluster as sclu
from skimage.morphology import binary_dilation, binary_erosion
from scipy.ndimage import maximum_filter, minimum_filter, label

COLOR_CLOSING = np.ones((3, 3))


def swap_columns(array):
    array[:, [0, 1]] = array[:, [1, 0]]
    return array


def connect_blobs(mask, overlap_size=1):
    if mask.dtype==np.bool:
        mask = convert_bool_to_binary(mask)
    labeled, _ = label(mask)
    maximum_filter_labeled1 = maximum_filter(labeled, overlap_size)
    max_val = np.max(labeled)+1
    labeled[labeled== 0] = max_val
    minimum_filter_labeled2 = minimum_filter(labeled, overlap_size)
    boolean_overlap = (maximum_filter_labeled1 != 0) * (minimum_filter_labeled2 != 0)
    labeled[boolean_overlap] = 1
    labeled[labeled == max_val] = 0
    return labeled.astype(bool).astype("int")


def extend_lines(matrix, extend_by=3):
    # Find the indices of the non-zero elements in the matrix
    indices = np.nonzero(matrix)

    # Extract the coordinates of the non-zero elements
    points = np.transpose(indices)

    # Extract the labels for the non-zero elements
    values = matrix[indices]

    # Calculate the slope and intercept of the line
    # slope, intercept = fit_line(points)

    # Initialize a copy of the matrix with all elements set to 0
    extended_matrix = np.zeros_like(matrix)

    # Iterate through the unique labels
    for label in np.unique(values):
        slope, intercept = fit_line(points[values == label])
        # Select the points with the current label
        label_points = points[values == label]
        # Calculate the points that the line goes through
        x1 = min(label_points[:, 1]) - extend_by
        # y1 = int(slope * x1 + intercept)
        x2 = max(label_points[:, 1]) + extend_by
        # y2 = int(slope * x2 + intercept)
        # Generate a sequence of x coordinates for the line
        x = np.linspace(x1, x2, abs(x2 - x1) + 1, dtype=int)
        # Calculate the corresponding y coordinates for the line
        y = (slope * x + intercept).astype(int)
        x_bounded_x = x[(x >= 0) & (x < matrix.shape[1])]
        y_bounded_x = y[(x >= 0) & (x < matrix.shape[1])]
        y_bounded_xy = y_bounded_x[(y_bounded_x >= 0) & (y_bounded_x < matrix.shape[0])]
        x_bounded_xy = x_bounded_x[(y_bounded_x >= 0) & (y_bounded_x < matrix.shape[0])]
        # Set the elements in the matrix corresponding to the extended line to the label value
        extended_matrix[y_bounded_xy,x_bounded_xy] = label
    extended_matrix[matrix!=0] = matrix[matrix!=0]
    return extended_matrix


def fit_line(points):
    """Fit a line to a set of points using the least squares method.

    Parameters
    ----------
    points : numpy array
        An Nx2 array, where the first column represents the y-values and the second column represents the x-values.

    Returns
    -------
    slope : float
        The slope of the fitted line.
    intercept : float
        The intercept of the fitted line.
    """
    # Extract the x and y values from the points array
    x = points[:, 1]
    y = points[:, 0]

    # Compute the mean of the x and y values
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Compute the slope and intercept using the least squares method
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    return slope, intercept

def create_circular_mask(image_dim, center=None, radius=None):
    h, w = image_dim[0], image_dim[1]
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask

def combine_masks(list_of_masks):
    # combined = [np.zeros(list_of_masks[0].shape,"int")+x for x in list_of_masks][0]
    combined = list_of_masks[0] + list_of_masks[1] + list_of_masks[2]
    combined = convert_bool_to_binary(combined.astype("bool"))
    return combined

def mask_object_colors(parameters, image):
    binary_color_masks = {x: None for x in parameters["colors_spaces"]}
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    hsv_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    for color in parameters["colors_spaces"]:
        binary_mask = np.zeros(image.shape[:-1], "int")
        for hsv_space in parameters["colors_spaces"][color]:
            mask1 = cv2.inRange(hsv_image, hsv_space[0], hsv_space[1])
            mask2 = cv2.inRange(hsv_image, hsv_space[2], hsv_space[3])
            binary_mask[(mask1 + mask2) != 0] = 1
        binary_mask = close_element(binary_mask, COLOR_CLOSING)
        binary_color_masks[color] = binary_mask
    return binary_color_masks

def close_element(mask, structure):
    mask = convert_bool_to_binary(binary_dilation(mask, structure))
    mask = convert_bool_to_binary(binary_erosion(mask, structure))
    return mask

def convert_bool_to_binary(bool_mask):
    binary_mask = np.zeros(bool_mask.shape,"int")
    binary_mask[bool_mask] = 1
    return binary_mask

def correct_by_boundry(hue, boundry):
    """
    HSV space has boundry (179,255,255). The hue value is periodic boundry, therefore this function retain the value to within the boundries.
    :param hue: The color in HSV
    :param boundry: in the case of HSV it is (179,255,255)
    :return: The values after correction.
    """
    boundry = boundry + 1
    final = np.zeros(3)
    for i in range(3):
        if hue[i] < 0:
            if i == 0:
                final[i] = boundry[i] + hue[i]
            else:
                final[i] = 0
        elif hue[i] >= boundry[i]:
            if i == 0:
                final[i] = np.abs(boundry[i] - hue[i])
            else:
                final[i] = boundry[i]
        else:
            final[i] = hue[i]
    return final

def most_common_hue(frame):
    """
    Finds the most common hue in the frame, uses clustering method.
    :return: The commmon hue in HSV
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ar = np.asarray(frame)
    shape = frame.shape
    ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)
    NUM_CLUSTERS = 5
    codes, dist = sclu.vq.kmeans(ar, NUM_CLUSTERS)
    vecs, dist = scipy.cluster.vq.vq(ar, codes)  # assign codes
    counts, bins = np.histogram(vecs, len(codes))  # count occurrences
    index_max = np.argmax(counts)  # find most frequent
    peak = codes[index_max].astype("int")
    print('most frequent color (background) is {})'.format(peak))
    return peak

def create_background_mask(image,background_color):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    BACKGROUND_MARGIN = np.array([15, 30, 20])
    BOUNDRY = np.array([179, 255, 255])
    lower_bg = correct_by_boundry(background_color - BACKGROUND_MARGIN, BOUNDRY)
    upper_bg = correct_by_boundry(background_color + BACKGROUND_MARGIN, BOUNDRY)
    mask = cv2.inRange(hsv, lower_bg, upper_bg)
    return mask


def biggest_contour(contours):
    areas = [cv2.contourArea(c) for c in contours]
    sorted_areas = np.sort(areas)
    cnt = contours[areas.index(sorted_areas[-1])]
    return cnt

def find_contour_center(contour):
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return np.array([cX,cY])

def get_xy_list_from_contour(contours):
    full_dastaset = []
    for contour in contours:
        xy_list = []
        for position in contour:
            [[x, y]] = position
            xy_list.append([x, y])
        full_dastaset.append(xy_list)
    return full_dastaset

def create_mask(frame,hsv_space):
    blurred = cv2.GaussianBlur(frame, (3, 3), 0)
    imgHSV = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(imgHSV, hsv_space[0], hsv_space[1])
    mask2 = cv2.inRange(imgHSV, hsv_space[2], hsv_space[3])
    return mask1+mask2

def obtain_dilated_contour(frame,hsv_space,iterations=2):
    """
    Create a mask based on a given HSV space, dilate it and find the contours of the mask.
    :param frame:
    :param hsv_space: A list of with lower HSV as first element, higher hsv as second.
     Since the Hue space in HSV could be set around 0, there is a need to provide a second HSV space.
     Therefore, the third and forth elements are the lower and higher HSV space respectively.
     If the Hue space is between 0 and 179, just set the second space as the first one.
    :return: The counturs of the mask.
    """
    mask = create_mask(frame,hsv_space)
    kernel = np.ones((3, 3), np.uint8)
    img_dilation = cv2.dilate(mask, kernel, iterations=iterations)
    contours, _ = cv2.findContours(img_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours


def obtain_skeleton(frame):
    """
    Obtain the skeleton of the contour, that was drawen before in green color (0,255,0).
    """
    rgb_lower = np.array([0, 255, 0])
    rgb_upper = np.array([0, 255, 0])
    mask = cv2.inRange(frame, rgb_lower, rgb_upper)
    mask_thinned = thin(mask)
    skeleton_image = skeletonize(mask_thinned, method='lee')
    return skeleton_image

def get_data(skeleton_image):
    xy_coords = np.flip(np.column_stack(np.where(skeleton_image > 0)), axis=1)
    x, y = np.array(xy_coords[:, 0]), np.array(xy_coords[:, 1])
    data = x, y
    return data


def crop_frame_by_contour(frame,absolute_min,absolute_max,margin,cap):
    """
    In order to reduce computational load, this function crop the current frame, based on the contour coordinates of the former frame.
    Croping is set to be N pixel before minimum X,Y, and N pixels after maximum X,Y.
    Since the with the margin the cropping coordinates ca exceed the frame dimensions, there is a need to limit it with limit frame.
    :param frame: The frame to be cropped.
    :param absolute_min: The lowest X and Y coordinates of the countour in the frame before.
    :param absolute_max:  The highest X and Y coordinates of the countour in the frame before.
    :param margin: The number of pixels to remove or add to absolute_min or absolute_min respectively.
    :param cap: The frame capture, used for limit_frame
    :return: A copy (instead if a reference) of the cropped frame.
    """
    frame_coordinates = np.array([(absolute_min[1] - margin[1]), (absolute_max[1] + margin[1]), (absolute_min[0] - margin[0]),(absolute_max[0] + margin[0])])
    frame_coordinates = limit_frame(frame_coordinates, cap, frame_coordinates=True)
    frame = frame[frame_coordinates[0]:frame_coordinates[1], frame_coordinates[2]:frame_coordinates[3]]
    return copy.copy(frame)


def limit_frame(np_array,cap,frame_coordinates=False):
    """
    Limit the coordinates for cropping frames, to the dimensions of the frame.
    :param np_array:
    :param cap: The frame capture.
    :param frame_coordinates:
    :return:
    """
    width = int(cap.get(3))
    height = int(cap.get(4))
    np_array[np_array < 0] = 0
    if frame_coordinates:
        if np_array[3] > width:
            np_array[3] = width
        if np_array[1] > height:
            np_array[1] = height
    else:
        if np_array[0] > width:
            np_array[0] = width
        if np_array[1] > height:
            np_array[1] = height
    return(np_array)

def write_contour_size_single(output_dir,frame_number,contour_size,first):
    """
    Writes the contour area size in a file called contour_sizes. This sheet will later be used to find frames with bad detection of the object.
    :param output_dir: output directory the save the sheet in.
    :param frame_number: The frame index/number.
    :param contour_size: The countur area size.
    :param first: If True, the function will first create a new file, if not it will only add new frames and contour sizes.
    """
    if first:
        log_file = open(os.path.join(output_dir, "contour_sizes.csv"), "w")
        log_file.write("frame_number,contour_size" + "\n")
        log_file.write(str(frame_number) + "," + str(contour_size) + "\n")
    else:
        log_file = open(os.path.join(output_dir, "contour_sizes.csv"), "a")
        log_file.write(str(frame_number)+","+str(contour_size) + "\n")



def write_contour_size(output_dir,frame_number,contour_class,first,no_contours =False):
    """
    Writes the contour area size in a file called contour_sizes. This sheet will later be used to find frames with bad detection of the object.
    This function is the same as write_contour_size_single, only here it is for detection of synchtonization experimets.
    :param output_dir: output directory the save the sheet in.
    :param frame_number: The frame index/number.
    :param contour_class: Wheter the left of right pendulum.
    :param contour_size: The countur area size.
    :param first: If True, the function will first create a new file, if not it will only add new frames and contour sizes.
    :param no_contours: If True, the None will be written.
    :return:
    """
    if first:
        log_file = open(os.path.join(output_dir, "contour_sizes.csv"), "w")
        log_file.write("frame_number,contour_size_left,contour_size_right" + "\n")
        log_file.write(str(frame_number) + "," + str(contour_class.cnt_area_left)+","+str(contour_class.cnt_area_right) + "\n")
    else:
        log_file = open(os.path.join(output_dir, "contour_sizes.csv"), "a")
        if no_contours: log_file.write(str(frame_number)+",None,None\n")
        else: log_file.write(str(frame_number) + "," + str(contour_class.cnt_area_left)+","+str(contour_class.cnt_area_right) + "\n")

def crop_frame_by_coordinates(frame,crop_coordinates):
    """
    Crop the frame by choosen coordinates.
    :param frame: Frame to cropped.
    """
    # x = 50
    # return frame[crop_coordinates[0]-x:crop_coordinates[1]+x,crop_coordinates[2]-x:crop_coordinates[3]+x]
    return frame[crop_coordinates[0]:crop_coordinates[1],crop_coordinates[2]:crop_coordinates[3]]

def remove_rectangle(frame,coordinates,element_color_by_name):
    """
    This functions creates a rectangle in red or white color, to remove pixels that disturb the detection of the object.
    :param frame: The frame to be change.
    :param coordinates: The coordinates of the rectangle's corners (top-left,bootom-right)
    :param element_color_by_name: The color of the obejct, if red rectangle will be white, if white rectangle will be red.
    """
    if element_color_by_name == "white": color = (0,0,255)
    elif element_color_by_name == "red": color = (255,255,255)
    else: color = (0,255,0)
    start_point = (coordinates[2],coordinates[0])
    end_point = (coordinates[3],coordinates[1])
    cv2.rectangle(frame, start_point, end_point, color, -1)
    cv2.rectangle(frame, start_point, end_point, color, 2)
    return frame

#
# def create_mask(frame,hsv_space):
#     hsv_lower1, hsv_upper1, hsv_lower2, hsv_upper2 = hsv_space
#     # blurred = cv2.GaussianBlur(frame, (3 ,3), 0)
#     blurred = frame
#     imgHSV = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
#     mask1 = cv2.inRange(imgHSV, hsv_lower1, hsv_upper1)
#     mask2 = cv2.inRange(imgHSV, hsv_lower2, hsv_upper2)
#     mask = mask1+mask2
#     mask = cv2.bitwise_not(mask)
#     masked = cv2.bitwise_and(frame, frame, mask=mask)
#     masked = cv2.cvtColor(masked, cv2.COLOR_HSV2BGR)
#     return masked,mask

def create_mask_with_contour(spaces,frame):
    image_contours = copy.copy(frame)
    for space in spaces:
        contours = obtain_dilated_contour(frame, space)
        image_contours = cv2.drawContours(image_contours, contours, -1, (0, 255, 0), -1)
    return image_contours

# def skeletonize(mask):
#     height, width = mask.shape
#     skel = np.zeros([height, width], dtype=np.uint8)  # [height,width,3]
#     kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
#     temp_nonzero = np.count_nonzero(mask)
#     while (np.count_nonzero(mask) != 0):
#         eroded = cv2.erode(mask, kernel)
#         cv2.imshow("eroded", eroded)
#         temp = cv2.dilate(eroded, kernel)
#         cv2.imshow("dilate", temp)
#         temp = cv2.subtract(mask, temp)
#         skel = cv2.bitwise_or(skel, temp)
#         mask = eroded.copy()
#     return skel


def find_lines(skel,frame):
    result = frame.copy()
    edges = cv2.Canny(skel, 50, 150)
    cv2.imshow("edges",edges)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,40,minLineLength=30,maxLineGap=30)
    i = 0
    for line in lines:
        for x1,y1,x2,y2 in line:
            i+=1
            cv2.line(result,(x1,y1),(x2,y2),(255,0,0),1)
    cv2.imshow("res",result)
    cv2.waitKey(0)


def combine_masks(list_of_masks):
    combined_mask = list_of_masks[0]
    for m in list_of_masks[1:]:
        combined_mask += m
    result = 255 * combined_mask
    result = result.clip(0, 255).astype("uint8")
    return result

