import cv2
import numpy as np
import os
import pickle
import datetime
from PIL import Image
import scipy.io as sio
import scipy.cluster as sclu
from scipy.ndimage import maximum_filter, minimum_filter, label, binary_fill_holes
from skimage.morphology import remove_small_objects, binary_closing, binary_opening, binary_dilation, binary_erosion
import matplotlib.pyplot as plt

COLOR_CLOSING = np.ones((3, 3))


def get_background_color(image):
    # convert a frame to PIL image
    im = Image.fromarray(image)
    colors = im.getcolors(im.size[0] * im.size[1])
    colors = sorted(colors, key=lambda x: x[0], reverse=True)
    background_color = colors[0][1]
    return background_color


def most_common_hue(image):
    """
    Finds the most common hue in the frame, uses clustering method.
    :return: The commmon hue in HSV
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ar = np.asarray(image)
    shape = image.shape
    ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)
    NUM_CLUSTERS = 5
    codes, dist = sclu.vq.kmeans(ar, NUM_CLUSTERS)
    vecs, dist = sclu.vq.vq(ar, codes)  # assign codes
    counts, bins = np.histogram(vecs, len(codes))  # count occurrences
    index_max = np.argmax(counts)  # find most frequent
    peak = codes[index_max].astype("int")
    # print('most frequent color (background) is {})'.format(peak))
    return peak


def get_farthest_point(point, contour, percentile=70, inverse=False):
    contour = swap_columns(contour.copy())
    distances = np.sqrt(np.sum(np.square(contour-point), 1))
    # take the average of the points above the 80 percentile
    percentile = np.percentile(distances, percentile) if not inverse else np.percentile(distances, 100-percentile)
    farthest_points = np.where(distances > percentile)[0] if not inverse else np.where(distances < percentile)[0]
    # farthest_points = np.argsort(distances)[-n_points:] if not inverse else np.argsort(distances)[:n_points]
    farthest_point = np.mean(contour[farthest_points], 0)#.astype(int)
    return farthest_point, np.max(distances)


def swap_columns(array):
    array[:, [0, 1]] = array[:, [1, 0]]
    return array


def connect_blobs(mask, overlap_size=1):
    mask = mask.astype(bool)
    labeled, _ = label(mask)
    maximum_filter_labeled1 = maximum_filter(labeled, overlap_size)
    max_val = np.max(labeled)+1
    labeled[labeled== 0] = max_val
    minimum_filter_labeled2 = minimum_filter(labeled, overlap_size)
    boolean_overlap = (maximum_filter_labeled1 != 0) * (minimum_filter_labeled2 != 0)
    labeled[boolean_overlap] = 1
    labeled[labeled == max_val] = 0
    return labeled.astype(bool)


def remove_small_blobs(bool_mask, min_size: int = 0):
    """
    Removes from the input mask all the blobs having less than N adjacent pixels.
    We set the small objects to the background label 0.
    """
    bool_mask = bool_mask.astype(bool)
    if min_size > 0:
        bool_mask = remove_small_objects(bool_mask, min_size=min_size)
    return bool_mask


def close_blobs(mask, structure):
    mask = binary_dilation(mask, structure).astype(np.uint8)
    mask = binary_erosion(mask, structure).astype(np.uint8)
    return mask


def clean_mask(mask, min_size, opening_size=None, fill_holes=True, circle_center_remove=None, circle_radius_remove=None):
    if fill_holes:
        mask = binary_fill_holes(mask.astype(np.uint8))
    if opening_size is not None:
        mask = binary_opening(mask.astype(np.uint8), np.ones((opening_size, opening_size)))
    mask = mask.astype(bool)
    if not (circle_center_remove is None or circle_radius_remove is None):
        inner_circle_mask = create_circular_mask(mask.shape, center=circle_center_remove, radius=circle_radius_remove * 0.9)
        outer_circle_mask = create_circular_mask(mask.shape, center=circle_center_remove, radius=circle_radius_remove * 3.5)
        mask[inner_circle_mask] = False
        mask[np.invert(outer_circle_mask)] = False
    mask = remove_small_blobs(mask, min_size)
    return mask


def mask_sharper(image, keep_mask, sobel_kernel_size, gradiant_threshold, opening_size, closing_size):
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel_size)
    sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel_size)
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_mask = (gradient_magnitude > gradiant_threshold).astype(int)
    sobel_mask = sobel_mask * keep_mask
    sobel_mask = binary_opening(sobel_mask, np.ones((opening_size, opening_size))).astype(int)
    sobel_mask = binary_closing(sobel_mask, np.ones((closing_size, closing_size))).astype(int)
    return sobel_mask.astype(bool)


def extend_lines(matrix, extend_by=3):
    indices = np.nonzero(matrix)
    points = np.transpose(indices)
    values = matrix[indices]
    extended_matrix = np.zeros_like(matrix)
    line_lengths = {}
    for label in np.unique(values):
        slope, intercept, line_length = fit_line(points[values == label])
        line_lengths[label] = line_length
        label_points = points[values == label]
        x1 = min(label_points[:, 1]) - extend_by
        x2 = max(label_points[:, 1]) + extend_by
        x = np.linspace(x1, x2, abs(x2 - x1) + 1, dtype=int)
        y = (slope * x + intercept).astype(int)
        x_bounded_x = x[(x >= 0) & (x < matrix.shape[1])]
        y_bounded_x = y[(x >= 0) & (x < matrix.shape[1])]
        y_bounded_xy = y_bounded_x[(y_bounded_x >= 0) & (y_bounded_x < matrix.shape[0])]
        x_bounded_xy = x_bounded_x[(y_bounded_x >= 0) & (y_bounded_x < matrix.shape[0])]
        extended_matrix[y_bounded_xy, x_bounded_xy] = label
    extended_matrix[matrix != 0] = matrix[matrix != 0]
    return extended_matrix, line_lengths


def fit_line(points):
    x = points[:, 1]
    y = points[:, 0]
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    line_length = np.sqrt((x[0] - x[-1]) ** 2 + (y[0] - y[-1]) ** 2)
    return slope, intercept, line_length


def mask_object_colors(image, parameters):
    color_spaces = [x for x in parameters["COLOR_SPACES"].keys() if x != "p"]
    color_masks = {x: None for x in color_spaces}
    for color_name in color_spaces:
        boolean_mask = np.full(image.shape[:-1], False, dtype="bool")
        for hsv_space in parameters["COLOR_SPACES"][color_name]:
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            hsv_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv_image, hsv_space[0], hsv_space[1]).astype(bool)
            mask2 = cv2.inRange(hsv_image, hsv_space[2], hsv_space[3]).astype(bool)
            boolean_mask[mask1+mask2] = True
        color_masks[color_name] = boolean_mask
    whole_object_mask = clean_mask(np.any(np.stack(list(color_masks.values()), axis=-1), axis=-1), parameters["MIN_SIZE_FOR_WHOLE_OBJECT"], fill_holes=False)
    # cv2.imshow("color_masks r", color_masks["r"].astype("uint8")*255)
    # cv2.waitKey(0)
    color_masks["r"] = mask_sharper(image, color_masks["r"], parameters["SOBEL_KERNEL_SIZE"],
                                          parameters["GRADIANT_THRESHOLD"], parameters["SPRINGS_ENDS_OPENING"], parameters["SPRINGS_ENDS_CLOSING"])
    return color_masks, whole_object_mask


def create_color_mask(image, color_spaces):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    hsv_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    binary_mask = np.zeros(image.shape[:-1], "int")
    for hsv_space in color_spaces:
        mask1 = cv2.inRange(hsv_image, hsv_space[0], hsv_space[1])
        mask2 = cv2.inRange(hsv_image, hsv_space[2], hsv_space[3])
        binary_mask[(mask1 + mask2) != 0] = 1
    binary_mask = close_blobs(binary_mask, COLOR_CLOSING)
    return binary_mask


def correct_by_boundry(hue, boundary):
    """
    HSV space has boundry (179,255,255). The hue value is periodic boundry, therefore this function retain the value to within the boundries.
    :param hue: The color in HSV
    :param boundary: in the case of HSV it is (179,255,255)
    :return: The values after correction.
    """
    boundary = boundary + 1
    final = np.zeros(3)
    for i in range(3):
        if hue[i] < 0:
            if i == 0:
                final[i] = boundary[i] + hue[i]
            else:
                final[i] = 0
        elif hue[i] >= boundary[i]:
            if i == 0:
                final[i] = np.abs(boundary[i] - hue[i])
            else:
                final[i] = boundary[i]
        else:
            final[i] = hue[i]
    return final


def no_overlap_background_sv_space(background_color, element_color_detected, margin, boundary):
    """
    Choosing the HSV space is a delicate work, therefore setting a margin for the SV isn't enough and there is a need to
     make sure there is no overlap between the background color and the HSV space of the object.
    :param background_color: Obtained 'most_common_hue'
    :param element_color_detected: The color of the element, chosen by the user.
    :param element_color_by_name: The element color the user provide (either 'red' or 'white') since there are different treatments for each one.
    :param margin: The margin to create the HSV space.
    :param boundary: The boundries of the space, (in the case of HSV it is (179,255,255))
    :return: return the HSV space (lower color, higher color).
    """
    background_margin = np.array([0, 30, 20])
    lower_bg = correct_by_boundry(background_color - background_margin, boundary)
    upper_bg = correct_by_boundry(background_color + background_margin, boundary)
    lower_element = correct_by_boundry(element_color_detected - margin, boundary)
    upper_element = correct_by_boundry(element_color_detected + margin, boundary)
    for i in range(1, 3):
        if i == 0 or i == 2:
            continue
        elif background_color[i] > element_color_detected[i]:
            if upper_element[i] > lower_bg[i]:
                upper_element[i] = lower_bg[i]
        elif background_color[i] < element_color_detected[i]:
            if lower_element[i] < upper_bg[i]:
                lower_element[i] = upper_bg[i]
    return lower_element, upper_element


def create_color_space_from_points(frame, points, margin, boundary, white=False):
    """
    Gets the points of the element (chosen by the user), the margin and the boundary and creates the HSV space,
    which will later be used to detect the object in the detection part.
    :param frame:
    :param white:
    :param points:  The points of the object, chosen by the user.
    :param margin: The margin around the color in the point, to create the space.
    :param boundary: The boundary of the space, in the case of HSV it is: 179,255,255.
    :return: Lower and upper colors of the space (there are two spaces, for cases where the space cotain 0 and below (hue is periodic))
    """
    image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    background_color = most_common_hue(frame)
    points_hsv = []
    for point in points:
        points_hsv.append(image_hsv[point[0], point[1], :])
    points_hsv = np.array(points_hsv)
    spaces = []
    for point_hsv in points_hsv:
        lower, upper = no_overlap_background_sv_space(background_color, point_hsv, margin, boundary)
        if white:
            lower[0], upper[0] = 0, 179
        lower1, upper1, lower2, upper2 = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
        for i in range(3):
            if lower[i] < upper[i]:
                lower1[i], lower2[i] = lower[i], lower[i]
                upper1[i], upper2[i] = upper[i], upper[i]
            elif lower[i] > upper[i]:
                lower1[i], lower2[i] = lower[i], 0
                upper1[i], upper2[i] = boundary[i], upper[i]
        spaces.append([lower1.astype("int"), upper1.astype("int"), lower2.astype("int"), upper2.astype("int")])
    return spaces


def calc_angles(points_to_measure, object_center, tip_point):
    ba = points_to_measure - object_center
    bc = (tip_point - object_center)
    ba_y = ba[:,0]
    ba_x = ba[:,1]
    dot = ba_y*bc[0] + ba_x*bc[1]
    det = ba_y*bc[1] - ba_x*bc[0]
    angles = np.arctan2(det, dot)
    return angles


def create_box_coordinates(center, margin, reduction_factor=1, resolution=(3840, 2160)):
    if len(center.shape) == 1:
        center = np.array([center])
    coordinates = np.zeros((center.shape[0], 4))
    for i in range(len(center)):
        y = [center[i][0] * reduction_factor - margin, center[i][0] * reduction_factor + margin]
        x = [center[i][1] * reduction_factor - margin, center[i][1] * reduction_factor + margin]
        if x[0] < 0: x[0] = 0
        if x[1] > resolution[0]: x[1] = resolution[0]
        if y[0] < 0: y[0] = 0
        if y[1] > resolution[1]: y[1] = resolution[1]
        coordinates[i] = [y[0], y[1], x[0], x[1]]
    return coordinates.astype(int)


def collect_points(image, n_points, show_color=False):
    def click_event(event, x, y, flags, params):
        if show_color:
            hue = image[y, x, 0]
            saturation = image[y, x, 1]
            value = image[y, x, 2]
            hsv = [hue, saturation, value]
            pickedColor = np.zeros((512, 512, 3), np.uint8)
            pickedColor[:] = hsv
            cv2.imshow("PickedColor", pickedColor)
            cv2.waitKey(30)
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([y, x])
    points = []
    cv2.imshow(f"Pick {n_points} pixels", image)
    cv2.setMouseCallback(f"Pick {n_points} pixels", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    points = np.array(points)
    if points.shape[0] != n_points:
        print(f"Wrong number of points picked, please pick exactly {n_points} points")
        del points
        points = collect_points(image, n_points)
    return points


def white_balance_bgr(img_bgr):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    avg_v = v.mean()
    scaling_factor = avg_v / 128.0
    balanced_v = np.clip(v * scaling_factor, 0, 255).astype(np.uint8)
    balanced_img_hsv = cv2.merge((h, s, balanced_v))
    balanced_img_bgr = cv2.cvtColor(balanced_img_hsv, cv2.COLOR_HSV2BGR)
    return balanced_img_bgr


def crop_frame(frame, coordinates):
    return frame[coordinates[0]:coordinates[1], coordinates[2]:coordinates[3]]


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


def process_image(image, alpha=2.5, beta=0, gradiant_threshold=10, sobel_kernel_size=1, blur_kernel=(7,7)):
    from scipy.ndimage import binary_fill_holes
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    image = white_balance_bgr(image)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # sharpening kernel
    image = cv2.filter2D(image, -1, kernel)
    # sobel mask:
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel_size)
    sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel_size)
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_mask = (gradient_magnitude > gradiant_threshold).astype("uint8")
    binary_fill_holes(sobel_mask, output=sobel_mask)
    # sharpen image:
    image = cv2.GaussianBlur(image, blur_kernel, 0)
    image[~(sobel_mask.astype(bool))] = [255, 255, 255]
    return image


def save_data(arrays, names, snapshot_data, parameters):
    output_path = parameters["OUTPUT_PATH"]
    continue_from_last = parameters["CONTINUE_FROM_LAST_SNAPSHOT"]
    os.makedirs(output_path, exist_ok=True)
    if snapshot_data["frame_count"]-1 == 0:
        for d, n in zip(arrays[:], names[:]):
            with open(os.path.join(output_path, str(n) + '.csv'), 'wb') as f:
                np.savetxt(f, d, delimiter=',')
                f.close()
    else:
        for d, n in zip(arrays[:], names[:]):
            with open(os.path.join(output_path, str(n) + '.csv'), 'a') as f:
                if continue_from_last:
                    line_count = get_csv_line_count(os.path.join(output_path, str(n) + '.csv'))
                    if line_count == snapshot_data["frame_count"]:
                        continue
                    else:
                        np.savetxt(f, d, delimiter=',')
                else:
                        np.savetxt(f, d, delimiter=',')
                f.close()


def get_csv_line_count(csv_file):
    with open(csv_file, 'r') as file:
        line_count = sum(1 for _ in file)
    return line_count


def convert_ants_centers_to_mathlab(output_dir):
    ants_centers_x = np.loadtxt(os.path.join(output_dir, "ants_centers_x.csv"), delimiter=",")
    ants_centers_y = np.loadtxt(os.path.join(output_dir, "ants_centers_y.csv"), delimiter=",")
    ants_centers = np.stack((ants_centers_x, ants_centers_y), axis=2)
    ants_centers_mat = np.zeros((ants_centers.shape[0], 1), dtype=np.object)
    for i in range(ants_centers.shape[0]):
        ants_centers_mat[i, 0] = ants_centers[i, :, :]
    sio.savemat(os.path.join(output_dir, "ants_centers.mat"), {"ants_centers": ants_centers_mat})
    matlab_script_path = "Z:\\Dor_Gabay\\ThesisProject\\scripts\\munkres_tracker\\"
    os.chdir(matlab_script_path)
    os.system("PATH=$PATH:'C:\Program Files\MATLAB\R2022a\bin'")
    execution_string = f"matlab -r ""ants_tracking('" + output_dir + "\\')"""
    os.system(execution_string)


def present_analysis_result(frame, calculations, springs, ants, video_name=" ", waitKey=1):
    image_to_illustrate = frame
    for point_red in springs.spring_ends_centers:
        image_to_illustrate = cv2.circle(image_to_illustrate, point_red.astype(int), 1, (0, 0, 255), 2)
    for point_blue in springs.spring_middle_part_centers:
        image_to_illustrate = cv2.circle(image_to_illustrate, point_blue.astype(int), 1, (255, 0, 0), 2)
    for count_, angle in enumerate(calculations.springs_angles_ordered):
        if angle != 0:
            if angle in springs.fixed_ends_edges_bundles_labels:
                point = springs.fixed_ends_edges_centers[springs.fixed_ends_edges_bundles_labels.index(angle)].astype(int)
                image_to_illustrate = cv2.circle(image_to_illustrate, point, 1, (0, 0, 0), 2)
                image_to_illustrate = cv2.putText(image_to_illustrate, str(count_), point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                image_to_illustrate = cv2.circle(image_to_illustrate, point, 1, (0, 0, 0), 2)
            if angle in springs.free_ends_edges_bundles_labels:
                point = springs.free_ends_edges_centers[springs.free_ends_edges_bundles_labels.index(angle)].astype(int)
                image_to_illustrate = cv2.circle(image_to_illustrate, point, 1, (0, 0, 0), 2)
    for label, ant_center_x, ant_center_y in zip(calculations.ants_attached_labels, calculations.ants_centers_x[0], calculations.ants_centers_y[0]):
        if label != 0:
            point = (int(ant_center_x), int(ant_center_y))
            image_to_illustrate = cv2.putText(image_to_illustrate, str(int(label)-1), point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    image_to_illustrate = cv2.circle(image_to_illustrate, springs.object_center_coordinates.astype(int), 1, (0, 0, 0), 2)
    image_to_illustrate = cv2.circle(image_to_illustrate, springs.needle_end.astype(int), 1, (0, 255, 0), 2)
    image_to_illustrate = white_balance_bgr(crop_frame(image_to_illustrate, springs.object_crop_coordinates))
    labeled_ants_cropped = crop_frame(ants.labeled_ants, springs.object_crop_coordinates)
    num_features = np.max(labeled_ants_cropped)+1
    cmap = plt.get_cmap('jet', num_features)
    mapped = cmap(labeled_ants_cropped)[:, :, :3]
    overlay_image = image_to_illustrate.copy()
    boolean = labeled_ants_cropped != 0
    boolean = np.repeat(boolean[:, :, np.newaxis], 3, axis=2)
    overlay_image[boolean] = (mapped[boolean] * 255).astype(np.uint8)
    if waitKey != -1:
        cv2.imshow(video_name, overlay_image)
        cv2.waitKey(waitKey)


def create_snapshot_data(parameters=None, snapshot_data=None, calculations=None, squares=None, springs=None, ants=None):
    if snapshot_data is None:
        if parameters["CONTINUE_FROM_LAST_SNAPSHOT"]\
                and os.path.exists(parameters["OUTPUT_PATH"])\
                and len([f for f in os.listdir(parameters["OUTPUT_PATH"]) if f.startswith("snap_data")]) != 0:
            snaps = [f for f in os.listdir(parameters["OUTPUT_PATH"]) if f.startswith("snap_data")]
            creation_time = [os.path.getctime(os.path.join(parameters["OUTPUT_PATH"], f)) for f in snaps]
            snapshot_data = pickle.load(open(os.path.join(parameters["OUTPUT_PATH"], snaps[np.argmax(creation_time)]), "rb"))
            parameters["STARTING_FRAME"] = snapshot_data["frame_count"]
            snapshot_data["current_time"] = datetime.datetime.now().strftime("%d.%m.%Y-%H%M")
        else:
            parameters["CONTINUE_FROM_LAST_SNAPSHOT"] = False
            snapshot_data = {"object_center_coordinates": parameters["OBJECT_CENTER_COORDINATES"][0],
                             "tip_point": None, "springs_angles_reference_order": None,
                             "sum_needle_radius": 0, "analysed_frame_count": 0, "frame_count": 0, "skipped_frames": 0,
                             "current_time": datetime.datetime.now().strftime("%d.%m.%Y-%H%M"),
                             "perspective_squares_coordinates": parameters["PERSPECTIVE_SQUARES_COORDINATES"],
                             "sum_ant_size": 0, "sum_num_ants": 0}
    else:
        snapshot_data["object_center_coordinates"] = springs.object_center_coordinates[[1, 0]]
        snapshot_data["tip_point"] = springs.needle_end
        snapshot_data["sum_needle_radius"] += int(springs.object_needle_radius)
        snapshot_data["sum_ant_size"] += ants.sum_ant_size
        snapshot_data["sum_num_ants"] += ants.sum_num_ants
        snapshot_data["springs_angles_reference_order"] = calculations.springs_angles_reference_order
        snapshot_data["perspective_squares_coordinates"] = swap_columns(squares.perspective_squares_properties[:, 0:2])
    return snapshot_data


def load_parameters(video_path, args):
    try:
        video_analysis_parameters = pickle.load(open(os.path.join(args["path"], "video_analysis_parameters.pickle"), 'rb'))[os.path.normpath(video_path)]
    except:
        raise ValueError("Video parameters for video: ", video_path, " not found. Please run the script with the flag --collect_parameters (-cp)")
    video_analysis_parameters["STARTING_FRAME"] = args["starting_frame"] if args["starting_frame"] is not None else video_analysis_parameters["STARTING_FRAME"]
    video_analysis_parameters["CONTINUE_FROM_LAST_SNAPSHOT"] = args["continue_from_last_snapshot"]
    sub_dirs = os.path.normpath(video_path).split(args["path"])[1].split(".MP4")[0].split("\\")
    video_analysis_parameters["OUTPUT_PATH"] = os.path.join(args["path"], "analysis_output", *sub_dirs)\
        if args["output_path"] is None else os.path.join(args["output_path"], *sub_dirs)
    return video_analysis_parameters


def insist_input_type(input_type, input_ask, str_list=None):
    if input_type == "yes_no":
        input_value = input(f"{input_ask} (y/n): ")
        while not input_value.isalpha() or str(input_value) not in ["y", "n"]:
            input_value = input("Please enter y or n: ")
        return str(input_value) == "y"
    if input_type == "str_list":
        input_value = input(f"{input_ask}: ")
        while not input_value.isalpha() or str(input_value) not in str_list:
            input_value = input(f"Please enter {str(str_list)}: ")
        return str(input_value)
    if input_type == "int":
        input_value = input(input_ask)
        while not input_value.isdigit():
            input_value = input("Please enter an integer: ")
        return int(input_value)
    if input_type == "float":
        input_value = input(input_ask)
        while not input_value.replace(".", "", 1).isdigit():
            input_value = input("Please enter a float: ")
        return float(input_value)


def calc_angle_matrix(a, b, c):
    """
    Calculate the angle between vectors a->b and b->c.
    a, b, and c are all 3D arrays of the same shape, where last dimension is the x, y coordinates.
    """
    ba = a - b
    bc = c - b
    ba_y = ba[:, :, 0]
    ba_x = ba[:, :, 1]
    dot = ba_y*bc[:, :, 0] + ba_x*bc[:, :, 1]
    det = ba_y*bc[:, :, 1] - ba_x*bc[:, :, 0]
    angles = np.arctan2(det, dot)
    return angles


def create_projective_transform_matrix(dst, dst_quality=None, quality_threshold=0.02, src_dimensions=(3840, 2160)):
    perfect_squares = dst_quality < quality_threshold if dst_quality is not None else np.full(dst.shape[0], True)
    PTMs = np.full((dst.shape[0], 3, 3), np.nan)
    w, h = src_dimensions
    for count, perfect_square in enumerate(perfect_squares):
        if np.all(perfect_square) and not np.any(np.isnan(dst[count])):
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



