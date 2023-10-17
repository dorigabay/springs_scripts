import cv2
import numpy as np
import os
import scipy.io as sio
import pickle
import datetime
from skimage.morphology import binary_dilation, binary_erosion, remove_small_objects
from scipy.ndimage import maximum_filter, minimum_filter, label, binary_fill_holes
import matplotlib.pyplot as plt


COLOR_CLOSING = np.ones((3, 3))


def find_circle(mask):
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image_gray = cv2.GaussianBlur(mask, (9, 9), 2)
    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    circles = cv2.HoughCircles(
        mask,  # Input image
        cv2.HOUGH_GRADIENT,  # Detection method (Hough Gradient method)
        dp=1,  # Inverse ratio of the accumulator resolution to the image resolution
        minDist=20,  # Minimum distance between detected centers of the circles
        param1=50,  # Upper threshold for the internal Canny edge detector
        param2=30,  # Threshold for center detection
        minRadius=10,  # Minimum radius of the detected circles
        maxRadius=100  # Maximum radius of the detected circles
    )

    if circles is not None:
        # Convert the circle coordinates to integer values
        circles = np.uint16(np.around(circles))

        # Draw the detected circles on the original image
        for circle in circles[0, :]:
            center = (circle[0], circle[1])  # Circle center
            radius = circle[2]  # Circle radius

            # Draw the circle outline
            cv2.circle(mask, center, radius, (0, 255, 0), 2)

        # Display the image with detected circles
        cv2.imshow('Detected Circles', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No circles were detected.")


def get_farthest_point(point, contour, n_points=50, inverse=False):
    contour = swap_columns(contour.copy())
    distances = np.sqrt(np.sum(np.square(contour-point), 1))
    # take the average of the 50 farthest points
    farthest_points = np.argsort(distances)[-n_points:] if not inverse else np.argsort(distances)[:n_points]
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


def clean_mask(mask, min_size, fill_holes=True, circle_center_remove=None, circle_radius_remove=None):
    mask = mask.astype(bool)
    if not (circle_center_remove is None or circle_radius_remove is None):
        inner_circle_mask = create_circular_mask(mask.shape, center=circle_center_remove, radius=circle_radius_remove * 0.9)
        outer_circle_mask = create_circular_mask(mask.shape, center=circle_center_remove, radius=circle_radius_remove * 2.5)
        mask[inner_circle_mask] = False
        mask[np.invert(outer_circle_mask)] = False
    if fill_holes:
        mask = binary_fill_holes(mask.astype(np.uint8)).astype(bool)
    mask = remove_small_blobs(mask, min_size)
    return mask


def extend_lines(matrix, extend_by=3):
    indices = np.nonzero(matrix)
    points = np.transpose(indices)
    values = matrix[indices]
    extended_matrix = np.zeros_like(matrix)
    for label in np.unique(values):
        slope, intercept = fit_line(points[values == label])
        label_points = points[values == label]
        x1 = min(label_points[:, 1]) - extend_by
        x2 = max(label_points[:, 1]) + extend_by
        x = np.linspace(x1, x2, abs(x2 - x1) + 1, dtype=int)
        y = (slope * x + intercept).astype(int)
        x_bounded_x = x[(x >= 0) & (x < matrix.shape[1])]
        y_bounded_x = y[(x >= 0) & (x < matrix.shape[1])]
        y_bounded_xy = y_bounded_x[(y_bounded_x >= 0) & (y_bounded_x < matrix.shape[0])]
        x_bounded_xy = x_bounded_x[(y_bounded_x >= 0) & (y_bounded_x < matrix.shape[0])]
        extended_matrix[y_bounded_xy,x_bounded_xy] = label
    extended_matrix[matrix!=0] = matrix[matrix!=0]
    return extended_matrix


def fit_line(points):
    x = points[:, 1]
    y = points[:, 0]
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return slope, intercept


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


# def mask_object_colors(parameters, image):
#     binary_color_masks = {x: None for x in parameters["colors_spaces"]}
#     blurred = cv2.GaussianBlur(image, (3, 3), 0)
#     hsv_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
#     for color in parameters["colors_spaces"]:
#         binary_mask = np.zeros(image.shape[:-1], "int")
#         for hsv_space in parameters["colors_spaces"][color]:
#             mask1 = cv2.inRange(hsv_image, hsv_space[0], hsv_space[1])
#             mask2 = cv2.inRange(hsv_image, hsv_space[2], hsv_space[3])
#             binary_mask[(mask1 + mask2) != 0] = 1
#         binary_mask = close_blobs(binary_mask, COLOR_CLOSING)
#         binary_color_masks[color] = binary_mask
#     return binary_color_masks


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
    output_path = parameters["output_path"]
    continue_from_last = parameters["continue_from_last_snapshot"]
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


def present_analysis_result(frame, calculations, springs, ants, video_name=" ", reduction_factor=1, waitKey=1):
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
    image_to_illustrate = cv2.circle(image_to_illustrate, springs.tip_point.astype(int), 1, (0, 255, 0), 2)
    image_to_illustrate = white_balance_bgr(crop_frame(image_to_illustrate, springs.object_crop_coordinates))
    labeled_ants_cropped = crop_frame(ants.labeled_ants, springs.object_crop_coordinates)
    num_features = np.max(labeled_ants_cropped)+1
    cmap = plt.get_cmap('jet', num_features)
    mapped = cmap(labeled_ants_cropped)[:, :, :3]
    overlay_image = image_to_illustrate.copy()
    boolean = labeled_ants_cropped != 0
    boolean = np.repeat(boolean[:, :, np.newaxis], 3, axis=2)
    overlay_image[boolean] = (mapped[boolean] * 255).astype(np.uint8)
    # plt.imshow(overlay_image.astype(np.uint8))
    # plt.title(video_name)
    # plt.show()
    if waitKey != -1:
        cv2.imshow(video_name, overlay_image)
        cv2.waitKey(waitKey)
    return image_to_illustrate


def create_snapshot_data(parameters=None, snapshot_data=None, calculations=None, squares=None, springs=None):
    if snapshot_data is None:
        if parameters["continue_from_last_snapshot"] and len([f for f in os.listdir(parameters["output_path"]) if f.startswith("snap_data")]) != 0:
            snaps = [f for f in os.listdir(parameters["output_path"]) if f.startswith("snap_data")]
            creation_time = [os.path.getctime(os.path.join(parameters["output_path"], f)) for f in snaps]
            snapshot_data = pickle.load(open(os.path.join(parameters["output_path"], snaps[np.argmax(creation_time)]), "rb"))
            parameters["starting_frame"] = snapshot_data["frame_count"]
            snapshot_data["current_time"] = datetime.datetime.now().strftime("%d.%m.%Y-%H%M")
        else:
            snapshot_data = {"object_center_coordinates": parameters["object_center_coordinates"][0],
                             "tip_point": None, "springs_angles_reference_order": None,
                             "sum_needle_radius": 0, "analysed_frame_count": 0, "frame_count": 0,"skipped_frames": 0,
                             "current_time": datetime.datetime.now().strftime("%d.%m.%Y-%H%M"),
                             "perspective_squares_coordinates": parameters["perspective_squares_coordinates"]}
    else:
        snapshot_data["object_center_coordinates"] = springs.object_center_coordinates[[1, 0]]
        snapshot_data["tip_point"] = springs.tip_point
        snapshot_data["sum_needle_radius"] += int(springs.object_needle_radius)
        snapshot_data["springs_angles_reference_order"] = calculations.springs_angles_reference_order
        snapshot_data["perspective_squares_coordinates"] = swap_columns(squares.perspective_squares_properties[:, 0:2])
    return snapshot_data


def load_parameters(video_path, args):
    try:
        video_analysis_parameters = pickle.load(open(os.path.join(args["path"], "video_analysis_parameters.pickle"), 'rb'))[os.path.normpath(video_path)]
    except:
        raise ValueError("Video parameters for video: ", video_path, " not found. Please run the script with the flag --collect_parameters (-cp)")
    video_analysis_parameters["starting_frame"] = args["starting_frame"] if args["starting_frame"] is not None else video_analysis_parameters["starting_frame"]
    video_analysis_parameters["continue_from_last_snapshot"] = args["continue_from_last_snapshot"]
    sub_dirs = os.path.normpath(video_path).split(args["path"])[1].split(".MP4")[0].split("\\")
    video_analysis_parameters["output_path"] = os.path.join(args["path"], "analysis_output", *sub_dirs)\
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


# def cut_access_data(video_n_frames, parameters):
#     print("Cutting access data", parameters["output_path"])
#     import numpy as np
#     data_names = ["perspective_squares_coordinates_x", "perspective_squares_coordinates_y", "perspective_squares_rectangle_similarity", "perspective_squares_squareness",
#                   "N_ants_around_springs", "size_ants_around_springs", "fixed_ends_coordinates_x", "fixed_ends_coordinates_y", "free_ends_coordinates_x",
#                   "free_ends_coordinates_y", "needle_part_coordinates_x", "needle_part_coordinates_y",
#                   "ants_centers_x", "ants_centers_y", "ants_attached_labels", "ants_attached_forgotten_labels"]
#     for data_name in data_names:
#         file = np.loadtxt(os.path.join(parameters["output_path"], data_name+".csv"), delimiter=",")
#         np.savetxt(os.path.join(parameters["output_path"], data_name+".csv"), file[:video_n_frames], delimiter=",")


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