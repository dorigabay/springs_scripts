import copy, os, pickle, re
import cv2
import numpy as np
import pandas as pd
import scipy.cluster as sclu

# local imports:
import video_analysis.utils as utils


def get_background_color(frame):
    from PIL import Image
    #conver a frame to PIL image
    im = Image.fromarray(frame)
    colors = im.getcolors(im.size[0] * im.size[1])
    colors = sorted(colors, key=lambda x: x[0], reverse=True)
    background_color = colors[0][1]
    return background_color


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
    vecs, dist = sclu.vq.vq(ar, codes)  # assign codes
    counts, bins = np.histogram(vecs, len(codes))  # count occurrences
    index_max = np.argmax(counts)  # find most frequent
    peak = codes[index_max].astype("int")
    print('most frequent color (background) is {})'.format(peak))
    return peak


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


def no_overlap_background_sv_space(background_color, element_color_detected, margin, boundry):
    """
    Choosing the HSV space is a delicate work, therefore setting a margin for the SV isn't enough and there is a need to
     make sure there is no overlap between the background color and the HSV space of the object.
    :param background_color: Obtained 'most_common_hue'
    :param element_color_detected: The color of the element, chosen by the user.
    :param element_color_by_name: The element color the user provide (either 'red' or 'white') since there are different treatments for each one.
    :param margin: The margin to create the HSV space.
    :param boundry: The boundries of the space, (in the case of HSV it is (179,255,255))
    :return: return the HSV space (lower color, higher color).
    """
    background_margin = np.array([0, 30, 20])
    lower_bg = correct_by_boundry(background_color - background_margin, boundry)
    upper_bg = correct_by_boundry(background_color + background_margin, boundry)
    lower_element = correct_by_boundry(element_color_detected - margin, boundry)
    upper_element = correct_by_boundry(element_color_detected + margin, boundry)
    for i in range(1, 3):
        if i == 0 or i == 2:
            continue
        elif background_color[i] > element_color_detected[i]:
            if upper_element[i] > lower_bg[i]: upper_element[i] = lower_bg[i]
        elif background_color[i] < element_color_detected[i]:
            if lower_element[i] < upper_bg[i]: lower_element[i] = upper_bg[i]
    return lower_element, upper_element


def create_color_space_from_points(frame, points, margin, boundary,white=False):
    """
    Gets the points of the element (chosen by the user), the margin and the boundary and creates the HSV space,
    which will later be used to detect the object in the detection part.
    :param points:  The points of the object, chosen by the user.
    :param margin: The margin around the color in the point, to create the space.
    :param boundary: The boundary of the space, in the case of HSV it is: 179,255,255.
    :return: Lower and upper colors of the space (there are two spaces, for cases where the space cotain 0 and below (hue is periodic))
    """
    image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    background_color = most_common_hue(frame)
    background_color2 = get_background_color(frame)
    print("background_color", background_color)
    print("background_color2", background_color2)
    points_hsv = []
    for point in points:
        points_hsv.append(image_hsv[point[0], point[1], :])
    points_hsv = np.array(points_hsv)
    spaces = []
    for point_hsv in points_hsv:
        lower, upper = no_overlap_background_sv_space(background_color, point_hsv, margin,boundary)
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


def get_parameters(output_folder, video_path):
    """
    If the preferences for analyzing the video already had been save, this function enables to access them.
    :param output_folder: The directory path of the output.
    :param vidpath: The directory path of the video.
    :return: The preferences.
    """
    video_name = os.path.basename(video_path).split(".")[0].split("_")[0]
    parameters = pd.read_pickle(os.path.join(output_folder, f"{video_name}_video_parameters.pickle"))
    parameters_normpath = {}
    keys_ends = {}
    for key in parameters:
        keys_ends[re.search(".*[/\\\\]([^/\\\\]+).[mM][pP]4", key).group(1)] = os.path.normpath(key)
        parameters_normpath[os.path.normpath(key)] = parameters[key]

    # #single time parameters addition:
    # key = list(parameters.keys())[0]
    # parameters[key]["ocm"] = 200
    # parameters[key]["pcm"] = 100
    # parameters[key]["object_center_coordinates"] = np.array((parameters[key]["crop_coordinates"][0]+200,parameters[key]["crop_coordinates"][2]+200)).reshape(1,2)
    # parameters[key]["perspective_squares_coordinates"] = np.array([[parameters[key]["purple_squares_coordinates"][0, 0]+100,parameters[key]["purple_squares_coordinates"][0, 2]+100],
    #                                                     [parameters[key]["purple_squares_coordinates"][1, 0]+100,parameters[key]["purple_squares_coordinates"][1, 2]+100],
    #                                                     [parameters[key]["purple_squares_coordinates"][2, 0]+100,parameters[key]["purple_squares_coordinates"][2, 2]+100],
    #                                                     [parameters[key]["purple_squares_coordinates"][3, 0]+100,parameters[key]["purple_squares_coordinates"][3, 2]+100]])
    # parameters[key]["max_ants_number"] = 100
    # parameters[key]["resolution"] = np.array((2160, 3840))
    # with open(os.path.join(output_folder, f"{video_name}_video_parameters.pickle"), 'wb') as handle:
    #     pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return parameters_normpath[keys_ends[video_name]]


def show_parameters_result(video, parameters, ocm=200, pcm=100):
    """
    Shows the results of choose the element color, cropping coordinates and so on.
    This function asks the user for confirmation, and return the answer (boolean)
    ocm: object crop margin size
    pcm: perspective squares crop margin size
    """
    image = utils.get_a_frame(video, parameters["starting_frame"])
    image = utils.neutrlize_colour(image, alpha=2.5, beta=0)
    image = utils.white_balance_bgr(image)
    complete_image = np.zeros((ocm * 2, ocm * 2 * 2, 3), dtype="uint8")
    object_crop_coordinates = utils.create_box_coordinates(parameters["object_center_coordinates"], ocm)[0]
    perspective_squares_crop_coordinates = utils.create_box_coordinates(parameters["perspective_squares_coordinates"], pcm)
    complete_image[0:ocm * 2, 0:ocm * 2] = image[object_crop_coordinates[0]:object_crop_coordinates[1],
                                                 object_crop_coordinates[2]:object_crop_coordinates[3]]
    perspective_first = image[perspective_squares_crop_coordinates[0][0]:perspective_squares_crop_coordinates[0][1],
            perspective_squares_crop_coordinates[0][2]:perspective_squares_crop_coordinates[0][3]]
    perspective_second = image[perspective_squares_crop_coordinates[1][0]:perspective_squares_crop_coordinates[1][1],
            perspective_squares_crop_coordinates[1][2]:perspective_squares_crop_coordinates[1][3]]
    perspective_third = image[perspective_squares_crop_coordinates[2][0]:perspective_squares_crop_coordinates[2][1],
            perspective_squares_crop_coordinates[2][2]:perspective_squares_crop_coordinates[2][3]]
    perspective_fourth = image[perspective_squares_crop_coordinates[3][0]:perspective_squares_crop_coordinates[3][1],
            perspective_squares_crop_coordinates[3][2]:perspective_squares_crop_coordinates[3][3]]
    complete_image[0:perspective_first.shape[0], ocm * 2:ocm * 2 + perspective_first.shape[1]] = perspective_first
    complete_image[0:perspective_second.shape[0], ocm * 2 + pcm * 2:600 + perspective_second.shape[1]] = perspective_second
    complete_image[pcm * 2:pcm * 2 + perspective_third.shape[0], ocm * 2 + pcm * 2:ocm * 2 + pcm * 2 + perspective_third.shape[1]] = perspective_third
    complete_image[pcm * 2:pcm * 2 + perspective_fourth.shape[0], ocm * 2:ocm * 2 + perspective_fourth.shape[1]] = perspective_fourth

    print("Are you happy with the result?:")
    binary_color_masks = utils.mask_object_colors(parameters, complete_image)
    for mask_color in binary_color_masks:
        while True:
            image_copy = copy.copy(complete_image)
            image_copy[binary_color_masks[mask_color].astype(bool)] = [0, 255, 0]
            cv2.imshow(mask_color, image_copy)
            cv2.waitKey(0)
            add_colors = str(input("Want to add more colors? (y/n/r to remove the last color)."))
            while add_colors not in ["y", "n", "r"]:
                add_colors = str(input("Please enter y or n:"))
            if add_colors == "y":
                parameters["colors_spaces"] = add_colors_parameters(image_copy,parameters["colors_spaces"],mask_color)
                binary_color_masks = utils.mask_object_colors(parameters, complete_image)
            elif add_colors == "r":
                parameters["colors_spaces"][mask_color].pop()
                binary_color_masks = utils.mask_object_colors(parameters, complete_image)
            else: break
    collect_again = str(input("Want to collect again? (y/n)"))
    while collect_again not in ["y", "n"]:
        collect_again = str(input("Please enter y or n:"))
    if collect_again == "y":
        return False
    else:
        return True


def add_colors_parameters(frame_with_mask,colors_dict, color_name):
    margin = np.array([12, 30, 30])
    boundary_hsv = np.array([179, 255, 255])
    print("Please pick the " + color_name + " element pixels:")
    points = utils.collect_points(frame_with_mask, n_points=1, show_color=True)
    print("Those are the colors you picked (HSV): \n",
          [list(cv2.cvtColor(frame_with_mask, cv2.COLOR_BGR2HSV)[i[0], i[1], :]) for i in points])
    color_spaces = create_color_space_from_points(frame_with_mask, points, margin, boundary_hsv)
    # print(f"color_spaces for color {color_name}: {color_spaces}")
    colors_dict[color_name] += color_spaces
    return colors_dict


def set_parameters(video, starting_frame=False, n_springs=20, ocm=200, pcm=100, reduction=0.25, resolution=(2160, 3840), max_ants_number=100):
    """Collects the parameters and returns them in a dictionary.
        ocm: object crop margin size
        pcm: perspective squares margin size
        reduction: reduction factor for the image
    """
    # load the frame:
    if starting_frame:
        print("What frame to start with: ")
        start_frame = int(input())
    else:
        start_frame = 0
    frame = utils.get_a_frame(video, starting_frame=start_frame)
    frame = utils.neutrlize_colour(frame,alpha=2.5, beta=0)
    frame = utils.white_balance_bgr(frame)
    # crop the frame:
    reduced_resolution = cv2.resize(frame, (0, 0), fx=reduction, fy=reduction)
    print("Please point on the center of the object, to get the initial crop coordinates.")
    object_center_coordinates = utils.collect_points(reduced_resolution, n_points=1).reshape(1, 2)
    print("Please point on ALL four perspective squares, to get the initial crop coordinates.")
    perspective_squares_coordinates = utils.collect_points(reduced_resolution, n_points=4)
    object_crop_coordinates = utils.create_box_coordinates(object_center_coordinates, ocm, reduction_factor=int(1 / reduction))[0]
    perspective_squares_crop_coordinates = utils.create_box_coordinates(perspective_squares_coordinates, pcm, reduction_factor=int(1/reduction))
    object_frame_cropped = frame[object_crop_coordinates[0]:object_crop_coordinates[1],
                    object_crop_coordinates[2]:object_crop_coordinates[3]]
    perspective_square_frame_cropped = frame[perspective_squares_crop_coordinates[0][0]:perspective_squares_crop_coordinates[0][1],
                    perspective_squares_crop_coordinates[0][2]:perspective_squares_crop_coordinates[0][3]]
    # collect the colors spaces:
    margin = np.array([12, 30, 30])
    boundary_hsv = np.array([179, 255, 255])
    colors = {}
    for color_name, short_name in zip(["blue", "red", "green", "purple"], ["b", "r", "g", "p"]):
        if color_name == "purple":
            frame_cropped = perspective_square_frame_cropped
        else:
            frame_cropped = object_frame_cropped
        print("Please pick the "+color_name+" element pixels:")
        points = utils.collect_points(frame_cropped, n_points=1, show_color=True)
        print("Those are the colors you picked (HSV): \n",
              [list(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[i[0], i[1], :]) for i in points])
        color_spaces = create_color_space_from_points(frame_cropped, points, margin, boundary_hsv)
        print(f"color_spaces for color {color_name}: {color_spaces}")
        colors[short_name] = color_spaces
    parameters = {"starting_frame": start_frame,
                  "object_center_coordinates": object_center_coordinates,
                  "perspective_squares_coordinates": perspective_squares_coordinates,
                  "ocm": ocm,
                  "pcm": pcm,
                  "colors_spaces": colors,
                  "n_springs": n_springs,
                  "max_ants_number": max_ants_number,
                  "resolution": np.array(resolution)}
    while not show_parameters_result(video, parameters):
        parameters = set_parameters(video, starting_frame=starting_frame, n_springs=n_springs)
    return parameters


def main(videos_to_analyse,output_folder,starting_frame=False, n_springs=None):
    output_path_parameters = os.path.join(output_folder, "parameters")
    os.makedirs(output_path_parameters, exist_ok=True)
    parameters = {}
    first = True
    if n_springs is None: n_springs = int(input("How many springs attached to the object?"))
    for video,i in zip(videos_to_analyse,range(len(videos_to_analyse))):
        print("Collecting prefernces for: ", video)
        video_name = os.path.basename(video).split(".")[0]
        reuse = 'n'
        if os.path.exists(os.path.join(output_path_parameters, f"{video_name}_video_parameters.pickle")):
            reuse = str(input("Found parameters for this video. Do you want to use them? (y/n)"))
            while reuse not in ["y", "n"]:
                reuse = str(input("Please enter y or n:"))
        if reuse == 'y':
            with open(os.path.join(output_path_parameters, f"{video_name}_video_parameters.pickle"), 'rb') as f:
                parameters[video] = pickle.load(f)[video]
                first = False
        elif reuse == 'n':
            if first:
                parameters[video] = set_parameters(video, starting_frame=starting_frame, n_springs=n_springs)
                first = False
            elif not first:
                parameters[video] = copy.copy(parameters[videos_to_analyse[i-1]])
                parameters[video]["starting_frame"] = int(input("What frame to start with: "))
                while not show_parameters_result(video, parameters[video]):
                    parameters[video] = set_parameters(video, starting_frame=starting_frame, n_springs=n_springs)
            with open(os.path.join(output_path_parameters, f"{video_name}_video_parameters.pickle"), 'wb') as f:
                pickle.dump({video: parameters[video]}, f)
    return output_path_parameters

