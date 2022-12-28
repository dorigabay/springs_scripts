import copy

import cv2
import numpy as np
import os
import pickle
import pandas as pd
import scipy.cluster as sclu
import scipy.misc
import scipy

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


def rectangle_coordinates(image, what_for):
    """
    Lets the user to select the points which define a rectangle. This function takes the most extreme.
    :param image: Basically the frame.
    :param what_for: A flag to use the function 'pick_points'
    :return: list of the coordinates.
    """
    points = pick_points(image, what_for)
    yx_min = np.min(points, axis=0)
    yx_max = np.max(points, axis=0)
    return [yx_min[0], yx_max[0], yx_min[1], yx_max[1]]


def pick_points(image, what_for):
    """
    Opens a window and enables the user to choose points.
    :param image: The image to select points from.
    :param what_for: A flag to select what is the porpuse: 'element_color', 'crop','remove_rectangle'
    :return: array of the points coordinates.
    """
    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if what_for == "element_color":
                hue = image[y, x, 0]
                saturation = image[y, x, 1]
                value = image[y, x, 2]
                hsv = [hue, saturation, value]
                pickedColor = np.zeros((512, 512, 3), np.uint8)
                pickedColor[:] = hsv
                cv2.imshow("PickedColor", pickedColor)
                cv2.waitKey(30)
            points.append([y, x])

    points = []
    if what_for == "element_color":
        cv2.imshow("Pick the element pixels (only one point)", image)
        cv2.setMouseCallback("Pick the element pixels (only one point)", click_event)
    elif what_for == "crop":
        cv2.imshow("Pick where to cut (only two points)", image)
        cv2.setMouseCallback("Pick where to cut (only two points)", click_event)
    elif what_for == "remove_rectangle":
        cv2.imshow("Pick where to remove pixels (only two points)", image)
        cv2.setMouseCallback("Pick where to remove pixels (only two points)", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if len(points) != 2 and (what_for == "crop" or what_for == "remove_rectangle"): pick_points(image, what_for)
    # if len(points) != 1 and what_for == "element_color": pick_points(image, what_for)
    return np.array(points)


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
    # if element_color_by_name=="white":
    #     margin[0]=180
    # if element_color_by_name=="red":
    #     element_color_detected[0]=0
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


def get_parameters(output_folder, vidpath):
    """
    If the preferences for analyzing the video already had been save, this function enables to access them.
    :param output_folder: The directory path of the output.
    :param vidpath: The directory path of the video.
    :return: The preferences.
    """
    preferences = pd.read_pickle(os.path.join(output_folder, "video_preferences.pickle"))
    # print(os.path.join(output_folder, "video_preferences.pickle"))
    preferences_normpath = {}
    for key in preferences:
        preferences_normpath[os.path.normpath(key)] = preferences[key]
    # print(preferences_normpath["\\".join(vidpath.split('\\'))])
    return preferences_normpath["\\".join(vidpath.split('\\'))]


def get_first_frame(video, starting_frame=0):
    """
    Gets the first frame for choosing points.
    The function reduces the resolution to the resolution that will be used int the detection part - this is important
    since if the resolution scaling will be changed in either functions, it will lead for discrepancy.
    :param video:
    :param starting_frame: The default is the first frame.
    :return:
    """
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (0, 0), fx=.3, fy=.3)
    return frame


def show_parameters_result(video, parameters):
    """
    Shows the results of choose the element color, cropping coordinates and so on.
    This function asks the user for confirmation, and return the answer (boolean)
    """
    image = get_first_frame(video, parameters["starting_frame"])
    image = image[parameters["crop_coordinates"][0]:parameters["crop_coordinates"][1],
            parameters["crop_coordinates"][2]:parameters["crop_coordinates"][3]]
    print("Are you happy with the result?:")
    import utils
    binary_color_masks = utils.mask_object_colors(parameters,image)
    for mask_color in binary_color_masks:
        while True:
            image_copy = copy.copy(image)
            image_copy[binary_color_masks[mask_color].astype(bool)] = [0,255,0]
            cv2.imshow(mask_color, image_copy)
            cv2.waitKey(0)
            if input("Want to add more colors? (type any character):"):
                parameters["colors_spaces"] = add_colors_parameters(image_copy,parameters["colors_spaces"],mask_color)
                binary_color_masks = utils.mask_object_colors(parameters,image)
            else: break

    if input("If not press any key, "
             "and set the parameters for this video again (press just 'ENTER' if you do happy with the parameters):"):
        return False
    else:
        return True


def crop_frame_by_coordinates(frame, crop_coordinates):
    """
    Crops the frame by the chosen coordinates
    """
    return frame[crop_coordinates[0]:crop_coordinates[1], crop_coordinates[2]:crop_coordinates[3]]

def add_colors_parameters(frame_with_mask,colors_dict, color_name):
    margin = np.array([12, 30, 30])
    boundary_hsv = np.array([179, 255, 255])
    print("Please pick the " + color_name + " element pixels:")
    points = pick_points(frame_with_mask, what_for="element_color")
    print("Those are the colors you picked (HSV): \n",
          [list(cv2.cvtColor(frame_with_mask, cv2.COLOR_BGR2HSV)[i[0], i[1], :]) for i in points])
    color_spaces = create_color_space_from_points(frame_with_mask, points, margin, boundary_hsv)
    # print(f"color_spaces for color {color_name}: {color_spaces}")
    colors_dict[color_name] += color_spaces
    return colors_dict

def set_parameters(video, starting_frame, stiff_object=False, crop_frame=False,image=False):
    """Collects the parameters and returns them in a dictionary.
    """
    if not image:
        # set starting frame
        if starting_frame:
            print("What frame to start with: ")
            starting_frame = int(input())
        else:
            starting_frame = 0
        frame = get_first_frame(video, starting_frame=starting_frame)
    else:
        frame = cv2.imread(video)
        frame = cv2.resize(frame, (0, 0), fx=.3, fy=.3)
    # set croping coordinates
    if crop_frame:
        print("Please select where to cut (mark two corners)")
        crop_coordinates = rectangle_coordinates(frame, what_for="crop")
        frame_cropped = crop_frame_by_coordinates(frame, crop_coordinates)
    else:
        crop_coordinates = None
    # set element color space for colors
    # if collect_element:
    margin = np.array([12, 30, 30])
    boundary_hsv = np.array([179, 255, 255])
    # NUMBER_OF_COLORS = int(input("Please enter the number of color to detect:"))
    colors = {}
    if not stiff_object:
        for color_name, short_name in zip(["red","green","blue"],["r","g","b"]):
            print("Please pick the "+color_name+" element pixels:")
            points = pick_points(frame_cropped, what_for="element_color")
            print("Those are the colors you picked (HSV): \n",
                  [list(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[i[0], i[1], :]) for i in points])
            color_spaces = create_color_space_from_points(frame_cropped, points, margin, boundary_hsv)
            print(f"color_spaces for color {color_name}: {color_spaces}")
            colors[short_name] = color_spaces
    elif stiff_object:
        print("Please pick the green element pixels:")
        points = pick_points(frame_cropped, what_for="element_color")
        print("Those are the colors you picked (HSV): \n",
              [list(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[i[0], i[1], :]) for i in points])
        color_spaces = create_color_space_from_points(frame_cropped, points, margin, boundary_hsv)
        print(f"color_spaces for color {color_name}: {color_spaces}")
        colors["g"] = color_spaces
    return {"crop_coordinates": crop_coordinates, "colors_spaces": colors,
            "starting_frame": starting_frame}

def main(videos_to_analyse,output_folder,starting_frame=False,collect_crop=False):
    parameters = {}
    first = True
    for video,i in zip(videos_to_analyse,range(len(videos_to_analyse))):
        print("Collecting prefernces for: ", video)
        if first:
            parameters[video] = set_parameters(video, starting_frame=starting_frame, crop_frame=collect_crop)
            first= False
        elif not first:
            parameters[video] = copy.copy(parameters[videos_to_analyse[i-1]])
        while not show_parameters_result(video, parameters[video]):
            parameters[video] = set_parameters(video, starting_frame=starting_frame, crop_frame=collect_crop)
    with open(os.path.join(output_folder, "video_preferences.pickle"), 'wb') as f:
        pickle.dump(parameters, f)




