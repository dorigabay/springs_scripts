import os
import pickle
import cv2
import numpy as np
import scipy.cluster as sclu
from PIL import Image
# local imports:
import video_analysis.utils as utils


GRADIANT_THRESHOLD = 5
NEUTRALIZE_COLOUR_ALPHA = 2.5
BLUR_KERNEL = (7, 7)
OBJECT_COORDINATES_MARGIN = 200
PERSPECTIVE_SQUARES_MARGIN = 100
IMAGE_RESIZE_FACTOR = 0.25
RESOLUTION = np.array([2160, 3840])
COLOR_SPACE_MARGIN = np.array([12, 30, 30])
HSV_SPACE_BOUNDARY = np.array([179, 255, 255])
MAX_ANTS_NUMBER = 100


class CollectAnalysisParameters:
    def __init__(self, videos_to_analyse, output_path, collect_starting_frame=False, external_parameters=None):
        self.n_springs = self.insist_input_type("int", "How many springs are used? (default is 20)")
        self.collect_starting_frame = self.insist_input_type("yes_no", "Do you want to collect a starting frame number for each video?")
        self.collect_analysis_frame = self.insist_input_type("yes_no", "Do you want to collect a frame number to analyze for each video?")
        self.collect_image_processing_parameters = self.insist_input_type("yes_no", "Do you want to collect image processing parameters for each video?")
        for count, video in enumerate(videos_to_analyse):
            print("Collecting color parameters for: ", video)
            # video_name = os.path.basename(video).split(".")[0]
            if not os.path.exists(os.path.join(output_path, "video_analysis_parameters.pickle")):
                video_parameters = self.collect_parameters(video)
                parameters = {os.path.normpath(video): video_parameters}
            else:
                # print("Found previous parameters file, loading it...")
                parameters = pickle.load(open(os.path.join(output_path, "video_analysis_parameters.pickle"), 'rb'))
                if not os.path.normpath(video) in parameters.keys():
                    print("There are no parameters for this video")
                    if count != 0 and self.insist_input_type("yes_no", "Should I use the previous video's parameters?"):
                        video_parameters = parameters[os.path.normpath(videos_to_analyse[count - 1])]
                    else:
                        video_parameters = parameters[os.path.normpath(videos_to_analyse[count - 1])]
                        video_parameters = self.collect_parameters(video, parameters=video_parameters)
                else:
                    print("Found previous parameters for this video, loading it...")
                    video_parameters = parameters[os.path.normpath(video)]
                    if self.insist_input_type("yes_no", "Do you want to edit the parameters for this video?"):
                        if count != 0 and self.insist_input_type("yes_no", "Should I use the previous video's parameters?"):
                            video_parameters = parameters[os.path.normpath(videos_to_analyse[count - 1])]
                        else:
                            video_parameters = self.collect_parameters(video, parameters=video_parameters)
                parameters[os.path.normpath(video)] = video_parameters
            pickle.dump(parameters, open(os.path.join(output_path, f"video_analysis_parameters.pickle"), 'wb'))

    def collect_parameters(self, video_path, parameters=None):
        while True:
            self.collect_keyboard_input_parameters(parameters=parameters)
            self.collect_point_on_screen_parameters(video_path, parameters=parameters)
            parameters = {"starting_frame": self.starting_frame, "n_springs": self.n_springs, "ocm": self.ocm,
                          "pcm": self.pcm, "image_resize_factor": self.image_resize_factor,
                          "resolution": np.array(self.resolution), "color_space_margin": self.color_space_margin,
                          "hsv_space_boundary": self.hsv_space_boundary, "max_ants_number": self.max_ants_number,
                          "object_center_coordinates": self.object_center_coordinates,
                          "perspective_squares_coordinates": self.perspective_squares_coordinates,
                          "colors_spaces": self.colors_spaces,
                          "GRADIANT_THRESHOLD": self.GRADIANT_THRESHOLD,
                          "NEUTRALIZE_COLOUR_ALPHA": self.NEUTRALIZE_COLOUR_ALPHA,
                          "BLUR_KERNEL": self.BLUR_KERNEL,}
            if not self.insist_input_type("yes_no", "Do you want to edit this video's parameters again?"):
                break
        return parameters

    def collect_keyboard_input_parameters(self, parameters=None):
        self.starting_frame = int(input("What frame to start with: ")) if self.collect_starting_frame else 0
        if self.collect_image_processing_parameters:
            self.GRADIANT_THRESHOLD = int(input("What is the gradiant threshold? (default is 5)"))
            self.NEUTRALIZE_COLOUR_ALPHA = int(input("What is the neutralize colour alpha? (default is 2)"))
            blur_kernel_size = int(input("What is the blur kernel size? (default is 7)"))
            self.BLUR_KERNEL = (blur_kernel_size, blur_kernel_size)
        else:
            self.GRADIANT_THRESHOLD = GRADIANT_THRESHOLD if parameters is None else parameters["GRADIANT_THRESHOLD"]
            self.NEUTRALIZE_COLOUR_ALPHA = NEUTRALIZE_COLOUR_ALPHA if parameters is None else parameters["NEUTRALIZE_COLOUR_ALPHA"]
            self.BLUR_KERNEL = BLUR_KERNEL if parameters is None else parameters["BLUR_KERNEL"]
        self.ocm = OBJECT_COORDINATES_MARGIN if parameters is None else parameters["ocm"]
        self.pcm = PERSPECTIVE_SQUARES_MARGIN if parameters is None else parameters["pcm"]
        self.image_resize_factor = IMAGE_RESIZE_FACTOR if parameters is None else parameters["image_resize_factor"]
        self.resolution = RESOLUTION if parameters is None else parameters["resolution"]
        self.color_space_margin = COLOR_SPACE_MARGIN if parameters is None else parameters["color_space_margin"]
        self.hsv_space_boundary = HSV_SPACE_BOUNDARY if parameters is None else parameters["hsv_space_boundary"]
        self.max_ants_number = MAX_ANTS_NUMBER if parameters is None else parameters["max_ants_number"]

    def collect_point_on_screen_parameters(self, video_path, parameters=None):
        self.colors_spaces = {"b": [], "r": [], "g": [], "p": []} if parameters is None else parameters["colors_spaces"]
        image = utils.get_a_frame(video_path, analysis_frame=int(input("What frame to analyze: ")) if self.collect_analysis_frame else self.starting_frame)
        image = utils.process_image(image, alpha=self.NEUTRALIZE_COLOUR_ALPHA,
                                           blur_kernel=self.BLUR_KERNEL,
                                           gradiant_threshold=self.GRADIANT_THRESHOLD)
        reduced_resolution = cv2.resize(image, (0, 0), fx=self.image_resize_factor, fy=self.image_resize_factor)
        if (parameters is None) or self.insist_input_type("yes_no", "Do you want to collect crop coordinates?"):
            print("Please point on the center of the object, to get the initial crop coordinates.")
            self.object_center_coordinates = utils.collect_points(reduced_resolution, n_points=1).reshape(1, 2) * (1 / self.image_resize_factor)
            print("Please point on ALL four perspective squares, to get the initial crop coordinates.")
            self.perspective_squares_coordinates = utils.collect_points(reduced_resolution, n_points=4) * (1 / self.image_resize_factor)
        else:
            self.object_center_coordinates = parameters["object_center_coordinates"]
            self.perspective_squares_coordinates = parameters["perspective_squares_coordinates"]
        if (parameters is None) or self.insist_input_type("yes_no", "Do you want to collect colors spaces?"):
            self.collect_or_edit_colors_spaces(image)

    def collect_or_edit_colors_spaces(self, image):
        collage_image = self.create_collage_image(image, self.ocm, self.pcm, self.object_center_coordinates, self.perspective_squares_coordinates)
        for color_name, color_short in zip(["blue", "red", "green", "purple"], self.colors_spaces.keys()):
            overlays_image = np.copy(collage_image)
            while True:
                if len(self.colors_spaces[color_short]) != 0:
                    overlays_image[utils.create_color_mask(collage_image, self.colors_spaces[color_short]).astype(bool)] = [0, 255, 0]
                    cv2.imshow(color_name, overlays_image)
                    cv2.waitKey(0)
                    ask_sentence = "Want to add more color spaces? (y/n/r to remove the last color)"
                    add_colors = self.insist_input_type("str_list", ask_sentence, str_list=["y", "n", "r"])
                else:
                    add_colors = "y"
                if add_colors == "n":
                    break
                elif add_colors == "r":
                    self.colors_spaces[color_short].pop()
                    overlays_image = np.copy(collage_image)
                else:
                    print("Please pick the " + color_name + " element pixels:")
                    points = utils.collect_points(overlays_image, n_points=1, show_color=True)
                    self.colors_spaces[color_short] += self.create_color_space_from_points(overlays_image, points, self.color_space_margin, self.hsv_space_boundary)

    def insist_input_type(self, input_type, input_ask, str_list=None):
        if input_type == "yes_no":
            input_value = str(input(f"{input_ask} (y/n): "))
            while input_value not in ["y", "n"]:
                input_value = str(input("Please enter y or n: "))
            return input_value == "y"
        if input_type == "str_list":
            input_value = str(input(f"{input_ask}: "))
            while input_value not in str_list:
                input_value = str(input(f"Please enter {str(str_list)}: "))
            return input_value
        if input_type == "int":
            input_value = int(input(input_ask))
            while not isinstance(input_value, int):
                input_value = int(input("Please enter an integer: "))
            return input_value

    def create_collage_image(self, frame, ocm, pcm, object_center_coordinates, perspective_squares_coordinates):
        complete_image = np.zeros((ocm * 2, ocm * 2 * 2, 3), dtype="uint8")
        object_crop_coordinates = utils.create_box_coordinates(object_center_coordinates, ocm)[0]
        perspective_squares_crop_coordinates = utils.create_box_coordinates(perspective_squares_coordinates, pcm)
        complete_image[0:ocm * 2, 0:ocm * 2] = frame[object_crop_coordinates[0]:object_crop_coordinates[1],
                                                     object_crop_coordinates[2]:object_crop_coordinates[3]]
        perspective_first = frame[perspective_squares_crop_coordinates[0][0]:perspective_squares_crop_coordinates[0][1],
                                  perspective_squares_crop_coordinates[0][2]:perspective_squares_crop_coordinates[0][3]]
        perspective_second = frame[perspective_squares_crop_coordinates[1][0]:perspective_squares_crop_coordinates[1][1],
                                   perspective_squares_crop_coordinates[1][2]:perspective_squares_crop_coordinates[1][3]]
        perspective_third = frame[perspective_squares_crop_coordinates[2][0]:perspective_squares_crop_coordinates[2][1],
                                  perspective_squares_crop_coordinates[2][2]:perspective_squares_crop_coordinates[2][3]]
        perspective_fourth = frame[perspective_squares_crop_coordinates[3][0]:perspective_squares_crop_coordinates[3][1],
                                   perspective_squares_crop_coordinates[3][2]:perspective_squares_crop_coordinates[3][3]]
        complete_image[0:perspective_first.shape[0], ocm * 2:ocm * 2 + perspective_first.shape[1]] = perspective_first
        complete_image[0:perspective_second.shape[0], ocm * 2 + pcm * 2:600 + perspective_second.shape[1]] = perspective_second
        complete_image[pcm * 2:pcm * 2 + perspective_third.shape[0], ocm * 2 + pcm * 2:ocm * 2 + pcm * 2 + perspective_third.shape[1]] = perspective_third
        complete_image[pcm * 2:pcm * 2 + perspective_fourth.shape[0], ocm * 2:ocm * 2 + perspective_fourth.shape[1]] = perspective_fourth
        return complete_image

    def get_background_color(self, frame):
        # convert a frame to PIL image
        im = Image.fromarray(frame)
        colors = im.getcolors(im.size[0] * im.size[1])
        colors = sorted(colors, key=lambda x: x[0], reverse=True)
        background_color = colors[0][1]
        return background_color

    def most_common_hue(self, frame):
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

    def correct_by_boundry(self, hue, boundry):
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

    def no_overlap_background_sv_space(self, background_color, element_color_detected, margin, boundry):
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
        lower_bg = self.correct_by_boundry(background_color - background_margin, boundry)
        upper_bg = self.correct_by_boundry(background_color + background_margin, boundry)
        lower_element = self.correct_by_boundry(element_color_detected - margin, boundry)
        upper_element = self.correct_by_boundry(element_color_detected + margin, boundry)
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

    def create_color_space_from_points(self, frame, points, margin, boundary, white=False):
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
        background_color = self.most_common_hue(frame)
        background_color2 = self.get_background_color(frame)
        print("background_color", background_color)
        print("background_color2", background_color2)
        points_hsv = []
        for point in points:
            points_hsv.append(image_hsv[point[0], point[1], :])
        points_hsv = np.array(points_hsv)
        spaces = []
        for point_hsv in points_hsv:
            lower, upper = self.no_overlap_background_sv_space(background_color, point_hsv, margin, boundary)
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

