import os
import pickle
import cv2
import numpy as np
import scipy.cluster as sclu
from PIL import Image
import matplotlib.pyplot as plt
# local imports:
import utils
from calculator import Calculation
from ants_detector import Ants
from springs_detector import Springs
from perspective_squares import PerspectiveSquares


GRADIANT_THRESHOLD = 5
NEUTRALIZE_COLOUR_ALPHA = 2.5
NEUTRALIZE_COLOUR_BETA = (7, 7)
BUNDLES_CLOSING_SIZE = 3
ANTS_MIN_SIZE = 50
ANTS_CLOSING_KERNEL = np.ones((5, 5))
ANTS_OBJECT_DILATION_SIZE = 3
ANTS_GRADIANT_THRESHOLD = 90
ANTS_NEUTRALIZE_COLOUR_ALPHA = 2
ANTS_NEUTRALIZE_COLOUR_BETA = 10
ANTS_SPRINGS_OVERLAP_SIZE = 3
ANTS_LOWER_HSV_VALUES = np.array([0, 0, 0])
ANTS_UPPER_HSV_VALUES = np.array([179, 255, 200])
OBJECT_COORDINATES_MARGIN = 200
PERSPECTIVE_SQUARES_MARGIN = 100
IMAGE_RESIZE_FACTOR = 0.25
RESOLUTION = np.array([2160, 3840])
COLOR_SPACE_MARGIN = np.array([12, 30, 30])
HSV_SPACE_BOUNDARY = np.array([179, 255, 255])
MAX_ANTS_NUMBER = 100


class CollectParameters:
    def __init__(self, video_paths, output_path, external_parameters=None):
        self.video_paths = video_paths
        self.output_path = output_path
        self.external_parameters = external_parameters
        # self.n_springs = utils.insist_input_type("int", "How many springs does the object has? (default is 20): ")
        self.n_springs = 20
        self.collect_starting_frame = False #utils.insist_input_type("yes_no", "Would you like to collect a starting frame number for each video?")
        # self.collect_image_processing_parameters = utils.insist_input_type("yes_no", "Would you like to collect image processing parameters for each video?")
        self.collect_image_processing_parameters = False
        # self.show_analysis_example = utils.insist_input_type("yes_no", "Would you like to show analysis example for each video?")
        self.show_analysis_example = True
        self.iterate_videos()

    def iterate_videos(self):
        for count, video in enumerate(self.video_paths):
            print("Collecting color parameters for: ", video)
            if not os.path.exists(os.path.join(self.output_path, "video_analysis_parameters.pickle")):
                video_parameters = self.collect_parameters(video)
                parameters = {os.path.normpath(video): video_parameters}
            else:
                parameters = pickle.load(open(os.path.join(self.output_path, "video_analysis_parameters.pickle"), 'rb'))
                if not os.path.normpath(video) in parameters.keys():
                    print("There are no parameters for this video")
                    if count != 0 and utils.insist_input_type("yes_no", "Should I use the previous video's parameters?"):
                        video_parameters = parameters[os.path.normpath(self.video_paths[count - 1])]
                        if utils.insist_input_type("yes_no", "Would you like to edit the parameters for this video?"):
                            video_parameters = self.collect_parameters(video, parameters=video_parameters)
                    else:
                        video_parameters = self.collect_parameters(video)
                else:
                    print("Found previous parameters for this video, loading it...")
                    video_parameters = parameters[os.path.normpath(video)]
                    # if True:
                    if utils.insist_input_type("yes_no", "Would you like to edit the parameters for this video?"):
                        if count != 0 and utils.insist_input_type("yes_no", "Should I use the previous video's parameters?"):
                            video_parameters = parameters[os.path.normpath(self.video_paths[count - 1])]
                        else:
                            video_parameters = self.collect_parameters(video, parameters=video_parameters)
                parameters[os.path.normpath(video)] = video_parameters
            pickle.dump(parameters, open(os.path.join(self.output_path, f"video_analysis_parameters.pickle"), 'wb'))

    def collect_parameters(self, video_path, parameters=None):
        while True:
            self.collect_image_preprocessing_parameters(parameters=parameters)
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, utils.insist_input_type("int", "From which frame would you like to collect the parameters?: "))
            # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            _, image = cap.read()
            processed_image = utils.process_image(image, alpha=self.NEUTRALIZE_COLOUR_ALPHA, blur_kernel=self.NEUTRALIZE_COLOUR_BETA, gradiant_threshold=self.GRADIANT_THRESHOLD)
            collage_image = self.collect_crop_coordinates(processed_image, parameters=parameters)
            self.collect_or_edit_colors_spaces(collage_image, parameters=parameters)
            parameters = {"starting_frame": self.starting_frame, "n_springs": self.n_springs, "ocm": self.ocm,
                          "pcm": self.pcm, "image_resize_factor": self.image_resize_factor,
                          "resolution": np.array(self.resolution), "color_space_margin": self.color_space_margin,
                          "hsv_space_boundary": self.hsv_space_boundary, "max_ants_number": self.max_ants_number,
                          "object_center_coordinates": self.object_center_coordinates,
                          "perspective_squares_coordinates": self.perspective_squares_coordinates,
                          "colors_spaces": self.colors_spaces,
                          "GRADIANT_THRESHOLD": self.GRADIANT_THRESHOLD, "ANTS_GRADIANT_THRESHOLD": self.ANTS_GRADIANT_THRESHOLD,
                          "NEUTRALIZE_COLOUR_ALPHA": self.NEUTRALIZE_COLOUR_ALPHA, "NEUTRALIZE_COLOUR_BETA": self.NEUTRALIZE_COLOUR_BETA,
                          "BUNDLES_CLOSING_SIZE": self.BUNDLES_CLOSING_SIZE,
                          "ANTS_SPRINGS_OVERLAP_SIZE": self.ANTS_SPRINGS_OVERLAP_SIZE, "ANTS_OBJECT_DILATION_SIZE": self.ANTS_OBJECT_DILATION_SIZE,
                          "ANTS_UPPER_HSV_VALUES": self.ANTS_UPPER_HSV_VALUES, "ANTS_LOWER_HSV_VALUES": self.ANTS_LOWER_HSV_VALUES, "ANTS_MIN_SIZE": self.ANTS_MIN_SIZE,
                          "ANTS_CLOSING_KERNEL": self.ANTS_CLOSING_KERNEL,
                          "ANTS_NEUTRALIZE_COLOUR_ALPHA": self.ANTS_NEUTRALIZE_COLOUR_ALPHA, "ANTS_NEUTRALIZE_COLOUR_BETA": self.ANTS_NEUTRALIZE_COLOUR_BETA,}
            if self.show_analysis_example:
                cap.set(cv2.CAP_PROP_POS_FRAMES, utils.insist_input_type("int", "second frame number?: "))
                _, second_image = cap.read()
                while True:
                    self.frame_center = np.array([3840 * utils.insist_input_type("float", "factor for x coordinates (default: 0.5): "), 2160 * utils.insist_input_type("float", "factor for y coordinates (default: 0.75): ")])
                    self.heights = [utils.insist_input_type("float", "height of fixed_end (default: 0.0025): "), 0.003]#, utils.insist_input_type("float", "height of needle part (default: 0.003): ")]
                    image_to_illustrate_first, needle_contour_first, image_cropped_first, object_center_coordinates_first, tip_point_first, crop_coordinates_first, image_processed_first = self.analysis_example(image.copy(), parameters)
                    image_to_illustrate_second, needle_contour_second, image_cropped_second, object_center_coordinates_second, tip_point_second, crop_coordinates_second, image_processed_second = self.analysis_example(second_image.copy(), parameters)

                # image_processed_first[needle_contour_first[:, 0].astype(int), needle_contour_first[:, 1].astype(int)] = [0, 0, 255]
                # image_processed_second[needle_contour_second[:, 0].astype(int), needle_contour_second[:, 1].astype(int)] = [0, 0, 255]
                # object_center_first = (object_center_coordinates_first - crop_coordinates_first[[2, 0]]).astype(int)
                # object_center_second = (object_center_coordinates_second - crop_coordinates_second[[2, 0]]).astype(int)
                # tip_point_first = (tip_point_first - crop_coordinates_first[[2, 0]]).astype(int)
                # tip_point_second = (tip_point_second - crop_coordinates_second[[2, 0]]).astype(int)
                # cv2.circle(image_processed_first, (object_center_first[0].astype(int), object_center_first[1].astype(int)), 3, (0, 255, 0), -1)
                # cv2.circle(image_processed_second, (object_center_second[0].astype(int), object_center_second[1].astype(int)), 3, (0, 255, 0), -1)
                # cv2.circle(image_processed_first, (tip_point_first[0].astype(int), tip_point_first[1].astype(int)), 3, (0, 255, 0), -1)
                # cv2.circle(image_processed_second, (tip_point_second[0].astype(int), tip_point_second[1].astype(int)), 3, (0, 255, 0), -1)
                # cv2.imshow("image_processed_first", image_to_illustrate_first)
                # cv2.waitKey(0)
                # cv2.imshow("image_processed_second", image_to_illustrate_second)
                # cv2.waitKey(0)
            if not utils.insist_input_type("yes_no", "Would you like to edit this video's parameters again? (This action won't delete what you've already collected)"):
                break
        return parameters

    def collect_image_preprocessing_parameters(self, parameters=None):
        self.starting_frame = utils.insist_input_type("int", "What frame to start with: ") if self.collect_starting_frame else 0
        if self.collect_image_processing_parameters:
            self.GRADIANT_THRESHOLD = utils.insist_input_type("int", f"What is the general gradiant threshold? (default is {GRADIANT_THRESHOLD}): ")
            self.NEUTRALIZE_COLOUR_ALPHA = utils.insist_input_type("float", f"What is the neutralize colour alpha? (default is {NEUTRALIZE_COLOUR_ALPHA}): ")
            blur_kernel_size = utils.insist_input_type("int", f"What is the blur kernel size? (default is {NEUTRALIZE_COLOUR_BETA[0]}):")
            self.NEUTRALIZE_COLOUR_BETA = (blur_kernel_size, blur_kernel_size)
            self.BUNDLES_CLOSING_SIZE = np.array([utils.insist_input_type("int", f"What is the bundles closing size? (default is {BUNDLES_CLOSING_SIZE}): ")] * 2)
            self.ANTS_GRADIANT_THRESHOLD = utils.insist_input_type("int", f"What is the ants gradiant threshold? (default is {ANTS_GRADIANT_THRESHOLD}): ")
            self.ANTS_SPRINGS_OVERLAP_SIZE = utils.insist_input_type("int", f"What is the ants springs overlap size? (default is {ANTS_SPRINGS_OVERLAP_SIZE}): ")
            self.ANTS_OBJECT_DILATION_SIZE = utils.insist_input_type("int", f"What is the ants object dilation size? (default is {ANTS_OBJECT_DILATION_SIZE}): ")
            self.ANTS_LOWER_HSV_VALUES = np.array([0]+[utils.insist_input_type("int", f"What is the ants lower {value_type} value for color space? (default is {value}): ")
                                                       for value_type, value in zip(["SATURATION", "VALUE"], ANTS_LOWER_HSV_VALUES[1:])])
            self.ANTS_UPPER_HSV_VALUES = np.array([179]+[utils.insist_input_type("int", f"What is the ants upper {value_type} value for color space? (default is {value}): ")
                                                         for value_type, value in zip(["SATURATION", "VALUE"], ANTS_UPPER_HSV_VALUES[1:])])
            self.ANTS_MIN_SIZE = utils.insist_input_type("int", f"What is the ants min size? (default is {ANTS_MIN_SIZE}): ")
            self.ANTS_CLOSING_KERNEL = np.ones([utils.insist_input_type("int", f"What is the ants closing kernel size? (default is {ANTS_CLOSING_KERNEL.shape[0]}): ")] * 2)
        else:
            self.GRADIANT_THRESHOLD = GRADIANT_THRESHOLD if parameters is None else parameters["GRADIANT_THRESHOLD"]
            self.NEUTRALIZE_COLOUR_ALPHA = NEUTRALIZE_COLOUR_ALPHA if parameters is None else parameters["NEUTRALIZE_COLOUR_ALPHA"]
            self.NEUTRALIZE_COLOUR_BETA = NEUTRALIZE_COLOUR_BETA if parameters is None else parameters["NEUTRALIZE_COLOUR_BETA"]
            self.BUNDLES_CLOSING_SIZE = BUNDLES_CLOSING_SIZE #if parameters is None else parameters["BUNDLES_CLOSING_KERNEL"]
            self.ANTS_GRADIANT_THRESHOLD = ANTS_GRADIANT_THRESHOLD #if parameters is None else parameters["ANTS_GRADIANT_THRESHOLD"]
            self.ANTS_SPRINGS_OVERLAP_SIZE = ANTS_SPRINGS_OVERLAP_SIZE #if parameters is None else parameters["ANTS_SPRINGS_OVERLAP_SIZE"]
            self.ANTS_OBJECT_DILATION_SIZE = ANTS_OBJECT_DILATION_SIZE #if parameters is None else parameters["ANTS_OBJECT_DILATION_SIZE"]
            self.ANTS_UPPER_HSV_VALUES = ANTS_UPPER_HSV_VALUES #if parameters is None else parameters["ANTS_UPPER_HSV_VALUES"]
            self.ANTS_LOWER_HSV_VALUES = ANTS_LOWER_HSV_VALUES #if parameters is None else parameters["ANTS_LOWER_HSV_VALUES"]
            self.ANTS_MIN_SIZE = ANTS_MIN_SIZE #if parameters is None else parameters["ANTS_MIN_SIZE"]
            self.ANTS_CLOSING_KERNEL = ANTS_CLOSING_KERNEL #if parameters is None else parameters["ANTS_CLOSING_KERNEL"]
        self.ocm = OBJECT_COORDINATES_MARGIN if parameters is None else parameters["ocm"]
        self.pcm = PERSPECTIVE_SQUARES_MARGIN if parameters is None else parameters["pcm"]
        self.image_resize_factor = IMAGE_RESIZE_FACTOR if parameters is None else parameters["image_resize_factor"]
        self.resolution = RESOLUTION if parameters is None else parameters["resolution"]
        self.color_space_margin = COLOR_SPACE_MARGIN if parameters is None else parameters["color_space_margin"]
        self.hsv_space_boundary = HSV_SPACE_BOUNDARY if parameters is None else parameters["hsv_space_boundary"]
        self.max_ants_number = MAX_ANTS_NUMBER if parameters is None else parameters["max_ants_number"]
        self.ANTS_NEUTRALIZE_COLOUR_ALPHA = ANTS_NEUTRALIZE_COLOUR_ALPHA if parameters is None else parameters["ANTS_NEUTRALIZE_COLOUR_ALPHA"]
        self.ANTS_NEUTRALIZE_COLOUR_BETA = ANTS_NEUTRALIZE_COLOUR_BETA if parameters is None else parameters["ANTS_NEUTRALIZE_COLOUR_BETA"]

    def collect_crop_coordinates(self, image, parameters=None):
        reduced_resolution_image = cv2.resize(image, (0, 0), fx=self.image_resize_factor, fy=self.image_resize_factor)
        # if False:
        if (parameters is None) or utils.insist_input_type("yes_no", "Would you to re-collect crop coordinates?"):
            print("Please point on the center of the object, to get the initial crop coordinates.")
            self.object_center_coordinates = utils.collect_points(reduced_resolution_image, n_points=1).reshape(1, 2) * (1 / self.image_resize_factor)
            print("Please point on ALL four perspective squares, to get the initial crop coordinates.")
            self.perspective_squares_coordinates = utils.collect_points(reduced_resolution_image, n_points=4) * (1 / self.image_resize_factor)
        else:
            self.object_center_coordinates = parameters["object_center_coordinates"]
            self.perspective_squares_coordinates = parameters["perspective_squares_coordinates"]
        try:
            collage_image = self.create_collage_image(image, self.ocm, self.pcm, self.object_center_coordinates, self.perspective_squares_coordinates)
        except:
            print("Couldn't create collage image, please re-collect crop coordinates far from the image's edges.")
            collage_image = self.collect_crop_coordinates(image, parameters=parameters)
        return collage_image

    def collect_or_edit_colors_spaces(self, collage_image, parameters=None):
        self.colors_spaces = {"b": [], "r": [], "g": [], "p": []} if parameters is None else parameters["colors_spaces"]
        if (parameters is None) or utils.insist_input_type("yes_no", "Would you like to edit the color spaces?"):
        # if False:
            for color_name, color_short in zip(["blue", "red", "green", "purple"], self.colors_spaces.keys()):
                overlays_image = np.copy(collage_image)
                while True:
                    if len(self.colors_spaces[color_short]) != 0:
                        overlays_image[utils.create_color_mask(collage_image, self.colors_spaces[color_short]).astype(bool)] = [0, 255, 0]
                        cv2.imshow(color_name, overlays_image)
                        cv2.waitKey(0)
                        ask_sentence = "Want to add more colors? (y/n or 'r' to remove the last color)"
                        add_colors = utils.insist_input_type("str_list", ask_sentence, str_list=["y", "n", "r"])
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

    def rotate(self, coordinates, angle_radians):
        new_x = coordinates[:, 0] * np.cos(angle_radians) - coordinates[:, 1] * np.sin(angle_radians)
        new_y = coordinates[:, 0] * np.sin(angle_radians) + coordinates[:, 1] * np.cos(angle_radians)
        return np.stack([new_x, new_y], axis=1)

    def plane_perpective_projection(self, coordinates, height, frame_center=np.array([3840 / 2, 2160 / 2])):
        # frame_center = np.array([3840 / 2, 2160 / 2])
        plane_to_center_distance = coordinates - frame_center
        plane_displacement_distance = plane_to_center_distance * (height / (1 - height))
        plane_correct_coordinates = coordinates - plane_displacement_distance
        return plane_correct_coordinates

    def analysis_example(self, image, parameters):
        parameters["continue_from_last_snapshot"] = False
        snapshot_data = utils.create_snapshot_data(parameters)
        snapshot_data["frame_count"] = 1
        squares = PerspectiveSquares(parameters, image, snapshot_data)
        springs = Springs(parameters, image, snapshot_data)
        ants = Ants(image, springs, squares)
        calculations = Calculation(parameters, snapshot_data, springs, ants)

        SQUARENESS_THRESHOLD = 0.0005
        RECTANGLE_SIMILARITY_THRESHOLD = 0.01
        QUALITY_THRESHOLD = SQUARENESS_THRESHOLD * RECTANGLE_SIMILARITY_THRESHOLD
        perspective_squares_quality = np.expand_dims(squares.perspective_squares_properties[:, 2] * squares.perspective_squares_properties[:, 3], axis=0)
        perspective_squares_coordinates = np.expand_dims(squares.perspective_squares_properties[:, 0:2], axis=0)[:, [0, 3, 1, 2], :]
        PTMs = utils.create_projective_transform_matrix(perspective_squares_coordinates, perspective_squares_quality, QUALITY_THRESHOLD, src_dimensions=(3840, 2160))

        fixed_ends_coordinates = utils.apply_projective_transform(springs.fixed_ends_edges_centers, PTMs)

        fixed_ends_coordinates = self.plane_perpective_projection(fixed_ends_coordinates, self.heights[0], self.frame_center)
        # fixed_ends_coordinates = springs.fixed_ends_edges_centers
        free_ends_coordinates = utils.apply_projective_transform(springs.free_ends_edges_centers, PTMs)
        # free_ends_coordinates = self.plane_perpective_projection(free_ends_coordinates, 0.025, frame_center)
        # free_ends_coordinates = springs.free_ends_edges_centers
        tip_point = utils.apply_projective_transform(springs.tip_point.reshape(1, 2), PTMs)
        tip_point = self.plane_perpective_projection(tip_point, self.heights[1], self.frame_center)
        # tip_point = springs.tip_point.reshape(1, 2)
        perspective_squares_coordinates = utils.apply_projective_transform(perspective_squares_coordinates, PTMs)

        image_to_illustrate = utils.present_analysis_result(image, calculations, springs, ants, waitKey=-1)
        labeled_ants_cropped = utils.crop_frame(ants.labeled_ants, springs.object_crop_coordinates)
        num_features = np.max(labeled_ants_cropped)+1
        cmap = plt.get_cmap('jet', num_features)
        mapped = cmap(labeled_ants_cropped)[:, :, :3]
        overlay_image = image_to_illustrate.copy()
        boolean = labeled_ants_cropped != 0
        boolean = np.repeat(boolean[:, :, np.newaxis], 3, axis=2)
        overlay_image[boolean] = (mapped[boolean]*255).astype(np.uint8)
        springs_length = np.linalg.norm(fixed_ends_coordinates - free_ends_coordinates, axis=1)
        print("springs_length: ", springs_length)
        # from data_analysis.utils import calc_angle_matrix
        nest_direction = fixed_ends_coordinates + np.array([0, 500])
        fixed_end_angle_to_nest = utils.calc_angle_matrix(np.expand_dims(tip_point.reshape(1, 2), axis=1), np.expand_dims(fixed_ends_coordinates, axis=1), np.expand_dims(nest_direction, axis=1)) + np.pi
        # print("fixed_end_angle_to_nest: ", fixed_end_angle_to_nest)
        # plt.imshow(overlay_image.astype(np.uint8))
        # plt.show()
        return image_to_illustrate, springs.needle_contour, springs.image_cropped, springs.object_center_coordinates, springs.tip_point, springs.object_crop_coordinates, springs.image_processed

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
        # print('most frequent color (background) is {})'.format(peak))
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
        # print("background_color", background_color)
        # print("background_color2", background_color2)
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

