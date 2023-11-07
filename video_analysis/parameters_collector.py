import os
import pickle
import cv2
import numpy as np
from scipy.ndimage import generate_binary_structure
# local imports:
import utils
from calculator import Calculation
from ants_detector import Ants
from springs_detector import Springs
from perspective_squares import PerspectiveSquares

NEUTRALIZE_COLOUR_ALPHA = 2.5
MAX_ANTS_NUMBER = 200
ANTS_SOBEL_KERNEL_SIZE = 3
ANTS_GRADIANT_THRESHOLD = 240
ANTS_CLOSING_KERNEL = np.ones((2, 2))
ANTS_MIN_SIZE = 40
ANTS_MAX_SIZE = 400
ANTS_MAX_LINE_LENGTH = 40
ANTS_OBJECT_DILATION_SIZE = 3
ANTS_SPRINGS_OVERLAP_SIZE = 3


class CollectParameters:
    def __init__(self, video_paths, output_path):
        self.video_paths = video_paths
        self.output_path = output_path
        self.n_springs = utils.insist_input_type("int", "How many springs does the object has? (default is 20): ")
        self.show_analysis_example = utils.insist_input_type("yes_no", "Would you like to show analysis example for each video?")
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
            parameters = self.collect_image_preprocessing_parameters(parameters=parameters)
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, utils.insist_input_type("int", "From which frame would you like to collect the parameters?: "))
            _, image = cap.read()
            processed_image = utils.process_image(image, alpha=parameters["NEUTRALIZE_COLOUR_ALPHA"],
                                                  blur_kernel=parameters["NEUTRALIZE_COLOUR_BETA"], gradiant_threshold=parameters["GRADIANT_THRESHOLD"])
            collage_image, parameters = self.collect_crop_coordinates(processed_image, parameters=parameters)
            parameters = self.collect_or_edit_color_spaces(collage_image, parameters=parameters)
            self.analysis_example(image, parameters)
            if not utils.insist_input_type("yes_no", "Would you like to edit this video's parameters again? (This action won't delete what you've already collected)"):
                break
        return parameters

    def collect_image_preprocessing_parameters(self, parameters=None):
        new_parameters = {} if parameters is None else parameters
        new_parameters["OUTPUT_PATH"] = self.output_path
        new_parameters["STARTING_FRAME"] = 0  # utils.insist_input_type("int", "What frame to start with: ") if self.collect_starting_frame else 0
        new_parameters["N_SPRINGS"] = self.n_springs
        new_parameters["PERSPECTIVE_SQUARES_COLOR_CLOSING"] = 5
        new_parameters["SQUARE_ON_BORDER_RATIO_THRESHOLD"] = 0.4
        new_parameters["SOBEL_KERNEL_SIZE"] = 3
        new_parameters["GRADIANT_THRESHOLD"] = 5
        new_parameters["SPRINGS_ENDS_OPENING"] = 2
        new_parameters["SPRINGS_ENDS_CLOSING"] = 4
        new_parameters["SPRINGS_MIDDLE_PART_OPENING"] = 4
        new_parameters["NEEDLE_RADIUS_TOLERANCE"] = 0.05
        new_parameters["LABELING_BINARY_STRUCTURE"] = generate_binary_structure(2, 2)
        new_parameters["MIN_SPRING_ENDS_SIZE"] = 50
        new_parameters["MIN_SPRING_MIDDLE_PART_SIZE"] = 150
        new_parameters["MIN_SIZE_FOR_WHOLE_OBJECT"] = 30
        new_parameters["SPRINGS_PARTS_OVERLAP_SIZE"] = 8
        new_parameters["NEUTRALIZE_COLOUR_BETA"] = (7, 7)
        new_parameters["BUNDLES_CLOSING_SIZE"] = 3
        new_parameters["FIRST_OPENING_STRUCTURE"] = np.ones((1, 1))
        new_parameters["SECOND_OPENING_STRUCTURE"] = np.ones((2, 2))
        new_parameters["ANTS_EXTENSION_LENGTH"] = 3
        new_parameters["ANTS_UPPER_HSV_VALUES"] = np.array([179, 255, 200])
        new_parameters["ANTS_LOWER_HSV_VALUES"] = np.array([0, 0, 0])
        new_parameters["OCM"] = 200
        new_parameters["PCM"] = 100
        new_parameters["IMAGE_RESIZE_FACTOR"] = 0.25
        new_parameters["RESOLUTION"] = np.array([2160, 3840])
        new_parameters["COLOR_SPACE_MARGIN"] = np.array([12, 30, 30])
        new_parameters["HSV_SPACE_BOUNDARY"] = np.array([179, 255, 255])
        new_parameters["ANTS_NEUTRALIZE_COLOUR_ALPHA"] = 2
        new_parameters["ANTS_NEUTRALIZE_COLOUR_BETA"] = 10
        new_parameters["NEUTRALIZE_COLOUR_ALPHA"] = NEUTRALIZE_COLOUR_ALPHA if parameters is None else parameters["NEUTRALIZE_COLOUR_ALPHA"]
        new_parameters["MAX_ANTS_NUMBER"] = MAX_ANTS_NUMBER if parameters is None else parameters["MAX_ANTS_NUMBER"]
        new_parameters["ANTS_SOBEL_KERNEL_SIZE"] = ANTS_SOBEL_KERNEL_SIZE if parameters is None else parameters["ANTS_SOBEL_KERNEL_SIZE"]
        new_parameters["ANTS_GRADIANT_THRESHOLD"] = ANTS_GRADIANT_THRESHOLD# if parameters is None else parameters["ANTS_GRADIANT_THRESHOLD"]
        new_parameters["ANTS_CLOSING_KERNEL"] = ANTS_CLOSING_KERNEL if parameters is None else parameters["ANTS_CLOSING_KERNEL"]
        new_parameters["ANTS_MIN_SIZE"] = ANTS_MIN_SIZE if parameters is None else parameters["ANTS_MIN_SIZE"]
        new_parameters["ANTS_MAX_SIZE"] = ANTS_MAX_SIZE if parameters is None else parameters["ANTS_MAX_SIZE"]
        new_parameters["ANTS_MAX_LINE_LENGTH"] = ANTS_MAX_LINE_LENGTH if parameters is None else parameters["ANTS_MAX_LINE_LENGTH"]
        new_parameters["ANTS_OBJECT_DILATION_SIZE"] = ANTS_OBJECT_DILATION_SIZE if parameters is None else parameters["ANTS_OBJECT_DILATION_SIZE"]
        new_parameters["ANTS_SPRINGS_OVERLAP_SIZE"] = ANTS_SPRINGS_OVERLAP_SIZE if parameters is None else parameters["ANTS_SPRINGS_OVERLAP_SIZE"]
        # if False:
        if utils.insist_input_type("yes_no", "Would you like to re-collect image processing parameters?"):
            # new_parameters["NEUTRALIZE_COLOUR_ALPHA"] = utils.insist_input_type("float", f"What is the neutralize colour alpha? (default is {NEUTRALIZE_COLOUR_ALPHA}): ")
            # new_parameters["MAX_ANTS_NUMBER"] = utils.insist_input_type("int", f"What is the max ants number expected to be tracked? (default is {MAX_ANTS_NUMBER}): ")
            # new_parameters["ANTS_SOBEL_KERNEL_SIZE"] = utils.insist_input_type("int", f"What is the ants sobel kernel size? (default is {ANTS_SOBEL_KERNEL_SIZE}): ")
            new_parameters["ANTS_GRADIANT_THRESHOLD"] = utils.insist_input_type("int", f"What is the ants gradiant threshold? (default is {ANTS_GRADIANT_THRESHOLD}): ")
            # new_parameters["ANTS_CLOSING_KERNEL"] = np.ones([utils.insist_input_type("int", f"What is the ants closing kernel size? (default is {ANTS_CLOSING_KERNEL.shape[0]}): ")] * 2)
            # new_parameters["ANTS_MIN_SIZE"] = utils.insist_input_type("int", f"What is the ants min size? (default is {ANTS_MIN_SIZE}): ")
            # new_parameters["ANTS_MAX_SIZE"] = utils.insist_input_type("int", f"What is the ants max size? (default is {ANTS_MAX_SIZE}): ")
            # new_parameters["ANTS_MAX_LINE_LENGTH"] = utils.insist_input_type("int", f"What is the ants max line length? (default is {ANTS_MAX_LINE_LENGTH}): ")
            # new_parameters["ANTS_OBJECT_DILATION_SIZE"] = utils.insist_input_type("int", f"What is the ants object dilation size? (default is {ANTS_OBJECT_DILATION_SIZE}): ")
            # new_parameters["ANTS_SPRINGS_OVERLAP_SIZE"] = utils.insist_input_type("int", f"What is the ants springs overlap size? (default is {ANTS_SPRINGS_OVERLAP_SIZE}): ")
        return new_parameters

    def collect_crop_coordinates(self, image, parameters):
        # if False:
        if ("OBJECT_CENTER_COORDINATES" not in parameters) or utils.insist_input_type("yes_no", "Would you to re-collect crop coordinates?"):
            reduced_resolution_image = cv2.resize(image, (0, 0), fx=parameters["IMAGE_RESIZE_FACTOR"], fy=parameters["IMAGE_RESIZE_FACTOR"])
            print("Please point on the center of the object, to get the initial crop coordinates.")
            parameters["OBJECT_CENTER_COORDINATES"] = utils.collect_points(reduced_resolution_image, n_points=1).reshape(1, 2) * (1 / parameters["IMAGE_RESIZE_FACTOR"])
            print("Please point on ALL four perspective squares, to get the initial crop coordinates.")
            parameters["PERSPECTIVE_SQUARES_COORDINATES"] = utils.collect_points(reduced_resolution_image, n_points=4) * (1 / parameters["IMAGE_RESIZE_FACTOR"])
        try:
            collage_image = self.create_collage_image(image, parameters["OCM"], parameters["PCM"],
                                                      parameters["OBJECT_CENTER_COORDINATES"], parameters["PERSPECTIVE_SQUARES_COORDINATES"])
        except:
            print("Couldn't create collage image, please re-collect crop coordinates far from the image's edges.")
            collage_image = self.collect_crop_coordinates(image, parameters=parameters)
        return collage_image, parameters

    def collect_or_edit_color_spaces(self, collage_image, parameters):
        # if False:
        if ("COLOR_SPACES" not in parameters) or utils.insist_input_type("yes_no", "Would you like to edit the color spaces?"):
            parameters["COLOR_SPACES"] = {"b": [], "r": [], "g": [], "p": []} if "COLOR_SPACES" not in parameters else parameters["COLOR_SPACES"]
            for color_name, color_short in zip(["blue", "red", "green", "purple"], parameters["COLOR_SPACES"].keys()):
                overlays_image = np.copy(collage_image)
                while True:
                    if len(parameters["COLOR_SPACES"][color_short]) != 0:
                        overlays_image[utils.create_color_mask(collage_image, parameters["COLOR_SPACES"][color_short]).astype(bool)] = [0, 255, 0]
                        cv2.imshow(color_name, overlays_image)
                        cv2.waitKey(0)
                        ask_sentence = "Want to add more colors? (y/n or 'r' to remove the last color)"
                        add_colors = utils.insist_input_type("str_list", ask_sentence, str_list=["y", "n", "r"])
                    else:
                        add_colors = "y"
                    if add_colors == "n":
                        break
                    elif add_colors == "r":
                        parameters["COLOR_SPACES"][color_short].pop()
                        overlays_image = np.copy(collage_image)
                    else:
                        print("Please pick the " + color_name + " element pixels:")
                        points = utils.collect_points(overlays_image, n_points=1, show_color=True)
                        parameters["COLOR_SPACES"][color_short] += utils.create_color_space_from_points(overlays_image, points,
                                                                                          parameters["COLOR_SPACE_MARGIN"], parameters["HSV_SPACE_BOUNDARY"])
        return parameters

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

    def analysis_example(self, image, parameters):
        if self.show_analysis_example:
            try:
                image = image.copy()
                parameters["CONTINUE_FROM_LAST_SNAPSHOT"] = False
                snapshot_data = utils.create_snapshot_data(parameters)
                snapshot_data["frame_count"] = 1
                squares = PerspectiveSquares(parameters, image, snapshot_data)
                springs = Springs(parameters, image, snapshot_data)
                ants = Ants(image, springs, squares)
                calculations = Calculation(parameters, snapshot_data, springs, ants)
                utils.present_analysis_result(image, calculations, springs, ants, waitKey=0)
            except:
                print("Error in analysis_example. Could not show example")


