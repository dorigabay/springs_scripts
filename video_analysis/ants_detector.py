import cv2
import numpy as np
from scipy.ndimage import label, center_of_mass
from skimage.morphology import remove_small_objects, binary_closing, binary_opening
from scipy.ndimage import maximum_filter, minimum_filter
# local imports:
import utils


class Ants:
    def __init__(self, image, springs, perspective_squares):
        self.parameters = springs.parameters
        self.previous_detections = springs.previous_detections
        self.label_ants(image, springs.whole_object_mask, perspective_squares.all_perspective_squares_mask)

    def label_ants(self, image, object_mask, perspective_squares_mask):
        image = cv2.convertScaleAbs(image, alpha=self.parameters["ANTS_NEUTRALIZE_COLOUR_ALPHA"], beta=self.parameters["ANTS_NEUTRALIZE_COLOUR_BETA"])
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=self.parameters["ANTS_SOBEL_KERNEL_SIZE"])
        sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=self.parameters["ANTS_SOBEL_KERNEL_SIZE"])
        gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        gradient_magnitude[object_mask > 0] = 0
        gradient_magnitude[perspective_squares_mask > 0] = 0
        sobel_mask = gradient_magnitude > self.parameters["ANTS_GRADIANT_THRESHOLD"]
        for i in range(1, 5):
            sobel_mask = binary_opening(sobel_mask, self.parameters["FIRST_OPENING_STRUCTURE"])
            sobel_mask[object_mask > 0] = False
            sobel_mask[perspective_squares_mask > 0] = False
            sobel_mask = binary_closing(sobel_mask, np.ones((i+1, i+1)))
        sobel_mask = binary_opening(sobel_mask, self.parameters["SECOND_OPENING_STRUCTURE"])
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, self.parameters["ANTS_LOWER_HSV_VALUES"], self.parameters["ANTS_UPPER_HSV_VALUES"]) > 0
        combined_mask = color_mask * sobel_mask
        combined_mask = binary_closing(combined_mask, self.parameters["ANTS_CLOSING_KERNEL"])
        combined_mask = remove_small_objects(combined_mask, self.parameters["ANTS_MIN_SIZE"])
        labeled_image, num_labels = label(combined_mask)
        labels_sizes = np.bincount(labeled_image.ravel())
        self.sum_ant_size = int(np.sum(labels_sizes[1:])) + self.previous_detections["sum_ant_size"]
        self.sum_num_ants = int(np.sum(labels_sizes[1:] > 0)) + self.previous_detections["sum_num_ants"]
        mean_ant_size = self.sum_ant_size / self.sum_num_ants
        large_labels = np.where(labels_sizes > mean_ant_size * 1.4)[0]
        for large_label in large_labels[1:]:
            if labels_sizes[large_label] > self.parameters["ANTS_MAX_SIZE"]:
                labeled_image[labeled_image == large_label] = 0
            else:
                label_mask = np.full(labeled_image.shape, False)
                label_mask[labeled_image == large_label] = True
                label_mask = binary_opening(label_mask.astype(bool), np.ones((3, 3)))
                large_label_labeled, large_labeled_num_labels = label(label_mask)
                if large_labeled_num_labels > 2:
                    large_label_labeled[large_label_labeled > 0] += np.max(labeled_image)
                    labeled_image[labeled_image == large_label] = large_label_labeled[labeled_image == large_label]
        maximum_filtered = maximum_filter(labeled_image, size=self.parameters["ANTS_OBJECT_DILATION_SIZE"])
        minimum_filtered = minimum_filter(maximum_filtered, size=self.parameters["ANTS_OBJECT_DILATION_SIZE"]) - labeled_image
        labeled_image, num_labels = label((labeled_image + minimum_filtered).astype(bool))
        self.labeled_ants, line_lengths = utils.extend_lines(labeled_image, extend_by=self.parameters["ANTS_EXTENSION_LENGTH"])
        for ant_label in line_lengths.keys():
            if line_lengths[ant_label] > self.parameters["ANTS_MAX_LINE_LENGTH"]:
                self.labeled_ants[self.labeled_ants == ant_label] = 0
        ants_centers = center_of_mass(labeled_image, labeled_image, range(1, num_labels+1))
        self.ants_centers = np.array(ants_centers)
        if self.ants_centers.shape[0] == 0:
            self.ants_centers = np.full((0, 2), np.nan)

