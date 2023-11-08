import cv2
import numpy as np
from scipy.ndimage import label, center_of_mass, distance_transform_edt
from skimage.morphology import binary_closing, binary_opening
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
# local imports:
import utils


class Ants:
    def __init__(self, image, springs, perspective_squares):
        self.parameters = springs.parameters
        self.previous_detections = springs.previous_detections
        circle_mask = self.create_circle_maks(image, springs.object_center_coordinates, springs.object_needle_radius)
        self.label_ants(image, springs.whole_object_mask, perspective_squares.all_perspective_squares_mask, circle_mask)

    def create_circle_maks(self, image, center, radius):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        center = tuple(center.astype(np.uint16))
        mask = cv2.circle(mask, center, int(radius*1.3), 255, -1)
        return mask > 0

    def sobel_masking(self, image, threshold, object_mask=None, perspective_squares_mask=None):
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=self.parameters["ANTS_SOBEL_KERNEL_SIZE"])
        sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=self.parameters["ANTS_SOBEL_KERNEL_SIZE"])
        gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        if object_mask is not None and perspective_squares_mask is not None:
            gradient_magnitude[object_mask > 0] = 0
            gradient_magnitude[perspective_squares_mask > 0] = 0
        sobel_mask = gradient_magnitude > threshold
        return sobel_mask

    def label_ants(self, image, object_mask, perspective_squares_mask, circle_maks):
        image = cv2.convertScaleAbs(image, alpha=self.parameters["ANTS_NEUTRALIZE_COLOUR_ALPHA"], beta=self.parameters["ANTS_NEUTRALIZE_COLOUR_BETA"])
        sobel_mask = self.sobel_masking(image, self.parameters["ANTS_GRADIANT_THRESHOLD"], object_mask, perspective_squares_mask)
        for i in range(1, 5):
            sobel_mask = binary_opening(sobel_mask, self.parameters["FIRST_OPENING_STRUCTURE"])
            sobel_mask[object_mask > 0] = False
            sobel_mask[perspective_squares_mask > 0] = False
            sobel_mask = binary_closing(sobel_mask, np.ones((i+1, i+1)))
        sobel_mask = binary_opening(sobel_mask, self.parameters["SECOND_OPENING_STRUCTURE"])
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, self.parameters["ANTS_LOWER_HSV_VALUES"], self.parameters["ANTS_UPPER_HSV_VALUES"]) > 0
        # color_mask = cv2.inRange(hsv, np.array([0, 3, 0]), self.parameters["ANTS_UPPER_HSV_VALUES"]) > 0
        combined_mask = color_mask * sobel_mask
        combined_mask = binary_closing(combined_mask, self.parameters["ANTS_CLOSING_KERNEL"])
        labeled_image, _ = label(combined_mask)

        labels_sizes = np.bincount(labeled_image.ravel())
        extreme_outliers_labels = np.where((labels_sizes > self.parameters["ANTS_MAX_SIZE"]) + (labels_sizes < self.parameters["ANTS_MIN_SIZE"]))[0]
        labeled_image[np.isin(labeled_image, extreme_outliers_labels)] = 0
        labeled_image, _ = label(labeled_image.astype(bool))
        labels_sizes = np.bincount(labeled_image.ravel())
        # sum_ant_size = int(np.sum(labels_sizes[1:])) + self.previous_detections["sum_ant_size"]
        # sum_num_ants = int(np.sum(labels_sizes[1:] > 0)) + self.previous_detections["sum_num_ants"]
        # mean_ant_size = sum_ant_size / sum_num_ants
        mean_ant_size = 110
        outlier_labels = np.where((labels_sizes > mean_ant_size * 1.6)+(labels_sizes < mean_ant_size * 0.7))[0]
        for outlier_label in outlier_labels[1:]:
            label_mask = labeled_image == outlier_label
            label_indexes = np.where(label_mask)
            upper_left = np.min(label_indexes[0]), np.min(label_indexes[1])
            lower_right = np.max(label_indexes[0]), np.max(label_indexes[1])
            label_image = image[upper_left[0]:lower_right[0], upper_left[1]:lower_right[1]]
            label_mask = label_mask[upper_left[0]:lower_right[0], upper_left[1]:lower_right[1]]
            label_image[~label_mask] = 255
            if not label_image.size == 0:
                # cv2.imshow("label_image", label_image)
                # cv2.waitKey(0)
                hsv = cv2.cvtColor(label_image, cv2.COLOR_BGR2HSV)
                color_mask = cv2.inRange(hsv,  np.array([0, 10, 0]), np.array([179, 255, 180])) > 0
                if labels_sizes[outlier_label] < mean_ant_size:
                    labeled_image[labeled_image == outlier_label] = 0
                    sub_labels_labeled, _ = label(color_mask)
                elif labels_sizes[outlier_label] > mean_ant_size:
                    label_image[~color_mask] = 255
                    # cv2.imshow("label_image", label_image)
                    # cv2.waitKey(0)
                    distance = distance_transform_edt(color_mask)
                    coordinates = peak_local_max(distance, footprint=np.ones((30, 30)), labels=color_mask)
                    mask = np.zeros(distance.shape, dtype=bool)
                    mask[coordinates.T[0], coordinates.T[1]] = True
                    markers, _ = label(mask)
                    sub_labels_labeled = watershed(-distance, markers, mask=color_mask)
                sub_labels_sizes = np.bincount(sub_labels_labeled.ravel())
                watershed_strange_sizes_labels = np.where((sub_labels_sizes > mean_ant_size * 1.5)+(sub_labels_sizes < mean_ant_size * 0.3))[0]
                sub_labels_labeled[np.isin(sub_labels_labeled, watershed_strange_sizes_labels)] = 0
                sub_labels_labeled[sub_labels_labeled > 0] += np.max(labeled_image)
                labeled_image[labeled_image == outlier_label] = 0
                labeled_image[upper_left[0]:lower_right[0], upper_left[1]:lower_right[1]][sub_labels_labeled > 0] = sub_labels_labeled[sub_labels_labeled > 0]
        unique_elements, indices = np.unique(labeled_image, return_inverse=True)
        labeled_image = indices.reshape(labeled_image.shape)
        num_labels = np.max(labeled_image)
        labels_sizes = np.bincount(labeled_image.ravel())
        small_labels_mask = np.isin(labeled_image, np.where(labels_sizes < mean_ant_size * 0.5)[0])
        labeled_image[small_labels_mask * circle_maks] = 0
        self.sum_ant_size = int(np.sum(labels_sizes[1:])) + self.previous_detections["sum_ant_size"]
        self.sum_num_ants = int(np.sum(labels_sizes[1:] > 0)) + self.previous_detections["sum_num_ants"]
        self.labeled_ants, line_lengths = utils.extend_lines(labeled_image, extend_by=self.parameters["ANTS_EXTENSION_LENGTH"])
        for ant_label in line_lengths.keys():
            if line_lengths[ant_label] > self.parameters["ANTS_MAX_LINE_LENGTH"]:
                self.labeled_ants[self.labeled_ants == ant_label] = 0
        ants_centers = center_of_mass(labeled_image, labeled_image, range(1, num_labels+1))
        self.ants_centers = np.array(ants_centers)
        if self.ants_centers.shape[0] == 0:
            self.ants_centers = np.full((0, 2), np.nan)

