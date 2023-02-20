import cv2
import numpy as np
from scipy.ndimage import label, generate_binary_structure, center_of_mass, maximum_filter, binary_fill_holes
from skimage.filters import threshold_otsu,threshold_local# threshold_li, threshold_triangle,threshold_yen
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects, binary_closing, binary_opening
#local imports:
from utils import close_element, convert_bool_to_binary, extend_lines, connect_blobs

OBJECT_DILATION_SIZE = 3
ANTS_OPENING_CLOSING_STRUCTURE = np.ones((4, 4))
MIN_ANTS_SIZE = 60
ANTS_SPRINGS_OVERLAP_SIZE = 10

class Ants:
    def __init__(self, image, springs):
        self.object_mask = self.create_object_mask(image.shape,springs)
        self.labeled_ants = self.label_ants(image,self.object_mask)

    def create_object_mask(self,image_dim,springs):
        # circle_mask = self.create_circular_mask(image_dim, center=springs.object_center, radius=springs.blue_radius*1.5)
        mask_object = springs.whole_object_mask != 0
        return mask_object

    def create_circular_mask(self, image_dim, center=None, radius=None):
        h,w = image_dim[0],image_dim[1]
        if center is None:  # use the middle of the image
            center = (int(w / 2), int(h / 2))
        if radius is None:  # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w - center[0], h - center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

        mask = dist_from_center <= radius
        return mask

    def label_ants(self,image,mask_object):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_val = np.array([0, 0, 0])
        upper_val = np.array([179, 100, 100])
        mask = cv2.inRange(hsv, lower_val, upper_val)
        mask = mask>0
        mask[mask_object] = False
        mask = binary_opening(mask, ANTS_OPENING_CLOSING_STRUCTURE)
        mask = binary_closing(mask, ANTS_OPENING_CLOSING_STRUCTURE )
        mask = remove_small_objects(mask, MIN_ANTS_SIZE)
        labeled_image, num_labels = label(connect_blobs(mask, overlap_size=4))
        labeled_image = clear_border(labeled_image)
        labeled_image = extend_lines(labeled_image, extend_by=5)
        return labeled_image

    # def extend_ants(self, labeled_image):
    #     lines = fit_line(labeled_image)
    #     print(lines)

    # def ants_coordinates(self, labeled_image):
    #     labels = list(np.unique(labeled_image))[1:]
    #     centers = np.array(center_of_mass(labeled_image, labels=labeled_image, index=labels)).astype("int")
    #     return centers, labels
    #
    # def track_ants(self, image_labeled, previous_labeled):
    #     """
    #     Takes the labeled image and matches the labels to the previous image labels,
    #     based on the proximity of the centers of mass. Then it corrects the labels,
    #     provides new labels for the new ants, and deletes the labels of the ants that disappeared.
    #     param image_labeled:
    #     :param previous_image_labeled:
    #     :return: The corrected labeled image.
    #     """
    #     labeled_image_center_of_mass = center_of_mass(image_labeled, labels=image_labeled,
    #                                                   index=np.unique(image_labeled)[1:])
    #     previous_labeled_center_of_mass = center_of_mass(previous_labeled, labels=previous_labeled,
    #                                                      index=np.unique(previous_labeled)[1:])
    #     labeled_image_center_of_mass = np.array(labeled_image_center_of_mass).astype("int")
    #     previous_image_labeled_center_of_mass = np.array(previous_labeled_center_of_mass).astype("int")
    #     # print("labeled_image_center_of_mass", labeled_image_center_of_mass)
    #     labeled_image_center_of_mass = labeled_image_center_of_mass
    #     previous_image_labeled_center_of_mass = previous_image_labeled_center_of_mass
    #     closest_pair = self.closetest_pair_of_points(labeled_image_center_of_mass,
    #                                                  previous_image_labeled_center_of_mass,previous_labeled)
    #     self.corrected_labeled_image = self.replace_labels(image_labeled, closest_pair)
    #     self.corrected_labels_center_of_mass = center_of_mass(self.corrected_labeled_image,
    #                     labels=self.corrected_labeled_image, index=np.unique(self.corrected_labeled_image)[1:])
    #
    # def replace_labels(self, image_labeled,closest_pair):
    #     # print("closest_pair", closest_pair)
    #     for pair in closest_pair:
    #         image_labeled[image_labeled == pair[0]] = pair[1]
    #     return image_labeled
    #
    # def closetest_pair_of_points(self, centers_current,centers_previous,previous_labeled):
    #     """
    #     Finds the closest pair of points between two sets of points.
    #     :param points1: The first set of points.
    #     :param points2: The second set of points.
    #     :return: The closest pair of points.
    #     """
    #     from scipy.spatial.distance import cdist
    #     distances = cdist(centers_current, centers_previous)
    #     # print("distances", distances)
    #     #get the column of each minimum value in each row:
    #     min_col = np.argmin(distances, axis=1)
    #     previous_labels = np.take(np.unique(previous_labeled)[1:], min_col)
    #     # print("len(min_col)", len(min_col))
    #     # print(list(zip(np.arange(len(min_col))+1, min_col+1)))
    #     return list(zip(np.arange(len(min_col))+1, previous_labels))