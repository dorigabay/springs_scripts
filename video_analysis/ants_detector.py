import cv2
import numpy as np
from scipy.ndimage import label, center_of_mass
from skimage.morphology import remove_small_objects, binary_closing, binary_opening
#local imports:
from video_analysis.utils import extend_lines, connect_blobs
from video_analysis.springs_detector import Springs

OBJECT_DILATION_SIZE = 5
ANTS_OPENING_CLOSING_STRUCTURE = np.ones((4, 4))
MIN_ANTS_SIZE = 50


class Ants(Springs):
    def __init__(self,parameters, image, previous_detections):
        super().__init__(parameters, image, previous_detections)
        self.label_ants(image, self.whole_object_mask_unconnected)

    def label_ants(self, image, object_mask):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_val = np.array([0, 0, 0])
        upper_val = np.array([179, 255, 200])
        mask = cv2.inRange(hsv, lower_val, upper_val)
        mask = mask>0

        mask[object_mask>0] = False
        mask = binary_opening(mask, ANTS_OPENING_CLOSING_STRUCTURE)
        mask = binary_closing(mask, ANTS_OPENING_CLOSING_STRUCTURE)
        mask = remove_small_objects(mask, MIN_ANTS_SIZE)

        labeled_image, num_labels = label(connect_blobs(mask, overlap_size=3))
        from scipy.ndimage import maximum_filter, minimum_filter
        maximum_filtered = maximum_filter(labeled_image, size=OBJECT_DILATION_SIZE)
        minimum_filtered = minimum_filter(maximum_filtered, size=OBJECT_DILATION_SIZE)-labeled_image
        labeled_image, num_labels = label((labeled_image+minimum_filtered).astype(bool))
        self.labeled_ants = extend_lines(labeled_image, extend_by=5)

        ants_centers = center_of_mass(labeled_image, labeled_image, range(1, num_labels+1))
        self.ants_centers = np.array(ants_centers)

    #
    # def create_circular_mask(self, image_dim, center=None, radius=None):
    #     h,w = image_dim[0],image_dim[1]
    #     if center is None:  # use the middle of the image
    #         center = (int(w / 2), int(h / 2))
    #     if radius is None:  # use the smallest distance between the center and image walls
    #         radius = min(center[0], center[1], w - center[0], h - center[1])
    #
    #     Y, X = np.ogrid[:h, :w]
    #     dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    #
    #     mask = dist_from_center <= radius
    #     return mask