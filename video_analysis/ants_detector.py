import cv2
import numpy as np
from scipy.ndimage import label, center_of_mass
from skimage.morphology import remove_small_objects, binary_closing, binary_opening
from scipy.ndimage import maximum_filter, minimum_filter

# local imports:
from video_analysis import utils
from video_analysis.springs_detector import Springs

OBJECT_DILATION_SIZE = 5
FIRST_OPENING_STRUCTURE = np.ones((1, 1))
SECOND_OPENING_STRUCTURE = np.ones((2, 2))
MIN_ANTS_SIZE = 20
SOBEL_KERNEL_SIZE = 1
GRADIANT_THRESHOLD = 90
LOWER_HSV_VALUES = np.array([0, 60, 0])
UPPER_HSV_VALUES = np.array([179, 255, 200])
ANTS_EXTENSION_LENGTH = 5


class Ants(Springs):
    def __init__(self,parameters, image, previous_detections):
        super().__init__(parameters, image, previous_detections)
        self.label_ants(image, self.whole_object_mask_unconnected, self.all_perspective_squares_mask)

    def label_ants(self, image, object_mask, perspective_squares_mask):
        image = utils.neutrlize_colour(utils.white_balance_bgr(np.copy(image)), alpha=2, beta=10)
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=SOBEL_KERNEL_SIZE)
        sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=SOBEL_KERNEL_SIZE)
        gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        gradient_magnitude[object_mask > 0] = 0
        gradient_magnitude[perspective_squares_mask > 0] = 0
        sobel_mask = gradient_magnitude > GRADIANT_THRESHOLD
        for i in range(1,4):
            sobel_mask = binary_opening(sobel_mask, FIRST_OPENING_STRUCTURE)
            sobel_mask[object_mask > 0] = False
            sobel_mask[perspective_squares_mask > 0] = False
            sobel_mask = binary_closing(sobel_mask, np.ones((i+1, i+1)))
        sobel_mask = binary_opening(sobel_mask, SECOND_OPENING_STRUCTURE)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, LOWER_HSV_VALUES, UPPER_HSV_VALUES) > 0

        combined_mask = color_mask * sobel_mask
        combined_mask = binary_closing(combined_mask, np.ones((2, 2)))
        combined_mask = remove_small_objects(combined_mask, MIN_ANTS_SIZE)
        labeled_image, num_labels = label(utils.connect_blobs(combined_mask, overlap_size=3))

        maximum_filtered = maximum_filter(labeled_image, size=OBJECT_DILATION_SIZE)
        minimum_filtered = minimum_filter(maximum_filtered, size=OBJECT_DILATION_SIZE) - labeled_image
        labeled_image, num_labels = label((labeled_image + minimum_filtered).astype(bool))
        self.labeled_ants = utils.extend_lines(labeled_image, extend_by=ANTS_EXTENSION_LENGTH)
        ants_centers = center_of_mass(labeled_image, labeled_image, range(1, num_labels+1))
        self.ants_centers = np.array(ants_centers)

        cv2.imshow('ants', cv2.resize(self.labeled_ants.astype(bool).astype(np.uint8) * 255, (0, 0), fx=0.3, fy=0.3))
        cv2.waitKey(1)