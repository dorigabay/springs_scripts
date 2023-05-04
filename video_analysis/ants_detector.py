import cv2
import numpy as np
from scipy.ndimage import label, center_of_mass, maximum_filter
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects, binary_closing, binary_opening
#local imports:
from general_video_scripts.utils import close_element, convert_bool_to_binary, extend_lines, connect_blobs

OBJECT_DILATION_SIZE = 3
ANTS_OPENING_CLOSING_STRUCTURE = np.ones((4, 4))
MIN_ANTS_SIZE = 60
ANTS_SPRINGS_OVERLAP_SIZE = 10

class Ants:
    def __init__(self, image, springs):
        # self.object_mask = self.create_object_mask(image.shape,springs)
        # self.labeled_ants = self.label_ants(image,self.object_mask)
        self.label_ants(image, springs.whole_object_mask_unconnected)

    # def create_object_mask(self,image_dim,springs):
    #     # circle_mask = self.create_circular_mask(image_dim, center=springs.object_center, radius=springs.blue_radius*1.5)
    #     mask_object = springs.whole_object_mask != 0
    #     return mask_object
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

    def label_ants(self, image, object_mask):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_val = np.array([0, 0, 0])
        upper_val = np.array([179, 150, 170])
        mask = cv2.inRange(hsv, lower_val, upper_val)
        mask = mask>0
        mask[object_mask>0] = False
        mask = binary_opening(mask, ANTS_OPENING_CLOSING_STRUCTURE)
        mask = binary_closing(mask, ANTS_OPENING_CLOSING_STRUCTURE)
        mask = remove_small_objects(mask, MIN_ANTS_SIZE)

        labeled_image, num_labels = label(connect_blobs(mask, overlap_size=4))
        labeled_image = clear_border(labeled_image)
        self.labeled_ants = extend_lines(labeled_image, extend_by=5)
        self.ant_centers = center_of_mass(labeled_image, labeled_image, range(1, num_labels + 1))


if __name__ == "__main__":
    import general_video_scripts.collect_color_parameters as ccp
    import general_video_scripts.utils as utils
    video_path = r"Z:\Dor_Gabay\ThesisProject\data\videos\15.9.22\plus0.3mm_force\S5280004.MP4"
    import pickle
    # parameters = ccp.set_parameters(video_path)
    parameters_path = r"Z:\Dor_Gabay\ThesisProject\data\videos\15.9.22\parameters\S5280004_video_parameters.pickle"
    parameters = pickle.load(open(parameters_path, "rb"))[video_path]
    print(parameters["colors_spaces"])

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    # # image = cv2.imread(r"Z:\Dor_Gabay\ThesisProject\data\pics\frame.png")
    image = ccp.neutrlize_colour(frame)
    masks = utils.mask_object_colors(parameters, image)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_val = np.array([0, 0, 0])
    upper_val = np.array([179, 150, 170])
    mask = cv2.inRange(hsv, lower_val, upper_val)
    mask = mask>0
    object_mask = np.zeros(mask.shape).astype(np.bool)
    for m in masks:
        object_mask[masks[m]>0] = True

    mask[object_mask] = False
    mask = binary_opening(mask, ANTS_OPENING_CLOSING_STRUCTURE)
    mask = binary_closing(mask, ANTS_OPENING_CLOSING_STRUCTURE)
    mask = remove_small_objects(mask, MIN_ANTS_SIZE)
    cv2.imshow("ants",mask.astype(np.uint8)*255)
    cv2.waitKey(0)
