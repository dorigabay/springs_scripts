import cv2
import copy
import numpy as np
from scipy.ndimage import label, generate_binary_structure, center_of_mass, maximum_filter, binary_fill_holes
from skimage.measure import regionprops, find_contours
from skimage.morphology import binary_dilation, binary_erosion, remove_small_objects
# local imports:
from utils import create_circular_mask, connect_blobs

WHOLE_OBJECT_CLOSING_SIZE = 4
# COLOR_CLOSING = np.ones((3, 3))
COLOR_CLOSING = 3
# BLUE_CLOSING = np.ones((5, 5))
BLUE_CLOSING = 9
BLUE_SIZE_DEVIATION = 0.85
BLUE_SIZE_DEVIATION_CENTER = 0.98
LABELING_BINARY_STRUCTURE = generate_binary_structure(2, 2)
MIN_GREEN_SIZE = 7
MIN_RED_SIZE = 300
BUNDLES_CLOSING = np.ones((10, 10))
# BUNDLES_CLOSING = 5
SPRINGS_PARTS_OVERLAP_SIZE = 10


class Springs:
    def __init__(self, parameters, image, previous_detections):
        self.binary_color_masks = self.mask_object_colors(parameters, image)
        # self.whole_object_mask = self.close_element(self.combine_masks(list(self.binary_color_masks.values())),np.ones((15,15)))
        self.whole_object_mask = connect_blobs(self.combine_masks(list(self.binary_color_masks.values())),WHOLE_OBJECT_CLOSING_SIZE)
        # cv2.imshow("whole_object_mask",self.whole_object_mask.astype(np.uint8)*255)
        # cv2.waitKey(1)
        self.object_center, self.tip_point, self.mask_blue_full, self.blue_radius =\
            self.detect_blue_stripe(self.binary_color_masks["b"], previous_detections = previous_detections)
        self.green_mask = self.clean_mask(self.binary_color_masks["g"],MIN_GREEN_SIZE)
        self.red_mask = self.clean_mask(self.binary_color_masks["r"],MIN_RED_SIZE)
        # cv2.imshow("red_mask",self.red_mask.astype(np.uint8)*255)
        # cv2.waitKey(1)
        self.red_labeled, self.green_labeled, self.fixed_ends_labeled, self.free_ends_labeled, self.red_centers, self.green_centers = \
            self.get_spring_parts(self.object_center,self.binary_color_masks["r"],self.green_mask)
        self.bundles_labeled, self.bundles_labels = self.create_bundles_labels()
        self.fixed_ends_bundles_labels, self.free_ends_bundles_labels, self.real_springs_bundles_labels = \
            self.assign_ends_to_bundles(self.bundles_labeled, self.fixed_ends_labeled,
                                          self.free_ends_labeled,self.red_centers, self.green_labeled)
        self.bundles_labeled_after_removal = self.remove_labels(self.real_springs_bundles_labels, self.bundles_labeled)
        self.fixed_ends_edges_centers, self.fixed_ends_edges_bundles_labels = \
            self.find_bounderies_touches(self.fixed_ends_labeled, self.red_labeled, self.bundles_labeled_after_removal)
        self.free_ends_edges_centers, self.free_ends_edges_bundles_labels = \
            self.find_bounderies_touches(self.free_ends_labeled, self.red_labeled, self.bundles_labeled_after_removal)

    def combine_masks(self,list_of_masks):
        combined = list_of_masks[0]+list_of_masks[1]+list_of_masks[2]
        combined = self.convert_bool_to_binary(combined.astype("bool"))
        return combined

    def numpy_masking(self, image, hsv_lower, hsv_upper):
        image_zeros = np.zeros(image.shape[:-1])
        bool_lower = (image[:,:,0] >= hsv_lower[0])&(image[:,:,1] >= hsv_lower[1])&(image[:,:,2] >= hsv_lower[2])
        bool_upper = (image[:,:,0] <= hsv_upper[0])&(image[:,:,1] <= hsv_upper[1])&(image[:,:,2] <= hsv_upper[2])
        image_zeros[bool_lower&bool_upper] = 1
        return image_zeros

    def mask_object_colors(self, parameters, image):
        binary_color_masks = {x:None for x in parameters["colors_spaces"]}
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        hsv_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        for color in parameters["colors_spaces"]:
            binary_mask = np.zeros(image.shape[:-1],"int")
            for hsv_space in parameters["colors_spaces"][color]:
                mask1_np = self.numpy_masking(hsv_image, hsv_space[0], hsv_space[1])
                mask2_np = self.numpy_masking(hsv_image, hsv_space[2], hsv_space[3])
                binary_mask[(mask1_np+mask2_np)!=0] = 1
            # binary_mask = self.close_element(binary_mask,COLOR_CLOSING)
            binary_mask = connect_blobs(binary_mask,COLOR_CLOSING)
            binary_color_masks[color] = binary_mask
        return binary_color_masks

    def close_element(self, mask, structure):
        mask = self.convert_bool_to_binary(binary_dilation(mask, structure))
        mask = self.convert_bool_to_binary(binary_erosion(mask, structure[:-2,:-2]))
        return mask


    def convert_bool_to_binary(self,bool_mask):
        binary_mask = np.zeros(bool_mask.shape, "int")
        binary_mask[bool_mask] = 1
        return binary_mask

    def detect_blue_stripe(self,mask_blue,closing_structure=BLUE_CLOSING,previous_detections=None):
        mask_blue_empty_closed = connect_blobs(mask_blue, closing_structure)
        #find the center of the blue stripe
        lableled,_ = label(mask_blue_empty_closed)
        bulb_prop = regionprops(lableled)
        biggest_blub = bulb_prop[np.argmax([x.area for x in bulb_prop])].label
        mask_blue_empty_closed[np.invert(lableled==biggest_blub)] = 0
        mask_blue_full = np.zeros(mask_blue_empty_closed.shape,"int")
        binary_fill_holes(mask_blue_empty_closed,output=mask_blue_full)
        inner_mask = mask_blue_full - mask_blue_empty_closed
        inner_mask_center = center_of_mass(inner_mask)
        # if the center is not found, use the previous detection
        if not np.isnan(np.sum(inner_mask_center)):
            object_center = [int(x) for x in inner_mask_center]
            object_center = np.array([object_center[1],object_center[0]])
        else: object_center = previous_detections[0]

        # if the length between the center and the edge is not as the previous detection,
        # use the previous detection for the center:
        farthest_point, blue_radius = self.find_farthest_point(object_center,mask_blue_full)
        # print(np.sqrt(np.sum(np.square(farthest_point-object_center))))
        BLUE_RADIUS_TOLERANCE = 0.05
        if previous_detections is not None:
            mean_radius = previous_detections[2]/previous_detections[3]
            # print("blue_radius",blue_radius)
            # print("previous_detections[2]",previous_detections[2])
            # tolerance = np.abs(blue_radius-previous_detections[2])/previous_detections[2]>BLUE_RADIUS_TOLERANCE
            # print("tolerance",tolerance)
            # print(np.abs(blue_radius - previous_detections[2]))
            if np.abs(blue_radius-mean_radius)/mean_radius>BLUE_RADIUS_TOLERANCE:
                object_center = previous_detections[0]
                farthest_point, blue_radius = self.find_farthest_point(object_center,mask_blue_full)
                if np.abs(blue_radius-mean_radius)/mean_radius>BLUE_RADIUS_TOLERANCE:
                    farthest_point = previous_detections[1]
                    blue_radius = np.sqrt(np.sum(np.square(farthest_point-object_center)))
                    if np.abs(blue_radius-mean_radius)/mean_radius>BLUE_RADIUS_TOLERANCE:
                        print("the problem is here")
                        exit("blue radius is not as expected")
        # if not previous_detections is None:
        #     if np.sum(mask_blue_full) < np.sum(previous_detections[1])*BLUE_SIZE_DEVIATION:
        #         mask_blue_full = previous_detections[1]
        # cv2.imshow("inner_mask", (mask_blue_empty_closed*255).astype(np.uint8))
        # cv2.waitKey(1)
        return object_center, farthest_point, mask_blue_full, blue_radius

    def find_farthest_point(self, point, mask):
        contour = (find_contours(mask)[0]).astype(int)
        point = np.array([point[1],point[0]])
        distances = np.sqrt(np.sum(np.square(contour-point),1))
        # take the average of the 50 farthest points
        farthest_points = np.argsort(distances)[-50:]
        farthest_point = np.mean(contour[farthest_points],0).astype(int)
        farthest_point = np.array([farthest_point[1],farthest_point[0]])
        return farthest_point, np.max(distances)

    def clean_mask(self,mask,min_size):
        inner_circle_mask = create_circular_mask(mask.shape, center=self.object_center, radius=self.blue_radius)
        outer_circle_mask = create_circular_mask(mask.shape, center=self.object_center, radius=self.blue_radius*3)
        mask[inner_circle_mask] = 0
        mask[np.invert(outer_circle_mask)] = 0
        mask = self.convert_bool_to_binary(binary_fill_holes(mask))
        mask = self.remove_small_blobs(mask, min_size)
        return mask

    def remove_small_blobs(self, binary_mask: np.ndarray, min_size: int = 0):
        """
        Removes from the input mask all the blobs having less than N adjacent pixels.
        We set the small objects to the background label 0.
        """
        if min_size > 0:
            dtype = binary_mask.dtype
            binary_mask = remove_small_objects(binary_mask.astype(bool), min_size=min_size)
            binary_mask = binary_mask.astype(dtype)
        return binary_mask

    def screen_small_labels(self, labeled, threshold=0.5):
        # calculate the number of pixels in each label, and remove labels with too few pixels, by the being smaller than the mean by a threshold
        label_counts = np.bincount(labeled.ravel())
        too_small = np.arange(len(label_counts))[label_counts < np.mean(label_counts[1:]) * threshold]
        labeled[np.isin(labeled, too_small)] = 0

        return label(labeled)

    def get_spring_parts(self,object_center,red_mask,green_mask):
        red_labeled, red_num_features = label(red_mask, LABELING_BINARY_STRUCTURE)
        red_labeled, red_num_features = self.screen_small_labels(red_labeled)
        red_centers = np.array(center_of_mass(red_labeled, labels=red_labeled, index=range(1, red_num_features + 1)))
        red_centers = self.swap_columns(red_centers)
        red_radii = np.sqrt(np.sum(np.square(red_centers - object_center), axis=1))

        green_labeled, green_num_features = label(green_mask, LABELING_BINARY_STRUCTURE)
        green_centers =np.array(center_of_mass(green_labeled,labels=green_labeled,index=range(1,green_num_features+1)))
        green_centers = self.swap_columns(green_centers)
        green_radii = np.sqrt(np.sum(np.square(green_centers - object_center), axis=1))
        fixed_ends_labels = np.array([x for x in range(1, green_num_features + 1)])[green_radii < (np.mean(red_radii))]
        free_ends_labels = np.array([x for x in range(1, green_num_features + 1)])[green_radii > np.mean(red_radii)]

        fixed_ends_labeled = copy.copy(green_labeled)
        fixed_ends_labeled[np.invert(np.isin(green_labeled, fixed_ends_labels))] = 0
        free_ends_labeled = copy.copy(green_labeled)
        free_ends_labeled[np.invert(np.isin(green_labeled, free_ends_labels))] = 0
        return red_labeled, green_labeled, fixed_ends_labeled, free_ends_labeled, red_centers, green_centers

    def swap_columns(self,array):
        array[:, [0, 1]] = array[:, [1, 0]]
        return array

    def create_bundles_labels(self,closing_structure=BUNDLES_CLOSING):
        #TODO: fix the bug that crushes the program when using connect_blobs
        all_parts_mask = self.red_mask + self.green_mask
        all_parts_mask = self.close_element(all_parts_mask,closing_structure)
        # all_parts_mask = connect_blobs(all_parts_mask, closing_structure)
        # cv2.imshow("all_parts_mask", self.convert_bool_to_binary(all_parts_mask.astype("bool")).astype(np.uint8)*255)
        # cv2.waitKey(1)
        labeled_image, num_features = label(all_parts_mask, generate_binary_structure(2, 2))
        fied_ends_centers = center_of_mass(self.fixed_ends_labeled, labels=self.fixed_ends_labeled,
                                         index=np.unique(self.fixed_ends_labeled)[1:])
        fied_ends_centers = np.array([np.array([x, y]).astype("int") for x, y in fied_ends_centers])
        self.bundles_centers = fied_ends_centers
        center = np.array([self.object_center[1],self.object_center[0]])
        tipp = np.array([self.tip_point[1],self.tip_point[0]])
        fied_ends_angles = self.calc_angles(fied_ends_centers, center, tipp)
        labeled_image_sorted = np.zeros(labeled_image.shape)
        for pnt,angle in zip(fied_ends_centers,fied_ends_angles):
            bundle_label = labeled_image[pnt[0],pnt[1]]
            if bundle_label != 0:
                labeled_image_sorted[labeled_image == bundle_label] = angle
        bundels_labels_fixed_centers = [labeled_image[x[0],x[1]] for x in fied_ends_centers]
        bad_bundels = np.unique(bundels_labels_fixed_centers)

        counts = np.array([bundels_labels_fixed_centers.count(x) for x in bundels_labels_fixed_centers])
        melted_bundles = np.unique(np.array(bundels_labels_fixed_centers)[counts > 1])
        for bad_label in melted_bundles:
            labeled_image_sorted[labeled_image==bad_label] = 0
        return labeled_image_sorted, fied_ends_angles

    def calc_angles(self, points_to_measure, object_center, tip_point):
        ba = points_to_measure - object_center
        bc = (tip_point - object_center)
        ba_y = ba[:,0]
        ba_x = ba[:,1]
        dot = ba_y*bc[0] + ba_x*bc[1]
        det = ba_y*bc[1] - ba_x*bc[0]
        angles = np.arctan2(det, dot)
        return angles

    def assign_ends_to_bundles(self,bundles_labeled,fixed_ends_labeled,free_ends_labeled,red_centers,green_labeled):
        fixed_ends_centers = self.swap_columns(np.array(
            center_of_mass(green_labeled, labels=green_labeled, index=list(np.unique(fixed_ends_labeled))[1:]), "int"))
        free_ends_centers = self.swap_columns(np.array(
            center_of_mass(green_labeled, labels=green_labeled, index=list(np.unique(free_ends_labeled))[1:]), "int"))
        fixed_ends_bundles_labels = []
        free_ends_bundles_labels = []
        red_bundles_labels = []
        for x1, y1 in fixed_ends_centers:
            fixed_ends_bundles_labels.append(bundles_labeled[y1, x1])
        for x1, y1 in free_ends_centers:
            free_ends_bundles_labels.append(bundles_labeled[y1, x1])
        for x1, y1 in red_centers.astype("int"):
            red_bundles_labels.append(bundles_labeled[y1, x1])
        labels_to_keep = self.screen_bundles(fixed_ends_bundles_labels, free_ends_bundles_labels, red_bundles_labels)
        return fixed_ends_bundles_labels, free_ends_bundles_labels, labels_to_keep

    def screen_bundles(self, fixed_labels, free_labels, red_labels):
        counts = np.array([fixed_labels.count(x) for x in fixed_labels])
        melted_bundles = np.unique(np.array(fixed_labels)[counts>1])
        for label in melted_bundles:
            fixed_labels = list(filter((label).__ne__, fixed_labels))
        ### Only bundles which has all 3 parts (free,fixed,red) will be consider real bundle
        real_springs_labels=list(set(fixed_labels).intersection(set(free_labels)).intersection(set(red_labels)))
        return real_springs_labels

    def remove_labels(self, labels_to_keep, labeled_image):
        labeled_image[np.isin(labeled_image, labels_to_keep, invert=True)] = 0
        return labeled_image

    def find_bounderies_touches(self, labeled1, labeled2, bundles_labeled):
        maximum_filter_labeled1 = maximum_filter(labeled1, SPRINGS_PARTS_OVERLAP_SIZE)
        maximum_filter_labeled2 = maximum_filter(labeled2, SPRINGS_PARTS_OVERLAP_SIZE)
        overlap_labeled = np.zeros(maximum_filter_labeled1.shape, "int")
        boolean_overlap = (maximum_filter_labeled1 != 0) * (maximum_filter_labeled2 != 0)
        overlap_labeled[boolean_overlap] = maximum_filter_labeled1[boolean_overlap]
        overlap_labels = list(np.unique(overlap_labeled))[1:]
        overlap_centers = self.swap_columns(np.array(
            center_of_mass(overlap_labeled, labels=overlap_labeled, index=overlap_labels)).astype("int"))
        overlap_bundles_labels = [bundles_labeled[x[1], x[0]] for x in overlap_centers]
        overlap_bundles_labels = self.remove_duplicates(overlap_bundles_labels)
        return overlap_centers, overlap_bundles_labels

    def remove_duplicates(self, labels):
        # turn duplicates labels to 0
        labels_array = np.array(labels)
        counts = np.array([labels.count(x) for x in labels])
        labels_array[counts > 1] = 0
        return list(labels_array)
