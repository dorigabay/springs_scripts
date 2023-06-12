import cv2
import copy
import numpy as np
from scipy.ndimage import label, generate_binary_structure, center_of_mass, maximum_filter, binary_fill_holes
from skimage.measure import regionprops, find_contours
from skimage.morphology import binary_dilation, binary_erosion, remove_small_objects
# local imports:
from video_analysis import utils

WHOLE_OBJECT_CLOSING_SIZE = 4
COLOR_CLOSING = 5
BLUE_CLOSING = 9
BLUE_SIZE_DEVIATION = 0.85
BLUE_SIZE_DEVIATION_CENTER = 0.98
LABELING_BINARY_STRUCTURE = generate_binary_structure(2, 2)
MIN_GREEN_SIZE = 7
MIN_RED_SIZE = 300
BUNDLES_CLOSING = np.ones((10, 10))
SPRINGS_PARTS_OVERLAP_SIZE = 10


class Springs:
    def __init__(self, parameters, image, previous_detections):
        self.n_springs = parameters["n_springs"]
        self.previous_detections = previous_detections
        binary_color_masks_connected, binary_color_masks_unconnected = self.mask_object_colors(parameters, image)
        self.whole_object_mask_unconnected = np.any(np.stack(list(binary_color_masks_unconnected.values()), axis=-1), axis=-1)
        self.whole_object_mask_connected = np.any(np.stack(list(binary_color_masks_connected.values()), axis=-1), axis=-1)
        self.object_center, self.tip_point, self.blue_radius,self.blue_area_size =self.detect_blue_stripe(binary_color_masks_connected["b"])
        green_mask = self.clean_mask(binary_color_masks_connected["g"],MIN_GREEN_SIZE)
        red_mask = self.clean_mask(binary_color_masks_connected["r"],MIN_RED_SIZE)
        red_labeled, green_labeled, fixed_ends_labeled, self.free_ends_labeled, self.red_centers, self.green_centers = \
            self.get_spring_parts(self.object_center,binary_color_masks_connected["r"],green_mask)
        self.bundles_labeled, self.bundles_labels = self.create_bundles_labels(red_mask,green_mask,fixed_ends_labeled)
        real_springs_bundles_labels = self.assign_ends_to_bundles(self.bundles_labeled, fixed_ends_labeled, self.free_ends_labeled, self.red_centers, green_labeled)
        bundles_labeled_after_removal = self.remove_labels(real_springs_bundles_labels, self.bundles_labeled)
        self.fixed_ends_edges_centers, self.fixed_ends_edges_bundles_labels = self.find_bounderies_touches(fixed_ends_labeled, red_labeled, bundles_labeled_after_removal)
        self.free_ends_edges_centers, self.free_ends_edges_bundles_labels = self.find_bounderies_touches(self.free_ends_labeled, red_labeled, bundles_labeled_after_removal)

    # def combine_masks(self,list_of_masks):
    #     # create a new array with all masks in new dimension
    #     combined = np.any(np.stack(list_of_masks, axis=-1), axis=-1)
    #     combined = list_of_masks[0]+list_of_masks[1]+list_of_masks[2]
    #     combined = combined.astype(np.bool).astype(np.uint8)
    #     return combined

    def mask_object_colors(self, parameters, image):
        boolean_color_masks_connected = {x:None for x in parameters["colors_spaces"]}
        boolean_color_masks_unconnected = {x:None for x in parameters["colors_spaces"]}
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        hsv_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        for color in parameters["colors_spaces"]:
            boolean_mask = np.full(hsv_image.shape[:-1], False, dtype="bool")
            for hsv_space in parameters["colors_spaces"][color]:
                mask1 = cv2.inRange(hsv_image, hsv_space[0], hsv_space[1]).astype(bool)
                mask2 = cv2.inRange(hsv_image, hsv_space[2], hsv_space[3]).astype(bool)
                boolean_mask[mask1+mask2] = True
            boolean_color_masks_unconnected[color] = boolean_mask
            boolean_mask = utils.connect_blobs(boolean_mask,COLOR_CLOSING)
            boolean_color_masks_connected[color] = boolean_mask
        return boolean_color_masks_connected, boolean_color_masks_unconnected

    def close_element(self, mask, structure):
        mask = binary_dilation(mask.astype(int), structure)
        mask = binary_erosion(mask.astype(int), structure[:-2,:-2])
        return mask.astype(bool)

    def detect_blue_stripe(self,mask_blue,closing_structure=BLUE_CLOSING):
        mask_blue_empty_closed = utils.connect_blobs(mask_blue, closing_structure)
        labeled,_ = label(mask_blue_empty_closed)
        blob_prop = regionprops(labeled)
        biggest_blob = blob_prop[np.argmax([x.area for x in blob_prop])].label
        mask_blue_empty_closed[np.invert(labeled==biggest_blob)] = 0
        mask_blue_full = np.zeros(mask_blue_empty_closed.shape,"uint8")
        binary_fill_holes(mask_blue_empty_closed,output=mask_blue_full)
        inner_mask = mask_blue_full - mask_blue_empty_closed
        inner_mask_center = center_of_mass(inner_mask)
        # if the center is not found, use the previous detection:
        if not np.isnan(np.sum(inner_mask_center)):
            object_center = [int(x) for x in inner_mask_center]
            object_center = np.array([object_center[1],object_center[0]])
        else: object_center = self.previous_detections[0]
        # if the length between the center and the edge is not as the previous detection,
        # use the previous detection for the center:
        mask_blue_full_contour = find_contours(mask_blue_full)[0].astype(int)
        farthest_point, blue_radius = utils.find_farthest_point(object_center, mask_blue_full_contour)

        BLUE_RADIUS_TOLERANCE = 0.05
        if self.previous_detections[0] is not None:
            mean_radius = self.previous_detections[3]/self.previous_detections[4]
            if np.abs(blue_radius-mean_radius)/mean_radius>BLUE_RADIUS_TOLERANCE:
                object_center = self.previous_detections[0]
                farthest_point, blue_radius = utils.find_farthest_point(object_center, mask_blue_full_contour)
                # farthest_point, blue_radius = utils.find_farthest_point(object_center,mask_blue_full_contours)
                if np.abs(blue_radius-mean_radius)/mean_radius>BLUE_RADIUS_TOLERANCE:
                    farthest_point = self.previous_detections[1]
                    blue_radius = np.sqrt(np.sum(np.square(farthest_point-object_center)))
                    if np.abs(blue_radius-mean_radius)/mean_radius>BLUE_RADIUS_TOLERANCE:
                        raise ValueError("There is a problem in the blue part detection")
        blue_area_size = np.sum(mask_blue_full)
        return object_center, farthest_point, blue_radius, blue_area_size

    def find_farthest_point(self, point, contour):
        point = np.array([point[1],point[0]])
        distances = np.sqrt(np.sum(np.square(contour-point),1))
        # take the average of the 50 farthest points
        farthest_points = np.argsort(distances)[-50:]
        farthest_point = np.mean(contour[farthest_points],0).astype(int)
        farthest_point = np.array([farthest_point[1],farthest_point[0]])
        return farthest_point, np.max(distances)

    def clean_mask(self,mask,min_size):
        mask = mask.astype(bool)
        inner_circle_mask = utils.create_circular_mask(mask.shape, center=self.object_center, radius=self.blue_radius)
        outer_circle_mask = utils.create_circular_mask(mask.shape, center=self.object_center, radius=self.blue_radius*3)
        mask[inner_circle_mask] = False
        mask[np.invert(outer_circle_mask)] = False
        mask = binary_fill_holes(mask.astype(np.uint8)).astype(bool)
        mask = self.remove_small_blobs(mask, min_size)
        return mask

    def remove_small_blobs(self, bool_mask, min_size: int = 0):
        """
        Removes from the input mask all the blobs having less than N adjacent pixels.
        We set the small objects to the background label 0.
        """
        bool_mask = bool_mask.astype(bool)
        if min_size > 0:
            # dtype = bool_mask.dtype
            bool_mask = remove_small_objects(bool_mask, min_size=min_size)
            # bool_mask = bool_mask.astype(dtype)
        return bool_mask

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
        red_centers = utils.swap_columns(red_centers)
        red_radii = np.sqrt(np.sum(np.square(red_centers - object_center), axis=1))

        green_labeled, green_num_features = label(green_mask, LABELING_BINARY_STRUCTURE)
        green_centers =np.array(center_of_mass(green_labeled,labels=green_labeled,index=range(1,green_num_features+1)))
        green_centers = utils.swap_columns(green_centers)
        green_radii = np.sqrt(np.sum(np.square(green_centers - object_center), axis=1))
        fixed_ends_labels = np.array([x for x in range(1, green_num_features + 1)])[green_radii < (np.mean(red_radii))]
        free_ends_labels = np.array([x for x in range(1, green_num_features + 1)])[green_radii > np.mean(red_radii)]

        fixed_ends_labeled = copy.copy(green_labeled)
        fixed_ends_labeled[np.invert(np.isin(green_labeled, fixed_ends_labels))] = 0
        free_ends_labeled = copy.copy(green_labeled)
        free_ends_labeled[np.invert(np.isin(green_labeled, free_ends_labels))] = 0
        return red_labeled, green_labeled, fixed_ends_labeled, free_ends_labeled, red_centers, green_centers

    def create_bundles_labels(self,red_mask,green_mask,fixed_ends_labeled,closing_structure=BUNDLES_CLOSING):
        #TODO: fix the bug that crushes the program when using connect_blobs
        all_parts_mask = red_mask + green_mask
        all_parts_mask = self.close_element(all_parts_mask,closing_structure)
        # all_parts_mask = utils.connect_blobs(all_parts_mask, closing_structure)
        labeled_image, num_features = label(all_parts_mask, generate_binary_structure(2, 2))
        fied_ends_centers = center_of_mass(fixed_ends_labeled, labels=fixed_ends_labeled,
                                         index=np.unique(fixed_ends_labeled)[1:])
        fied_ends_centers = np.array([np.array([x, y]).astype("int") for x, y in fied_ends_centers])
        self.bundles_centers = fied_ends_centers
        center = np.array([self.object_center[1],self.object_center[0]])
        tipp = np.array([self.tip_point[1],self.tip_point[0]])
        fied_ends_angles = utils.calc_angles(fied_ends_centers, center, tipp)
        labeled_image_sorted = np.zeros(labeled_image.shape)
        for pnt,angle in zip(fied_ends_centers,fied_ends_angles):
            bundle_label = labeled_image[pnt[0],pnt[1]]
            if bundle_label != 0:
                labeled_image_sorted[labeled_image == bundle_label] = angle
        bundels_labels_fixed_centers = [labeled_image[x[0],x[1]] for x in fied_ends_centers]
        # bad_bundels = np.unique(bundels_labels_fixed_centers)

        counts = np.array([bundels_labels_fixed_centers.count(x) for x in bundels_labels_fixed_centers])
        melted_bundles = np.unique(np.array(bundels_labels_fixed_centers)[counts > 1])
        for bad_label in melted_bundles:
            labeled_image_sorted[labeled_image==bad_label] = 0
        return labeled_image_sorted, fied_ends_angles

    # def calc_angles(self, points_to_measure, object_center, tip_point):
    #     ba = points_to_measure - object_center
    #     bc = (tip_point - object_center)
    #     ba_y = ba[:,0]
    #     ba_x = ba[:,1]
    #     dot = ba_y*bc[0] + ba_x*bc[1]
    #     det = ba_y*bc[1] - ba_x*bc[0]
    #     angles = np.arctan2(det, dot)
    #     return angles

    def assign_ends_to_bundles(self,bundles_labeled,fixed_ends_labeled,free_ends_labeled,red_centers,green_labeled):
        fixed_ends_centers = utils.swap_columns(np.array(
            center_of_mass(green_labeled, labels=green_labeled, index=list(np.unique(fixed_ends_labeled))[1:]), "int"))
        free_ends_centers = utils.swap_columns(np.array(
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
        bundles_labels = self.screen_bundles(fixed_ends_bundles_labels, free_ends_bundles_labels, red_bundles_labels)
        return  bundles_labels

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
        self.bundles_labels = labels_to_keep
        return labeled_image

    def find_bounderies_touches(self, labeled1, labeled2, bundles_labeled):
        maximum_filter_labeled1 = maximum_filter(labeled1, SPRINGS_PARTS_OVERLAP_SIZE)
        maximum_filter_labeled2 = maximum_filter(labeled2, SPRINGS_PARTS_OVERLAP_SIZE)
        overlap_labeled = np.zeros(maximum_filter_labeled1.shape, "int")
        boolean_overlap = (maximum_filter_labeled1 != 0) * (maximum_filter_labeled2 != 0)
        overlap_labeled[boolean_overlap] = maximum_filter_labeled1[boolean_overlap]
        overlap_labels = list(np.unique(overlap_labeled))[1:]
        overlap_centers = utils.swap_columns(np.array(
            center_of_mass(overlap_labeled, labels=overlap_labeled, index=overlap_labels)).astype("int"))
        overlap_bundles_labels = [bundles_labeled[x[1], x[0]] for x in overlap_centers]
        overlap_bundles_labels = self.remove_duplicates(overlap_bundles_labels)
        return overlap_centers, overlap_bundles_labels

    def remove_duplicates(self,labels):
        # turn duplicates labels to 0
        labels_array = np.array(labels)
        counts = np.array([labels.count(x) for x in labels])
        labels_array[counts > 1] = 0
        return list(labels_array)

