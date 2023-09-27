import cv2
import copy
import numpy as np
from scipy.ndimage import label, generate_binary_structure, center_of_mass, maximum_filter, binary_fill_holes
from skimage.measure import regionprops, find_contours
# local imports:
import utils


# GRADIANT_THRESHOLD = 5
# NEUTRALIZE_COLOUR_ALPHA = 2.5
# BLUR_KERNEL = (7, 7)
COLOR_CLOSING = 3
NEEDLE_CLOSING = 3
NEEDLE_RADIUS_TOLERANCE = 0.05
LABELING_BINARY_STRUCTURE = generate_binary_structure(2, 2)
MIN_SPRING_ENDS_SIZE = 80
MIN_SPRING_MIDDLE_PART_SIZE = 300
MIN_SIZE_FOR_WHOLE_OBJECT = 30
BUNDLES_CLOSING = np.ones((3, 3))
SPRINGS_PARTS_OVERLAP_SIZE = 10


# class Springs(PerspectiveSquares):
class Springs:
    def __init__(self, parameters, image, previous_detections):
        # super().__init__(parameters, image, previous_detections)
        self.parameters = parameters
        self.image = image
        self.previous_detections = previous_detections
        self.n_springs = self.parameters["n_springs"]
        image_cropped, self.object_crop_coordinates = self.prepare_frame(self.image)
        springs_properties = self.get_springs_properties(image_cropped)
        self.springs_coordinates_transformation(self.image, springs_properties)

    def prepare_frame(self, frame):
        if (self.previous_detections["skipped_frames"] >= 25) or self.previous_detections["frame_count"] == 0:
            object_crop_coordinates = np.array([0, self.parameters["resolution"][0], 0, self.parameters["resolution"][1]])
        else:
            object_crop_coordinates = utils.create_box_coordinates(self.previous_detections["object_center_coordinates"], self.parameters["ocm"])[0]
        image_processed = utils.crop_frame_by_coordinates(frame, object_crop_coordinates)
        image_processed = utils.process_image(image_processed, alpha=self.parameters["NEUTRALIZE_COLOUR_ALPHA"],
                                                               blur_kernel=self.parameters["NEUTRALIZE_COLOUR_BETA"],
                                                               gradiant_threshold=self.parameters["GRADIANT_THRESHOLD"])
        return image_processed, object_crop_coordinates

    def get_springs_properties(self, image_cropped):
        binary_color_masks_connected, binary_color_masks_unconnected = self.mask_object_colors(self.parameters, image_cropped)
        whole_object_mask_unconnected = self.clean_mask(np.any(np.stack(list(binary_color_masks_unconnected.values()), axis=-1), axis=-1),
                                                        MIN_SIZE_FOR_WHOLE_OBJECT, remove_needle=False, fill_holes=False)
        self.object_center_coordinates, self.tip_point, self.object_needle_radius, self.object_needle_area_size =self.detect_object_needle(binary_color_masks_connected["g"])
        spring_ends_mask = self.clean_mask(binary_color_masks_connected["r"], MIN_SPRING_ENDS_SIZE)
        spring_middle_part_mask = self.clean_mask(binary_color_masks_connected["b"], MIN_SPRING_MIDDLE_PART_SIZE)
        spring_middle_part_labeled, spring_ends_labeled, self.fixed_ends_labeled, self.free_ends_labeled, spring_middle_part_centers, spring_ends_centers = \
            self.get_spring_parts(self.object_center_coordinates, spring_middle_part_mask, spring_ends_mask)
        self.bundles_labeled, bundles_labels = self.create_bundles_labels(spring_middle_part_mask, spring_ends_mask, self.fixed_ends_labeled)
        self.bundles_labels = self.assign_ends_to_bundles(self.bundles_labeled, self.fixed_ends_labeled, self.free_ends_labeled, spring_middle_part_centers, spring_ends_labeled)
        self.bundles_labeled[np.isin(self.bundles_labeled, self.bundles_labels, invert=True)] = 0  # remove bundles that are not connected to any end
        # bundles_labeled_after_removal = self.remove_labels(self.bundles_labels, self.bundles_labeled)
        self.free_ends_labeled[self.bundles_labeled == 0] = 0
        self.fixed_ends_labeled[self.bundles_labeled == 0] = 0
        fixed_ends_edges_centers, self.fixed_ends_edges_bundles_labels = self.find_boundaries_touches(self.fixed_ends_labeled, spring_middle_part_labeled, self.bundles_labeled)
        free_ends_edges_centers, self.free_ends_edges_bundles_labels = self.find_boundaries_touches(self.free_ends_labeled, spring_middle_part_labeled, self.bundles_labeled)
        properties = [whole_object_mask_unconnected, spring_middle_part_labeled, spring_ends_labeled,
                      spring_middle_part_centers, spring_ends_centers, fixed_ends_edges_centers, free_ends_edges_centers]
        return properties

    def springs_coordinates_transformation(self, image, springs_properties):
        whole_object_mask_unconnected, spring_middle_part_labeled, spring_ends_labeled,\
        spring_middle_part_centers, spring_ends_centers, fixed_ends_edges_centers, free_ends_edges_centers\
            = springs_properties
        y_addition, x_addition = self.object_crop_coordinates[0], self.object_crop_coordinates[2]
        self.fixed_ends_edges_centers = fixed_ends_edges_centers + [x_addition, y_addition]
        self.free_ends_edges_centers = free_ends_edges_centers + [x_addition, y_addition]
        self.spring_middle_part_centers = spring_middle_part_centers + [x_addition, y_addition] # used only for visualization
        self.spring_ends_centers = spring_ends_centers + [x_addition, y_addition] # used only for visualization
        self.object_center_coordinates = self.object_center_coordinates + [x_addition, y_addition]
        self.tip_point = self.tip_point + [x_addition, y_addition]
        self.whole_object_mask_unconnected = np.full(image.shape[:2], False, dtype=np.bool)
        self.whole_object_mask_unconnected[self.object_crop_coordinates[0]:self.object_crop_coordinates[1],self.object_crop_coordinates[2]:self.object_crop_coordinates[3]] = whole_object_mask_unconnected > 0

    def mask_object_colors(self, parameters, image):
        object_colors = [x for x in parameters["colors_spaces"].keys() if x != "p"]
        boolean_color_masks_connected = {x:None for x in object_colors}
        boolean_color_masks_unconnected = {x:None for x in object_colors}
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        hsv_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        for color_name in object_colors:
            boolean_mask = np.full(hsv_image.shape[:-1], False, dtype="bool")
            for hsv_space in parameters["colors_spaces"][color_name]:
                mask1 = cv2.inRange(hsv_image, hsv_space[0], hsv_space[1]).astype(bool)
                mask2 = cv2.inRange(hsv_image, hsv_space[2], hsv_space[3]).astype(bool)
                boolean_mask[mask1+mask2] = True
            boolean_color_masks_unconnected[color_name] = boolean_mask
            boolean_mask = utils.connect_blobs(boolean_mask,COLOR_CLOSING)
            boolean_color_masks_connected[color_name] = boolean_mask
        return boolean_color_masks_connected, boolean_color_masks_unconnected

    def detect_object_needle(self, needle_mask, closing_structure=NEEDLE_CLOSING):
        needle_mask_empty_closed = utils.connect_blobs(needle_mask, closing_structure)
        labeled, _ = label(needle_mask_empty_closed)
        needle_prop = regionprops(labeled)
        biggest_blob = needle_prop[np.argmax([x.area for x in needle_prop])].label
        needle_mask_empty_closed[np.invert(labeled==biggest_blob)] = 0
        needle_mask_full = np.zeros(needle_mask_empty_closed.shape,"uint8")
        binary_fill_holes(needle_mask_empty_closed,output=needle_mask_full)
        inner_mask = needle_mask_full - needle_mask_empty_closed
        inner_mask_center = center_of_mass(inner_mask)
        # if the center is not found, use the previous detection:
        if not np.isnan(np.sum(inner_mask_center)):
            object_center = [int(x) for x in inner_mask_center]
            object_center = np.array([object_center[1],object_center[0]])
        else: object_center = self.previous_detections["object_center_coordinates"]
        # if the length between the center and the edge is not as the previous detection,
        # use the previous detection for the center:
        needle_mask_full_contour = find_contours(needle_mask_full)[0].astype(int)
        farthest_point, needle_radius = utils.find_farthest_point(object_center, needle_mask_full_contour)
        if self.previous_detections["tip_point"] is not None:
            mean_radius = self.previous_detections["sum_needle_radius"]/self.previous_detections["analysed_frame_count"]
            if np.abs(needle_radius-mean_radius)/mean_radius>NEEDLE_RADIUS_TOLERANCE:
                object_center = self.previous_detections["object_center_coordinates"]
                farthest_point, needle_radius = utils.find_farthest_point(object_center, needle_mask_full_contour)
                if np.abs(needle_radius-mean_radius)/mean_radius>NEEDLE_RADIUS_TOLERANCE:
                    farthest_point = self.previous_detections["analysed_frame_count"]
                    needle_radius = np.sqrt(np.sum(np.square(farthest_point-object_center)))
                    if np.abs(needle_radius-mean_radius)/mean_radius>NEEDLE_RADIUS_TOLERANCE:
                        raise ValueError("There is a problem in the needle part detection")
        needle_area_size = np.sum(needle_mask_full)
        return object_center, farthest_point, needle_radius, needle_area_size

    def find_farthest_point(self, point, contour):
        point = np.array([point[1],point[0]])
        distances = np.sqrt(np.sum(np.square(contour-point),1))
        # take the average of the 50 farthest points
        farthest_points = np.argsort(distances)[-50:]
        farthest_point = np.mean(contour[farthest_points],0).astype(int)
        farthest_point = np.array([farthest_point[1],farthest_point[0]])
        return farthest_point, np.max(distances)

    def clean_mask(self, mask, min_size, remove_needle=True, fill_holes=True):
        mask = mask.astype(bool)
        if remove_needle:
            inner_circle_mask = utils.create_circular_mask(mask.shape, center=self.object_center_coordinates, radius=self.object_needle_radius * 0.9)
            outer_circle_mask = utils.create_circular_mask(mask.shape, center=self.object_center_coordinates, radius=self.object_needle_radius * 2.5)
            mask[inner_circle_mask] = False
            mask[np.invert(outer_circle_mask)] = False
        if fill_holes:
            mask = binary_fill_holes(mask.astype(np.uint8)).astype(bool)
        mask = utils.remove_small_blobs(mask, min_size)
        return mask

    def screen_small_labels(self, labeled, threshold=0.5):
        # calculate the number of pixels in each label, and remove labels with too few pixels, by the being smaller than the mean by a threshold
        label_counts = np.bincount(labeled.ravel())
        too_small = np.arange(len(label_counts))[label_counts < np.mean(label_counts[1:]) * threshold]
        labeled[np.isin(labeled, too_small)] = 0
        return label(labeled)

    def get_spring_parts(self, object_center, spring_middle_part_mask, spring_ends_mask):
        spring_middle_part_labeled, spring_middle_part_num_features = label(spring_middle_part_mask, LABELING_BINARY_STRUCTURE)
        spring_middle_part_labeled, spring_middle_part_num_features = self.screen_small_labels(spring_middle_part_labeled)
        spring_middle_part_centers = np.array(center_of_mass(spring_middle_part_labeled, labels=spring_middle_part_labeled, index=range(1, spring_middle_part_num_features + 1)))
        spring_middle_part_centers = utils.swap_columns(spring_middle_part_centers)
        spring_middle_part_radii = np.sqrt(np.sum(np.square(spring_middle_part_centers - object_center), axis=1))

        spring_ends_labeled, spring_ends_num_features = label(spring_ends_mask, LABELING_BINARY_STRUCTURE)
        spring_ends_centers =np.array(center_of_mass(spring_ends_labeled,labels=spring_ends_labeled,index=range(1,spring_ends_num_features+1)))
        spring_ends_centers = utils.swap_columns(spring_ends_centers)
        spring_ends_radii = np.sqrt(np.sum(np.square(spring_ends_centers - object_center), axis=1))
        fixed_ends_labels = np.array([x for x in range(1, spring_ends_num_features + 1)])[spring_ends_radii < (np.mean(spring_middle_part_radii))]
        free_ends_labels = np.array([x for x in range(1, spring_ends_num_features + 1)])[spring_ends_radii > np.mean(spring_middle_part_radii)]

        fixed_ends_labeled = copy.copy(spring_ends_labeled)
        fixed_ends_labeled[np.invert(np.isin(spring_ends_labeled, fixed_ends_labels))] = 0
        free_ends_labeled = copy.copy(spring_ends_labeled)
        free_ends_labeled[np.invert(np.isin(spring_ends_labeled, free_ends_labels))] = 0
        if np.sum(fixed_ends_labeled) == 0:
            raise ValueError("No springs fixed ends were found."
                             "\n Try to change the threshold (MIN_SPRING_ENDS_SIZE), or the closing structure (SPRING_ENDS_CLOSING) or to collect the color parameters again")
        if np.sum(free_ends_labeled) == 0:
            raise ValueError("No springs free ends were found"
                             "\n Try to change the threshold (MIN_SPRING_ENDS_SIZE), or the closing structure (SPRING_ENDS_CLOSING) or to collect the color parameters again")
        return spring_middle_part_labeled, spring_ends_labeled, fixed_ends_labeled, free_ends_labeled, spring_middle_part_centers, spring_ends_centers

    def create_bundles_labels(self,spring_middle_part_mask,spring_ends_mask,fixed_ends_labeled,closing_structure=BUNDLES_CLOSING):
        #TODO: fix the bug that crushes the program when using connect_blobs
        all_parts_mask = spring_middle_part_mask + spring_ends_mask
        # all_parts_mask = self.close_element(all_parts_mask,closing_structure)
        # all_parts_mask = utils.connect_blobs(all_parts_mask, closing_structure)
        labeled_image, num_features = label(all_parts_mask, generate_binary_structure(2, 2))
        fixed_ends_centers = center_of_mass(fixed_ends_labeled, labels=fixed_ends_labeled,
                                         index=np.unique(fixed_ends_labeled)[1:])
        fixed_ends_centers = np.array([np.array([x, y]).astype("int") for x, y in fixed_ends_centers])
        self.bundles_centers = fixed_ends_centers
        center = np.array([self.object_center_coordinates[1], self.object_center_coordinates[0]])
        tip = np.array([self.tip_point[1],self.tip_point[0]])
        fixed_ends_angles = utils.calc_angles(fixed_ends_centers, center, tip)
        labeled_image_sorted = np.zeros(labeled_image.shape)
        for pnt,angle in zip(fixed_ends_centers,fixed_ends_angles):
            bundle_label = labeled_image[pnt[0],pnt[1]]
            if bundle_label != 0:
                labeled_image_sorted[labeled_image == bundle_label] = angle
        bundles_labels_fixed_centers = [labeled_image[x[0],x[1]] for x in fixed_ends_centers]
        counts = np.array([bundles_labels_fixed_centers.count(x) for x in bundles_labels_fixed_centers])
        melted_bundles = np.unique(np.array(bundles_labels_fixed_centers)[counts > 1])
        for bad_label in melted_bundles:
            labeled_image_sorted[labeled_image==bad_label] = 0
        return labeled_image_sorted, fixed_ends_angles

    def assign_ends_to_bundles(self,bundles_labeled, fixed_ends_labeled, free_ends_labeled, spring_middle_part_centers, spring_ends_labeled):

        fixed_ends_centers = utils.swap_columns(np.array(
            center_of_mass(spring_ends_labeled, labels=spring_ends_labeled, index=list(np.unique(fixed_ends_labeled))[1:]), "int"))
        free_ends_centers = utils.swap_columns(np.array(
            center_of_mass(spring_ends_labeled, labels=spring_ends_labeled, index=list(np.unique(free_ends_labeled))[1:]), "int"))
        fixed_ends_bundles_labels = []
        free_ends_bundles_labels = []
        spring_middle_part_bundles_labels = []
        for x1, y1 in fixed_ends_centers:
            fixed_ends_bundles_labels.append(bundles_labeled[y1, x1])
        for x1, y1 in free_ends_centers:
            free_ends_bundles_labels.append(bundles_labeled[y1, x1])
        for x1, y1 in spring_middle_part_centers.astype("int"):
            spring_middle_part_bundles_labels.append(bundles_labeled[y1, x1])
        bundles_labels = self.screen_bundles(fixed_ends_bundles_labels, free_ends_bundles_labels, spring_middle_part_bundles_labels)
        return bundles_labels

    def screen_bundles(self, fixed_labels, free_labels, spring_middle_part_labels):
        counts = np.array([fixed_labels.count(x) for x in fixed_labels])
        melted_bundles = np.unique(np.array(fixed_labels)[counts>1])
        for label in melted_bundles:
            fixed_labels = list(filter((label).__ne__, fixed_labels))
        ### Only bundles which has all 3 parts (free,fixed,spring_middle_part) will be consider real bundle
        real_springs_labels=list(set(fixed_labels).intersection(set(free_labels)).intersection(set(spring_middle_part_labels)))
        return real_springs_labels

    # def remove_labels(self, labels_to_keep, labeled_image):
    #     labeled_image[np.isin(labeled_image, labels_to_keep, invert=True)] = 0
    #     self.bundles_labels = labels_to_keep
    #     return labeled_image

    # def clear_labels(self, labeled_image, mask):
    #     labeled_image[mask == 0] = 0
    #     return labeled_image

    def find_boundaries_touches(self, labeled1, labeled2, bundles_labeled):
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

    def remove_duplicates(self, labels):
        # turn duplicates labels to 0
        labels_array = np.array(labels)
        counts = np.array([labels.count(x) for x in labels])
        labels_array[counts > 1] = 0
        return list(labels_array)

