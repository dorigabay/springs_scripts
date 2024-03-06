import cv2
import copy
import numpy as np
from scipy.ndimage import label, generate_binary_structure, center_of_mass, maximum_filter, binary_fill_holes, binary_erosion, binary_dilation
from skimage.measure import regionprops, find_contours
# local imports:
import utils


class Springs:
    def __init__(self, parameters, frame, checkpoint):
        self.parameters = parameters
        self.frame = frame
        self.checkpoint = checkpoint
        frame_cropped, self.object_crop_coordinates = self.prepare_frame(self.frame)
        springs_properties = self.get_properties(frame_cropped)
        self.transform_coordinates(self.frame, springs_properties)

    def prepare_frame(self, frame):
        if (self.checkpoint.skipped_frames >= 25) or self.checkpoint.frame_count == 0:
            object_crop_coordinates = np.array([0, self.parameters["RESOLUTION"][0], 0, self.parameters["RESOLUTION"][1]])
        else:
            object_crop_coordinates = utils.create_box_coordinates(self.checkpoint.object_center_coordinates, self.parameters["OCM"])[0]
        frame_cropped = utils.crop_frame(frame, object_crop_coordinates)
        frame_cropped = utils.process_image(frame_cropped, alpha=self.parameters["NEUTRALIZE_COLOUR_ALPHA"],
                                            blur_kernel=self.parameters["NEUTRALIZE_COLOUR_BETA"],
                                            gradiant_threshold=self.parameters["GRADIANT_THRESHOLD"])
        return frame_cropped, object_crop_coordinates

    def get_properties(self, frame):
        color_masks, whole_object_mask = utils.mask_object_colors(frame, self.parameters)
        self.object_center_coordinates, self.needle_end, self.object_needle_radius, self.object_needle_area_size = self.detect_needle(color_masks["g"])
        spring_ends_mask = utils.clean_mask(color_masks["r"], self.parameters["MIN_SPRING_ENDS_SIZE"],
                                            circle_center_remove=self.object_center_coordinates, circle_radius_remove=self.object_needle_radius)
        spring_middle_part_mask = utils.clean_mask(color_masks["b"], self.parameters["MIN_SPRING_MIDDLE_PART_SIZE"], self.parameters["SPRINGS_MIDDLE_PART_OPENING"],
                                                   circle_center_remove=self.object_center_coordinates, circle_radius_remove=self.object_needle_radius)
        spring_middle_part_labeled, spring_ends_labeled, self.fixed_ends_labeled, self.free_ends_labeled, spring_middle_part_centers, spring_ends_centers = \
            self.get_parts(self.object_center_coordinates, spring_middle_part_mask, spring_ends_mask)
        self.bundles_labeled, bundles_labels = self.create_bundles(spring_middle_part_mask, spring_ends_mask, self.fixed_ends_labeled)
        self.bundles_labels = self.assign_parts_to_bundles(self.bundles_labeled, self.fixed_ends_labeled, self.free_ends_labeled, spring_middle_part_centers, spring_ends_labeled)
        self.bundles_labeled[np.isin(self.bundles_labeled, self.bundles_labels, invert=True)] = 0  # remove bundles that has no ends
        self.free_ends_labeled[self.bundles_labeled == 0] = 0
        self.fixed_ends_labeled[self.bundles_labeled == 0] = 0
        fixed_ends_edges_centers, self.fixed_ends_edges_bundles_labels = self.find_fixed_end_edge(self.fixed_ends_labeled, self.object_center_coordinates, self.bundles_labeled)
        free_ends_edges_centers, self.free_ends_edges_bundles_labels = self.find_free_end_edge(self.free_ends_labeled, spring_middle_part_labeled, self.bundles_labeled)
        properties = [whole_object_mask, spring_middle_part_labeled, spring_ends_labeled,
                      spring_middle_part_centers, spring_ends_centers, fixed_ends_edges_centers, free_ends_edges_centers]
        return properties

    def detect_needle(self, needle_mask):
        needle_mask_empty_closed = needle_mask
        labeled, _ = label(needle_mask_empty_closed)
        needle_prop = regionprops(labeled)
        biggest_blob = needle_prop[np.argmax([x.area for x in needle_prop])].label
        needle_mask_empty_closed[np.invert(labeled == biggest_blob)] = 0
        needle_mask_full = np.zeros(needle_mask_empty_closed.shape, "uint8")
        binary_fill_holes(needle_mask_empty_closed, output=needle_mask_full)
        inner_mask = needle_mask_full - needle_mask_empty_closed
        inner_mask_center = center_of_mass(inner_mask)
        if not np.isnan(np.sum(inner_mask_center)):
            object_center = np.array([list(inner_mask_center)[1], list(inner_mask_center)[0]])
        else:
            object_center = self.checkpoint.object_center_coordinates
        # If the length between the center and the edge aren't as the previous detection,
        # use the previous detection for the center, or the edge:
        needle_mask_full_contour = find_contours(needle_mask_full)[0]
        farthest_point, needle_radius = utils.get_farthest_point(object_center, needle_mask_full_contour, percentile=80)
        if self.checkpoint.tip_point is not None:
            mean_radius = self.checkpoint.sum_needle_radius / self.checkpoint.analysed_frame_count
            if np.abs(needle_radius - mean_radius) / mean_radius > self.parameters["NEEDLE_RADIUS_TOLERANCE"]:
                object_center = self.checkpoint.object_center_coordinates
                farthest_point, needle_radius = utils.get_farthest_point(object_center, needle_mask_full_contour, percentile=80)
                if np.abs(needle_radius - mean_radius) / mean_radius > self.parameters["NEEDLE_RADIUS_TOLERANCE"]:
                    farthest_point = self.checkpoint.tip_point
                    needle_radius = np.sqrt(np.sum(np.square(farthest_point - object_center)))
                    if np.abs(needle_radius - mean_radius) / mean_radius > self.parameters["NEEDLE_RADIUS_TOLERANCE"]:
                        raise ValueError("There is a problem in the needle part detection")
        needle_area_size = np.sum(needle_mask_full)
        return object_center, farthest_point, needle_radius, needle_area_size

    def screen_labels(self, labeled, threshold=0.5):
        # Calculate label sizes, and remove labels smaller than the mean size:
        label_counts = np.bincount(labeled.ravel())
        too_small = np.arange(len(label_counts))[label_counts < np.mean(label_counts[1:]) * threshold]
        labeled[np.isin(labeled, too_small)] = 0
        return label(labeled)

    def get_parts(self, object_center, spring_middle_part_mask, spring_ends_mask):
        spring_middle_part_labeled, spring_middle_part_num_features = label(spring_middle_part_mask, self.parameters["LABELING_BINARY_STRUCTURE"])
        spring_middle_part_labeled, spring_middle_part_num_features = self.screen_labels(spring_middle_part_labeled)
        spring_middle_part_centers = np.array(center_of_mass(spring_middle_part_labeled, labels=spring_middle_part_labeled, index=range(1, spring_middle_part_num_features + 1)))
        spring_middle_part_centers = utils.swap_columns(spring_middle_part_centers)
        spring_middle_part_radii = np.sqrt(np.sum(np.square(spring_middle_part_centers - object_center), axis=1))
        spring_ends_labeled, spring_ends_num_features = label(spring_ends_mask, self.parameters["LABELING_BINARY_STRUCTURE"])
        spring_ends_centers = np.array(center_of_mass(spring_ends_labeled, labels=spring_ends_labeled, index=range(1, spring_ends_num_features + 1)))
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

    def create_bundles(self, spring_middle_part_mask, spring_ends_mask, fixed_ends_labeled):
        all_parts_mask = spring_middle_part_mask + spring_ends_mask
        all_parts_mask = utils.connect_blobs(all_parts_mask, self.parameters["BUNDLES_CLOSING_SIZE"])
        labeled_image, num_features = label(all_parts_mask, generate_binary_structure(2, 2))
        fixed_ends_centers = center_of_mass(fixed_ends_labeled, labels=fixed_ends_labeled, index=np.unique(fixed_ends_labeled)[1:])
        fixed_ends_centers = np.array([np.array([x, y]).astype("int") for x, y in fixed_ends_centers])
        self.bundles_centers = fixed_ends_centers
        center = np.array([self.object_center_coordinates[1], self.object_center_coordinates[0]])
        tip = np.array([self.needle_end[1], self.needle_end[0]])
        fixed_ends_angles = utils.calc_angles(fixed_ends_centers, center, tip)
        labeled_image_sorted = np.zeros(labeled_image.shape)
        for pnt, angle in zip(fixed_ends_centers, fixed_ends_angles):
            bundle_label = labeled_image[pnt[0], pnt[1]]
            if bundle_label != 0:
                labeled_image_sorted[labeled_image == bundle_label] = angle
        bundles_labels_fixed_centers = [labeled_image[x[0], x[1]] for x in fixed_ends_centers]
        counts = np.array([bundles_labels_fixed_centers.count(x) for x in bundles_labels_fixed_centers])
        melted_bundles = np.unique(np.array(bundles_labels_fixed_centers)[counts > 1])
        for bad_label in melted_bundles:
            labeled_image_sorted[labeled_image == bad_label] = 0
        return labeled_image_sorted, fixed_ends_angles

    def assign_parts_to_bundles(self, bundles_labeled, fixed_ends_labeled, free_ends_labeled, spring_middle_part_centers, spring_ends_labeled):
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
        melted_bundles = np.unique(np.array(fixed_labels)[counts > 1])
        for label in melted_bundles:
            fixed_labels = list(filter((label).__ne__, fixed_labels))
        ### Only bundles which has all 3 parts (free,fixed,spring_middle_part) will be consider real bundle
        real_springs_labels = list(set(fixed_labels).intersection(set(free_labels)).intersection(set(spring_middle_part_labels)))
        return real_springs_labels

    def find_fixed_end_edge(self, labeled, object_center, bundles_labeled):
        edge_points = np.full((len(np.unique(labeled)[1:]), 2), 0, np.float32)
        for count, lab in enumerate(np.unique(labeled)[1:]):
            label_mask = labeled == lab
            label_contour = find_contours(label_mask)[0]
            farthest_point, length = utils.get_farthest_point(object_center, label_contour, percentile=70, inverse=True)
            edge_points[count] = farthest_point
        overlap_bundles_labels = [bundles_labeled[int(y), int(x)] for x, y in edge_points]
        overlap_bundles_labels = self.remove_duplicates(overlap_bundles_labels)
        return edge_points, overlap_bundles_labels

    def find_free_end_edge(self, labeled1, labeled2, bundles_labeled):
        maximum_filter_labeled1 = maximum_filter(labeled1, self.parameters["SPRINGS_PARTS_OVERLAP_SIZE"])
        maximum_filter_labeled2 = maximum_filter(labeled2, self.parameters["SPRINGS_PARTS_OVERLAP_SIZE"])
        overlap_labeled = np.full(maximum_filter_labeled1.shape, 0, np.float32)
        boolean_overlap = (maximum_filter_labeled1 != 0) * (maximum_filter_labeled2 != 0)
        overlap_labeled[boolean_overlap] = maximum_filter_labeled1[boolean_overlap]
        overlap_labels = list(np.unique(overlap_labeled))[1:]
        overlap_centers = utils.swap_columns(np.array(center_of_mass(overlap_labeled, labels=overlap_labeled, index=overlap_labels)))
        overlap_bundles_labels = [bundles_labeled[int(y), int(x)] for x, y in overlap_centers]
        overlap_bundles_labels = self.remove_duplicates(overlap_bundles_labels)
        return overlap_centers, overlap_bundles_labels

    def remove_duplicates(self, labels):
        labels_array = np.array(labels)
        counts = np.array([labels.count(x) for x in labels])
        labels_array[counts > 1] = 0
        return list(labels_array)

    def transform_coordinates(self, frame, springs_properties):
        whole_object_mask, spring_middle_part_labeled, spring_ends_labeled, \
            spring_middle_part_centers, spring_ends_centers, fixed_ends_edges_centers, free_ends_edges_centers = springs_properties
        y_addition, x_addition = self.object_crop_coordinates[0], self.object_crop_coordinates[2]
        self.fixed_ends_edges_centers = fixed_ends_edges_centers + [x_addition, y_addition]
        self.free_ends_edges_centers = free_ends_edges_centers + [x_addition, y_addition]
        self.spring_middle_part_centers = spring_middle_part_centers + [x_addition, y_addition]  # used only for visualization
        self.spring_ends_centers = spring_ends_centers + [x_addition, y_addition]  # used only for visualization
        self.object_center_coordinates = self.object_center_coordinates + [x_addition, y_addition]
        self.needle_end = self.needle_end + [x_addition, y_addition]
        self.whole_object_mask = np.full(frame.shape[:2], False, dtype=bool)
        self.whole_object_mask[self.object_crop_coordinates[0]:self.object_crop_coordinates[1],
                               self.object_crop_coordinates[2]:self.object_crop_coordinates[3]] = whole_object_mask > 0
