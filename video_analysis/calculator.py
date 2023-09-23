import pickle
import os
import numpy as np
from scipy.ndimage import maximum_filter
# local imports:
from video_analysis import utils


ANTS_SPRINGS_OVERLAP_SIZE = 5


class Calculation:
    def __init__(self, parameters, previous_detections, springs, ants):
        self.parameters = parameters
        self.previous_detections = previous_detections
        self.ants = ants
        self.springs = springs
        self.max_ants_number = self.parameters["max_ants_number"]
        self.make_ants_centers()
        self.initiate_springs_angles_matrix()
        self.springs_angles_ordered = self.match_springs(self.springs_angles_reference_order)
        self.N_ants_around_springs, self.size_ants_around_springs = self.occupied_springs(self.springs_angles_ordered)
        self.fixed_ends_coordinates_x, self.fixed_ends_coordinates_y, self.free_ends_coordinates_x, \
            self.free_ends_coordinates_y, self.needle_part_coordinates_x, self.needle_part_coordinates_y =\
            self.reorder_coordinates(self.springs_angles_ordered)

    def make_ants_centers(self):
        ants_centers_x = self.ants.ants_centers[:, 1]
        ants_centers_y = self.ants.ants_centers[:, 0]
        ants_centers_x = ants_centers_x.reshape(1, ants_centers_x.shape[0])
        ants_centers_y = ants_centers_y.reshape(1, ants_centers_y.shape[0])
        empty_row = np.empty((1, self.max_ants_number))
        empty_row[:] = np.nan
        self.ants_centers_x = np.copy(empty_row)
        self.ants_centers_y = np.copy(empty_row)
        self.ants_centers_x[:, :ants_centers_x.shape[1]] = ants_centers_x
        self.ants_centers_y[:, :ants_centers_y.shape[1]] = ants_centers_y

    def initiate_springs_angles_matrix(self):
        #TODO: make the 3rd line in this function to be functional (if np.sum(np.abs(linspace_angles-self.previous_detections[2]))==0:)
        linspace_angles = np.linspace(-np.pi, np.pi - np.pi / self.springs.n_springs, self.springs.n_springs).reshape(1, self.springs.n_springs)
        if self.previous_detections["springs_angles_reference_order"] is not None:
            if np.sum(np.abs(linspace_angles - self.previous_detections["springs_angles_reference_order"])) == 0:
                if len(self.springs.bundles_labels) == self.springs.n_springs:
                    self.springs_angles_reference_order = np.sort(self.springs.bundles_labels).reshape(1, self.springs.n_springs)
                else:
                    self.springs_angles_reference_order = self.previous_detections["springs_angles_reference_order"]
            else:
                self.springs_angles_reference_order = self.previous_detections["springs_angles_reference_order"]
        elif len(self.springs.bundles_labels) == self.springs.n_springs:
            self.springs_angles_reference_order = np.sort(self.springs.bundles_labels).reshape(1, self.springs.n_springs)
        else:
            self.springs_angles_reference_order = linspace_angles
            min_index = np.argmin(np.abs(np.array(self.springs.bundles_labels)))
            diff = np.abs(np.array(self.springs.bundles_labels)[min_index])
            self.springs_angles_reference_order -= diff

    def match_springs(self, springs_angles_reference_order):
        subtraction_combinations = np.cos(springs_angles_reference_order[0][:, np.newaxis] - self.springs.bundles_labels)
        sort = np.argsort(subtraction_combinations, axis=1)[0]
        current_springs_angles = list(np.array(self.springs.bundles_labels)[sort])
        assigned_angles_classes = []
        for value in current_springs_angles:
            cos_diff = np.cos(springs_angles_reference_order - value)
            assigned_class = np.argsort(cos_diff)[0][-1]
            if assigned_class in assigned_angles_classes:
                assigned_class = np.argsort(cos_diff)[0][-2]
            assigned_angles_classes.append(assigned_class)
        new_springs_row = np.array([np.nan for x in range(self.springs.n_springs)])
        for angle_class, angle in zip(assigned_angles_classes, current_springs_angles):
            new_springs_row[angle_class] = angle
        return new_springs_row

    def occupied_springs(self, springs_order):
        dilated_ends = maximum_filter(self.springs.free_ends_labeled, ANTS_SPRINGS_OVERLAP_SIZE)
        labeled_ants_cropped = utils.crop_frame_by_coordinates(self.ants.labeled_ants, self.springs.object_crop_coordinates)
        dilated_ants = maximum_filter(labeled_ants_cropped, ANTS_SPRINGS_OVERLAP_SIZE)
        joints_labels = np.unique(dilated_ends[((dilated_ends != 0) * (dilated_ants != 0))])

        spring_ends_occupied = []
        for label in joints_labels:
            bundle_label = np.unique(self.springs.bundles_labeled[dilated_ends == label])
            bundle_label = bundle_label[np.where(bundle_label != 0)]
            self.springs.bundles_labeled[dilated_ends == label] = bundle_label[0]
            spring_ends_occupied.append(bundle_label[0])

        N_ants_around_springs = np.zeros(self.springs.n_springs).astype(np.uint8)
        size_ants_around_springs = np.zeros(self.springs.n_springs).astype(np.uint32)
        self.ants_attached_labels = np.full(self.max_ants_number, 0).astype(np.uint8)
        self.ants_attached_forgotten_labels = np.full(self.max_ants_number, 0).astype(np.uint8)
        for spring_end in spring_ends_occupied:
            spring_position = springs_order == spring_end
            ants_on_spring_end = np.unique(dilated_ants[(self.springs.bundles_labeled == spring_end) * (dilated_ends != 0)])[1:]
            if np.sum(spring_position) == 1:
                springs_forgotten = self.ants_attached_labels[ants_on_spring_end - 1]
                ants_on_two_springs = ants_on_spring_end[springs_forgotten != 0]
                springs_forgotten = springs_forgotten[springs_forgotten != 0]
                self.ants_attached_forgotten_labels[ants_on_two_springs - 1] = springs_forgotten
                self.ants_attached_labels[ants_on_spring_end - 1] = np.arange(1, self.springs.n_springs + 1)[spring_position][0]
            N_ants_around_springs[spring_position] = len(ants_on_spring_end)
            size_ants_around_springs[spring_position] = np.sum(np.isin(self.springs.labeled_ants, ants_on_spring_end))
        return N_ants_around_springs.reshape(1, self.springs.n_springs), size_ants_around_springs.reshape(1, self.springs.n_springs)

    def reorder_coordinates(self, springs_order):
        fixed_ends_x, fixed_ends_y, free_ends_x, free_ends_y = \
            np.empty(self.springs.n_springs), np.empty(self.springs.n_springs), np.empty(self.springs.n_springs), np.empty(self.springs.n_springs)
        fixed_ends_x[:] = np.nan
        fixed_ends_y[:] = np.nan
        free_ends_x[:] = np.nan
        free_ends_y[:] = np.nan
        for label in self.springs.bundles_labels:
            if label != 0:
                index_springs = springs_order == label
                if label in self.springs.fixed_ends_edges_bundles_labels:
                    index = self.springs.fixed_ends_edges_bundles_labels.index(label)
                    fixed_ends_x[index_springs] = self.springs.fixed_ends_edges_centers[index, 0]
                    fixed_ends_y[index_springs] = self.springs.fixed_ends_edges_centers[index, 1]
                if label in self.springs.free_ends_edges_bundles_labels:
                    index = self.springs.free_ends_edges_bundles_labels.index(label)
                    free_ends_x[index_springs] = self.springs.free_ends_edges_centers[index, 0]
                    free_ends_y[index_springs] = self.springs.free_ends_edges_centers[index, 1]

        blue_part_x, blue_part_y = np.empty(2), np.empty(2)
        blue_part_x[:] = np.nan
        blue_part_y[:] = np.nan
        blue_part_x[0] = self.springs.object_center_coordinates[0]
        blue_part_x[1] = self.springs.tip_point[0]
        blue_part_y[0] = self.springs.object_center_coordinates[1]
        blue_part_y[1] = self.springs.tip_point[1]
        return fixed_ends_x.reshape(1, self.springs.n_springs), fixed_ends_y.reshape(1, self.springs.n_springs), free_ends_x.reshape(1,
            self.springs.n_springs), free_ends_y.reshape(1, self.springs.n_springs), blue_part_x.reshape(1, 2), blue_part_y.reshape(1, 2)


def save_data(output_path, snapshot_data, parameters, calculations=None, continue_from_last=False):
    max_ants = parameters["max_ants_number"]
    n_springs = parameters["n_springs"]
    if calculations is None:
        empty_springs = np.full((1, n_springs), np.nan)
        empty_2_values = np.full((1, 2), np.nan)
        empty_ants = np.full((1, max_ants), np.nan)
        arrays = [empty_springs for _ in range(6)] + [empty_2_values, empty_2_values] + [empty_ants for _ in range(4)]
    else:
        arrays = [calculations.N_ants_around_springs, calculations.size_ants_around_springs,
                  calculations.fixed_ends_coordinates_x, calculations.fixed_ends_coordinates_y,
                  calculations.free_ends_coordinates_x, calculations.free_ends_coordinates_y,
                  calculations.needle_part_coordinates_x, calculations.needle_part_coordinates_y,
                  calculations.ants_centers_x, calculations.ants_centers_y,
                  calculations.ants_attached_labels.reshape(1, max_ants),
                  calculations.ants_attached_forgotten_labels.reshape(1, max_ants)]
    names = ["N_ants_around_springs", "size_ants_around_springs",
             "fixed_ends_coordinates_x", "fixed_ends_coordinates_y", "free_ends_coordinates_x",
             "free_ends_coordinates_y", "needle_part_coordinates_x", "needle_part_coordinates_y",
             "ants_centers_x", "ants_centers_y", "ants_attached_labels", "ants_attached_forgotten_labels"]
    utils.save_data(output_path, arrays, names, snapshot_data, continue_from_last)
    pickle.dump(snapshot_data, open(os.path.join(output_path, f'snap_data_{snapshot_data["current_time"]}.pickle'), "wb"))
