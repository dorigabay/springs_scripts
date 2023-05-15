import copy
import numpy as np
from scipy.ndimage import  maximum_filter


ANTS_SPRINGS_OVERLAP_SIZE = 10


class Calculation:
    def __init__(self, springs, ants, previous_detections=None):
        self.n_springs = springs.n_springs
        self.blue_area_size = springs.blue_area_size
        self.make_ants_centers(ants)
        self.initiate_springs_angles_matrix(springs, previous_detections)
        self.match_springs(springs, self.springs_angles_reference_order)
        self.N_ants_around_springs, self.size_ants_around_springs = self.occupied_springs(springs, ants, self.springs_angles_ordered)
        self.fixed_ends_coordinates_x, self.fixed_ends_coordinates_y, self.free_ends_coordinates_x,\
            self.free_ends_coordinates_y, self.blue_part_coordinates_x, self.blue_part_coordinates_y = self.reorder_coordinates(springs, self.springs_angles_ordered)

    def make_ants_centers(self, ants):
        ants_centers_x = ants.ants_centers[:, 1]
        ants_centers_y = ants.ants_centers[:, 0]
        ants_centers_x = ants_centers_x.reshape(1, ants_centers_x.shape[0])
        ants_centers_y = ants_centers_y.reshape(1, ants_centers_y.shape[0])
        empty_row = np.empty((1, 100))
        empty_row[:] = np.nan
        self.ants_centers_x = copy.copy(empty_row)
        self.ants_centers_y = copy.copy(empty_row)
        self.ants_centers_x[:, :ants_centers_x.shape[1]] = ants_centers_x
        self.ants_centers_y[:, :ants_centers_y.shape[1]] = ants_centers_y

    def initiate_springs_angles_matrix(self,springs, previous_detections):
        if previous_detections is None:
            if len(springs.bundles_labels)!=self.n_springs:
                exit("First frame should have exactly 20 springs. Different number of springs were detect,"
                     " please start the process from a different frame.")
            else:
                self.springs_angles_reference_order = np.sort(springs.bundles_labels).reshape(1,self.n_springs)
        else:
            self.springs_angles_reference_order = previous_detections[4]

    def match_springs(self, springs, springs_angles_reference_order):
        current_springs_angles = list(springs.bundles_labels)
        assigned_angles_classes = []
        for value in current_springs_angles:
            cos_diff = np.cos(springs_angles_reference_order - value)
            assigned_class = np.argmax(cos_diff)
            assigned_angles_classes.append(assigned_class)
        new_springs_row = np.array([np.nan for x in range(self.n_springs)])
        for angle_class, angle in zip(assigned_angles_classes, current_springs_angles):
            new_springs_row[angle_class] = angle
        self.springs_angles_ordered =  new_springs_row

        a = [springs.bundles_labels,springs.blue_area_size,springs.free_ends_labeled,springs.fixed_ends_edges_bundles_labels,
             springs.fixed_ends_edges_centers,springs.free_ends_edges_bundles_labels,springs.free_ends_edges_centers,
             springs.object_center,springs.tip_point,]
    # def calc_springs_lengths(self, springs, springs_order):# fixed_ends_coor, free_ends_coor, fixed_ends_labels, free_ends_labels):
    #     springs_length = np.empty((20))
    #     springs_length[:] = np.nan
    #     found_in_both = list(set(springs.fixed_ends_edges_bundles_labels).
    #                          intersection(set(springs.free_ends_edges_bundles_labels)))
    #     for label in found_in_both:
    #         if label != 0:
    #             coor_free = springs.free_ends_edges_centers[springs.free_ends_edges_bundles_labels.index(label)]
    #             coor_fixed = springs.fixed_ends_edges_centers[springs.fixed_ends_edges_bundles_labels.index(label)]
    #             distance = np.sqrt(np.sum(np.square(coor_free - coor_fixed)))
    #             index_springs = springs_order == label
    #             springs_length[index_springs] = distance
    #     return springs_length.reshape(1,20)

    def ants_centers(self, ants):
        self.ants_centers_coordinates = ants.ants_centers

    def occupied_springs(self, springs, ants, springs_order):
        dialated_ends = maximum_filter(springs.free_ends_labeled,ANTS_SPRINGS_OVERLAP_SIZE)
        dialated_ants = maximum_filter(ants.labeled_ants,ANTS_SPRINGS_OVERLAP_SIZE)
        joints = ((dialated_ends != 0) * (dialated_ants != 0))
        self.joints = joints
        ends_occupied = np.unique(springs.bundles_labeled[joints])
        ends_occupied = ends_occupied[np.where(ends_occupied != 0)]
        N_ants_around_springs = np.zeros(self.n_springs)
        size_ants_around_springs = np.zeros(self.n_springs)
        self.ants_attached_labels = [np.nan for x in range(self.n_springs)]
        for end in ends_occupied:
            spring_position = springs_order==end
            ends_i = np.unique(dialated_ants[(springs.bundles_labeled == end)*(dialated_ends != 0)])[1:]
            if np.sum(spring_position)==1:
                self.ants_attached_labels[np.arange(self.n_springs)[spring_position][0]] = ends_i
            N_ants_around_springs[spring_position] = len(ends_i)
            size_ants_around_springs[spring_position] = np.sum(np.isin(ants.labeled_ants,ends_i))
        return N_ants_around_springs.reshape(1,self.n_springs), size_ants_around_springs.reshape(1,self.n_springs)

    def reorder_coordinates(self,springs,springs_order):
        fixed_ends_x,fixed_ends_y,free_ends_x,free_ends_y = np.empty(self.n_springs),np.empty(self.n_springs),np.empty(self.n_springs),np.empty(self.n_springs)
        fixed_ends_x[:] = np.nan
        fixed_ends_y[:] = np.nan
        free_ends_x[:] = np.nan
        free_ends_y[:] = np.nan
        for label in springs.bundles_labels:
            if label !=0:
                index_springs = springs_order == label
                if label in springs.fixed_ends_edges_bundles_labels:
                    index = springs.fixed_ends_edges_bundles_labels.index(label)
                    fixed_ends_x[index_springs] = springs.fixed_ends_edges_centers[index,0]
                    fixed_ends_y[index_springs] = springs.fixed_ends_edges_centers[index,1]
                if label in springs.free_ends_edges_bundles_labels:
                    index = springs.free_ends_edges_bundles_labels.index(label)
                    free_ends_x[index_springs] = springs.free_ends_edges_centers[index,0]
                    free_ends_y[index_springs] = springs.free_ends_edges_centers[index,1]

        blue_part_x, blue_part_y = np.empty(2),np.empty(2)
        blue_part_x[:] = np.nan
        blue_part_y[:] = np.nan
        blue_part_x[0] = springs.object_center[0]
        blue_part_x[1] = springs.tip_point[0]
        blue_part_y[0] = springs.object_center[1]
        blue_part_y[1] = springs.tip_point[1]
        return fixed_ends_x.reshape(1,self.n_springs),fixed_ends_y.reshape(1,self.n_springs),free_ends_x.reshape(1,self.n_springs),\
            free_ends_y.reshape(1,self.n_springs),blue_part_x.reshape(1,2),blue_part_y.reshape(1,2)

    # def stack_coordinates(self, fixed_ends_x, fixed_ends_y, free_ends_x, free_ends_y, blue_part_x, blue_part_y):
    #     self.fixed_ends_coordinates_x = np.vstack([self.fixed_ends_coordinates_x, fixed_ends_x])
    #     self.fixed_ends_coordinates_y = np.vstack([self.fixed_ends_coordinates_y, fixed_ends_y])
    #     self.free_ends_coordinates_x = np.vstack([self.free_ends_coordinates_x, free_ends_x])
    #     self.free_ends_coordinates_y = np.vstack([self.free_ends_coordinates_y, free_ends_y])
    #     self.blue_part_coordinates_x = np.vstack([self.blue_part_coordinates_x, blue_part_x])
    #     self.blue_part_coordinates_y = np.vstack([self.blue_part_coordinates_y, blue_part_y])

    def calc_springs_angles(self,springs, springs_order):
        nest_direction = np.array([springs.object_center[0], springs.object_center[1] - 100])
        fixed_ends_angles_to_nest = springs.calc_angles(springs.fixed_ends_edges_centers,springs.object_center,nest_direction)
        free_ends_angles_to_object = springs.calc_angles(springs.free_ends_edges_centers,springs.object_center,springs.tip_point)
        fixed_ends_angles_to_object = springs.calc_angles(springs.fixed_ends_edges_centers,springs.object_center,springs.tip_point)
        nan_array = np.empty((self.n_springs))
        nan_array[:] = np.nan
        angles_to_nest, angles_to_object_free, angles_to_object_fixed = copy.copy(nan_array), copy.copy(nan_array), copy.copy(nan_array)
        for label in springs.bundles_labels:
            if label !=0:
                index_springs = springs_order == label
                if label in springs.fixed_ends_edges_bundles_labels:
                    angles_to_nest[index_springs] = fixed_ends_angles_to_nest[springs.fixed_ends_edges_bundles_labels==label]
                    angles_to_object_fixed[index_springs] = fixed_ends_angles_to_object[springs.fixed_ends_edges_bundles_labels==label]
                if label in springs.free_ends_edges_bundles_labels:
                    angles_to_object_free[index_springs] = free_ends_angles_to_object[springs.free_ends_edges_bundles_labels==label]
        angles_to_nest,angles_to_object_free,angles_to_object_fixed =\
            angles_to_nest.reshape(1,self.n_springs),angles_to_object_free.reshape(1,self.n_springs),angles_to_object_fixed.reshape(1,self.n_springs)
        return angles_to_nest, angles_to_object_free, angles_to_object_fixed

    # def add_blank_row(self,number_of_rows):
    #     empty_row = np.empty((self.n_springs))
    #     empty_row[:] = np.nan
    #     ref = number_of_rows
    #     #for the self matrices, if shape[0] is as count then do nothing, else add a row of nans.
    #     # if ref != self.springs_length.shape[0]:
    #     #     self.springs_length = np.vstack((self.springs_length,empty_row))
    #     # else:
    #     #     self.N_ants_around_springs[-1,:] = empty_row
    #
    #     if ref != self.N_ants_around_springs.shape[0]:
    #         self.N_ants_around_springs = np.vstack((self.N_ants_around_springs,empty_row))
    #     else:
    #         self.N_ants_around_springs[-1,:] = empty_row
    #
    #     if ref != self.size_ants_around_springs.shape[0]:
    #         self.size_ants_around_springs = np.vstack((self.size_ants_around_springs,empty_row))
    #     else:
    #         self.size_ants_around_springs[-1,:] = empty_row
    #
    #     # if ref != self.angles_to_nest.shape[0]:
    #     #     self.angles_to_nest = np.vstack((self.angles_to_nest,empty_row))
    #     # else:
    #     #     self.angles_to_nest[-1,:] = empty_row
    #     #
    #     # if ref != self.angles_to_object_free.shape[0]:
    #     #     self.angles_to_object_free = np.vstack((self.angles_to_object_free,empty_row))
    #     # else:
    #     #     self.angles_to_object_free[-1,:] = empty_row
    #     #
    #     # if ref != self.angles_to_object_fixed.shape[0]:
    #     #     self.angles_to_object_fixed = np.vstack((self.angles_to_object_fixed,empty_row))
    #     # else:
    #     #     self.angles_to_object_fixed[-1,:] = empty_row
    #     if ref != self.fixed_ends_coordinates_x.shape[0]:
    #         self.fixed_ends_coordinates_x = np.vstack((self.fixed_ends_coordinates_x,empty_row))
    #         self.fixed_ends_coordinates_y = np.vstack((self.fixed_ends_coordinates_y,empty_row))
    #         self.free_ends_coordinates_x = np.vstack((self.free_ends_coordinates_x,empty_row))
    #         self.free_ends_coordinates_y = np.vstack((self.free_ends_coordinates_y,empty_row))
    #         blue_part_empty_row = np.empty((2))
    #         blue_part_empty_row[:] = np.nan
    #         self.blue_part_coordinates_x = np.vstack((self.blue_part_coordinates_x,blue_part_empty_row))
    #         self.blue_part_coordinates_y = np.vstack((self.blue_part_coordinates_y,blue_part_empty_row))
    #
    #
    # def clear_data(calculations):
    #     # calculations.springs_length = calculations.springs_length[-1,:]
    #     calculations.N_ants_around_springs = calculations.N_ants_around_springs[-1,:]
    #     calculations.size_ants_around_springs = calculations.size_ants_around_springs[-1,:]
    #     # calculations.angles_to_nest = calculations.angles_to_nest[-1,:]
    #     # calculations.angles_to_object_free = calculations.angles_to_object_free[-1,:]
    #     # calculations.angles_to_object_fixed = calculations.angles_to_object_fixed[-1,:]
    #     calculations.fixed_ends_coordinates_x = calculations.fixed_ends_coordinates_x[-1,:]
    #     calculations.fixed_ends_coordinates_y = calculations.fixed_ends_coordinates_y[-1,:]
    #     calculations.free_ends_coordinates_x = calculations.free_ends_coordinates_x[-1,:]
    #     calculations.free_ends_coordinates_y = calculations.free_ends_coordinates_y[-1,:]
    #     calculations.blue_part_coordinates_x = calculations.blue_part_coordinates_x[-1,:]
    #     calculations.blue_part_coordinates_y = calculations.blue_part_coordinates_y[-1,:]


