import numpy as np
from scipy.ndimage import  maximum_filter
#local imports:


ANTS_SPRINGS_OVERLAP_SIZE = 10

class Calculation:
    def __init__(self, springs, ants):
        self.initiate_springs_angles_matrix(springs)
        springs_order = self.springs_angles_matrix[-1, :]
        self.springs_length = self.calc_springs_lengths(springs, springs_order)
        self.N_ants_around_springs, self.size_ants_around_springs =\
            self.occupied_springs(springs, ants, springs_order)
        self.springs_angles_to_nest, self.springs_angles_to_object = self.calc_springs_angles(springs, springs_order)
    def initiate_springs_angles_matrix(self,springs):
        if len(springs.bundles_labels)!=20:
            exit("First frame should have exactly 20 springs. Different number of springs were detect,"
                 " please start the process from a different frame.")
        self.springs_angles_matrix = np.sort(springs.bundles_labels).reshape(1,20)

    def make_calculations(self,springs,ants):
        self.springs_angles_matrix = np.vstack([self.springs_angles_matrix, self.match_springs(springs)])

        springs_order = self.springs_angles_matrix[-1, :]
        self.springs_length = np.vstack([self.springs_length, self.calc_springs_lengths(springs, springs_order)])
        N_ants_around_springs, size_ants_around_springs =\
            self.occupied_springs(springs,ants,springs_order)
        self.N_ants_around_springs = np.vstack([self.N_ants_around_springs, N_ants_around_springs])
        self.size_ants_around_springs = np.vstack([self.size_ants_around_springs, size_ants_around_springs])
        angles_to_nest, angles_to_object = self.calc_springs_angles(springs, springs_order)
        self.springs_angles_to_nest = np.vstack([self.springs_angles_to_nest, angles_to_nest])
        self.springs_angles_to_object = np.vstack([self.springs_angles_to_object, angles_to_object])

    def match_springs(self, springs):
        current_springs_angles = list(springs.bundles_labels)
        previous_springs_angles_mean = self.springs_angles_matrix[0, :]
        assigned_angles_classes = []
        for value in current_springs_angles:
            cos_diff = np.cos(previous_springs_angles_mean - value)
            assigned_class = np.argmax(cos_diff)
            assigned_angles_classes.append(assigned_class)
        new_springs_row = np.array([np.nan for x in range(20)])
        for angle_class, angle in zip(assigned_angles_classes, current_springs_angles):
            new_springs_row[angle_class] = angle
        return new_springs_row

    def calc_springs_lengths(self, springs, springs_order):# fixed_ends_coor, free_ends_coor, fixed_ends_labels, free_ends_labels):
        springs_length = np.empty((20))
        springs_length[:] = np.nan
        found_in_both = list(set(springs.fixed_ends_edges_bundles_labels).
                             intersection(set(springs.free_ends_edges_bundles_labels)))
        for label in found_in_both:
            if label != 0:
                coor_free = springs.free_ends_edges_centers[springs.free_ends_edges_bundles_labels.index(label)]
                coor_fixed = springs.fixed_ends_edges_centers[springs.fixed_ends_edges_bundles_labels.index(label)]
                distance = np.sqrt(np.sum(np.square(coor_free - coor_fixed)))
                index_springs = springs_order == label
                springs_length[index_springs] = distance
        return springs_length.reshape(1,20)

    def occupied_springs(self, springs, ants, springs_order):
        dialated_ends = maximum_filter(springs.free_ends_labeled,ANTS_SPRINGS_OVERLAP_SIZE)
        dialated_ants = maximum_filter(ants.labaled_ants,ANTS_SPRINGS_OVERLAP_SIZE)
        joints = ((dialated_ends != 0) * (dialated_ants != 0))
        self.joints = joints
        # import cv2
        # cv2.imshow("joints", joints.astype(np.uint8)*255)
        # cv2.waitKey(1)
        ends_occupied = np.sort(np.unique(springs.bundles_labeled[joints*(springs.bundles_labeled!=0)]))
        N_ants_around_springs = np.zeros(20)
        size_ants_around_springs = np.zeros(20)
        # N_ants_around_springs = np.empty((20))
        # N_ants_around_springs[:] = np.nan
        # size_ants_around_springs = np.empty((20))
        # size_ants_around_springs[:] = np.nan
        for end in ends_occupied:
            index_springs = springs_order==end
            ends_i = np.unique(dialated_ants[(springs.bundles_labeled==end)*(dialated_ends != 0)])[1:]
            N_ants_around_springs[index_springs] = len(ends_i)
            size_ants_around_springs[index_springs] = np.sum(np.isin(ants.labaled_ants,ends_i))
        return N_ants_around_springs.reshape(1,20), size_ants_around_springs.reshape(1,20)

    def calc_springs_angles(self,springs, springs_order):
        nest_direction = np.array([springs.object_center[0], springs.object_center[1] - 100])
        fixed_ends_angles_to_nest = springs.calc_angles(springs.fixed_ends_edges_centers,springs.object_center,nest_direction)
        free_ends_angles_to_object = springs.calc_angles(springs.free_ends_edges_centers,springs.object_center,springs.tip_point)
        angles_to_nest = np.empty((20))
        angles_to_nest[:] = np.nan
        angles_to_object = np.empty((20))
        angles_to_object[:] = np.nan
        for label in springs.bundles_labels:
            if label !=0:
                index_springs = springs_order == label
                if label in springs.fixed_ends_edges_bundles_labels:
                    # print(label)
                    # print(springs.fixed_ends_edges_bundles_labels==label)
                    # print(springs.free_ends_edges_bundles_labels==label)
                    # print(springs.free_ends_edges_bundles_labels)
                    angles_to_nest[index_springs] = fixed_ends_angles_to_nest[springs.fixed_ends_edges_bundles_labels==label]
                if label in springs.free_ends_edges_bundles_labels:
                    # print("angle to object",free_ends_angles_to_object[springs.free_ends_edges_bundles_labels==label])
                    angles_to_object[index_springs] = free_ends_angles_to_object[springs.free_ends_edges_bundles_labels==label]
        angles_to_nest = angles_to_nest.reshape(1,20)
        angles_to_object = angles_to_nest.reshape(1,20)
        return angles_to_nest, angles_to_object

    def add_blank_row(self):
        empty_row = np.empty((20))
        empty_row[:] = np.nan
        ref = self.springs_length.shape[0]
        #for the self matrices, if shape[0] is as count then do nothing, else add a row of nans.
        if ref != self.N_ants_around_springs.shape[0]:
            self.N_ants_around_springs = np.vstack((self.N_ants_around_springs,empty_row))
        if ref != self.springs_angles_to_nest.shape[0]:
            self.springs_angles_to_nest = np.vstack((self.springs_angles_to_nest,empty_row))
        if ref != self.size_ants_around_springs.shape[0]:
            self.size_ants_around_springs = np.vstack((self.size_ants_around_springs,empty_row))
        if ref != self.springs_angles_to_object.shape[0]:
            self.springs_angles_to_object = np.vstack((self.springs_angles_to_object,empty_row))
        # if ref != self.springs_length.shape[0]:
        #     self.springs_length = np.vstack((self.springs_length,empty_row))
        # for matrix in [self.springs_length, self.springs_angles_to_nest, self.springs_angles_to_object,
        #                self.N_ants_around_springs, self.size_ants_around_springs]:
        #     if matrix.shape[0] != ref:
        #         print("adding row to matrix")
        #         print(matrix.shape)
        #         matrix = np.vstack((matrix,empty_row))

    def clear_data(calculations):
        # data_arrays_names = ["springs_length", "N_ants_around_springs", "size_ants_around_springs",
        #                      "springs_angles_to_nest", "springs_angles_to_object"]
        # for dat in data_arrays_names:
        #     exec(f"calculations.{dat} = calculations.{dat}[-1,:]")
        calculations.springs_length = calculations.springs_length[-1,:]
        calculations.N_ants_around_springs = calculations.N_ants_around_springs[-1,:]
        calculations.size_ants_around_springs = calculations.size_ants_around_springs[-1,:]
        calculations.springs_angles_to_nest = calculations.springs_angles_to_nest[-1,:]
        calculations.springs_angles_to_object = calculations.springs_angles_to_object[-1,:]
