import copy
import numpy as np
from scipy.ndimage import  maximum_filter
from video_analysis.ants_detector import Ants
from video_analysis.springs_detector import Springs
ANTS_SPRINGS_OVERLAP_SIZE = 10


class Calculation(Ants):
    def __init__(self, parameters, image, previous_detections=None):
        super().__init__(parameters, image, previous_detections)
        self.make_ants_centers()
        self.initiate_springs_angles_matrix()
        self.match_springs(self.springs_angles_reference_order)
        self.N_ants_around_springs, self.size_ants_around_springs = self.occupied_springs(self.springs_angles_ordered)
        self.fixed_ends_coordinates_x, self.fixed_ends_coordinates_y, self.free_ends_coordinates_x,\
            self.free_ends_coordinates_y, self.blue_part_coordinates_x, self.blue_part_coordinates_y = self.reorder_coordinates(self.springs_angles_ordered)

    def make_ants_centers(self):
        ants_centers_x = self.ants_centers[:, 1]
        ants_centers_y = self.ants_centers[:, 0]
        ants_centers_x = ants_centers_x.reshape(1, ants_centers_x.shape[0])
        ants_centers_y = ants_centers_y.reshape(1, ants_centers_y.shape[0])
        empty_row = np.empty((1, 100))
        empty_row[:] = np.nan
        self.ants_centers_x = copy.copy(empty_row)
        self.ants_centers_y = copy.copy(empty_row)
        self.ants_centers_x[:, :ants_centers_x.shape[1]] = ants_centers_x
        self.ants_centers_y[:, :ants_centers_y.shape[1]] = ants_centers_y

    def initiate_springs_angles_matrix(self):
        linspace_angles = np.linspace(-np.pi, np.pi-np.pi/self.n_springs, self.n_springs).reshape(1, self.n_springs)
        if self.previous_detections[2] is not None:
            if np.sum(linspace_angles-self.previous_detections[2])==0:
                if len(self.bundles_labels) == self.n_springs:
                    self.springs_angles_reference_order = np.sort(self.bundles_labels).reshape(1, self.n_springs)
                else:
                    self.springs_angles_reference_order = self.previous_detections[2]
            else:
                self.springs_angles_reference_order = self.previous_detections[2]
        elif len(self.bundles_labels) == self.n_springs:
            self.springs_angles_reference_order = np.sort(self.bundles_labels).reshape(1, self.n_springs)
        else:
            self.springs_angles_reference_order = linspace_angles

    def match_springs(self, springs_angles_reference_order):
        current_springs_angles = list(self.bundles_labels)
        assigned_angles_classes = []
        for value in current_springs_angles:
            cos_diff = np.cos(springs_angles_reference_order - value)
            assigned_class = np.argmax(cos_diff)
            assigned_angles_classes.append(assigned_class)
        new_springs_row = np.array([np.nan for x in range(self.n_springs)])
        for angle_class, angle in zip(assigned_angles_classes, current_springs_angles):
            new_springs_row[angle_class] = angle
        self.springs_angles_ordered =  new_springs_row

    def ants_centers(self, ants):
        self.ants_centers_coordinates = ants.ants_centers

    def occupied_springs(self, springs_order):
        dilated_ends = maximum_filter(self.free_ends_labeled,ANTS_SPRINGS_OVERLAP_SIZE)
        dilated_ants = maximum_filter(self.labeled_ants,ANTS_SPRINGS_OVERLAP_SIZE)
        joints = ((dilated_ends != 0) * (dilated_ants != 0))
        self.joints = joints
        ends_occupied = np.unique(self.bundles_labeled[joints])
        ends_occupied = ends_occupied[np.where(ends_occupied != 0)]
        N_ants_around_springs = np.zeros(self.n_springs)
        size_ants_around_springs = np.zeros(self.n_springs)
        self.ants_attached_labels = np.array([np.nan for x in range(100)])
        for end in ends_occupied:
            spring_position = springs_order==end
            ends_i = np.unique(dilated_ants[(self.bundles_labeled == end)*(dilated_ends != 0)])[1:]
            if np.sum(spring_position)==1:
                self.ants_attached_labels[ends_i-1] = np.arange(self.n_springs)[spring_position][0]
            N_ants_around_springs[spring_position] = len(ends_i)
            size_ants_around_springs[spring_position] = np.sum(np.isin(self.labeled_ants,ends_i))
        return N_ants_around_springs.reshape(1,self.n_springs), size_ants_around_springs.reshape(1,self.n_springs)

    def reorder_coordinates(self,springs_order):
        fixed_ends_x,fixed_ends_y,free_ends_x,free_ends_y = np.empty(self.n_springs),np.empty(self.n_springs),np.empty(self.n_springs),np.empty(self.n_springs)
        fixed_ends_x[:] = np.nan
        fixed_ends_y[:] = np.nan
        free_ends_x[:] = np.nan
        free_ends_y[:] = np.nan
        for label in self.bundles_labels:
            if label !=0:
                index_springs = springs_order == label
                if label in self.fixed_ends_edges_bundles_labels:
                    index = self.fixed_ends_edges_bundles_labels.index(label)
                    fixed_ends_x[index_springs] = self.fixed_ends_edges_centers[index,0]
                    fixed_ends_y[index_springs] = self.fixed_ends_edges_centers[index,1]
                if label in self.free_ends_edges_bundles_labels:
                    index = self.free_ends_edges_bundles_labels.index(label)
                    free_ends_x[index_springs] = self.free_ends_edges_centers[index,0]
                    free_ends_y[index_springs] = self.free_ends_edges_centers[index,1]

        blue_part_x, blue_part_y = np.empty(2),np.empty(2)
        blue_part_x[:] = np.nan
        blue_part_y[:] = np.nan
        blue_part_x[0] = self.object_center[0]
        blue_part_x[1] = self.tip_point[0]
        blue_part_y[0] = self.object_center[1]
        blue_part_y[1] = self.tip_point[1]
        return fixed_ends_x.reshape(1,self.n_springs),fixed_ends_y.reshape(1,self.n_springs),free_ends_x.reshape(1,self.n_springs),\
            free_ends_y.reshape(1,self.n_springs),blue_part_x.reshape(1,2),blue_part_y.reshape(1,2)



