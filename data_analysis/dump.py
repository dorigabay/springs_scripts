import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label,sum_labels
from scipy.signal import savgol_filter
import os
from data_analysis import utils

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pickle

class PostProcessing:
    def __init__(self, directory, calibration_model=None):
        self.directory = directory
        if calibration_model is None:
            self.calibration_mode = True
        else: self.calibration_mode = False
        self.load_data(directory)
        self.N_ants_proccessing()
        self.calc_distances()
        self.repeat_values()
        self.calc_angle()
        self.calibration_model = calibration_model
        if not self.calibration_mode:
            # self.object_center_to_fixed_end_distance, self.object_center_to_fixed_end_distance_bias_equations = self.norm_values(self.object_center_to_fixed_end_distance,self.fixed_end_angle_to_nest,bias_bool=self.rest_bool)
            self.find_fixed_coordinates(bias_equations_free=self.free_end_angle_to_blue_part_bias_equations,
                                        bias_equations_fixed=self.fixed_end_angle_to_blue_part_bias_equations)
            self.calc_pulling_angle(zero_angles=True)
            self.calc_spring_length()
            self.calc_force(calibration_model)


    def load_data(self,directory):
        # init function
        print("loading data from:", directory)
        self.norm_size = pickle.load(open(os.path.join(directory,"blue_median_area.pickle"), "rb"))
        directory = os.path.join(directory, "raw_analysis")+"\\"
        self.N_ants_around_springs = np.loadtxt(os.path.join(directory,"N_ants_around_springs.csv"), delimiter=",")
        self.size_ants_around_springs = np.loadtxt(os.path.join(directory,"size_ants_around_springs.csv"), delimiter=",")
        fixed_ends_coordinates_x = np.loadtxt(os.path.join(directory,"fixed_ends_coordinates_x.csv"), delimiter=",")
        fixed_ends_coordinates_y = np.loadtxt(os.path.join(directory,"fixed_ends_coordinates_y.csv"), delimiter=",")
        free_ends_coordinates_x = np.loadtxt(os.path.join(directory,"free_ends_coordinates_x.csv"), delimiter=",")
        free_ends_coordinates_y = np.loadtxt(os.path.join(directory,"free_ends_coordinates_y.csv"), delimiter=",")
        if self.calibration_mode:
            self.fixed_ends_coordinates = np.stack((np.expand_dims(fixed_ends_coordinates_x,1), np.expand_dims(fixed_ends_coordinates_y,1)), axis=2)
            self.free_ends_coordinates = np.stack((np.expand_dims(free_ends_coordinates_x,1), np.expand_dims(free_ends_coordinates_y,1)), axis=2)
            self.N_ants_around_springs = np.expand_dims(self.N_ants_around_springs,1)
            self.size_ants_around_springs = np.expand_dims(self.size_ants_around_springs,1)
            self.num_of_springs = 1
        else:
            self.fixed_ends_coordinates = np.stack((fixed_ends_coordinates_x, fixed_ends_coordinates_y), axis=2)
            self.free_ends_coordinates = np.stack((free_ends_coordinates_x, free_ends_coordinates_y), axis=2)
            self.num_of_springs = self.N_ants_around_springs.shape[1]
        blue_part_coordinates_x = np.loadtxt(os.path.join(directory,"blue_part_coordinates_x.csv"), delimiter=",")
        blue_part_coordinates_y = np.loadtxt(os.path.join(directory,"blue_part_coordinates_y.csv"), delimiter=",")
        blue_part_coordinates = np.stack((blue_part_coordinates_x, blue_part_coordinates_y), axis=2)
        self.object_center = blue_part_coordinates[:, 0, :]
        self.blue_tip_coordinates = blue_part_coordinates[:, -1, :]
        self.num_of_frames = self.N_ants_around_springs.shape[0]

    def N_ants_proccessing(self):
        def smoothing_n_ants(array):
            for col in range(array.shape[1]):
                array[:,col] = np.abs(np.round(savgol_filter(array[:,col], 31, 2)))
            return array
        undetected_springs_for_long_time = utils.filter_continuity(utils.convert_bool_to_binary(
            np.isnan(self.fixed_ends_coordinates[:, :, 0])),min_size=8)
        self.N_ants_around_springs = np.round(utils.interpolate_data(self.N_ants_around_springs,
                                                    utils.find_cells_to_interpolate(self.N_ants_around_springs)))
        self.N_ants_around_springs[np.isnan(self.N_ants_around_springs)] = 0
        self.N_ants_around_springs = smoothing_n_ants(self.N_ants_around_springs)
        self.N_ants_around_springs[undetected_springs_for_long_time] = np.nan
        all_small_attaches = np.zeros(self.N_ants_around_springs.shape,int)
        for n in np.unique(self.N_ants_around_springs)[1:]:
            if not np.isnan(n):
                short_attaches = utils.filter_continuity(self.N_ants_around_springs==n,max_size=30)
                all_small_attaches[short_attaches] = 1
        self.N_ants_around_springs = np.round(utils.interpolate_data(self.N_ants_around_springs,
                                                                     all_small_attaches.astype(bool)))
        self.rest_bool = self.N_ants_around_springs == 0

    def bound_angle(self,angle):
        angle[angle > 2 * np.pi] -= 2 * np.pi
        return angle

    def calc_distances(self):
        object_center = np.repeat(self.object_center[:, np.newaxis, :], self.num_of_springs, axis=1)
        self.blue_length = np.nanmedian(np.linalg.norm(self.blue_tip_coordinates - self.object_center, axis=1))
        self.object_center_to_free_end_distance = np.linalg.norm(self.free_ends_coordinates - object_center, axis=2)
        self.object_center_to_fixed_end_distance = np.linalg.norm(self.fixed_ends_coordinates - object_center, axis=2)
        # self.norm_length = np.nanmedian(self.object_center_to_fixed_end_distance)
        # print("norm_size is: ", self.norm_size)
        # print("norm length is: ", self.norm_length)
        # print("blue length is: ", self.blue_length)
        # print("blue length/norm length is: ", self.blue_length/self.norm_length)
        # print("blue length/norm size is: ", self.blue_length/self.norm_size)
        # self.object_center_to_free_end_distance /= self.norm_size
        # self.object_center_to_fixed_end_distance /= self.norm_size

    def repeat_values(self):
        nest_direction = np.stack((self.object_center[:, 0], self.object_center[:, 1]-100), axis=1)
        self.nest_direction_repeated = np.repeat(nest_direction[:, np.newaxis, :], self.free_ends_coordinates.shape[1], axis=1)
        self.object_center_repeated = copy.copy(np.repeat(self.object_center[:, np.newaxis, :], self.free_ends_coordinates.shape[1], axis=1))
        self.blue_tip_coordinates_repeated = np.repeat(self.blue_tip_coordinates[:, np.newaxis, :], self.free_ends_coordinates.shape[1], axis=1)

    def calc_angle(self):
        # init function
        #angles to the nest
        self.free_end_angle_to_nest = utils.calc_angle_matrix(self.free_ends_coordinates, self.object_center_repeated, self.nest_direction_repeated)+np.pi
        # print("free_end_angle_to_nest: min = {}, max = {}".format(np.nanmin(self.free_end_angle_to_nest), np.nanmax(self.free_end_angle_to_nest)))
        self.fixed_end_angle_to_nest = utils.calc_angle_matrix(self.fixed_ends_coordinates, self.object_center_repeated, self.nest_direction_repeated)+np.pi
        # print("fixed_end_angle_to_nest: min = {}, max = {}".format(np.nanmin(self.fixed_end_angle_to_nest), np.nanmax(self.fixed_end_angle_to_nest)))
        self.blue_part_angle_to_nest = utils.calc_angle_matrix(self.nest_direction_repeated, self.object_center_repeated, self.blue_tip_coordinates_repeated)+np.pi
        # print("blue_part_angle_to_nest: min = {}, max = {}".format(np.nanmin(self.blue_part_angle_to_nest), np.nanmax(self.blue_part_angle_to_nest)))
        # angles to the blue part
        self.free_end_angle_to_blue_part = utils.calc_angle_matrix(self.blue_tip_coordinates_repeated, self.object_center_repeated, self.free_ends_coordinates)+np.pi
        # print("free_end_angle_to_blue_part: min = {}, max = {}".format(np.nanmin(self.free_end_angle_to_blue_part), np.nanmax(self.free_end_angle_to_blue_part)))
        self.fixed_end_angle_to_blue_part = utils.calc_angle_matrix(self.blue_tip_coordinates_repeated, self.object_center_repeated,self.fixed_ends_coordinates)+np.pi
        # print("fixed_end_angle_to_blue_part: min = {}, max = {}".format(np.nanmin(self.fixed_end_angle_to_blue_part), np.nanmax(self.fixed_end_angle_to_blue_part)))
        # self.find_column_on_boundry()
        _, self.free_end_angle_to_blue_part_bias_equations = self.norm_values(self.free_end_angle_to_nest,
                                                       self.free_end_angle_to_blue_part,
                                                       bias_bool=self.rest_bool,
                                                       find_boundary_column=True)
        _, self.fixed_end_angle_to_blue_part_bias_equations = self.norm_values(self.fixed_end_angle_to_nest,
                                                        self.fixed_end_angle_to_blue_part,
                                                        bias_bool=self.rest_bool,
                                                        find_boundary_column=True)

    def find_column_on_boundry(self,X):
        columns_on_boundry = []
        for s in range(X.shape[1]):
            no_nans_fixed_end_angle_to_blue_part = X[:, s][~np.isnan(X[:, s])]
            if np.sum(np.abs(np.diff(no_nans_fixed_end_angle_to_blue_part))>np.pi) > 10:
                columns_on_boundry.append(s)
        if len(columns_on_boundry) == 1:
            column_on_boundry = columns_on_boundry[0]
        else:
            raise ValueError("more than one column on boundry")
        return column_on_boundry

    def norm_values(self, X, Y, bias_bool=None, find_boundary_column=False, bias_equations=None):
        # takes array of x and y values and returns the normalized values
        # X and Y should have the same shape
        # X is the independent variable, that creates the bias over Y
        # iterates over the second axis
        if bias_bool is not None:
            X_bias = copy.deepcopy(X)
            Y_bias = copy.deepcopy(Y)
            X_bias[np.invert(bias_bool)] = np.nan
            Y_bias[np.invert(bias_bool)] = np.nan
        else:
            X_bias = X
            Y_bias = Y
        # subtract the median of the y values from the y values
        # print("Y_bias[:, 0] min: {}, max{}:".format(np.nanmin(Y_bias[:, 0]), np.nanmax(Y_bias[:, 0])))
        if find_boundary_column:
            column_on_boundary = self.find_column_on_boundry(Y)
            # remove nanmedian from all columns except the one on the boundary
            # Y_bias[:, np.arange(Y_bias.shape[1]) != column_on_boundary] -= np.nanmedian(Y_bias[:, np.arange(Y_bias.shape[1]) != column_on_boundary], axis=0)
            above_nonan = Y_bias[:, column_on_boundary] > np.pi
            # below_nonan = Y_bias[:, column_on_boundary] < np.pi
            Y_bias[above_nonan, column_on_boundary] -= 2*np.pi
            # Y_bias[below_nonan, column_on_boundary] += np.pi
        # print("Y_bias[:, 0] min: {}, max{}:".format(np.nanmin(Y_bias[:, 0]), np.nanmax(Y_bias[:, 0])))
        Y_bias -= np.nanmedian(Y_bias, axis=0)
        normed_Y = np.zeros(Y.shape)
        import pandas as pd
        if bias_equations is None:
            bias_equations = []
            for i in range(X.shape[1]):
                df = pd.DataFrame({"x": X_bias[:, i], "y": Y_bias[:, i]}).dropna()
                bias_equation = utils.deduce_bias_equation(df["x"], df["y"])
                bias_equations.append(bias_equation)
        if find_boundary_column:
            Y = copy.deepcopy(Y)
            # remove nanmedian from all columns except the one on the boundary
            above = Y[:, column_on_boundary] > np.pi
            # below = Y[:, column_on_boundary] < np.pi
            Y[above, column_on_boundary] -= 2*np.pi
            # Y[below, column_on_boundary] += np.pi
            # print("Y[:, 0] min: {}, max{}:".format(np.nanmin(Y[:, 0]), np.nanmax(Y[:, 0])))
        for i in range(self.num_of_springs):
            bias_equation = bias_equations[i]
            normed_Y[:, i] = utils.normalize(Y[:, i], X[:, i], bias_equation)
        if find_boundary_column:
            # print("normed_Y[:, 0] min: {}, max{}:".format(np.nanmin(normed_Y[:, 0]), np.nanmax(normed_Y[:, 0])))
            # above = normed_Y[:, column_on_boundary] > 0
            below = normed_Y[:, column_on_boundary] < 0
            # normed_Y[above, column_on_boundary] -= np.pi
            normed_Y[below, column_on_boundary] += 2*np.pi
            # normed_Y += np.pi
        # print(normed_Y)
        # print("normed_Y[:, 0] min: {}, max{}:".format(np.nanmin(normed_Y[:, 0]), np.nanmax(normed_Y[:, 0])))
        return normed_Y, bias_equations

    def find_fixed_coordinates(self,bias_equations_free=None,bias_equations_fixed=None):
        def calc_fixed(distance_to_object_center,angle_to_blue=None,end_type=None):
            if end_type== "free":
                # print("3"*50)
                angle_to_blue, self.free_end_angle_to_blue_bias_equations = \
                    self.norm_values(self.free_end_angle_to_nest, self.free_end_angle_to_blue_part,
                                     bias_bool=self.rest_bool, bias_equations=bias_equations_free, find_boundary_column=True)
                distance_to_object_center[~self.rest_bool] = np.nan
                median_distance = np.nanmedian(distance_to_object_center, axis=0)
            elif end_type == "fixed":
                # spring_length = np.linalg.norm(self.free_ends_coordinates - self.fixed_ends_coordinates, axis=2)
                # bottom_5_percent_quantile = np.nanquantile(spring_length[self.rest_bool].flatten(), 0.95)
                # angle_to_blue = copy.copy(self.fixed_end_angle_to_blue_part)
                # angle_to_blue[(spring_length > bottom_5_percent_quantile) & ~self.rest_bool] = np.nan
                # angle_to_blue = np.nanmedian(angle_to_blue, axis=0)
                # angle_to_blue = np.repeat(angle_to_blue[np.newaxis, :], self.num_of_frames, axis=0)
                angle_to_blue, self.fixed_end_angle_to_blue_bias_equations = \
                    self.norm_values(self.fixed_end_angle_to_nest, self.fixed_end_angle_to_blue_part,
                                        bias_bool=self.rest_bool, bias_equations=bias_equations_fixed, find_boundary_column=True)

                # print("angle_to_blue",angle_to_blue)
                # print("distance_to_object_center",distance_to_object_center)
                median_distance = np.nanmedian(distance_to_object_center, axis=0)
            median_distance = np.repeat(median_distance[np.newaxis, :], self.num_of_frames, axis=0)
            # subtract the angle of the blue part end to the nest from the angle of the fixed end to the blue part
            angle_to_blue_part_normed = self.bound_angle(self.blue_part_angle_to_nest+angle_to_blue-np.pi/2)
            # find the fixed end coordinates with fixed_end_angle_to_blue_part_normed and median_distance_to_fixed_end
            fixed_coordinates = self.object_center_repeated + np.stack((np.cos(angle_to_blue_part_normed) * median_distance,
                                                                        np.sin(angle_to_blue_part_normed) * median_distance), axis=2)
            return fixed_coordinates
        # find fixed coordinates
        self.fixed_end_fixed_coordinates = calc_fixed(self.object_center_to_fixed_end_distance,end_type="fixed")
        self.free_end_fixed_coordinates = calc_fixed(self.object_center_to_free_end_distance,end_type="free")
        # angles of fixed coordinates to the nest
        # show fixed coordinates on the first image
        # import cv2
        # video_path = "Z:\\Dor_Gabay\\ThesisProject\\data\\videos\\15.9.22\\plus0.3mm_force\\S5280006.MP4"
        # video = cv2.VideoCapture(video_path)
        # ret, frame = video.read()
        # for i in range(self.num_of_springs):
        #     cv2.circle(frame, (int(self.fixed_end_fixed_coordinates[0, i, 0]),int(self.fixed_end_fixed_coordinates[0, i, 1])), 5, (0, 0, 255), 2)
        #     cv2.circle(frame, (int(self.free_end_fixed_coordinates[0, i, 0]),int(self.free_end_fixed_coordinates[0, i, 1])), 5, (0, 255, 0), 2)
        # cv2.imshow("frame", frame)
        # cv2.waitKey(0)
        self.calc_fixed_coordinates_angles()

    def calc_fixed_coordinates_angles(self):
        self.free_end_fixed_coordinates_angle_to_nest = utils.calc_angle_matrix(self.free_end_fixed_coordinates, self.object_center_repeated, self.nest_direction_repeated)+np.pi
        # print("free_end_fixed_coordinates_angle_to_nest: min: {}, max: {}".format(np.nanmin(self.free_end_fixed_coordinates_angle_to_nest),np.nanmax(self.free_end_fixed_coordinates_angle_to_nest)))
        self.fixed_end_fixed_coordinates_angle_to_nest = utils.calc_angle_matrix(self.fixed_end_fixed_coordinates, self.object_center_repeated, self.nest_direction_repeated)+np.pi
        # print("fixed_end_fixed_coordinates_angle_to_nest: min: {}, max: {}".format(np.nanmin(self.fixed_end_fixed_coordinates_angle_to_nest),np.nanmax(self.fixed_end_fixed_coordinates_angle_to_nest)))
        # angles of fixed coordinates to the blue part
        self.free_end_fixed_coordinates_angle_to_blue_part = utils.calc_angle_matrix(self.blue_tip_coordinates_repeated,
                                                                   self.object_center_repeated,
                                                                   self.free_end_fixed_coordinates)+np.pi
        # print("free_end_fixed_coordinates_angle_to_blue_part: min: {}, max: {}".format(np.nanmin(self.free_end_fixed_coordinates_angle_to_blue_part),np.nanmax(self.free_end_fixed_coordinates_angle_to_blue_part)))
        self.fixed_end_fixed_coordinates_angle_to_blue_part = utils.calc_angle_matrix(self.blue_tip_coordinates_repeated,
                                                                    self.object_center_repeated,
                                                                    self.fixed_end_fixed_coordinates)+np.pi
        # print("fixed_end_fixed_coordinates_angle_to_blue_part: min: {}, max: {}".format(np.nanmin(self.fixed_end_fixed_coordinates_angle_to_blue_part),np.nanmax(self.fixed_end_fixed_coordinates_angle_to_blue_part)))

    def calc_pulling_angle(self,zero_angles=False):
        # pulling angle between fixed coordinates of free end and fixed end
        self.pulling_angle = utils.calc_angle_matrix(self.free_end_fixed_coordinates,self.fixed_end_fixed_coordinates,self.object_center_repeated)
        # print(self.pulling_angle)
        # print(f"pulling_angle: min: {np.nanmin(self.pulling_angle)}, max: {np.nanmax(self.pulling_angle)}")
        above = self.pulling_angle > 0
        below = self.pulling_angle < 0
        self.pulling_angle[above] = self.pulling_angle[above] - np.pi
        self.pulling_angle[below] = self.pulling_angle[below] + np.pi
        if zero_angles:
            median_pulling_angle_at_rest = copy.copy(self.pulling_angle)
            median_pulling_angle_at_rest[np.invert(self.rest_bool)] = np.nan
            median_spring_length_at_rest = np.nanmedian(median_pulling_angle_at_rest, axis=0)
            # self.pulling_angle -= np.nanmedian(self.pulling_angle, axis=0)
            self.pulling_angle -= median_spring_length_at_rest
        # if zero_angle is None:
        #     self.pulling_angle -= np.nanmedian(self.pulling_angle,axis=0)
        # self.pulling_angle
        # self.pulling_angle += np.pi
        # print(f"pulling_angle: min: {np.nanmin(self.pulling_angle)}, max: {np.nanmax(self.pulling_angle)}")
        # self.pulling_angle = self.bound_angle(self.pulling_angle)
        # print(f"pulling_angle: min: {np.nanmin(self.pulling_angle)}, max: {np.nanmax(self.pulling_angle)}")
        # median_pulling_angle = copy.copy(self.pulling_angle)
        # median_pulling_angle[np.invert(self.rest_bool)] = np.nan
        # median_pulling_angle = np.nanmedian(median_pulling_angle,axis=0)
        # median_pulling_angle = np.repeat(median_pulling_angle[np.newaxis, :], self.num_of_frames, axis=0)
        # self.pulling_angle -= median_pulling_angle

    def calc_spring_length(self,bias_equations=None,zero_length=None,first_calib=False):
        # print(self.free_end_fixed_coordinates.shape)
        # self.spring_length = np.linalg.norm(self.free_end_fixed_coordinates - self.object_center_repeated , axis=2)
        self.spring_length = np.linalg.norm(self.free_end_fixed_coordinates - self.fixed_end_fixed_coordinates , axis=2)
        # print(f"spring_length: min: {np.nanmin(self.spring_length)}, max: {np.nanmax(self.spring_length)}, mean: {np.nanmean(self.spring_length)}")
        self.spring_length = utils.interpolate_data(self.spring_length,
                                                       utils.find_cells_to_interpolate(self.spring_length))
        if bias_equations is not None:
            self.spring_length = \
            self.norm_values(self.free_end_fixed_coordinates_angle_to_nest, self.spring_length, bias_bool=self.rest_bool, bias_equations=bias_equations,
                             find_boundary_column=False)[0]
        else:
            self.spring_length = self.norm_values(self.free_end_fixed_coordinates_angle_to_nest, self.spring_length, bias_bool=self.rest_bool, find_boundary_column=False)[0]
        self.spring_length /= self.norm_size
        if zero_length is not None:
            self.spring_length /= zero_length
            print(f" calib spring length min: {np.nanmin(self.spring_length)}, max: {np.nanmax(self.spring_length)}, mean: {np.nanmean(self.spring_length)}")
        elif not first_calib:
            median_spring_length_at_rest = copy.copy(self.spring_length)
            median_spring_length_at_rest[np.invert(self.rest_bool)] = np.nan
            median_spring_length_at_rest = np.nanmedian(median_spring_length_at_rest, axis=0)
            # print("median_spring_length_at_rest: {}".format(median_spring_length_at_rest))
            self.spring_length /= median_spring_length_at_rest
            print(f"spring length min: {np.nanmin(self.spring_length)}, max: {np.nanmax(self.spring_length)}, mean: {np.nanmean(self.spring_length)}")

            # self.spring_length /= np.nanmedian(self.spring_length[self.rest_bool], axis=0)
            # print("median_spring_length_at_rest: {}".format(np.nanmedian(self.spring_length[self.rest_bool], axis=0)))
        # self.spring_length /= self.blue_length
        # print(f"spring_length: min: {np.nanmin(self.spring_length)}, max: {np.nanmax(self.spring_length)}, mean: {np.nanmean(self.spring_length)}")
        # print(f"spring_length: min: {np.nanmin(self.spring_length)}, max: {np.nanmax(self.spring_length)}, mean: {np.nanmean(self.spring_length)}")
        # norm self.spring_length by subtracting the length of the blue part
        if zero_length is not None:
            # print("zero_length: {}".format(zero_length))
            self.spring_extension = self.spring_length.flatten() - 1#np.repeat(zero_length/zero_length, self.num_of_frames, axis=0)
            # self.spring_extension = self.spring_length.flatten() - np.repeat(zero_length, self.num_of_frames, axis=0)
            print(f"FINAL min extension: {np.nanmin(self.spring_extension)}, max extension: {np.nanmax(self.spring_extension)}, mean extension: {np.nanmean(self.spring_extension)}")
        elif first_calib:
            # median_spring_length_at_rest = copy.copy(self.spring_length)
            # median_spring_length_at_rest[np.invert(self.rest_bool)] = np.nan
            # median_spring_length_at_rest = np.nanmedian(median_spring_length_at_rest,axis=0)
            # self.spring_extension = self.spring_length - np.repeat(median_spring_length_at_rest[np.newaxis, :],
            #                                                        self.num_of_frames, axis=0)
            pass
        elif not first_calib:
            # print("zero_length: {}".format(median_spring_length_at_rest))
            # median_spring_length_at_rest = copy.copy(self.spring_length)
            # median_spring_length_at_rest[np.invert(self.rest_bool)] = np.nan
            # median_spring_length_at_rest = np.nanmedian(median_spring_length_at_rest, axis=0)/np.nanmedian(median_spring_length_at_rest, axis=0)
            # print("median_spring_length_at_rest: ",median_spring_length_at_rest)
            self.spring_extension = self.spring_length -1# np.repeat(median_spring_length_at_rest[np.newaxis, :], self.num_of_frames, axis=0)
            print(f"FINAL min extension: {np.nanmin(self.spring_extension)}, max extension: {np.nanmax(self.spring_extension)}, mean extension: {np.nanmean(self.spring_extension)}")
        # print("np.nanmax(self.spring_extension)/np.nanmin(self.spring_extension): ",np.nanmax(self.spring_extension)/np.nanmin(self.spring_extension))
        # print("nanmax spring_extension / norm_size: ",np.nanmax(self.spring_extension)/self.norm_size)



    def calc_calibration_force(self, calibration_weight):
        self.calibration_force_direction = (self.fixed_end_fixed_coordinates_angle_to_nest).flatten()
        above = self.calibration_force_direction > np.pi
        self.calibration_force_direction[above] = self.calibration_force_direction[above] - 2 * np.pi
        # self.calibration_force_direction *= -1
        # print(f"force direction shape: {self.calibration_force_direction.shape}")
        G = 9.81
        weight_in_Kg = calibration_weight*1e-3
        self.calibration_force_magnitude = np.repeat(weight_in_Kg * G, self.num_of_frames)
        # print(f"force magnitude shape: {self.calibration_force_magnitude.shape}")
        # self.calibration_force = self.force_direction#self.force_magnitude *

    def plot_pulling_angle_to_nest_angle(self,X,bool,spring):
        # save_path = os.path.join(self.output_path, "plots","weights")
        # os.makedirs(save_path, exist_ok=True)
        # for processed in self.processed_list:
        title = str(spring)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        spring = 0
        # x = self.fixed_end_fixed_coordinates_angle_to_nest[:,spring][bool]
        x = np.arange(0,self.num_of_frames)[bool]
        # above = x > np.pi
        # below = x < np.pi
        # x[above] = x[above] - np.pi
        # x[below] = x[below] + np.pi
        # x -= np.pi

        y1 = X[:,0]
        above = y1 > np.pi
        below = y1 < np.pi
        y1[above] = y1[above] - np.pi
        y1[below] = y1[below] + np.pi
        y1 -= np.pi
        ax1.plot(x, y1, "o",alpha=0.5)
        ax1.set_xlabel("Direction to the nest")
        ax1.set_ylabel("Pulling direction")
        ax1.set_title(f"spring: {title}")

        y2 = X[:,1]
        ax2.plot(x, y2, "o",alpha=0.5)
        ax2.set_xlabel("Direction to the nest")
        ax2.set_ylabel("Spring extension")

        y_pred = self.calibration_model.predict(X)
        y3 = y_pred#[:,0]
        # direction_pred = y_pred[:, 0]
        # above = y3 > 0
        # below = y3 < 0
        # y3[above] = y3[above] - np.pi
        # y3[below] = y3[below] + np.pi
        # direction_pred[above] = direction_pred[above] - np.pi
        # direction_pred[below] = direction_pred[below] + np.pi
        ax3.plot(x, y3, "o",alpha=0.5)
        # ax3.plot(x, direction_pred, "o", color="red",alpha=0.1)
        ax3.set_xlabel("Direction to the nest")
        ax3.set_ylabel("Force direction")

        y4 = y_pred[:, 1]
        # extent_pred = y_pred[:, 1]
        ax4.plot(x, y4, "o",alpha=0.5)
        # ax4.plot(x, extent_pred, "o", color="red",alpha=0.1)
        ax4.set_xlabel("Direction to the nest")
        ax4.set_ylabel("Force magnitude")

        fig.tight_layout()
        fig.set_size_inches(7.5, 5)
        plt.show()
        # fig.savefig(os.path.join(save_path, f"weight_{title}.png"))
        pass
    def calc_force(self,model):
        self.force_direction = np.zeros((self.num_of_frames,self.num_of_springs))
        self.force_magnitude = np.zeros((self.num_of_frames,self.num_of_springs))
        self.force = np.zeros((self.num_of_frames,self.num_of_springs))
        # self.object. = np.zeros((self.num_of_frames,self.num_of_springs,2))
        for s in range(self.num_of_springs):
            X = np.stack((self.pulling_angle[:,s],self.spring_extension[:,s]),axis=1)
            un_nan_bool = ~np.isnan(X).any(axis=1)
            X = X[un_nan_bool]
            forces_predicted = model.predict(X)
            self.force[un_nan_bool,s] = forces_predicted
            # self.force_direction[un_nan_bool,s] = forces_predicted[:,0]
            # self.force_magnitude[un_nan_bool,s] = forces_predicted[:,1]
            # self.plot_pulling_angle_to_nest_angle(X,bool=un_nan_bool,spring=s)
        print(f"X pulling_angle min {np.nanmin(X[:,0])}, max {np.nanmax(X[:,0])}, mean {np.nanmean(X[:,0])}")
        print(f"X spring_extension min {np.nanmin(X[:,1])}, max {np.nanmax(X[:,1])}, mean {np.nanmean(X[:,1])}")

        # import cv2
        # video_path = "Z:\\Dor_Gabay\\ThesisProject\\data\\videos\\15.9.22\\plus0.3mm_force\\S5280006.MP4"
        # video = cv2.VideoCapture(video_path)
        # point = (int(self.object_center_repeated[0, 0, 0]), int(self.object_center_repeated[0, 0, 1]))
        # point2 = (int(self.object_center_repeated[0, 0, 0]), int(self.object_center_repeated[0, 0, 1] + 20))
        # point3 = (int(self.object_center_repeated[0, 0, 0]), int(self.object_center_repeated[0, 0, 1] + 40))
        # point4 = (int(self.object_center_repeated[0, 0, 0]), int(self.object_center_repeated[0, 0, 1] + 60))
        # # ret, frame = video.read()
        # for f in range(10000):
        #     if f > 5000:
        #         ret, frame = video.read()
        #         cv2.putText(frame, f"pulling angle: {self.pulling_angle[f, 0]}", point, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #                     (0, 0, 255), 2)
        #         cv2.putText(frame, f"spring_extension: {self.spring_extension[f,0]}", point2,
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        #         cv2.putText(frame, f"force_direction: {self.force_direction[f,0]}", point3,
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        #         cv2.putText(frame, f"force_magnitude: {self.force_magnitude[f,0]}", point4,
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        #         cv2.imshow("frame", frame)
        #         cv2.waitKey(10)

        # print(f"y_pred force_direction min {np.nanmin(forces_predicted[:,0])}, max {np.nanmax(forces_predicted[:,0])}, mean {np.nanmean(forces_predicted[:,0])}")
        # print(f"y_pred force_magnitude min {np.nanmin(forces_predicted[:,1])}, max {np.nanmax(forces_predicted[:,1])}, mean {np.nanmean(forces_predicted[:,1])}")
        # print(f"norm length: {self.norm_length}")
        # print("force_direction: min: {}, max: {}".format(np.nanmin(self.force_direction),np.nanmax(self.force_direction)))
        # print("force_magnitude: min: {}, max: {}".format(np.nanmin(self.force_magnitude),np.nanmax(self.force_magnitude)))

    def save_data(self, directory):
        print("saving data to:", directory)
        directory = os.path.join(directory, "post_processed_data/")
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.savetxt(f"{directory}N_ants_around_springs.csv", self.N_ants_around_springs, delimiter=",")
        np.savetxt(f"{directory}spring_length.csv", self.spring_length, delimiter=",")
        np.savetxt(f"{directory}fixed_end_fixed_coordinates_angle_to_nest.csv", self.fixed_end_fixed_coordinates_angle_to_nest, delimiter=",")
        np.savetxt(f"{directory}pulling_angle.csv", self.pulling_angle, delimiter=",")
        np.savetxt(f"{directory}force_direction.csv", self.force_direction, delimiter=",")
        np.savetxt(f"{directory}force_magnitude.csv", self.force_magnitude, delimiter=",")
        np.savetxt(f"{directory}force.csv", self.force, delimiter=",")

########################################################################################################################
class Calibration:
    def __init__(self, directories, weights, output_path, degree=None):
        print("Creating calibration model...")
        self.output_path = output_path
        self.directories = directories
        self.weights = weights
        self.get_bias_equation()
        self.get_data()
        # self.plot_angles_to_force()
        if degree is None:
            self.find_best_degree()
            # self.calibration_model(self.degree)
        self.plot_fitting_results()
        self.plot_pulling_angle_to_nest_angle()

    def get_bias_equation(self):
        dir = self.directories[0]
        print("Creating bias equation from: ", dir)
        processed = PostProcessing(dir)
        # print("8"*50)
        # print(np.nanmax(processed.free_end_angle_to_blue_part), np.nanmin(processed.free_end_angle_to_blue_part))
        # normed_angles, self.angle_bias_equation = processed.norm_values(processed.free_end_angle_to_nest,
        #                                                                 processed.free_end_angle_to_blue_part,
        #                                                                 bias_bool=processed.rest_bool,
        #                                                                 find_boundary_column=True)
        # normed_angles, self.angle_bias_equation_fixed = processed.norm_values(processed.fixed_end_angle_to_nest,
        #                                                                 processed.fixed_end_angle_to_blue_part,
        #                                                                 bias_bool=processed.rest_bool,
        #                                                                 find_boundary_column=True)
        # print(np.nanmax(processed.free_end_angle_to_blue_part), np.nanmin(processed.free_end_angle_to_blue_part))
        self.free_end_angle_to_blue_part_bias_equations = processed.free_end_angle_to_blue_part_bias_equations
        self.fixed_end_angle_to_blue_part_bias_equations = processed.fixed_end_angle_to_blue_part_bias_equations
        processed.find_fixed_coordinates(bias_equations_free=self.free_end_angle_to_blue_part_bias_equations,
                                            bias_equations_fixed=self.fixed_end_angle_to_blue_part_bias_equations)
        processed.spring_length = np.linalg.norm(processed.free_ends_coordinates - processed.fixed_end_fixed_coordinates, axis=2)
        processed.spring_length = utils.interpolate_data(processed.spring_length,
                                                    utils.find_cells_to_interpolate(processed.spring_length))
        # print("0"*50)
        _, self.length_bias_equation = processed.norm_values(processed.free_end_fixed_coordinates_angle_to_nest, processed.spring_length,
                                                             bias_bool=processed.rest_bool,
                                                             find_boundary_column=False)
        # processed.calc_pulling_angle()
        processed.calc_spring_length(bias_equations=self.length_bias_equation,first_calib=True)
        self.zero_length = np.nanmedian(processed.spring_length.flatten())
        print("zero length: ", self.zero_length)

    def get_data(self):
        print("Getting data from... ")
        calibration_force_direction = np.array(())
        calibration_force_magnitude = np.array(())
        pulling_angle = np.array(())
        extension = np.array(())
        weights = np.array(())
        nest_direction = np.array(())
        self.processed_list = []
        # from data_analysis.data_preparation import PostProcessing
        first = False
        for dir, weight in zip(self.directories[:], self.weights[:]):
            processed = PostProcessing(dir)
            processed.find_fixed_coordinates(bias_equations_free=self.free_end_angle_to_blue_part_bias_equations,
                                            bias_equations_fixed=self.fixed_end_angle_to_blue_part_bias_equations)
            processed.calc_pulling_angle()
            processed.calc_spring_length(zero_length=self.zero_length, bias_equations=self.length_bias_equation)
            processed.calc_calibration_force(weight)
            processed.weight = weight
            calibration_force_magnitude = np.append(calibration_force_magnitude, processed.calibration_force_magnitude)
            pulling_angle = np.append(pulling_angle, (processed.pulling_angle).flatten())
            extension = np.append(extension, processed.spring_extension.flatten())
            if first:
                # pulling_angle = np.append(pulling_angle,np.zeros(processed.pulling_angle.shape))
                # extension = np.append(extension,np.zeros(processed.spring_extension.shape))
                calibration_force_direction = np.append(calibration_force_direction, np.zeros(processed.calibration_force_direction.shape))
                # calibration_force_direction = np.append(calibration_force_direction, processed.calibration_force_direction)
                first = False
            else:
                calibration_force_direction = np.append(calibration_force_direction, processed.calibration_force_direction)
            weights = np.append(weights, np.repeat(weight, len(processed.calibration_force_magnitude)))
            nest_direction = np.append(nest_direction, processed.fixed_end_fixed_coordinates_angle_to_nest.flatten())
            data = np.vstack((pulling_angle,
                              extension,
                              calibration_force_direction,
                              calibration_force_magnitude,
                              weights,
                              nest_direction
                              )).transpose()
            self.processed_list.append(processed)

            # import cv2
            # video_name = processed.directory.split("\\")[-1]
            # video_path = "Z:\\Dor_Gabay\\ThesisProject\\data\\videos\\calibration2\\{}.MP4".format(video_name)
            # if "3" in video_name:
            #     video = cv2.VideoCapture(video_path)
            #     point = (int(processed.object_center_repeated[0, 0, 0]), int(processed.object_center_repeated[0, 0, 1]))
            #     point2 = (int(processed.object_center_repeated[0, 0, 0]), int(processed.object_center_repeated[0, 0, 1] + 20))
            #     point3 = (int(processed.object_center_repeated[0, 0, 0]), int(processed.object_center_repeated[0, 0, 1] + 40))
            #     point4 = (int(processed.object_center_repeated[0, 0, 0]), int(processed.object_center_repeated[0, 0, 1] + 60))
            #     # ret, frame = video.read()
            #     for f in range(1500):
            #         ret, frame = video.read()
            #         cv2.putText(frame, f"pulling angle: {processed.pulling_angle[f, 0]}", point, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #                     (0, 0, 255), 2)
            #         print(processed.spring_extension[f])
            #         cv2.putText(frame, f"pulling angle: {processed.spring_extension[f]}", point2,
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            #         cv2.putText(frame, f"force_direction: {processed.calibration_force_direction[f]}", point3,
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            #         cv2.putText(frame, f"force_magnitude: {processed.calibration_force_magnitude[f]}", point4,
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            #         cv2.imshow("frame", frame)
            #         cv2.waitKey(20)

        #remove nan rows
        # print("shape of data before removing nan rows: ", data.shape)
        data = data[~np.isnan(data).any(axis=1)]
        # print("shape of data after removing nan rows: ", data.shape)
        self.X = data[:, 0:2]
        # self.y = np.sin(data[:, 2]) * data[:, 3]
        # self.y = data[:, 2:4]
        self.y = np.sin(data[:, 2]) * data[:, 3]
        self.weights_labels = data[:, 4]
        self.nest_direction = data[:, 5]


    def plot_pulling_angle_to_nest_angle(self):
        save_path = os.path.join(self.output_path, "plots","weights")
        os.makedirs(save_path, exist_ok=True)
        for processed in self.processed_list:
            title = np.round(processed.weight, 3)
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

            x = self.nest_direction[self.weights_labels == processed.weight]
            above = x > np.pi
            below = x < np.pi
            x[above] = x[above] - np.pi
            x[below] = x[below] + np.pi
            x -= np.pi

            y1 = self.X[self.weights_labels == processed.weight][:, 0]
            above = y1 > np.pi
            below = y1 < np.pi
            y1[above] = y1[above] - np.pi
            y1[below] = y1[below] + np.pi
            y1 -= np.pi
            ax1.plot(x, y1, "o",alpha=0.5)
            ax1.set_xlabel("Direction to the nest")
            ax1.set_ylabel("Pulling direction")
            ax1.set_title(f"weight: {title}")

            y2 = self.X[self.weights_labels == processed.weight][:, 1]
            ax2.plot(x, y2, "o",alpha=0.5)
            ax2.set_xlabel("Direction to the nest")
            ax2.set_ylabel("Spring extension")

            y_pred = self.model.predict(self.X[self.weights_labels == processed.weight, :])
            y3 = self.y[self.weights_labels == processed.weight]#[:, 0]
            direction_pred = y_pred#[:, 0]
            # above = y3 > 0
            # below = y3 < 0
            # y3[above] = y3[above] - np.pi
            # y3[below] = y3[below] + np.pi
            # direction_pred[above] = direction_pred[above] - np.pi
            # direction_pred[below] = direction_pred[below] + np.pi
            ax3.plot(x, y3, "o",alpha=0.5)
            ax3.plot(x, direction_pred, "o", color="red",alpha=0.1)
            ax3.set_xlabel("Direction to the nest")
            ax3.set_ylabel("Force")

            y4 = self.y[self.weights_labels == processed.weight]#[:, 1]
            extent_pred = y_pred#[:, 1]
            # ax4.plot(x, y4, "o",alpha=0.5)
            # ax4.plot(x, extent_pred, "o", color="red",alpha=0.1)
            ax4.plot()
            ax4.set_xlabel("Direction to the nest")
            ax4.set_ylabel("Force magnitude")

            fig.tight_layout()
            fig.set_size_inches(7.5, 5)
            fig.savefig(os.path.join(save_path, f"weight_{title}.png"))
        pass

    def test_correlation(self,calibration_model):
        print("-"*60)
        data_dir = "Z:\\Dor_Gabay\\ThesisProject\\data\\exp_collected_data\\15.9.22\\plus0.3mm_force\\S5280006\\"
        object = PostProcessing(data_dir, calibration_model=calibration_model)
        # total_force = np.sin(object.force_direction) * object.force_magnitude
        total_force = object.force
        net_force = np.nansum(total_force, axis=1)
        import pandas as pd
        velocity_spaced = np.nanmedian(utils.calc_angular_velocity(object.fixed_end_fixed_coordinates_angle_to_nest, diff_spacing=20), axis=1)
        corr_df = pd.DataFrame({"net_force": net_force, "angular_velocity": velocity_spaced})
        corr_df = corr_df.dropna()
        correlation_score = corr_df.corr()
        # print(correlation_score)
        return correlation_score

    def find_best_degree(self):
        number_degrees = list(range(1, 3))
        plt_mean_squared_error = []
        plt_r_squared = []
        corr_scores = []
        best_degrees = []
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(self.X, self.y, self.weights_labels, test_size=0.2, random_state=42)
        # X_train_t, X_val, y_train_t, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        # print("X_train shape: ", X_train.shape)
        for degree in number_degrees:
            model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            plt_mean_squared_error.append(mean_squared_error(y_test, y_pred, squared=False))
            plt_r_squared.append(model.score(X_test, y_test))
            # corr_scores.append(self.test_correlation(model))
            # print(corr_scores[-1])
        best_degrees.append(number_degrees[np.argmin(plt_mean_squared_error)])
        # best_degrees.append(number_degrees[np.argmax([corr_scores[i].iloc[0, 1] for i in range(len(corr_scores))])])
        # print(f"best correlation score: {np.max([corr_scores[i].iloc[0, 1] for i in range(len(corr_scores))])}")
        self.degree = max(set(best_degrees), key=best_degrees.count)

        print("Best polynomial degree is for force calibration equation: ", self.degree)
        self.model = make_pipeline(PolynomialFeatures(degree=self.degree), LinearRegression())
        self.model.fit(X_train, y_train)
        # save model
        with open(os.path.join(self.output_path,"calibration_model.pkl"), "wb") as f:
            pickle.dump(self.model, f)
        y_pred = self.model.predict(X_test)
        y_pred = np.repeat(y_pred[:, np.newaxis], 3, axis=1)
        print(f"X_train pulling_angle min: {np.min(X_train[:,0])}, X_train max: {np.max(X_train[:,0])}, mean: {np.mean(X_train[:,0])}")
        print(f"X_train spring_extension min: {np.min(X_train[:,1])}, X_train max: {np.max(X_train[:,1])}, mean: {np.mean(X_train[:,1])}")
        # print(f"y_train force direction min: {np.min(y_train[:,0])}, y_train max: {np.max(y_train[:,0])}, mean: {np.mean(y_train[:,0])}")
        # print(f"y_train force magnitude min: {np.min(y_train[:,1])}, y_train max: {np.max(y_train[:,1])}, mean: {np.mean(y_train[:,1])}")
        print("r squared (before removing out layers): ", self.model.score(X_test, y_test))
        y_test = np.repeat(y_test[:, np.newaxis], 3, axis=1)
        print("mean squared error (before removing out layers): ", mean_squared_error(y_test, y_pred, squared=False))
        self.ploting_fitting_results_data = (number_degrees, plt_mean_squared_error, plt_r_squared,y_test,y_pred, weights_test)

    def plot_fitting_results(self):
        number_degrees, plt_mean_squared_error, plt_r_squared, y_test, y_pred, weights_test = self.ploting_fitting_results_data
        save_dir = os.path.join(self.output_path, "plots")
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving plots to {save_dir}")
        for true,pred,name in \
                zip([np.abs(y_test[:, 0])*y_test[:, 1],y_test[:, 0],y_test[:, 1]],
                    [np.abs(y_pred[:, 0])*y_pred[:, 1],y_pred[:, 0],y_pred[:, 1]],
                    ["angle_times_extension","angle","extension"]):
            fig, ax = plt.subplots()
            ax.scatter(true,pred, c=weights_test, cmap="viridis")
            from matplotlib import cm
            cmap = cm.get_cmap('viridis', 10)
            sm = cm.ScalarMappable(cmap=cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm)
            cbar.set_label('Weight (g)')
            ax.set_xlabel("y_true")
            ax.set_ylabel("y_predicted")
            plt.savefig(os.path.join(save_dir, f"pred_true_comparison-{name}.png"))
            plt.clf()

        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Degree")
        ax1.set_ylabel("Mean Squared Error", color="red")
        ax1.plot(number_degrees, plt_mean_squared_error, color="red")
        ax1.scatter(number_degrees, plt_mean_squared_error, color="green")
        ax2 = ax1.twinx()
        ax2.set_ylabel("R squared", color="blue")
        ax2.plot(number_degrees, plt_r_squared, color="blue")
        from matplotlib.ticker import MaxNLocator
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(save_dir, "mean_squared_error.png"))
        plt.clf()

def iter_dirs(spring_type_directories,function):
    springs_analysed = {}
    for spring_type in spring_type_directories:
        springs_analysed[spring_type] = {}
        for video_data in spring_type_directories[spring_type]:
            video_name = video_data.split("\\")[-2]
            object = function(video_data)
            # object.save_data(video_data)
            # springs_analysed[spring_type][video_name] = object
    return springs_analysed

def collect_normalization_length(video_path,data_dir):
    import cv2
    from general_video_scripts.collect_color_parameters import neutrlize_colour
    video_folder = "\\".join(video_path.split("\\")[:-1])
    video_name = video_path.split("\\")[-1].split(".")[0]
    preferences_path = "\\".join(video_path.split("\\")[:-2]+["video_preferences.pickle"])
    # with open(os.path.join(video_folder,"parameters", f"{video_name}_video_parameters.pickle"), "rb") as f:
    with open(os.path.normpath(preferences_path), "rb") as f:
        video_parameters = pickle.load(f)[video_path]
    video = cv2.VideoCapture(video_path)
    # collect the fixed ends mask
    from video_analysis.springs_detector import Springs
    areas_medians = []
    blue_areas_medians = []
    # while frame is not None:
    for i in range(100):
        ret, frame = video.read()
        frame = neutrlize_colour(frame)
        springs = Springs(video_parameters,image=frame,previous_detections=None)
        fixed_ends_labeled = springs.bundles_labeled
        # turn to zero all labels that are not in springs.fixed_ends_bundles_labels , without a loop
        fixed_ends_labeled[np.isin(fixed_ends_labeled, springs.fixed_ends_bundles_labels, invert=True)] = 0
        # find the area size of each label
        from scipy.ndimage import label
        fixed_ends_labeled, num_labels = label(fixed_ends_labeled)
        # find the label with the biggest area
        from scipy.ndimage import sum
        areas = sum(fixed_ends_labeled, fixed_ends_labeled, index=np.arange(1, num_labels + 1))/np.arange(1, num_labels + 1)
        blue_labeled, num_labels = label(springs.mask_blue_full)
        blue_areas = sum(blue_labeled, blue_labeled, index=np.arange(1, num_labels + 1))/np.arange(1, num_labels + 1)
        if len(areas) > 0 and len(blue_areas) > 0:
            median_area = np.median(areas)
            blue_median_area = np.median(blue_areas)
            areas_medians.append(median_area)
            blue_areas_medians.append(blue_median_area)
    # save the median area in data folder
    median_medians = np.median(areas_medians)
    blue_median_medians = np.median(blue_areas_medians)
    # print(f"median area is {median_medians}, median blue area is {blue_median_medians}")
    # print("median area/median blue area", median_medians/blue_median_medians)
    with open(os.path.join(data_dir, "blue_median_area.pickle"), "wb") as f:
        pickle.dump(blue_median_medians, f)
    return blue_median_medians

if __name__ == "__main__":
    data_dir = "Z:\\Dor_Gabay\\ThesisProject\\data\\exp_collected_data\\15.9.22\\plus0.3mm_force\\S5280006\\"

    calibration_dir1 = "Z:\\Dor_Gabay\\ThesisProject\\data\\calibration\\post_slicing\\calibration1\\"
    directories_1 = [os.path.join(calibration_dir1, o) for o in os.listdir(calibration_dir1)
                     if os.path.isdir(os.path.join(calibration_dir1, o)) and "_sliced" in os.path.basename(o)]
    weights1 = list(np.array([0.10606, 0.14144, 0.16995, 0.19042, 0.16056, 0.15082]) - 0.10506)

    calibration_dir2 = "Z:\\Dor_Gabay\\ThesisProject\\data\\calibration\\post_slicing\\calibration2\\"
    directories_2 = [os.path.join(calibration_dir2, o) for o in os.listdir(calibration_dir2)
                     if os.path.isdir(os.path.join(calibration_dir2, o)) and "_sliced" in os.path.basename(o) and "6" not in os.path.basename(o)]# and "9" not in os.path.basename(o)]
    # weights2 = list(np.array([0.10582,0.13206,0.15650,0.18405,0.21030,0.46612])-0.10582)
    # weights2 = list(np.array([0.10582,0.13206,0.15650,0.18405,0.46612])-0.10582)
    weights2 = list(np.array([0.10582,0.13206,0.15650,0.18405,0.46612])-0.10582)

    # video_dir = "Z:\\Dor_Gabay\\ThesisProject\\data\\videos\\calibration2\\"
    # videos_paths = [os.path.join(video_dir, o) for o in os.listdir(video_dir)
    #                 if os.path.isfile(os.path.join(video_dir, o)) and ".MP4" in os.path.basename(o) and "6" not in os.path.basename(o)]
    # for video_path,dir  in zip(videos_paths,directories_2):
    #     print(collect_normalization_length(video_path,dir))

    # video_path = "Z:\\Dor_Gabay\\ThesisProject\\data\\videos\\15.9.22\\plus0.3mm_force\\S5280006.MP4"
    # print(collect_normalization_length(video_path, data_dir))
    #
    # all_dirs = [directories_1[0],directories_2[0],*directories_1[1:],*directories_2[1:]]
    # all_weights = [weights1[0],weights2[0],*weights1[1:],*weights2[1:]]
    # calibration_model = Calibration(directories=all_dirs,weights=all_weights,
    #                                 output_path="Z:\\Dor_Gabay\\ThesisProject\\data\\calibration\\post_slicing\\").model
    # calibration_model = Calibration(directories=[directories_2[0]], weights=[weights2[0]],
    #                                 output_path=calibration_dir2).model
    calibration_model = Calibration(directories=directories_2, weights=weights2,
                                    output_path=calibration_dir2).model
    # #load calibration model
    # print(f"Loading calibration model from: {calibration_dir2}calibration_model.pkl")
    # print("-"*60)
    print("-"*60)
    calibration_model_loaded = pickle.load(open(calibration_dir2+"calibration_model.pkl","rb"))
    # object = PostProcessing(data_dir,calibration_model=calibration_model_loaded)
    # object.save_data(data_dir)
    correlation_score = Calibration.test_correlation(Calibration,calibration_model_loaded)
    print(f"best correlation score: {correlation_score.iloc[0, 1]}")





    #     def attaches_events(self,N_ants_array):
    #         N_ants_array = copy.copy(N_ants_array)
    #         N_ants_array = utils.column_dilation(N_ants_array)
    #         diff = np.vstack((np.diff(N_ants_array,axis=0),np.zeros(N_ants_array.shape[1]).reshape(1,N_ants_array.shape[1])))
    #         labeled_single_attaches,num_labels_single = label(utils.convert_bool_to_binary(N_ants_array==1))
    #         labeled_single_attaches_adapted = np.vstack((labeled_single_attaches,np.zeros(labeled_single_attaches.shape[1]).reshape(1,labeled_single_attaches.shape[1])))[1:,:]
    #         labels_0_to_1 = np.unique(labeled_single_attaches_adapted[(diff==1)&(N_ants_array==0)])
    #         labels_1_to_0 = np.unique(labeled_single_attaches_adapted[(diff==-1)&(N_ants_array==1)])
    #         self.labeled_zero_to_one,_ = label(np.isin(labeled_single_attaches,labels_0_to_1))
    #         self.labeled_zero_to_one = self.labeled_zero_to_one[:,list(range(0,self.labeled_zero_to_one.shape[1],2))]
    #         self.labeled_one_to_zero,_ = label(np.isin(labeled_single_attaches,labels_1_to_0))
    #         self.labeled_one_to_zero = self.labeled_one_to_zero[:,list(range(0,self.labeled_one_to_zero.shape[1],2))]
    #
    #     def fiter_attaches_events(self, labeled):
    #         EVENT_LENGTH = 150
    #         PRE_EVENT_LENGTH = 15
    #         ar = self.labeled_zero_to_one
    #         events_to_keep = []
    #         self.events_sts = []
    #         for event in np.unique(ar):
    #             frames = np.where(ar[:,:]==event)[0]
    #             # print(event,":",len(frames))
    #             pre_frames = np.arange(frames[0]-PRE_EVENT_LENGTH,frames[0]-1,1)
    #             spring = np.where(ar[:,:]==event)[1][0]
    #             # if len(frames)>EVENT_LENGTH:
    #             if True:
    #                 if np.sum(np.isnan(self.N_ants_around_springs[frames[0]-10:frames[0],spring]))==0:
    #                     pre_frames_lengths = np.take(self.springs_length_processed[:,spring],pre_frames)
    #                     pre_frames_median = np.nanmedian(pre_frames_lengths)
    #                     # if (not pre_frames_median<self.rest_length*0.8) & (not pre_frames_median>self.rest_length*1.2):
    #                     if True:
    #                         self.events_sts.append(np.nanstd(pre_frames_lengths)/pre_frames_median)
    #                         # if not np.std()
    #                         events_to_keep.append(event)
    #         labeled = copy.copy(labeled)
    #         labeled[np.invert(np.isin(labeled,events_to_keep))] = 0
    #         labeled = utils.column_dilation(labeled)
    #         labeled, _ =  label(labeled!=0)
    #         labeled = labeled[:,list(range(0,labeled.shape[1],2))]
    #         # print(.shape)
    #         return labeled
    #
    #     def clean_ant_switching(self,N_ants_array):
    #         N_ants_array = copy.copy(N_ants_array)
    #         N_ants_array = utils.column_dilation(N_ants_array)
    #         labeled_all_attaches,num_labels_all = label(utils.convert_bool_to_binary(N_ants_array>=1))
    #
    #         labels_to_remove = np.unique(labeled_all_attaches[N_ants_array>1&np.isnan(N_ants_array)])
    #         N_ants_array[np.isin(labeled_all_attaches,labels_to_remove)] = 0
    #         N_ants_array = N_ants_array[:,list(range(0,N_ants_array.shape[1],2))]
    #         return N_ants_array
