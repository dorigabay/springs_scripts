import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
from data_analysis import utils
import pickle
from calibration.utils import bound_angle


class PostProcessing:

    def __init__(self, directory, calibration_model):
        self.directory = directory
        self.load_data(directory)
        self.n_ants_processing()
        self.calc_distances()
        self.repeat_values()
        self.calc_angle()
        self.calibration_model = calibration_model
        self.find_fixed_coordinates(bias_equations_free=self.free_end_angle_to_blue_part_bias_equations,
                                    bias_equations_fixed=self.fixed_end_angle_to_blue_part_bias_equations)
        self.calc_spring_length()
        self.calc_pulling_angle(zero_angles=True)
        self.calc_force(calibration_model)
        self.calculations()

    def load_data(self,directory):
        print("-" * 60)
        print("loading data from:", directory)
        self.norm_size = pickle.load(open(os.path.join(directory,"blue_median_area.pickle"), "rb"))
        directory = os.path.join(directory, "raw_analysis")+"\\"
        self.N_ants_around_springs = np.loadtxt(os.path.join(directory,"N_ants_around_springs.csv"), delimiter=",")
        self.num_of_springs = self.N_ants_around_springs.shape[1]
        self.size_ants_around_springs = np.loadtxt(os.path.join(directory,"size_ants_around_springs.csv"), delimiter=",")
        fixed_ends_coordinates_x = np.loadtxt(os.path.join(directory,"fixed_ends_coordinates_x.csv"), delimiter=",")
        fixed_ends_coordinates_y = np.loadtxt(os.path.join(directory,"fixed_ends_coordinates_y.csv"), delimiter=",")
        self.fixed_ends_coordinates = np.stack((fixed_ends_coordinates_x, fixed_ends_coordinates_y), axis=2)
        free_ends_coordinates_x = np.loadtxt(os.path.join(directory,"free_ends_coordinates_x.csv"), delimiter=",")
        free_ends_coordinates_y = np.loadtxt(os.path.join(directory,"free_ends_coordinates_y.csv"), delimiter=",")
        self.free_ends_coordinates = np.stack((free_ends_coordinates_x, free_ends_coordinates_y), axis=2)
        blue_part_coordinates_x = np.loadtxt(os.path.join(directory,"blue_part_coordinates_x.csv"), delimiter=",")
        blue_part_coordinates_y = np.loadtxt(os.path.join(directory,"blue_part_coordinates_y.csv"), delimiter=",")
        blue_part_coordinates = np.stack((blue_part_coordinates_x, blue_part_coordinates_y), axis=2)
        self.object_center = blue_part_coordinates[:, 0, :]
        self.blue_tip_coordinates = blue_part_coordinates[:, -1, :]
        self.num_of_frames = self.N_ants_around_springs.shape[0]

    def smoothing_n_ants(self, array):
        for col in range(array.shape[1]):
            array[:,col] = np.abs(np.round(savgol_filter(array[:,col], 31, 2)))
        return array

    def n_ants_processing(self):
        undetected_springs_for_long_time = utils.filter_continuity(utils.convert_bool_to_binary(
            np.isnan(self.fixed_ends_coordinates[:, :, 0])),min_size=8)
        self.N_ants_around_springs = np.round(utils.interpolate_data(self.N_ants_around_springs,
                                                    utils.find_cells_to_interpolate(self.N_ants_around_springs)))
        self.N_ants_around_springs[np.isnan(self.N_ants_around_springs)] = 0
        self.N_ants_around_springs = self.smoothing_n_ants(self.N_ants_around_springs)
        self.N_ants_around_springs[undetected_springs_for_long_time] = np.nan
        all_small_attaches = np.zeros(self.N_ants_around_springs.shape,int)
        for n in np.unique(self.N_ants_around_springs)[1:]:
            if not np.isnan(n):
                short_attaches = utils.filter_continuity(self.N_ants_around_springs==n,max_size=30)
                all_small_attaches[short_attaches] = 1
        self.N_ants_around_springs = np.round(utils.interpolate_data(self.N_ants_around_springs,
                                                                     all_small_attaches.astype(bool)))
        self.rest_bool = self.N_ants_around_springs == 0

    def calc_distances(self):
        object_center = np.repeat(self.object_center[:, np.newaxis, :], self.num_of_springs, axis=1)
        self.blue_length = np.nanmedian(np.linalg.norm(self.blue_tip_coordinates - self.object_center, axis=1))
        self.object_center_to_free_end_distance = np.linalg.norm(self.free_ends_coordinates - object_center, axis=2)
        self.object_center_to_fixed_end_distance = np.linalg.norm(self.fixed_ends_coordinates - object_center, axis=2)

    def repeat_values(self):
        nest_direction = np.stack((self.object_center[:, 0], self.object_center[:, 1]-100), axis=1)
        self.nest_direction_repeated = np.repeat(nest_direction[:, np.newaxis, :], self.free_ends_coordinates.shape[1], axis=1)
        self.object_center_repeated = copy.copy(np.repeat(self.object_center[:, np.newaxis, :], self.free_ends_coordinates.shape[1], axis=1))
        self.blue_tip_coordinates_repeated = np.repeat(self.blue_tip_coordinates[:, np.newaxis, :], self.free_ends_coordinates.shape[1], axis=1)

    def calc_angle(self):
        self.free_end_angle_to_nest = utils.calc_angle_matrix(self.free_ends_coordinates, self.object_center_repeated, self.nest_direction_repeated)+np.pi
        self.fixed_end_angle_to_nest = utils.calc_angle_matrix(self.fixed_ends_coordinates, self.object_center_repeated, self.nest_direction_repeated)+np.pi
        self.blue_part_angle_to_nest = utils.calc_angle_matrix(self.nest_direction_repeated, self.object_center_repeated, self.blue_tip_coordinates_repeated)+np.pi
        self.free_end_angle_to_blue_part = utils.calc_angle_matrix(self.blue_tip_coordinates_repeated, self.object_center_repeated, self.free_ends_coordinates)+np.pi
        self.fixed_end_angle_to_blue_part = utils.calc_angle_matrix(self.blue_tip_coordinates_repeated, self.object_center_repeated,self.fixed_ends_coordinates)+np.pi
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
        # if len(columns_on_boundry) == 1:
        #     column_on_boundry = columns_on_boundry[0]
        # else:
        #     raise ValueError("more than one column on boundry")
        return columns_on_boundry

    def norm_values(self, X, Y, bias_bool=None, find_boundary_column=False, bias_equations=None):
        if bias_bool is not None:
            X_bias = copy.deepcopy(X)
            Y_bias = copy.deepcopy(Y)
            X_bias[np.invert(bias_bool)] = np.nan
            Y_bias[np.invert(bias_bool)] = np.nan
        else:
            X_bias = X
            Y_bias = Y
        if find_boundary_column:
            columns_on_boundary = self.find_column_on_boundry(Y)
            for col in columns_on_boundary:
                above_nonan = Y_bias[:, col] > np.pi
                Y_bias[above_nonan, col] -= 2*np.pi
            # above_nonan = Y_bias[:, column_on_boundary] > np.pi
            # Y_bias[above_nonan, column_on_boundary] -= 2*np.pi
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
            for col in columns_on_boundary:
                above = Y[:, col] > np.pi
                Y[above, col] -= 2*np.pi
            # above = Y[:, column_on_boundary] > np.pi
            # Y[above, column_on_boundary] -= 2*np.pi
        for i in range(self.num_of_springs):
            bias_equation = bias_equations[i]
            normed_Y[:, i] = utils.normalize(Y[:, i], X[:, i], bias_equation)
        if find_boundary_column:
            for col in columns_on_boundary:
                below = normed_Y[:, col] < 0
                normed_Y[below, col] += 2*np.pi
            # below = normed_Y[:, column_on_boundary] < 0
            # normed_Y[below, column_on_boundary] += 2*np.pi
        return normed_Y, bias_equations

    # def norm_values(self, X, Y, bias_bool=None, find_boundary_column=False, bias_equations=None):
    #     if bias_bool is not None:
    #         X_bias = copy.deepcopy(X)
    #         Y_bias = copy.deepcopy(Y)
    #         X_bias[np.invert(bias_bool)] = np.nan
    #         Y_bias[np.invert(bias_bool)] = np.nan
    #     else:
    #         X_bias = X
    #         Y_bias = Y
    #     if find_boundary_column:
    #         columns_on_boundary = self.find_column_on_boundry(Y)
    #         for col in columns_on_boundary:
    #             above_nonan = Y_bias[:, col] > np.pi
    #             Y_bias[above_nonan, col] -= 2*np.pi
    #         # above_nonan = Y_bias[:, column_on_boundary] > np.pi
    #         # Y_bias[above_nonan, column_on_boundary] -= 2*np.pi
    #     Y_bias -= np.nanmedian(Y_bias, axis=0)
    #     normed_Y = np.zeros(Y.shape)
    #     import pandas as pd
    #     if bias_equations is None:
    #         bias_equations = []
    #         for i in range(X.shape[1]):
    #             df = pd.DataFrame({"x": X_bias[:, i], "y": Y_bias[:, i]}).dropna()
    #             bias_equation = utils.deduce_bias_equation(df["x"], df["y"])
    #             bias_equations.append(bias_equation)
    #     if find_boundary_column:
    #         Y = copy.deepcopy(Y)
    #         for col in columns_on_boundary:
    #             above = Y[:, col] > np.pi
    #             Y[above, col] -= 2*np.pi
    #         # above = Y[:, column_on_boundary] > np.pi
    #         # Y[above, column_on_boundary] -= 2*np.pi
    #     for i in range(self.num_of_springs):
    #         bias_equation = bias_equations[i]
    #         normed_Y[:, i] = utils.normalize(Y[:, i], X[:, i], bias_equation)
    #     if find_boundary_column:
    #         for col in columns_on_boundary:
    #             below = normed_Y[:, col] < 0
    #             normed_Y[below, col] += 2*np.pi
    #         # below = normed_Y[:, column_on_boundary] < 0
    #         # normed_Y[below, column_on_boundary] += 2*np.pi
    #     return normed_Y, bias_equations

    def find_fixed_coordinates(self,bias_equations_free,bias_equations_fixed):
        def calc_fixed(distance_to_object_center,end_type):
            median_distance = None  # removable line
            angle_to_blue = None  # removable line
            if end_type== "free":
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
                median_distance = np.nanmedian(distance_to_object_center, axis=0)
            median_distance = np.repeat(median_distance[np.newaxis, :], self.num_of_frames, axis=0)
            angle_to_blue_part_normed = bound_angle(self.blue_part_angle_to_nest+angle_to_blue-np.pi/2)
            fixed_coordinates = self.object_center_repeated + np.stack((np.cos(angle_to_blue_part_normed) * median_distance,
                                                                        np.sin(angle_to_blue_part_normed) * median_distance), axis=2)
            return fixed_coordinates
        self.fixed_end_fixed_coordinates = calc_fixed(self.object_center_to_fixed_end_distance,end_type="fixed")
        self.free_end_fixed_coordinates = calc_fixed(self.object_center_to_free_end_distance,end_type="free")

        # self.video_path = "Z:\\Dor_Gabay\\ThesisProject\\data\\videos\\15.9.22\\plus0.3mm_force\\S5280006.MP4"
        # frame = utils.load_first_frame(self.video_path)
        # import cv2
        # for fix, free in zip(self.fixed_end_fixed_coordinates[0], self.free_end_fixed_coordinates[0]):
        #     print(fix, free)
        #     frame = cv2.circle(frame, tuple(fix.astype(int)), 5, (0, 0, 255), -1)
        #     frame = cv2.circle(frame, tuple(free.astype(int)), 5, (0, 255, 0), -1)
        # cv2.imshow("frame", frame)
        # cv2.waitKey(0)

        self.calc_fixed_coordinates_angles()

    def calc_fixed_coordinates_angles(self):
        self.free_end_fixed_coordinates_angle_to_nest = utils.calc_angle_matrix(self.free_end_fixed_coordinates, self.object_center_repeated, self.nest_direction_repeated)+np.pi
        self.fixed_end_fixed_coordinates_angle_to_nest = utils.calc_angle_matrix(self.fixed_end_fixed_coordinates, self.object_center_repeated, self.nest_direction_repeated)+np.pi
        self.free_end_fixed_coordinates_angle_to_blue_part = utils.calc_angle_matrix(self.blue_tip_coordinates_repeated,
                                                                   self.object_center_repeated,
                                                                   self.free_end_fixed_coordinates)+np.pi
        self.fixed_end_fixed_coordinates_angle_to_blue_part = utils.calc_angle_matrix(self.blue_tip_coordinates_repeated,
                                                                    self.object_center_repeated,
                                                                    self.fixed_end_fixed_coordinates)+np.pi

    def calc_pulling_angle(self,zero_angles=False):
        self.pulling_angle = utils.calc_angle_matrix(self.free_end_fixed_coordinates,self.fixed_end_fixed_coordinates,self.object_center_repeated)+np.pi
        self.pulling_angle, _ = self.norm_values(self.free_end_fixed_coordinates_angle_to_nest, self.pulling_angle, bias_bool=self.rest_bool, find_boundary_column=True)
        self.pulling_angle = self.pulling_angle - np.pi
        above = self.pulling_angle > 0
        below = self.pulling_angle < 0
        self.pulling_angle[above] = self.pulling_angle[above] - np.pi
        self.pulling_angle[below] = self.pulling_angle[below] + np.pi

        pulling_angle_at_rest = copy.copy(self.pulling_angle)
        pulling_angle_at_rest[~self.rest_bool] = np.nan
        # pulling_angle_at_rest -= np.nanmedian(pulling_angle_at_rest,axis=0)

        # pulling_angle_at_rest_diff = pulling_angle_at_rest - np.nanmean(pulling_angle_at_rest,axis=0)
        pulling_angle_at_rest_diff = np.diff(pulling_angle_at_rest,axis=0)
        lines_with_all_nans = np.all(np.isnan(pulling_angle_at_rest_diff),axis=1)
        pulling_angle_at_rest_mean_diff = np.nanmean(pulling_angle_at_rest_diff[~lines_with_all_nans],axis=1)

        pulling_angle_at_rest_mean_diff_matrix = np.zeros_like(pulling_angle_at_rest)
        pulling_angle_at_rest_mean_diff_matrix[0, :] = np.nan
        pulling_angle_at_rest_mean_diff_matrix[1:, :][lines_with_all_nans] = np.nan
        pulling_angle_at_rest_mean_diff_matrix[1:, :][~lines_with_all_nans, :] = pulling_angle_at_rest_mean_diff[:, np.newaxis]
        pulling_angle_at_rest_mean_diff_matrix = utils.interpolate_data(pulling_angle_at_rest_mean_diff_matrix, np.isnan(pulling_angle_at_rest_mean_diff_matrix))
        # print(pulling_angle_at_rest_mean_diff_matrix)
        # self.pulling_angle -= np.nanmedian(pulling_angle_at_rest,axis=0) - pulling_angle_at_rest_mean_diff_matrix
        self.pulling_angle -= np.nanmedian(pulling_angle_at_rest,axis=0)
        self.negative_pulling_angle = self.pulling_angle < 0
        # self.pulling_angle[self.negative_pulling_angle] *= -1
        # print(self.pulling_angle)
        # self.pulling_angle = self.pulling_angle[self.pulling_angle>0][:1000]

        # median_pulling_angle_at_rest = np.nanmedian(pulling_angle_at_rest, axis=0)
        # pulling_angle_at_rest -= median_pulling_angle_at_rest
        # print(pulling_angle_at_rest)
        # pulling_angle_at_rest -= pulling_angle_at_rest_mean_diff[:,np.newaxis]
        # # mean_diff = np.nanmean(np.diff(median_pulling_angle_at_rest),axis=1)
        # print(pulling_angle_at_rest_mean_diff.shape, pulling_angle_at_rest.shape)
        # # median_pulling_angle_at_rest_without_diff = median_pulling_angle_at_rest_without_diff - mean_diff[:,np.newaxis]
        #
        # self.pulling_angle -= pulling_angle_at_rest
        import pandas as pd
        # angular_velocity_moving_median = pd.Series(np.nanmedian(
        #     utils.calc_angular_velocity(self.fixed_end_fixed_coordinates_angle_to_nest, diff_spacing=20), axis=1)).rolling(50).median()
        # median_pulling_angle_at_rest_above = np.nanmedian(median_pulling_angle_at_rest[angular_velocity_moving_median > 0,:], axis=0)
        # all lines between 5750 and 10450 are below
        # median_pulling_angle_at_rest_below = np.nanmedian(median_pulling_angle_at_rest[5750:10450, :], axis=0)
        # one = np.nanmedian(median_pulling_angle_at_rest[5750:10450, :], axis=0)
        # two = np.nanmedian(median_pulling_angle_at_rest[:5750, :], axis=0)
        # three = np.nanmedian(median_pulling_angle_at_rest[10450:, :], axis=0)
        # median_pulling_angle_at_rest_matrix = np.zeros(self.pulling_angle.shape)
        # median_pulling_angle_at_rest_matrix[5750:10450,:] = one
        # median_pulling_angle_at_rest_matrix[:5750,:] = two
        # median_pulling_angle_at_rest_matrix[10450:,:] = three
        # print(median_pulling_angle_at_rest_below)
        # all line not between 5750 and 10450 are above
        # median_pulling_angle_at_rest_above = np.nanmedian(median_pulling_angle_at_rest[np.logical_and(5750 < np.arange(self.num_of_frames), np.arange(self.num_of_frames) > 10450),:], axis=0)
        # print(median_pulling_angle_at_rest_above)
        # median_pulling_angle_at_rest_below = np.nanmedian(median_pulling_angle_at_rest[angular_velocity_moving_median < 0,:], axis=0)
        # median_pulling_angle_at_rest_matrix[np.logical_and(5750 < np.arange(self.num_of_frames), np.arange(self.num_of_frames) > 10450),:] = median_pulling_angle_at_rest_below
        # median_pulling_angle_at_rest_above_no_nan = median_pulling_angle_at_rest_above[~np.isnan(median_pulling_angle_at_rest_above)]
        # un_nan_len_above = np.sum(np.invert(np.isnan(median_pulling_angle_at_rest_above)))
        # median_pulling_angle_at_rest_below = median_pulling_angle_at_rest[self.angular_velocity < 0]
        # median_pulling_angle_at_rest_below_no_nan = median_pulling_angle_at_rest_below[~np.isnan(median_pulling_angle_at_rest_below)]
        # un_nan_len_below = np.sum(np.invert(np.isnan(median_pulling_angle_at_rest_below)))
        # max_un_nan = len(median_pulling_angle_at_rest_above_no_nan) if len(median_pulling_angle_at_rest_above_no_nan) < len(median_pulling_angle_at_rest_below_no_nan) else len(median_pulling_angle_at_rest_below_no_nan)
        # median_pulling_angle_at_rest = np.concatenate((np.random.choice(median_pulling_angle_at_rest_above_no_nan, max_un_nan, replace=False), np.random.choice(median_pulling_angle_at_rest_below_no_nan, max_un_nan, replace=False)))
        # median_pulling_angle_at_rest = np.median(median_pulling_angle_at_rest)
        # self.pulling_angle -= median_pulling_angle_at_rest_matrix

        # self.pulling_angle_copy = copy.copy(self.pulling_angle)
        # win = 100
        # import pandas as pd
        # self.pulling_angle_interpolated = utils.interpolate_data(self.pulling_angle,~self.rest_bool)
        # pd_pulling_angle = pd.DataFrame(self.pulling_angle_interpolated)
        # rolling_median = pd_pulling_angle.rolling(win).median()
        #
        #
        # self.calc_force(calibration_model)
        # net_force = np.nansum(self.total_force, axis=1)
        #
        # velocity_spaced = np.nanmedian(
        #     utils.calc_angular_velocity(self.fixed_end_fixed_coordinates_angle_to_nest, diff_spacing=20), axis=1)
        # corr_df = pd.DataFrame({"net_force": net_force, "angular_velocity": velocity_spaced})
        # corr_df = corr_df.dropna()
        # correlation_score = corr_df.rolling(win).corr()
        # correlation_score = correlation_score.loc[correlation_score.index.get_level_values(1) == "net_force"]["angular_velocity"]
        # correlation_score = correlation_score[correlation_score.index.get_level_values(0)%100==0]
        # under_median = correlation_score[correlation_score < np.nanmedian(correlation_score)].index.get_level_values(0)
        # diff_under_median = np.diff(under_median)
        # print(diff_under_median>100)
        # #create na array, with the size of the rolling median
        # fixed_median = np.zeros(rolling_median.shape)
        # fixed_median[:,:] = np.nan
        # for i in under_median:
        #     fixed_median[i-win:i,:] = rolling_median.loc[i,:].values
        # #whatever is left empty, fill with the median
        # for col in rolling_median.columns:
        #     fixed_median[np.where(np.isnan(fixed_median[:,col])),col] = median_pulling_angle_at_rest[col]
        # self.pulling_angle = self.pulling_angle_interpolated - fixed_median

        # if zero_angles:
        #     rest_slots = utils.column_dilation(self.rest_bool)
        #     from scipy.ndimage import label
        #     labeled, _ = label(rest_slots!=0)
        #     labeled = labeled[:,list(range(0,labeled.shape[1],2))]
        #     corrs = []
        #     # sizes = [5,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,1250,1500,1750,2000]
        #     sizes = [1950,2000,2050,2100,2150,2200,2250,2300,2350,2400,2450,2500,2550,2600,2650,2700,2750,2800,2850,2900,2950,3000]
        #     # sizes = list(range(5,2000,5))
        #     for size in sizes:
        #         self.pulling_angle = copy.copy(self.pulling_angle_copy)
        #         for col in range(labeled.shape[1]):
        #             slots = np.unique(labeled[:,col])[1:]
        #             slots_idx = [np.where(labeled[:,col]==s)[0] for s in slots]
        #             slots_median = [np.nanmedian(self.pulling_angle[labeled[:,col]==s,col]) for s in slots]
        #             medians.append(np.nanmedian(slots_median))
        #             big_slots = [(s,i,m) for s,i,m in zip(slots,slots_idx,slots_median) if len(i)>size]
        #             for i, s in enumerate(big_slots):
        #                 slot, slot_idx, slot_median = s
        #                 bottom_dist = int(slot_idx[0]-big_slots[i-1][1][-1]/2) if i>0 else slot_idx[0]
        #                 top_dist = int(big_slots[i+1][1][0]-slot_idx[-1]/2) if i<len(big_slots)-1 else len(self.pulling_angle)-slot_idx[-1]
        #                 # print(bottom_dist, top_dist)
        #                 self.pulling_angle[bottom_dist:top_dist,col] -= slot_median
        #         self.calc_force(self.calibration_model)
        #         corrs.append(self.test_correlation())
        #     best_size = sizes[np.argmax(corrs)]
        #     print("best size is: {}".format(best_size))
        #     self.pulling_angle = copy.copy(self.pulling_angle_copy)
        #     for col in range(labeled.shape[1]):
        #         slots = np.unique(labeled[:, col])[1:]
        #         slots_idx = [np.where(labeled[:, col] == s)[0] for s in slots]
        #         slots_median = [np.nanmedian(self.pulling_angle[labeled[:, col] == s, col]) for s in slots]
        #         medians.append(np.nanmedian(slots_median))
        #         big_slots = [(s, i, m) for s, i, m in zip(slots, slots_idx, slots_median) if len(i) > best_size]
        #         for i, s in enumerate(big_slots):
        #             slot, slot_idx, slot_median = s
        #             bottom_dist = int(slot_idx[0] - big_slots[i - 1][1][-1] / 2) if i > 0 else slot_idx[0]
        #             top_dist = int(big_slots[i + 1][1][0] - slot_idx[-1] / 2) if i < len(big_slots) - 1 else len(
        #                 self.pulling_angle) - slot_idx[-1]
        #             # print(bottom_dist, top_dist)
        #             self.pulling_angle[bottom_dist:top_dist, col] -= slot_median
            #     for i, slot in enumerate(slots):
            #         for s in slots[i:]:
            #             s_idx = np.where(labeled[:, col] == s)[0]
            #             if len(slot_idx) > 1000:
            #                 self.slot_median = np.nanmedian(self.pulling_angle[s_idx, col])
            #                 break
            #         if slot != slots[-1]:
            #             next_slot_idx = np.where(labeled[:,col]==slots[i+1])[0]
            #             self.pulling_angle[slot_idx[-1]+1:next_slot_idx[0],col] -= self.slot_median
            #         elif slot == slots[-1]:
            #             self.pulling_angle[slot_idx[-1]+1:,col] -= self.slot_median
            # median_pulling_angle_at_rest = copy.copy(self.pulling_angle)
            # median_pulling_angle_at_rest[np.invert(self.rest_bool)] = np.nan
            # median_pulling_angle_at_rest = np.nanmedian(median_pulling_angle_at_rest, axis=0)
            # print(np.array(medians))
            # print(median_pulling_angle_at_rest)
            # self.pulling_angle -= median_pulling_angle_at_rest

    def calc_spring_length(self):
        self.spring_length = np.linalg.norm(self.free_end_fixed_coordinates - self.fixed_end_fixed_coordinates , axis=2)
        self.spring_length = utils.interpolate_data(self.spring_length,
                                                       utils.find_cells_to_interpolate(self.spring_length))
        self.spring_length = self.norm_values(self.fixed_end_angle_to_nest, self.spring_length, bias_bool=self.rest_bool, find_boundary_column=False)[0]
        self.spring_length /= self.norm_size
        median_spring_length_at_rest = copy.copy(self.spring_length)
        median_spring_length_at_rest[np.invert(self.rest_bool)] = np.nan
        median_spring_length_at_rest = np.nanmedian(median_spring_length_at_rest, axis=0)
        self.spring_length /= median_spring_length_at_rest
        self.spring_extension = self.spring_length -1# np.repeat(median_spring_length_at_rest[np.newaxis, :], self.num_of_frames, axis=0)
        # self.spring_extension = self.spring_length - np.repeat(median_spring_length_at_rest[np.newaxis, :], self.num_of_frames, axis=0)

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

    def calc_force(self,model):
        self.angular_force = True
        self.force_direction = np.zeros((self.num_of_frames,self.num_of_springs))
        self.force_magnitude = np.zeros((self.num_of_frames,self.num_of_springs))
        self.total_force = np.zeros((self.num_of_frames,self.num_of_springs))
        for s in range(self.num_of_springs):
            X = np.stack((self.pulling_angle[:,s],self.spring_extension[:,s]),axis=1)
            un_nan_bool = ~np.isnan(X).any(axis=1)
            X = X[un_nan_bool]
            forces_predicted = model.predict(X)
            if not self.angular_force:
                self.force_direction[un_nan_bool,s] = forces_predicted[:,0]
                self.force_magnitude[un_nan_bool,s] = forces_predicted[:,1]
            else:
                self.total_force[un_nan_bool,s] = forces_predicted
        # self.total_force[self.negative_pulling_angle] *= -1
        self.total_force *= -1
        # import cv2
        # video_path = "Z:\\Dor_Gabay\\ThesisProject\\data\\videos\\15.9.22\\plus0.3mm_force\\S5280006.MP4"
        # video = cv2.VideoCapture(video_path)
        # point = (int(self.object_center_repeated[0, 0, 0]), int(self.object_center_repeated[0, 0, 1]))
        # point2 = (int(self.object_center_repeated[0, 0, 0]), int(self.object_center_repeated[0, 0, 1] + 20))
        # point3 = (int(self.object_center_repeated[0, 0, 0]), int(self.object_center_repeated[0, 0, 1] + 40))
        # point4 = (int(self.object_center_repeated[0, 0, 0]), int(self.object_center_repeated[0, 0, 1] + 60))
        # for f in range(10000):
        #     if f > 5000:
        #         ret, frame = video.read()
        #         cv2.putText(frame, f"pulling angle: {self.pulling_angle[f, 0]}", point, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #                     (0, 0, 255), 2)
        #         cv2.putText(frame, f"spring_extension: {self.spring_extension[f,0]}", point2,
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        #         if not self.angular_force:
        #             cv2.putText(frame, f"force_direction: {self.force_direction[f,0]}", point3,
        #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        #             cv2.putText(frame, f"force_magnitude: {self.force_magnitude[f,0]}", point4,
        #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        #         else:
        #             cv2.putText(frame, f"total_force: {self.total_force[f,0]}", point3,
        #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        #         cv2.imshow("frame", frame)
        #         cv2.waitKey(10)

        spring = 0
        # print(f"min pulling_angel: {np.nanmin(self.pulling_angle[:, spring])}, max pulling_angel: {np.nanmax(self.pulling_angle[:, spring])}, mean pulling_angel: {np.nanmean(self.pulling_angle[:, spring])}")
        # print(f"min spring_extension: {np.nanmin(self.spring_extension[:, spring])}, max spring_extension: {np.nanmax(self.spring_extension[:, spring])}, mean spring_extension: {np.nanmean(self.spring_extension[:, spring])}")
        if not self.angular_force:
            self.total_force = np.sin(self.force_direction) * self.force_magnitude
            # print(f"y_pred force_direction min {np.nanmin(self.force_direction[:,spring])}, max {np.nanmax(self.force_direction[:,spring])}, mean {np.nanmean(self.force_direction[:,spring])}")
            # print(f"y_pred force_magnitude min {np.nanmin(self.force_magnitude[:,spring])}, max {np.nanmax(self.force_magnitude[:,spring])}, mean {np.nanmean(self.force_magnitude[:,spring])}")
        # print(f"norm length: {self.norm_length}")
        else:
            pass
            # print(f"total_force: min: {np.nanmin(self.total_force[spring])}, max: {np.nanmax(self.total_force[spring])}")

    def save_data(self, directory):
        print("-" * 60)
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
        np.savetxt(f"{directory}total_force.csv", self.total_force, delimiter=",")

    def test_correlation(self):
        print("-"*60)
        # total_force = np.sin(object.force_direction) * object.force_magnitude
        # total_force = self.total_force
        net_force = np.nansum(self.total_force, axis=1)
        import pandas as pd
        velocity_spaced = np.nanmedian(utils.calc_angular_velocity(self.fixed_end_fixed_coordinates_angle_to_nest, diff_spacing=20), axis=1)
        corr_df = pd.DataFrame({"net_force": net_force, "angular_velocity": velocity_spaced})
        corr_df = corr_df.dropna()
        correlation_score = corr_df.corr()["net_force"]["angular_velocity"]
        print(f"correlation score between net force and angular velocity: {correlation_score}")
        return correlation_score

    def calculations(self):
        # self.velocity = np.nanmedian(utils.calc_angular_velocity(self.fixed_end_angle_to_nest, diff_spacing=1), axis=1)
        self.angular_velocity = np.nanmedian(utils.calc_angular_velocity(self.fixed_end_fixed_coordinates_angle_to_nest, diff_spacing=20), axis=1)
        self.total_n_ants = np.nansum(self.N_ants_around_springs, axis=1)
        self.sum_pulling_angle = np.nansum(self.pulling_angle, axis=1)
        self.angle_to_nest = np.nansum(self.fixed_end_fixed_coordinates_angle_to_nest, axis=1)
        self.spring_extension = np.nanmean(self.spring_length, axis=1)
        # self.total_force = np.sin(self.force_direction) * self.force_magnitude
        # print(np.sum(np.isnan(self.pulling_angle[:30000]))/(self.pulling_angle[:30000].shape[0]*self.pulling_angle[:30000].shape[1]))
        # print(np.sum(self.total_force==0),self.total_force.shape[0]*self.total_force.shape[1])
        self.total_force[np.isnan(self.pulling_angle)] = np.nan
        self.net_force = np.nansum(self.total_force,axis=1)
        # print("total force at frame 9882:", self.total_force[9882])
        # self.net_force = np.nansum(self.force, axis=1)

if __name__=="__main__":
    calibration_dir = "Z:\\Dor_Gabay\\ThesisProject\\data\\calibration\\post_slicing\\calibration_perfect2\\sliced_videos\\"
    data_dir = "Z:\\Dor_Gabay\\ThesisProject\\data\\exp_collected_data\\15.9.22\\plus0.3mm_force\\S5280006\\"
    calibration_model = pickle.load(open(calibration_dir + "calibration_model.pkl", "rb"))
    object = PostProcessing(data_dir, calibration_model=calibration_model)
    object.test_correlation()
    object.calculations()

    from data_analysis import plots
    object.video_name = "S5280006"
    plots.plot_overall_behavior(object, start=0, end=None, window_size=50, title="S5280006", output_dir=data_dir)
    # object.save_data(data_dir)
    print("-"*60)

