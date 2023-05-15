import copy
import numpy as np
import pandas as pd
import os
import pickle
from scipy.signal import savgol_filter
from calibration.utils import bound_angle
from data_analysis import utils

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

class PostProcessing:

    def __init__(self, directory, calibration_model,length_models=None, angle_models=None):
        # self.frame_center = np.array([1920, 540])/2
        self.directory = directory
        self.load_data(directory)
        self.n_ants_processing()
        self.calc_distances()
        self.repeat_values()
        self.calc_angle()
        self.calibration_model = calibration_model
        self.find_fixed_coordinates(bias_equations_free=self.free_end_angle_to_blue_part_bias_equations,
                                    bias_equations_fixed=self.fixed_end_angle_to_blue_part_bias_equations)
        # if length_models is not None:
                # self.find_best_sectioning()
            # length_models, angle_models = self.calc_pull_and_length()
        self.calc_spring_length(models=length_models)
        self.calc_pulling_angle(models=angle_models)
        self.calc_force(calibration_model)
        self.calculations()

    def load_data(self,directory):
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

    def norm_values_coordinates(self, X, Y, fit_idx):
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import train_test_split
        model = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
        X_fit = X[fit_idx,:]
        Y_fit = Y[fit_idx]
        X_fit = X_fit[~np.isnan(Y_fit)]
        Y_fit = Y_fit[~np.isnan(Y_fit)]
        # X_train, X_test, y_train, y_test = train_test_split(X_fit, Y_fit, test_size=0.3, random_state=42)
        model.fit(X_fit, Y_fit)
        # print("mean_squared: ", mean_squared_error(y_test, model.predict(X_test)))

        Y_pred = model.predict(X[~np.isnan(Y)])
        Y_normed = Y
        Y_normed[~np.isnan(Y)] -= Y_pred
        return Y_normed

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
        Y_bias -= np.nanmedian(Y_bias, axis=0)
        normed_Y = np.zeros(Y.shape)
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
        for i in range(self.num_of_springs):
            bias_equation = bias_equations[i]
            normed_Y[:, i] = utils.normalize(Y[:, i], X[:, i], bias_equation)
        if find_boundary_column:
            for col in columns_on_boundary:
                below = normed_Y[:, col] < 0
                normed_Y[below, col] += 2*np.pi
        return normed_Y, bias_equations

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

    # def norm_to_sections(self,array,section_min_size):
    #     import scipy.ndimage as ndi
    #     if_rest = copy.copy(array)
    #     if_rest[~self.rest_bool] = np.nan
    #     for col in range(if_rest.shape[1]):
    #         label_min_size = section_min_size
    #         values = if_rest[:, col]
    #         if label_min_size > np.sum(~np.isnan(values)):
    #             label_min_size = np.sum(~np.isnan(values))
    #         labeled, n_labels = ndi.label(~np.isnan(values))
    #         labeled_nan, n_labels_nan = ndi.label(np.isnan(values))
    #         if labeled_nan[0] == 1:
    #             labeled_nan[(labeled_nan != 1) + (labeled_nan != 0)] -= 1
    #         labeled_nan = labeled_nan + labeled
    #         for i in range(1, n_labels + 1):
    #             if np.sum(labeled == i) < label_min_size:
    #                 labeled[labeled == i] = i + 1
    #                 labeled_nan[labeled_nan == i] = i + 1
    #             else:
    #                 values[labeled_nan == i] = np.nanmedian(values[labeled == i])
    #         if_rest[:, col] = values
    #     return if_rest

    # def find_best_sectioning(self):
    #     self.calc_spring_length()
    #     self.calc_pulling_angle(zero_angles=True)
    #     saved_pulling_angle = copy.copy(self.pulling_angle)
    #     saved_spring_length = copy.copy(self.spring_length)
    #     corr_results = []
    #     section_sizes = range(1,70000,100)
    #     for section_size in section_sizes:
    #         self.pulling_angle = copy.copy(saved_pulling_angle)
    #         pulling_angle_if_rest = self.norm_to_sections(self.pulling_angle, section_size)
    #         self.pulling_angle -= pulling_angle_if_rest
    #
    #         self.spring_length = copy.copy(saved_spring_length)
    #         median_spring_length_at_rest = self.norm_to_sections(self.spring_length, section_size)
    #         self.spring_length /= median_spring_length_at_rest
    #         self.spring_extension = self.spring_length - 1
    #
    #         self.calc_force(calibration_model)
    #         self.calculations()
    #         corr = self.test_correlation()
    #         print("corr: ",corr)
    #         corr_results.append(corr)
    #
    #     self.best_section_size = section_sizes[np.argmax(corr_results)]
    #     self.pulling_angle = copy.copy(saved_pulling_angle)
    #     pulling_angle_if_rest = self.norm_to_sections(self.pulling_angle, self.best_section_size)
    #     self.pulling_angle -= pulling_angle_if_rest
    #
    #     self.spring_length = copy.copy(saved_spring_length)
    #     median_spring_length_at_rest = self.norm_to_sections(self.spring_length, 1000)
    #     self.spring_length /= median_spring_length_at_rest
    #     self.spring_extension = self.spring_length - 1
    #
    #     self.calc_force(calibration_model)
    #     self.calculations()

    # def calc_pull_and_length(self):
    #     X_length = np.concatenate((self.fixed_ends_coordinates, np.expand_dims(self.fixed_end_angle_to_nest, axis=2)),axis=2)
    #     # self.free_end_angle_to_nest[self.free_end_angle_to_nest > np.pi] -= 2 * np.pi
    #     # self.free_end_angle_to_nest[self.free_end_angle_to_nest <0] *= -1
    #     # print(self.free_end_angle_to_nest)
    #     X_angle = np.concatenate((self.fixed_ends_coordinates,np.expand_dims(self.free_end_angle_to_nest,axis=2)),axis=2)
    #     # X_angle[X_angle[:,:,2]>np.pi, :, 2] -= 2*np.pi
    #     # X_angle[X_angle[:, :, 2] > np.pi, :, 2] *= -1
    #     y_length = np.linalg.norm(self.free_ends_coordinates - self.fixed_ends_coordinates, axis=2)
    #     pulling_angle = utils.calc_angle_matrix(self.free_ends_coordinates, self.fixed_ends_coordinates,
    #                                             self.object_center_repeated)
    #     above = pulling_angle > 0
    #     below = pulling_angle < 0
    #     pulling_angle[above] = pulling_angle[above] - np.pi
    #     pulling_angle[below] = pulling_angle[below] + np.pi
    #     # print(X_length)
    #     # print(pulling_angle)
    #     y_angle = pulling_angle
    #     idx = self.rest_bool
    #     models_lengths = []
    #     models_angles = []
    #     for col in range(y_length.shape[1]):
    #         X_length_fit = X_length[idx[:, col], col,:]
    #         X_angle_fit = X_angle[idx[:, col], col,:]
    #         y_length_fit = y_length[idx[:, col], col]
    #         y_angle_fit = y_angle[idx[:, col], col]
    #         X_length_fit = X_length_fit[~np.isnan(y_length_fit)]
    #         X_angle_fit = X_angle_fit[~np.isnan(y_angle_fit)]
    #         y_length_fit = y_length_fit[~np.isnan(y_length_fit)]
    #         y_angle_fit = y_angle_fit[~np.isnan(y_angle_fit)]
    #         model_length = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
    #         model_angle = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
    #         models_lengths.append(model_length.fit(X_length_fit, y_length_fit))
    #         models_angles.append(model_angle.fit(X_angle_fit, y_angle_fit))

    #     return models_lengths, models_angles

    def calc_pulling_angle(self,models):
        self.pulling_angle = utils.calc_pulling_angle_matrix(self.fixed_ends_coordinates,self.object_center_repeated,
                                                             self.free_ends_coordinates)

        angles_to_nest = np.expand_dims(self.fixed_end_angle_to_nest, axis=2)
        fixed_end_distance = np.expand_dims(self.object_center_to_fixed_end_distance, axis=2)
        free_end_distance = np.expand_dims(self.object_center_to_free_end_distance, axis=2)
        object_center = np.expand_dims(self.object_center_repeated, axis=2)
        blue_length = np.expand_dims(np.repeat(np.expand_dims(np.linalg.norm(
            self.blue_tip_coordinates - self.object_center, axis=1), axis=1), self.num_of_springs, axis=1), axis=2)
        X = np.concatenate((np.sin(angles_to_nest), np.cos(angles_to_nest)),axis=2)#,fixed_end_distance,self.object_center_repeated), axis=2)
        rest_not_nan_idx = ~(np.isnan(self.pulling_angle) + np.isnan(X).any(axis=2)) * self.rest_bool
        not_nan_idx = ~(np.isnan(self.pulling_angle) + np.isnan(X).any(axis=2))
        for col in range(self.pulling_angle.shape[1]):
            model = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
            X_fit = X[rest_not_nan_idx[:,col],col,:]
            y_fit = self.pulling_angle[rest_not_nan_idx[:,col],col]
            # model = models[col]
            model.fit(X_fit,y_fit)
            y_pred = model.predict(X[not_nan_idx[:,col],col,:])
            self.pulling_angle[not_nan_idx[:,col],col] -= y_pred
        pulling_angle_if_rest = copy.copy(self.pulling_angle)
        pulling_angle_copy = copy.copy(self.pulling_angle)
        pulling_angle_if_rest[~self.rest_bool] = np.nan
        pulling_angle_copy[self.rest_bool] = np.nan
        self.pulling_angle -= np.nanmedian(pulling_angle_if_rest, axis=0)

    # def calc_pulling_angle(self,models):
    #     # self.pulling_angle = utils.calc_angle_matrix(self.free_ends_coordinates, self.fixed_ends_coordinates,
    #     #                         self.object_center_repeated)
    #     # above = self.pulling_angle > 0
    #     # below = self.pulling_angle < 0
    #     # self.pulling_angle[above] = self.pulling_angle[above] - np.pi
    #     # self.pulling_angle[below] = self.pulling_angle[below] + np.pi
    #     # # X = np.concatenate((self.fixed_ends_coordinates, np.expand_dims(self.free_end_angle_to_nest, axis=2)), axis=2)
    #     # X = np.concatenate((self.fixed_ends_coordinates,self.free_ends_coordinates, self.object_center_repeated), axis=2)
    #     # y = self.pulling_angle
    #     # for col in range(y.shape[1]):
    #     #     y_pred = models[col].predict(X[~np.isnan(y[:, col]), col, :])
    #     #     self.pulling_angle[~np.isnan(y[:, col]), col] -= y_pred
    #
    #     self.pulling_angle = utils.calc_angle_matrix(self.free_end_fixed_coordinates,self.fixed_end_fixed_coordinates,self.object_center_repeated)+np.pi
    #     self.pulling_angle, _ = self.norm_values(self.free_end_fixed_coordinates_angle_to_nest, self.pulling_angle, bias_bool=self.rest_bool, find_boundary_column=True)
    #     self.pulling_angle = self.pulling_angle - np.pi
    #     above = self.pulling_angle > 0
    #     below = self.pulling_angle < 0
    #     self.pulling_angle[above] = self.pulling_angle[above] - np.pi
    #     self.pulling_angle[below] = self.pulling_angle[below] + np.pi
    #     pulling_angle_if_rest = copy.copy(self.pulling_angle)
    #     pulling_angle_if_rest[~self.rest_bool] = np.nan
    #     self.pulling_angle -= np.nanmedian(pulling_angle_if_rest, axis=0)
    #
    #     # self.pulling_angle -= pulling_angle_if_rest

    # def calc_spring_length(self,models):
    #     self.spring_length = np.linalg.norm(self.free_ends_coordinates - self.fixed_end_fixed_coordinates, axis=2)
    #     self.spring_length /= self.norm_size
    #     self.spring_length = utils.interpolate_data(self.spring_length,utils.find_cells_to_interpolate(self.spring_length))
    #     median_spring_length_at_rest = copy.copy(self.spring_length)
    #     median_spring_length_at_rest[np.invert(self.rest_bool)] = np.nan
    #     median_spring_length_at_rest = np.nanmedian(median_spring_length_at_rest, axis=0)
    #     self.spring_length /= median_spring_length_at_rest
    #     angles_to_nest = np.expand_dims(self.fixed_end_fixed_coordinates_angle_to_nest, axis=2)
    #     X = np.concatenate((np.sin(angles_to_nest), np.cos(angles_to_nest),self.object_center_repeated), axis=2)
    #     rest_not_nan_idx = ~(np.isnan(self.spring_length)+np.isnan(X[:,:,0]))*self.rest_bool
    #     for col in range(self.spring_length.shape[1]):
    #         model = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
    #         X_fit = X[rest_not_nan_idx[:,col],col,:]
    #         y_fit = self.spring_length[rest_not_nan_idx[:,col],col]
    #         model.fit(X_fit,y_fit)
    #         # model = models[col]
    #         y_pred = model.predict(X[rest_not_nan_idx[:,col],col,:])
    #         self.spring_length[rest_not_nan_idx[:,col],col] -= y_pred
    #
    #     self.spring_extension = self.spring_length -1
    #     print("spring extension at rest: ", np.nanmedian(self.spring_extension[self.rest_bool]))
    #     print("spring extension at no rest: ", np.nanmedian(self.spring_extension[~self.rest_bool]))

    def calc_spring_length(self,models):
        self.spring_length = np.linalg.norm(self.free_ends_coordinates - self.fixed_end_fixed_coordinates , axis=2)
        # self.spring_length = np.linalg.norm(self.free_ends_coordinates - self.fixed_ends_coordinates, axis=2)
        self.spring_length = utils.interpolate_data(self.spring_length,
                                                       utils.find_cells_to_interpolate(self.spring_length))
        self.spring_length = self.norm_values(self.fixed_end_fixed_coordinates_angle_to_nest, self.spring_length, bias_bool=self.rest_bool, find_boundary_column=False)[0]
        self.spring_length /= self.norm_size
        median_spring_length_at_rest = copy.copy(self.spring_length)
        median_spring_length_at_rest[np.invert(self.rest_bool)] = np.nan
        median_spring_length_at_rest = np.nanmedian(median_spring_length_at_rest, axis=0)
        self.spring_length /= median_spring_length_at_rest
        self.spring_extension = self.spring_length -1
        # print("spring extension at rest: ", np.nanmedian(self.spring_extension[self.rest_bool]))
        # print("spring extension at no rest: ", np.nanmedian(self.spring_extension[~self.rest_bool]))

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

        if not self.angular_force:
            self.total_force = np.sin(self.force_direction) * self.force_magnitude
            # self.total_force[self.negative_pulling_angle] *= -1
            self.total_force *= -1
        else:
            self.total_force *= -1

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

    def calculations(self):
        self.net_force = np.nansum(self.total_force, axis=1)
        self.net_force = np.array(pd.Series(self.net_force).rolling(window=5,center=True).median())
        frame_velocity = utils.calc_angular_velocity(self.fixed_end_fixed_coordinates_angle_to_nest, diff_spacing=20)/20
        # for col in range(frame_velocity.shape[1]):
        #     frame_velocity[:,col] = pd.Series(frame_velocity[:,col]).rolling(self.velocity_spacing).median()
        self.angular_velocity = np.nanmedian(frame_velocity, axis=1)
        self.total_n_ants = np.nansum(self.N_ants_around_springs, axis=1)
        self.sum_pulling_angle = np.nansum(self.pulling_angle, axis=1)
        self.angle_to_nest = np.nansum(self.fixed_end_fixed_coordinates_angle_to_nest, axis=1)
        self.spring_extension = np.nanmean(self.spring_length, axis=1)
        self.total_force[np.isnan(self.pulling_angle)] = np.nan

    def test_correlation(self):
        corr_df = pd.DataFrame({"net_force": self.net_force, "angular_velocity": self.angular_velocity})
        corr_df = corr_df.dropna()
        correlation_score = corr_df.corr()["net_force"]["angular_velocity"]
        print(f"correlation score between net force and angular velocity: {correlation_score}")
        return correlation_score

def create_model():
    calibration_dir = "Z:\\Dor_Gabay\\ThesisProject\\data\\calibration\\post_slicing\\calibration_perfect2\\sliced_videos\\"
    calibration_model = pickle.load(open(calibration_dir + "calibration_model.pkl", "rb"))
    X = None
    y_length = None
    y_angle = None
    idx = None
    # for count,i in enumerate([1,3,4,5]):#,6,7,8,9]):
    for count,i in enumerate([1]):#,6,7,8,9]):
        data_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\exp_collected_data\\15.9.22\\plus0.3mm_force\\S528000{i}\\"
        object = PostProcessing(data_dir, calibration_model=calibration_model)
        if count == 0:
            angles_to_nest = np.expand_dims(object.fixed_end_fixed_coordinates_angle_to_nest,axis=2)
            X = np.concatenate((np.sin(angles_to_nest),np.cos(angles_to_nest),object.object_center_repeated),axis=2)
            y_length = np.linalg.norm(object.free_end_fixed_coordinates - object.fixed_end_fixed_coordinates,axis=2)
            y_length /= object.norm_size
            # pulling_angle = utils.calc_angle_matrix(object.free_ends_coordinates,object.fixed_ends_coordinates,object.object_center_repeated)
            pulling_angle = utils.calc_angle_matrix(object.free_end_fixed_coordinates,
                                                         object.fixed_end_fixed_coordinates,
                                                         object.object_center_repeated)
            above = pulling_angle > 0
            below = pulling_angle < 0
            pulling_angle[above] = pulling_angle[above] - np.pi
            pulling_angle[below] = pulling_angle[below] + np.pi
            # pulling_angle[pulling_angle<0] *= -1
            pulling_angle_if_rest = copy.copy(pulling_angle)
            pulling_angle_if_rest[~object.rest_bool] = np.nan
            pulling_angle -= np.nanmedian(pulling_angle_if_rest, axis=0)
            y_angle = pulling_angle
            idx = object.rest_bool
        else:
            angles_to_nest = np.expand_dims(object.fixed_end_fixed_coordinates_angle_to_nest,axis=2)
            X = np.concatenate((X,np.concatenate((np.sin(angles_to_nest),np.cos(angles_to_nest),object.object_center_repeated),axis=2)),axis=0)
            y_length = np.concatenate((y_length,np.linalg.norm(object.free_end_fixed_coordinates - object.fixed_end_fixed_coordinates , axis=2)),axis=0)
            y_length /= object.norm_size
            # pulling_angle = utils.calc_angle_matrix(object.free_ends_coordinates, object.fixed_ends_coordinates,
            #                                         object.object_center_repeated)
            pulling_angle = utils.calc_angle_matrix(object.free_end_fixed_coordinates,
                                                         object.fixed_end_fixed_coordinates,
                                                         object.object_center_repeated)
            above = pulling_angle > 0
            below = pulling_angle < 0
            pulling_angle[above] = pulling_angle[above] - np.pi
            pulling_angle[below] = pulling_angle[below] + np.pi
            pulling_angle_if_rest = copy.copy(pulling_angle)
            pulling_angle_if_rest[~object.rest_bool] = np.nan
            pulling_angle -= np.nanmedian(pulling_angle_if_rest, axis=0)
            y_angle = np.concatenate((y_angle,pulling_angle),axis=0)
            idx = np.concatenate((idx,object.rest_bool),axis=0)
    models_lengths = []
    models_angles = []
    for col in range(y_length.shape[1]):
        X_fit = X[idx[:,col],col]
        y_length_fit = y_length[idx[:,col],col]
        y_angle_fit = y_angle[idx[:,col],col]
        X_fit = X_fit[~np.isnan(y_length_fit)]
        y_length_fit = y_length_fit[~np.isnan(y_length_fit)]
        y_angle_fit = y_angle_fit[~np.isnan(y_angle_fit)]
        model_length = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
        model_angle = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
        models_lengths.append(model_length.fit(X_fit, y_length_fit))
        models_angles.append(model_angle.fit(X_fit, y_angle_fit))
    return models_lengths,models_angles

if __name__=="__main__":
    # models_lengths,models_angles = create_model()
    for i in [1,3,4,5,6,7,8,9]:
    # for i in range(9,10):
        print("-" * 60)
        calibration_dir = "Z:\\Dor_Gabay\\ThesisProject\\data\\calibration\\post_slicing\\calibration_perfect2\\sliced_videos\\"
        # calibration_dir = "Z:\\Dor_Gabay\\ThesisProject\\data\\calibration\\post_slicing\\calibration2\\"
        data_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\exp_collected_data\\15.9.22\\plus0.3mm_force\\S528000{i}\\"
        calibration_model = pickle.load(open(calibration_dir + "calibration_model.pkl", "rb"))
        object = PostProcessing(data_dir, calibration_model=calibration_model)#, length_models=models_lengths, angle_models=models_angles)
        correlation_score = object.test_correlation()
        # object.calculations()

        from data_analysis import plots
        object.video_name = data_dir.split("\\")[-2]
        out = "Z:\\Dor_Gabay\\ThesisProject\\data\\exp_collected_data\\15.9.22\\plus0.3mm_force\\figures_adjustments2\\"
        os.makedirs(out, exist_ok=True)

        plots.plot_overall_behavior(object, start=0, end=None, window_size=200,
                                    title=object.video_name+"_corr"+str(np.round(correlation_score,2)),
                                    output_dir=out)
        # object.save_data(data_dir)
        # print("-"*60)

