import copy
import numpy as np
import pandas as pd
import os
import pickle
from scipy.signal import savgol_filter
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from line_profiler_pycharm import profile

# local imports:
from data_analysis import utils


class PostProcessing:
    def __init__(self, directory, sub_dirs_sets, calibration_model, two_vars=True, slicing_info=None, arrangements=None):
        self.frame_size = (1920, 1080)
        self.directory = directory
        self.sub_dirs_names = [sub_dirs_sets[0]+sub_dirs_sets[x] for x in range(1,len(sub_dirs_sets))][0]
        self.calibration_model = calibration_model
        self.two_vars = two_vars
        self.sub_dirs_sets = sub_dirs_sets
        self.load_data(slicing_info=slicing_info, arrangements=arrangements)
        self.n_ants_processing()
        self.calc_distances()
        self.repeat_values()
        self.calc_angle()
        self.create_model()
        self.calc_spring_length()
        self.calc_pulling_angle()
        self.calc_force(calibration_model)
        # self.profile_ants()

    def load_data(self, slicing_info=None, arrangements=None):
        print("loading data...")
        # print([os.path.join(self.directory, "ant_tracking", self.ant_assignments_sub_dirs_names[x], "ants_assigned_to_springs.npz")for x in range(len(self.ant_assignments_sub_dirs_names))])
        # self.ants_assigned_to_springs = [np.load(os.path.join(self.directory, "ant_tracking", self.ant_assignments_sub_dirs_names[x], "ants_assigned_to_springs.npz"))["arr_0"]
        #                                             for x in range(len(self.ant_assignments_sub_dirs_names))]
        # self.ants_assigned_set_sizes = self.join_ants_assigned_to_springs()
        # print("self.ants_assigned_to_springs.shape: ", self.ants_assigned_to_springs.shape)
        self.N_ants_around_springs = np.concatenate([np.loadtxt(os.path.join(self.directory, sub_dir, "raw_analysis", "N_ants_around_springs.csv"), delimiter=",")
                                                     for sub_dir in self.sub_dirs_names], axis=0)
        self.size_ants_around_springs = np.concatenate([np.loadtxt(os.path.join(self.directory, sub_dir, "raw_analysis", "size_ants_around_springs.csv"), delimiter=",")
                                                        for sub_dir in self.sub_dirs_names], axis=0)
        self.fixed_ends_coordinates = np.concatenate([np.stack((np.loadtxt(os.path.join(self.directory, sub_dir, "raw_analysis", "fixed_ends_coordinates_x.csv"), delimiter=","),
                                                                np.loadtxt(os.path.join(self.directory, sub_dir, "raw_analysis", "fixed_ends_coordinates_y.csv"), delimiter=",")),
                                                               axis=2) for sub_dir in self.sub_dirs_names], axis=0)
        self.free_ends_coordinates = np.concatenate([np.stack((np.loadtxt(os.path.join(self.directory, sub_dir, "raw_analysis", "free_ends_coordinates_x.csv"), delimiter=","),
                                                               np.loadtxt(os.path.join(self.directory, sub_dir, "raw_analysis", "free_ends_coordinates_y.csv"), delimiter=",")),
                                                              axis=2) for sub_dir in self.sub_dirs_names], axis=0)
        blue_part_coordinates = np.concatenate(
            [np.loadtxt(os.path.join(self.directory, sub_dir, "raw_analysis", "blue_part_coordinates_x.csv"), delimiter=",") for sub_dir in self.sub_dirs_names], axis=0)
        blue_part_coordinates = np.stack((blue_part_coordinates, np.concatenate(
            [np.loadtxt(os.path.join(self.directory, sub_dir, "raw_analysis", "blue_part_coordinates_y.csv"), delimiter=",") for sub_dir in self.sub_dirs_names], axis=0)), axis=2)
        self.object_center = blue_part_coordinates[:, 0, :]
        self.blue_tip_coordinates = blue_part_coordinates[:, -1, :]
        self.norm_size = np.array([pickle.load(open(os.path.join(self.directory, sub_dir, "raw_analysis", "blue_median_area.pickle"), "rb")) for sub_dir in self.sub_dirs_names])
        self.num_of_frames_per_video = np.array(
            [np.loadtxt(os.path.join(self.directory, sub_dir, "raw_analysis", "N_ants_around_springs.csv"), delimiter=",").shape[0] for sub_dir in self.sub_dirs_names])
        self.make_sets_idx()
        self.num_of_frames = np.sum(self.num_of_frames_per_video).astype(int)
        self.num_of_springs = self.N_ants_around_springs.shape[1]
        self.slice_data(slicing_info) if slicing_info is not None else None
        self.rearrange_data(arrangements) if arrangements is not None else None

    def make_sets_idx(self):
        self.sets_idx = []
        videos_count = 0
        for count, data_set in enumerate(self.sub_dirs_sets):
            if count == 0: self.sets_idx.append([0, np.sum(self.num_of_frames_per_video[:len(data_set)])])
            else: self.sets_idx.append([self.sets_idx[-1][-1], np.sum(self.num_of_frames_per_video[:videos_count+len(data_set)])])
            videos_count += len(data_set)


    def join_ants_assigned_to_springs(self):
        set_sizes = [x.shape[0] for x in self.ants_assigned_to_springs]
        for count, ants_assigned_to_spring in enumerate(self.ants_assigned_to_springs):

            if count == 0: self.ants_assigned_to_springs = ants_assigned_to_spring
            else:
                add_this = np.hstack((np.zeros(self.ants_assigned_to_springs.shape[0],ants_assigned_to_spring.shape[1]), ants_assigned_to_spring))
                self.ants_assigned_to_springs = np.append(self.ants_assigned_to_springs, add_this, axis=0)
        self.ants_assigned_to_springs = self.ants_assigned_to_springs.astype(int)
        return set_sizes

    def rearrange_data(self, arrangements):
        n = self.num_of_springs
        for count, arrangement in enumerate(arrangements):
            if arrangement != None:
                start_frame = np.sum(self.num_of_frames_per_video[:count])
                end_frame = np.sum(self.num_of_frames_per_video[:count + 1])
                rearrangement = np.append(np.arange(arrangement, n), np.arange(0, arrangement))
                self.N_ants_around_springs[start_frame:end_frame, :] = self.N_ants_around_springs[start_frame:end_frame, rearrangement]
                self.size_ants_around_springs[start_frame:end_frame, :] = self.size_ants_around_springs[start_frame:end_frame, rearrangement]
                self.fixed_ends_coordinates[start_frame:end_frame, :, :] = self.fixed_ends_coordinates[start_frame:end_frame, rearrangement, :]
                self.free_ends_coordinates[start_frame:end_frame, :, :] = self.free_ends_coordinates[start_frame:end_frame, rearrangement, :]

    def slice_data(self, slicing_info):
        for count, slice in enumerate(slicing_info):
            if slice != None:
                start_frame = slice[0] + np.sum(self.num_of_frames_per_video[:count])
                end_frame = slice[1] + np.sum(self.num_of_frames_per_video[:count])
                self.N_ants_around_springs[start_frame:end_frame, :] = np.nan
                self.size_ants_around_springs[start_frame:end_frame, :] = np.nan
                self.fixed_ends_coordinates[start_frame:end_frame, :] = np.nan
                self.free_ends_coordinates[start_frame:end_frame, :] = np.nan
                self.blue_tip_coordinates[start_frame:end_frame, :] = np.nan
                self.object_center[start_frame:end_frame, :] = np.nan

    def n_ants_processing(self):
        for count, set_idx in enumerate(self.sets_idx):
            start_frame = set_idx[0]
            end_frame = set_idx[1]
            undetected_springs_for_long_time = utils.filter_continuity(utils.convert_bool_to_binary(np.isnan(self.fixed_ends_coordinates[start_frame:end_frame, :, 0])), min_size=8)
            self.N_ants_around_springs[start_frame:end_frame, :] = np.round(utils.interpolate_data(self.N_ants_around_springs[start_frame:end_frame, :],
                utils.find_cells_to_interpolate(self.N_ants_around_springs[start_frame:end_frame, :])))
            self.N_ants_around_springs[start_frame:end_frame, :][np.isnan(self.N_ants_around_springs[start_frame:end_frame, :])] = 0
            self.N_ants_around_springs[start_frame:end_frame, :] = self.smoothing_n_ants(self.N_ants_around_springs[start_frame:end_frame, :])
            self.N_ants_around_springs[start_frame:end_frame, :][undetected_springs_for_long_time] = np.nan
            all_small_attaches = np.zeros(self.N_ants_around_springs[start_frame:end_frame, :].shape)
            for n in np.unique(self.N_ants_around_springs[start_frame:end_frame, :])[1:]:
                if not np.isnan(n):
                    short_attaches = utils.filter_continuity(self.N_ants_around_springs[start_frame:end_frame, :] == n, max_size=30)
                    all_small_attaches[short_attaches] = 1
            self.N_ants_around_springs[start_frame:end_frame, :] = \
                np.round(utils.interpolate_data(self.N_ants_around_springs[start_frame:end_frame, :], all_small_attaches.astype(bool)))
        self.rest_bool = self.N_ants_around_springs == 0
        # undetected_springs_for_long_time = utils.filter_continuity(utils.convert_bool_to_binary(
        #     np.isnan(self.fixed_ends_coordinates[:, :, 0])), min_size=8)
        # self.N_ants_around_springs = np.round(utils.interpolate_data(self.N_ants_around_springs,
        #                                                              utils.find_cells_to_interpolate(
        #                                                                  self.N_ants_around_springs)))
        # self.N_ants_around_springs[np.isnan(self.N_ants_around_springs)] = 0
        # self.N_ants_around_springs = self.smoothing_n_ants(self.N_ants_around_springs)
        # self.N_ants_around_springs[undetected_springs_for_long_time] = np.nan
        # all_small_attaches = np.zeros(self.N_ants_around_springs.shape, int)
        # for n in np.unique(self.N_ants_around_springs)[1:]:
        #     if not np.isnan(n):
        #         short_attaches = utils.filter_continuity(self.N_ants_around_springs == n, max_size=30)
        #         all_small_attaches[short_attaches] = 1
        # self.N_ants_around_springs = np.round(utils.interpolate_data(self.N_ants_around_springs,
        #                                                              all_small_attaches.astype(bool)))
        # self.size_ants_around_springs = np.round(utils.interpolate_data(self.size_ants_around_springs,
        #                                                                 utils.find_cells_to_interpolate(
        #                                                                     self.size_ants_around_springs)))
        # median_size = np.nanmedian(self.size_ants_around_springs)
        # self.small_blobs = self.size_ants_around_springs < median_size * 0.3

    def smoothing_n_ants(self, array):
        for col in range(array.shape[1]):
            array[:, col] = np.abs(np.round(savgol_filter(array[:, col], 31, 2)))
        return array

    def smoothing_fixed_ends(self, array):
        for col in range(array.shape[1]):
            x_smoothed = pd.DataFrame(array[:, col, 0]).rolling(20, center=True, min_periods=1).mean().values[:, 0]
            y_smoothed = pd.DataFrame(array[:, col, 1]).rolling(20, center=True, min_periods=1).mean().values[:, 0]
            array[:, col] = np.vstack((x_smoothed, y_smoothed)).T
        return array

    def calc_distances(self):
        object_center = np.repeat(self.object_center[:, np.newaxis, :], self.num_of_springs, axis=1)
        blue_tip_coordinates = np.repeat(self.blue_tip_coordinates[:, np.newaxis, :], self.num_of_springs, axis=1)
        self.blue_length = np.nanmedian(np.linalg.norm(self.blue_tip_coordinates - self.object_center, axis=1))
        self.object_center_to_free_end_distance = np.linalg.norm(self.free_ends_coordinates - object_center, axis=2)
        self.object_center_to_fixed_end_distance = np.linalg.norm(self.fixed_ends_coordinates - object_center, axis=2)
        self.object_blue_tip_to_fixed_end_distance = np.linalg.norm(self.fixed_ends_coordinates - blue_tip_coordinates, axis=2)
        # for count,video in enumerate(self.num_of_frames_per_video):
        #     start = np.sum(self.num_of_frames_per_video[:count])
        #     print(self.object_blue_tip_to_fixed_end_distance[start, :])

    def repeat_values(self):
        nest_direction = np.stack((self.object_center[:, 0], self.object_center[:, 1] - 100), axis=1)
        self.nest_direction_repeated = np.repeat(nest_direction[:, np.newaxis, :], self.free_ends_coordinates.shape[1],axis=1)
        self.object_center_repeated = copy.copy(np.repeat(self.object_center[:, np.newaxis, :], self.free_ends_coordinates.shape[1], axis=1))
        self.blue_tip_coordinates_repeated = np.repeat(self.blue_tip_coordinates[:, np.newaxis, :], self.free_ends_coordinates.shape[1], axis=1)

    def calc_angle(self):
        self.free_end_angle_to_nest = utils.calc_angle_matrix(self.free_ends_coordinates, self.object_center_repeated, self.nest_direction_repeated) + np.pi
        self.fixed_end_angle_to_nest = utils.calc_angle_matrix(self.fixed_ends_coordinates, self.object_center_repeated, self.nest_direction_repeated) + np.pi
        self.blue_part_angle_to_nest = utils.calc_angle_matrix(self.nest_direction_repeated, self.object_center_repeated, self.blue_tip_coordinates_repeated) + np.pi
        self.free_end_angle_to_blue_part = utils.calc_angle_matrix(self.blue_tip_coordinates_repeated, self.object_center_repeated, self.free_ends_coordinates) + np.pi
        self.fixed_end_angle_to_blue_part = utils.calc_angle_matrix(self.blue_tip_coordinates_repeated, self.object_center_repeated, self.fixed_ends_coordinates) + np.pi
        self.fixed_to_blue_angle_change = utils.calc_pulling_angle_matrix(self.blue_tip_coordinates_repeated, self.object_center_repeated, self.fixed_ends_coordinates)

    def create_model(self):
        # start = np.sum(self.num_of_frames_per_video[:2])
        y_length = np.linalg.norm(self.free_ends_coordinates - self.fixed_ends_coordinates, axis=2)
        angles_to_nest = np.expand_dims(self.fixed_end_angle_to_nest, axis=2)
        fixed_end_distance = np.expand_dims(self.object_center_to_fixed_end_distance, axis=2)
        # fixed_to_tip_distance = np.expand_dims(self.object_blue_tip_to_fixed_end_distance, axis=2)
        # fixed_to_blue_angle_change = np.expand_dims(
        #     np.repeat(np.expand_dims(np.nanmedian(self.fixed_to_blue_angle_change, axis=1), axis=1),
        #               self.num_of_springs, axis=1), axis=2)
        # object_center = self.object_center_repeated
        blue_length = np.expand_dims(np.repeat(np.expand_dims(np.linalg.norm(
            self.blue_tip_coordinates - self.object_center, axis=1), axis=1), self.num_of_springs, axis=1), axis=2)
        # X = np.concatenate((np.sin(angles_to_nest), np.cos(angles_to_nest), object_center, fixed_end_distance,
        #                     fixed_to_tip_distance, fixed_to_blue_angle_change, blue_length), axis=2)
        X = np.concatenate((np.sin(angles_to_nest), np.cos(angles_to_nest), blue_length, fixed_end_distance), axis=2)
        # X = np.concatenate((np.sin(angles_to_nest), np.cos(angles_to_nest), blue_length, fixed_end_distance,
        #                     fixed_to_blue_angle_change), axis=2)
        y_angle = utils.calc_pulling_angle_matrix(self.fixed_ends_coordinates,
                                                  self.object_center_repeated,
                                                  self.free_ends_coordinates)
        idx = self.rest_bool
        self.models_lengths = []
        self.models_angles = []
        not_nan_idx = ~(np.isnan(y_angle) + np.isnan(X).any(axis=2)) * idx
        for col in range(y_length.shape[1]):
            X_fit = X[not_nan_idx[:, col], col]
            y_length_fit = y_length[not_nan_idx[:, col], col]
            y_angle_fit = y_angle[not_nan_idx[:, col], col]
            model_length = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
            model_angle = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
            self.models_lengths.append(model_length.fit(X_fit, y_length_fit))
            self.models_angles.append(model_angle.fit(X_fit, y_angle_fit))

    def norm_values(self, matrix, models=None):
        angles_to_nest = np.expand_dims(self.fixed_end_angle_to_nest, axis=2)
        fixed_end_distance = np.expand_dims(self.object_center_to_fixed_end_distance, axis=2)
        # object_center = self.object_center_repeated
        # fixed_to_tip_distance = np.expand_dims(self.object_blue_tip_to_fixed_end_distance, axis=2)
        # fixed_to_blue_angle_change = np.expand_dims(
        #     np.repeat(np.expand_dims(np.nanmedian(self.fixed_to_blue_angle_change, axis=1), axis=1),
        #               self.num_of_springs, axis=1), axis=2)
        blue_length = np.expand_dims(np.repeat(np.expand_dims(np.linalg.norm(
            self.blue_tip_coordinates - self.object_center, axis=1), axis=1), self.num_of_springs, axis=1), axis=2)
        # X = np.concatenate((np.sin(angles_to_nest), np.cos(angles_to_nest), object_center, fixed_end_distance,
        #                     fixed_to_tip_distance, fixed_to_blue_angle_change, blue_length,), axis=2)
        X = np.concatenate((np.sin(angles_to_nest), np.cos(angles_to_nest), blue_length, fixed_end_distance), axis=2)
        not_nan_idx = ~(np.isnan(matrix) + np.isnan(X).any(axis=2))
        prediction_matrix = np.zeros(matrix.shape)
        for col in range(matrix.shape[1]):
            model = models[col]
            prediction_matrix[not_nan_idx[:, col], col] = model.predict(X[not_nan_idx[:, col], col, :])
        return prediction_matrix

    def calc_pulling_angle(self):
        self.pulling_angle = utils.calc_pulling_angle_matrix(self.fixed_ends_coordinates, self.object_center_repeated,
                                                             self.free_ends_coordinates)
        pred_pulling_angle = self.norm_values(self.pulling_angle, self.models_angles)
        self.pulling_angle -= pred_pulling_angle
        # for count, frames_set in enumerate(self.num_of_frames_per_video):
        #     start_frame = np.sum(self.num_of_frames_per_video[:count])
        #     end_frame = np.sum(self.num_of_frames_per_video[:count + 1])
        for count, set_idx in enumerate(self.sets_idx):
            start_frame = set_idx[0]
            end_frame = set_idx[1]
            pulling_angle_if_rest = copy.copy(self.pulling_angle[start_frame:end_frame])
            pulling_angle_copy = copy.copy(self.pulling_angle[start_frame:end_frame])
            pulling_angle_if_rest[~self.rest_bool[start_frame:end_frame]] = np.nan
            pulling_angle_copy[self.rest_bool[start_frame:end_frame]] = np.nan
            self.pulling_angle[start_frame:end_frame] -= np.nanmedian(pulling_angle_if_rest, axis=0)

    def calc_spring_length(self):
        self.spring_length = np.linalg.norm(self.free_ends_coordinates - self.fixed_ends_coordinates, axis=2)
        pred_spring_length = self.norm_values(self.spring_length, self.models_lengths)
        self.spring_length /= pred_spring_length
        # for count, frames_set in enumerate(self.num_of_frames_per_video):
        #     start_frame = np.sum(self.num_of_frames_per_video[:count])
        #     end_frame = np.sum(self.num_of_frames_per_video[:count + 1])
        for count, set_idx in enumerate(self.sets_idx):
            start_frame = set_idx[0]
            end_frame = set_idx[1]
            self.spring_length[start_frame:end_frame] = utils.interpolate_data(self.spring_length[start_frame:end_frame],
                                                                               utils.find_cells_to_interpolate(self.spring_length[start_frame:end_frame]))
            self.spring_length[start_frame:end_frame] /= self.norm_size[count]
            # self.spring_length /= self.norm_size
            median_spring_length_at_rest = copy.copy(self.spring_length[start_frame:end_frame])
            median_spring_length_at_rest[np.invert(self.rest_bool[start_frame:end_frame])] = np.nan
            median_spring_length_at_rest = np.nanmedian(median_spring_length_at_rest, axis=0)
            self.spring_length[start_frame:end_frame] /= median_spring_length_at_rest
        self.spring_extension = self.spring_length - 1

    def calc_force(self, model):
        self.force_direction = np.zeros((self.num_of_frames, self.num_of_springs))
        self.force_magnitude = np.zeros((self.num_of_frames, self.num_of_springs))
        self.total_force = np.zeros((self.num_of_frames, self.num_of_springs))
        for s in range(self.num_of_springs):
            X = np.stack((self.pulling_angle[:, s], self.spring_extension[:, s]), axis=1)
            un_nan_bool = ~np.isnan(X).any(axis=1)
            X = X[un_nan_bool]
            forces_predicted = model.predict(X)
            if self.two_vars:
                self.force_direction[un_nan_bool, s] = forces_predicted[:, 0]
                self.force_magnitude[un_nan_bool, s] = forces_predicted[:, 1]
            else:
                self.total_force[un_nan_bool, s] = forces_predicted
        if self.two_vars:
            self.total_force = np.sin(self.force_direction) * self.force_magnitude
            self.total_force *= -1
        else:
            self.total_force *= -1

    def profile_ants(self):
        print("Profiling ants...")
        self.ants_assigned_to_springs = self.ants_assigned_to_springs[:, :-1]
        profiles = np.full(5, np.nan)  # ant, spring, start, end, precedence
        for ant in range(self.ants_assigned_to_springs.shape[1]):
            attachment = self.ants_assigned_to_springs[:, ant]
            events_springs = np.split(attachment, np.arange(len(attachment[1:]))[np.diff(attachment) != 0] + 1)
            events_frames = np.split(np.arange(len(attachment)), np.arange(len(attachment[1:]))[np.diff(attachment) != 0] + 1)
            precedence = 0
            for event in range(len(events_springs)):
                if events_springs[event][0] != 0 and len(events_springs[event]) > 1:
                    precedence += 1
                    start = events_frames[event][0]
                    end = events_frames[event][-1]
                    profiles = np.vstack(
                        (profiles, np.array([ant + 1, events_springs[event][0], start, end, precedence])))
        profiles = profiles[1:, :]
        self.all_profiles_force_magnitude = np.full((len(profiles), 10000), np.nan)
        self.all_profiles_force_direction = np.full((len(profiles), 10000), np.nan)
        self.all_profiles_ants_number = np.full((len(profiles), 10000), np.nan)
        self.all_profiles_angle_to_nest = np.full((len(profiles), 10000), np.nan)
        self.all_profiles_precedence = profiles[:, 4]
        for profile in range(len(profiles)):
            # ant = int(profiles[profile, 0] - 1)
            spring = int(profiles[profile, 1])
            start = int(profiles[profile, 2])
            end = int(profiles[profile, 3])
            if not end - start + 1 > 10000:
                self.all_profiles_force_magnitude[profile, 0:end - start + 1] = self.force_magnitude[start:end + 1, int(spring - 1)]
                self.all_profiles_force_direction[profile, 0:end - start + 1] = self.force_direction[start:end + 1, int(spring - 1)]
                self.all_profiles_ants_number[profile, 0:end - start + 1] = self.N_ants_around_springs[start:end + 1, int(spring - 1)]
                self.all_profiles_angle_to_nest[profile, 0:end - start + 1] = self.fixed_end_angle_to_nest[start:end + 1, int(spring - 1)]
            else:
                self.all_profiles_precedence[profile] = np.nan

    def save_data(self):
        if self.two_vars:
            save_path = os.path.join(self.directory, "two_vars_post_processing")
        else:
            save_path = os.path.join(self.directory, "one_var_post_processing")
        for count, data_set in enumerate(self.sub_dirs_sets):
            set_save_path = os.path.join(save_path, f"{data_set[0]}-{data_set[-1]}")
            os.makedirs(set_save_path, exist_ok=True)
            print("-" * 60)
            print("saving data to:", set_save_path)
            print(self.N_ants_around_springs[self.sets_idx[count][0]:self.sets_idx[count][1]].shape)
            np.savez_compressed(os.path.join(set_save_path, "N_ants_around_springs.npz"), self.N_ants_around_springs[self.sets_idx[count][0]:self.sets_idx[count][1]])
            np.savez_compressed(os.path.join(set_save_path, "fixed_end_angle_to_nest.npz"), self.fixed_end_angle_to_nest[self.sets_idx[count][0]:self.sets_idx[count][1]])
            np.savez_compressed(os.path.join(set_save_path, "force_direction.npz"), self.force_direction[self.sets_idx[count][0]:self.sets_idx[count][1]])
            np.savez_compressed(os.path.join(set_save_path, "force_magnitude.npz"), self.force_magnitude[self.sets_idx[count][0]:self.sets_idx[count][1]])
        # np.savez_compressed(os.path.join(save_path, "all_profiles_force_magnitude.npy"),
        #                     self.all_profiles_force_magnitude)
        # np.savez_compressed(os.path.join(save_path, "all_profiles_force_direction.npy"),
        #                     self.all_profiles_force_direction)
        # np.savez_compressed(os.path.join(save_path, "all_profiles_ants_number.npy"), self.all_profiles_ants_number)
        # np.savez_compressed(os.path.join(save_path, "all_profiles_angle_to_nest.npy"), self.all_profiles_angle_to_nest)
        # np.savez_compressed(os.path.join(save_path, "all_profiles_precedence.npy"), self.all_profiles_precedence)

    def calculations(self):
        self.net_force = np.nansum(self.total_force, axis=1)
        self.net_force = np.array(pd.Series(self.net_force).rolling(window=5, center=True).median())
        self.net_magnitude = np.nansum(self.force_magnitude, axis=1)
        self.net_magnitude = np.array(pd.Series(self.net_magnitude).rolling(window=5, center=True).median())
        self.angular_velocity = np.nanmedian(
            utils.calc_angular_velocity(self.fixed_end_angle_to_nest, diff_spacing=20) / 20, axis=1)
        self.total_n_ants = np.nansum(self.N_ants_around_springs, axis=1)
        self.sum_pulling_angle = np.nansum(self.pulling_angle, axis=1)
        self.angle_to_nest = np.nansum(self.fixed_end_angle_to_nest, axis=1)
        self.spring_extension = np.nanmean(self.spring_length, axis=1)
        self.total_force[np.isnan(self.pulling_angle)] = np.nan

    def test_correlation(self):
        corr_df = pd.DataFrame({"net_force": self.net_force, "angular_velocity": self.angular_velocity})
        corr_df = corr_df.dropna()
        correlation_score = corr_df.corr()["net_force"]["angular_velocity"]
        print(f"correlation score between net force and angular velocity: {correlation_score}")
        return correlation_score


if __name__ == "__main__":
    calibration_dir = "Z:\\Dor_Gabay\\ThesisProject\\data\\calibration\\post_slicing\\calibration_perfect2\\sliced_videos\\"
    data_dir = "Z:\\Dor_Gabay\\ThesisProject\\data\\analysed_with_tracking\\15.9.22\\plus0.3mm_force"
    calibration_model = pickle.load(open(calibration_dir + "calibration_model.pkl", "rb"))
    # sub_dirs_names = [f"S528000{i}" for i in [3, 4, 5, 6, 7, 8, 9]]
    # sets = [(3, 4, 5, 6, 7), (8, 9)]
    sub_dirs_sets = [[f"S528000{i}" for i in [3, 4, 5, 6, 7]], [f"S528000{i}" for i in [8, 9]]]
    arrangements = [0, 0, 0, 0, 0, 19, 0]
    slicing_info = [None, None, None, [30000, 68025], [0, 30000], None, None]
    two_vars = True
    object = PostProcessing(data_dir,
                            sub_dirs_sets=sub_dirs_sets,
                            calibration_model=calibration_model,
                            two_vars=True,
                            slicing_info=slicing_info,
                            arrangements=arrangements)
    object.save_data()

    from data_analysis import plots

    object.calculations()
    correlation_score = object.test_correlation()
    figures_path = os.path.join(data_dir, "figures_two_vars")
    os.makedirs(figures_path, exist_ok=True)
    plots.plot_overall_behavior(object, start=0, end=None, window_size=200,
                                title="all_videos_corr " + str(np.round(correlation_score, 2)),
                                output_dir=figures_path)

    # def save_as_mathlab_matrix(output_dir):
    #     ants_centers_x = np.loadtxt(os.path.join(output_dir, "ants_centers_x.csv"), delimiter=",")
    #     ants_centers_y = np.loadtxt(os.path.join(output_dir, "ants_centers_y.csv"), delimiter=",")
    #     ants_centers = np.stack((ants_centers_x, ants_centers_y), axis=2)
    #     ants_centers_mat = np.zeros((ants_centers.shape[0], 1), dtype=np.object)
    #     for i in range(ants_centers.shape[0]):
    #         ants_centers_mat[i, 0] = ants_centers[i, :, :]
    #     sio.savemat(os.path.join(output_dir, "ants_centers.mat"), {"ants_centers": ants_centers_mat})
    #     matlab_script_path = "Z:\\Dor_Gabay\\ThesisProject\\scripts\\munkres_tracker\\"
    #     os.chdir(matlab_script_path)
    #     os.system("PATH=$PATH:'C:\Program Files\MATLAB\R2022a\bin'")
    #     execution_string = f"matlab -r ""ants_tracking('" + output_dir + "\\')"""
    #     os.system(execution_string)
    #
    #
    # def save_blue_areas_median(output_dir):
    #     blue_area_sizes = np.loadtxt(os.path.join(output_dir, "blue_area_sizes.csv"), delimiter=",")
    #     median_blue_area_size = np.nanmedian(blue_area_sizes)
    #     with open(os.path.join(output_dir, "blue_median_area.pickle"), 'wb') as f:
    #         pickle.dump(median_blue_area_size, f)

    # for i in [1, 3, 4, 5, 6, 7, 8, 9]:
    #     path = os.path.join(data_dir, f"S528000{i}")
    #     # save_blue_areas_median(os.path.join(path,"raw_analysis"))
    #     # save_as_mathlab_matrix(os.path.join(path,"raw_analysis"))
    #     # matlab_script_path = "Z:\\Dor_Gabay\\ThesisProject\\scripts\\munkres_tracker\\"
    #     # os.chdir(matlab_script_path)
    #     # os.system("PATH=$PATH:'C:\Program Files\MATLAB\R2022a\bin'")
    #     # execution_string = f"matlab -r ""ants_tracking('" +os.path.join(path,"raw_analysis") + "\\')"""
    #     # os.system(execution_string)
    #     if i == 7:
    #         objects.append(
    #             PostProcessing(path, calibration_model=calibration_model, slice=[30000, 68025], two_vars=two_vars,
    #                            correct_tracking=False))
    #     elif i == 6:
    #         objects.append(
    #             PostProcessing(path, calibration_model=calibration_model, slice=[0, 30000], two_vars=two_vars,
    #                            correct_tracking=False))
    #         # objects.append(PostProcessing(path, calibration_model=calibration_model,slice=[0,30000],two_vars=two_vars,correct_tracking=True))
    #     else:
    #         objects.append(
    #             PostProcessing(path, calibration_model=calibration_model, two_vars=two_vars, correct_tracking=False))
    # models_lengths, models_angles = create_model(objects)
    # for object in objects:
    #     print("-" * 60)
    #     object.calc_spring_length(models=models_lengths)
    #     object.calc_pulling_angle(models=models_angles)
    #     object.calc_force(calibration_model)
    #     object.save_data()
    #
    #     # correlation plots:
    #     object.calculations()
    #     correlation_score = object.test_correlation()
    #     object.video_name = object.directory.split("\\")[-1]
    #     if two_vars:
    #         object.figures_output_path = os.path.join(data_dir, "figures_two_vars_magnitude2")
    #     else:
    #         object.figures_output_path = os.path.join(data_dir, "figures_one_var")
    #     os.makedirs(object.figures_output_path, exist_ok=True)
    #     plots.plot_overall_behavior(object, start=0, end=None, window_size=200,
    #                                 title=object.video_name + "_corr" + str(np.round(correlation_score, 2)),
    #                                 output_dir=object.figures_output_path)

    # def create_model(objects):
    #     X = None
    #     y_length = None
    #     y_angle = None
    #     idx = None
    #     for count,object in enumerate(objects):
    #         if count == 0:
    #             y_length = np.linalg.norm(object.free_ends_coordinates - object.fixed_ends_coordinates , axis=2)
    #             angles_to_nest = np.expand_dims(object.fixed_end_angle_to_nest, axis=2)
    #             fixed_end_distance = np.expand_dims(object.object_center_to_fixed_end_distance, axis=2)
    #             fixed_to_tip_distance = np.expand_dims(object.object_blue_tip_to_fixed_end_distance, axis=2)
    #             fixed_to_blue_angle_change = np.expand_dims(np.repeat(np.expand_dims(np.nanmedian(object.fixed_to_blue_angle_change,axis=1),axis=1),object.num_of_springs,axis=1),axis=2)
    #             object_center = object.object_center_repeated
    #             blue_length = np.expand_dims(np.repeat(np.expand_dims(np.linalg.norm(
    #                 object.blue_tip_coordinates - object.object_center, axis=1), axis=1), object.num_of_springs, axis=1), axis=2)
    #             X = np.concatenate((np.sin(angles_to_nest), np.cos(angles_to_nest),object_center,fixed_end_distance,
    #                                 fixed_to_tip_distance,fixed_to_blue_angle_change,blue_length), axis=2)
    #             y_angle = utils.calc_pulling_angle_matrix(object.fixed_ends_coordinates,
    #                                                                  object.object_center_repeated,
    #                                                                  object.free_ends_coordinates)
    #             idx = object.rest_bool
    #         else:
    #             y_length = np.concatenate((y_length,np.linalg.norm(object.free_ends_coordinates - object.fixed_ends_coordinates , axis=2)),axis=0)
    #             angles_to_nest = np.expand_dims(object.fixed_end_angle_to_nest,axis=2)
    #             fixed_end_distance = np.expand_dims(object.object_center_to_fixed_end_distance, axis=2)
    #             fixed_to_tip_distance = np.expand_dims(object.object_blue_tip_to_fixed_end_distance, axis=2)
    #             fixed_to_blue_angle_change = np.expand_dims(np.repeat(np.expand_dims(np.nanmedian(object.fixed_to_blue_angle_change,axis=1),axis=1),object.num_of_springs,axis=1),axis=2)
    #             object_center = object.object_center_repeated
    #             blue_length = np.expand_dims(np.repeat(np.expand_dims(np.linalg.norm(
    #                 object.blue_tip_coordinates - object.object_center, axis=1), axis=1), object.num_of_springs, axis=1), axis=2)
    #             X = np.concatenate((X,np.concatenate((np.sin(angles_to_nest),np.cos(angles_to_nest),object_center,
    #                                                   fixed_end_distance,fixed_to_tip_distance,fixed_to_blue_angle_change,
    #                                                   blue_length),axis=2)),axis=0)
    #             y_angle = np.concatenate((y_angle,utils.calc_pulling_angle_matrix(object.fixed_ends_coordinates,
    #                                                                     object.object_center_repeated,
    #                                                                     object.free_ends_coordinates)),axis=0)
    #             idx = np.concatenate((idx,object.rest_bool),axis=0)
    #     models_lengths = []
    #     models_angles = []
    #     not_nan_idx = ~(np.isnan(y_angle) + np.isnan(X).any(axis=2))*idx
    #     for col in range(y_length.shape[1]):
    #         X_fit = X[not_nan_idx[:,col],col]
    #         y_length_fit = y_length[not_nan_idx[:,col],col]
    #         y_angle_fit = y_angle[not_nan_idx[:,col],col]
    #         model_length = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
    #         model_angle = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
    #         models_lengths.append(model_length.fit(X_fit, y_length_fit))
    #         models_angles.append(model_angle.fit(X_fit, y_angle_fit))
    #     return models_lengths,models_angles

    # def load_data(self, slice=None):
    #
    #     directory = os.path.join(self.directory, "raw_analysis")
    #     print("loading data from:", directory)
    #     # if (not os.path.exists(os.path.join(self.directory, "two_vars_post_processing", "tracking_data_corrected.mat"))) or self.correct_tracking:
    #     #     print("Correcting tracking data since corrected file does not exist or it was requested...")
    #     #     self.correct_tracked_ants()
    #     # self.tracked_ants = sio.loadmat(os.path.join(self.directory, "two_vars_post_processing", "tracking_data_corrected.mat"))["tracked_blobs_matrix"]
    #     # self.ants_attached_labels  = np.loadtxt(os.path.join(directory, "ants_attached_labels.csv"), delimiter=",")
    #     if os.path.exists(os.path.join(self.directory,"blue_median_area.pickle")):
    #         self.norm_size = pickle.load(open(os.path.join(self.directory,"blue_median_area.pickle"), "rb"))
    #     else:
    #         self.norm_size = pickle.load(open(os.path.join(directory,"blue_median_area.pickle"), "rb"))
    #     self.N_ants_around_springs = np.loadtxt(os.path.join(directory,"N_ants_around_springs.csv"), delimiter=",")
    #     self.num_of_springs = self.N_ants_around_springs.shape[1]
    #     self.size_ants_around_springs = np.loadtxt(os.path.join(directory,"size_ants_around_springs.csv"), delimiter=",")
    #     fixed_ends_coordinates_x = np.loadtxt(os.path.join(directory,"fixed_ends_coordinates_x.csv"), delimiter=",")
    #     fixed_ends_coordinates_y = np.loadtxt(os.path.join(directory,"fixed_ends_coordinates_y.csv"), delimiter=",")
    #     self.fixed_ends_coordinates = np.stack((fixed_ends_coordinates_x, fixed_ends_coordinates_y), axis=2)
    #     free_ends_coordinates_x = np.loadtxt(os.path.join(directory,"free_ends_coordinates_x.csv"), delimiter=",")
    #     free_ends_coordinates_y = np.loadtxt(os.path.join(directory,"free_ends_coordinates_y.csv"), delimiter=",")
    #     self.free_ends_coordinates = np.stack((free_ends_coordinates_x, free_ends_coordinates_y), axis=2)
    #     blue_part_coordinates_x = np.loadtxt(os.path.join(directory,"blue_part_coordinates_x.csv"), delimiter=",")
    #     blue_part_coordinates_y = np.loadtxt(os.path.join(directory,"blue_part_coordinates_y.csv"), delimiter=",")
    #     blue_part_coordinates = np.stack((blue_part_coordinates_x, blue_part_coordinates_y), axis=2)
    #     self.object_center = blue_part_coordinates[:, 0, :]
    #     self.blue_tip_coordinates = blue_part_coordinates[:, -1, :]
    #     self.num_of_frames = self.N_ants_around_springs.shape[0]
    #     # self.num_of_frames = 1000
    #     if slice is not None:
    #         self.slice_data(*slice)
    #     # self.correct_tracked_ants()

    # @profile
    # def correct_tracked_ants(self):
    #     "This function removes all new labels that were create inside the frame, and not on the boundries (10% of the frame)"
    #     def test_if_in_boundries(coords):
    #         if coords[0] > self.frame_size[1] * 0.1 and coords[0] < self.frame_size[0] - self.frame_size[1] * 0.1 and coords[1] > self.frame_size[1] * 0.1 and coords[1] < self.frame_size[1] * 0.9:
    #             return True
    #         else:
    #             return False
    #     directory = os.path.join(self.directory, "raw_analysis") + "\\"
    #     self.tracked_ants = sio.loadmat(os.path.join(directory, "tracking_data.mat"))["tracked_blobs_matrix"]
    #     # self.tracked_ants = self.tracked_ants[:, :, 0:1000]
    #     self.frame_size = (1920, 1080)
    #     self.num_of_frames = self.tracked_ants.shape[2]
    #
    #     frames_arange = np.repeat(np.arange(self.tracked_ants.shape[2])[np.newaxis, :], self.tracked_ants.shape[0], axis=0)
    #     ants_arange = np.repeat(np.arange(self.tracked_ants.shape[0])[np.newaxis, :], self.tracked_ants.shape[2], axis=0).T
    #     self.tracked_ants[np.isnan(self.tracked_ants)] = 0
    #
    #     appeared_labels = np.array([])
    #     appeared_labels_idx = []
    #     appeared_last_coords = np.array([]).reshape(0, 2)
    #     disappeared_labels = []
    #     disappeared_labels_idx = []
    #     disappeared_labels_partners = []
    #     disappeared_labels_partners_relative_distance = []
    #     disappeared_labels_partners_idx = []
    #
    #     for frame in range(1,self.num_of_frames-1):
    #         print(frame)
    #         if frame+20>=self.num_of_frames:
    #             break
    #
    #         if frame >= 11: min_frame = frame - 10
    #         else: min_frame = 1
    #
    #         labels_in_window = set(self.tracked_ants[:,2,min_frame:frame+20].flatten())
    #         labels_scanned_for_appearing = set(self.tracked_ants[:,2,min_frame-1].flatten())
    #         labels_now = set(self.tracked_ants[:, 2, frame])
    #         labels_before = set(self.tracked_ants[:, 2, frame - 1])
    #         appearing_labels = labels_in_window.difference(labels_scanned_for_appearing)
    #         disappearing_labels = labels_before.difference(labels_now)
    #
    #         appeared_not_collected = appearing_labels.difference(set(appeared_labels))
    #         for label in appeared_not_collected:
    #             label_idx = np.isin(self.tracked_ants[:, 2, :], label)
    #             label_idx = (ants_arange[label_idx],frames_arange[label_idx])
    #             coords = self.tracked_ants[label_idx[0], 0:2, label_idx[1]]
    #             first_appear = np.argmin(label_idx[1])
    #             if test_if_in_boundries(coords[first_appear, :]):
    #                 appeared_labels = np.append(appeared_labels,label)
    #                 median_coords = coords[first_appear,:].reshape(1,2)
    #                 appeared_last_coords = np.concatenate((appeared_last_coords, median_coords), axis=0)
    #                 appeared_labels_idx.append(label_idx)
    #
    #         currently_appearing = np.array([x for x in range(len(appeared_labels)) if appeared_labels[x] in appearing_labels])
    #         if len(currently_appearing)>0:
    #             currently_appearing_labels = appeared_labels[currently_appearing]
    #             currently_appearing_idx = [appeared_labels_idx[x] for x in currently_appearing]
    #             currently_appearing_last_coords = appeared_last_coords[currently_appearing]
    #             for label in disappearing_labels:
    #                 label_idx = np.where(self.tracked_ants[:, 2, :] == label)
    #                 coords = self.tracked_ants[label_idx[0], 0:2, label_idx[1]]
    #                 last_appear = np.argmax(label_idx[1])
    #                 if test_if_in_boundries(coords[last_appear, :]):
    #                     if currently_appearing_last_coords.shape[0] > 0:
    #                         distances = np.linalg.norm(currently_appearing_last_coords - coords[last_appear, :], axis=1)
    #                         closest_appearing_label_argmin = np.argsort(distances)[0]
    #                         if label==currently_appearing_labels[closest_appearing_label_argmin] \
    #                                 and len(distances)>1:
    #                             closest_appearing_label_argmin = np.argsort(distances)[1]
    #                         partner_label = currently_appearing_labels[closest_appearing_label_argmin]
    #                         distance_closest = distances[closest_appearing_label_argmin]
    #                         idx_closest_appearing_label = currently_appearing_idx[closest_appearing_label_argmin]
    #                         if distance_closest<300:
    #                             disappeared_labels.append(label)
    #                             disappeared_labels_idx.append(label_idx)
    #                             disappeared_labels_partners.append(partner_label)
    #                             disappeared_labels_partners_relative_distance.append(distance_closest)
    #                             disappeared_labels_partners_idx.append(idx_closest_appearing_label)
    #     for i in range(len(disappeared_labels)):
    #         idx = disappeared_labels_partners_idx[i]
    #         self.tracked_ants[idx[0], 2, idx[1]] = disappeared_labels[i]
    #         if disappeared_labels_partners[i] in disappeared_labels:
    #             disappeared_labels[disappeared_labels.index(disappeared_labels_partners[i])] = disappeared_labels[i]
    #         duplicated_idxs = np.arange(len(disappeared_labels_partners))[np.array(disappeared_labels_partners)==disappeared_labels_partners[i]]
    #         duplicated_labels = [disappeared_labels[x] for x in duplicated_idxs]
    #         disappeared_labels = [np.min(duplicated_labels) if x in duplicated_labels else x for x in disappeared_labels]
    #     for unique_label in np.unique(self.tracked_ants[:, 2, :]):
    #         occurrences = np.count_nonzero(self.tracked_ants[:, 2, :] == unique_label)
    #         if occurrences < 20:
    #             idx = np.where(self.tracked_ants[:, 2, :] == unique_label)
    #             self.tracked_ants[idx[0], 2, idx[1]] = 0
    #     if not os.path.exists(os.path.join(self.directory, "two_vars_post_processing")):
    #         os.makedirs(os.path.join(self.directory, "two_vars_post_processing"))
    #     sio.savemat(
    #         os.path.join(os.path.join(self.directory, "two_vars_post_processing"), "tracking_data_corrected.mat"),
    #         {"tracked_blobs_matrix": self.tracked_ants})
    #
    # def assign_ants_to_springs(self):
    #     self.ants_assigned_to_springs = np.zeros((self.num_of_frames, np.nanmax(self.tracked_ants[:,2,:]).astype(np.int32)))
    #     for i in range(self.num_of_frames):
    #         labels = self.tracked_ants[~np.isnan(self.ants_attached_labels[i]),2,i]
    #         labels = labels[~np.isnan(labels)].astype(int)
    #         springs = self.ants_attached_labels[i,~np.isnan(self.ants_attached_labels[i])].astype(int)
    #         self.ants_assigned_to_springs[i,labels-1] = springs+1
    #     self.interpolate_assigned_ants()
    #
    # def interpolate_assigned_ants(self,max_gap=30):
    #     aranged_frames = np.arange(self.num_of_frames)
    #     for ant in range(self.ants_assigned_to_springs.shape[1]):
    #         array = self.ants_assigned_to_springs[:,ant]
    #         closing_bool = binary_closing(array.astype(bool).reshape(1,self.num_of_frames), np.ones((1, max_gap)))
    #         closing_bool = closing_bool.reshape(closing_bool.shape[1])
    #         xp = aranged_frames[~closing_bool+(array!=0)]
    #         fp = array[xp]
    #         x = aranged_frames[~closing_bool+(array==0)]
    #         self.ants_assigned_to_springs[x,ant] = np.round(np.interp(x, xp, fp))
    #
    # def remove_artificial_cases(self):
    #     print("Removing artificial cases...")
    #     unreal_ants_attachments = np.full(self.N_ants_around_springs.shape, np.nan)
    #     switch_attachments = np.full(self.N_ants_around_springs.shape, np.nan)
    #     unreal_detachments = np.full(self.N_ants_around_springs.shape, np.nan)
    #     for ant in range(self.ants_assigned_to_springs.shape[1]):
    #         attachment = self.ants_assigned_to_springs[:,ant]
    #         if not np.all(attachment == 0):
    #             first_attachment = np.where(attachment != 0)[0][0]
    #             for count, spring in enumerate(np.unique(attachment)[1:]):
    #                 frames = np.where(attachment==spring)[0]
    #                 sum_assign = np.sum(self.ants_assigned_to_springs[frames[0]:frames[-1]+3,:] == spring, axis=1)
    #                 real_sum = self.N_ants_around_springs[frames[0]:frames[-1]+3,int(spring-1)]
    #                 if frames[0] == first_attachment and np.all(sum_assign[0:2] != real_sum[0:2]):
    #                     unreal_ants_attachments[frames,int(spring-1)] = ant
    #                 elif np.all(sum_assign[0:2] != real_sum[0:2]):
    #                     switch_attachments[frames,int(spring-1)] = ant
    #                 else:
    #                     if np.all(sum_assign[-1] != real_sum[-1]):
    #                         unreal_detachments[frames[-1], int(spring-1)] = ant
    #     for frame in range(self.ants_assigned_to_springs.shape[0]):
    #         detach_ants = np.unique(unreal_detachments[frame,:])
    #         for detach_ant in detach_ants[~np.isnan(detach_ants)]:
    #             spring = np.where(unreal_detachments[frame,:] == detach_ant)[0][0]
    #             assign_ant = unreal_ants_attachments[frame,spring]
    #             if not np.isnan(assign_ant):
    #                 self.ants_assigned_to_springs[unreal_ants_attachments[:,spring]==assign_ant,int(detach_ant)] = spring+1
    #                 self.ants_assigned_to_springs[unreal_ants_attachments[:,spring]==assign_ant,int(assign_ant)] = 0
    #         switch_ants = np.unique(switch_attachments[frame, :])
    #         for switch_ant in switch_ants[~np.isnan(switch_ants)]:
    #             spring = np.where(switch_attachments[frame,:] == switch_ant)[0][0]
    #             switch_from_spring = np.where(switch_attachments[frame-1,:] == switch_ant)[0]
    #             switch_from_spring = switch_from_spring[switch_from_spring != spring]
    #             if len(switch_from_spring) > 0:
    #                 switch_frames = np.where(switch_attachments[:,spring] == switch_ant)[0]
    #                 self.ants_assigned_to_springs[switch_frames, int(switch_ant)] = switch_from_spring[0]
    #     self.ants_assigned_to_springs = self.ants_assigned_to_springs[:,:-1]

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

    # def correct_tracked_ants(self):
    #     "This function removes all new labels that were create inside the frame, and not on the boundries (10% of the frame)"
    #     directory = os.path.join(self.directory, "raw_analysis")+"\\"
    #     self.tracked_ants = sio.loadmat(os.path.join(directory, "tracking_data.mat"))["tracked_blobs_matrix"]
    #     self.frame_size = (1920, 1080)
    #     self.num_of_frames = self.tracked_ants.shape[2]
    #     MIN_DISTANCE = 250
    #     self.tracked_ants[np.isnan(self.tracked_ants)] = 0
    #     labels_scanned = set(self.tracked_ants[:,2,0])
    #     # for frame in range(1500):
    #     for frame in range(self.num_of_frames):
    #         if frame==self.num_of_frames-1:
    #             break
    #         print("frame:", frame, end="\r")
    #         labels = set(self.tracked_ants[:,2,frame])
    #         labels_not_in_labels_scanned = labels.difference(labels_scanned)
    #         for label in labels_not_in_labels_scanned:
    #             coords = self.tracked_ants[self.tracked_ants[:,2,frame]==label,0:2,frame][0,:].reshape(1,2)
    #             if coords[0,0]>self.frame_size[1]*0.1 and coords[0,0]<self.frame_size[0]-self.frame_size[1]*0.1  and coords[0,1]>self.frame_size[1]*0.1 and coords[0,1]<self.frame_size[1]*0.9:
    #                 all_coords = self.tracked_ants[:,0:2,frame]
    #                 distances = np.linalg.norm(all_coords-coords, axis=1)
    #                 distances_argsort = np.argsort(distances)
    #                 closest_distance = distances[distances_argsort[1]]
    #                 closest_label = self.tracked_ants[distances_argsort[1],2,frame]
    #                 disappeared_labels_before = self.tracked_ants[:,2,frame-1][np.isin(self.tracked_ants[:,2,frame-1],list(set(self.tracked_ants[:, 2, frame - 1].flatten()).difference(labels)))]
    #                 set_after = set(self.tracked_ants[:,2,frame+1].flatten())
    #                 disappeared_labels_after = self.tracked_ants[:,2,frame+1][np.isin(self.tracked_ants[:,2,frame+1],list(set_after.difference(labels)))]
    #                 disappeared_labels_before = disappeared_labels_before[np.isin(disappeared_labels_before,np.array(list(labels_scanned)))]
    #                 disappeared_labels_after = disappeared_labels_after[np.isin(disappeared_labels_after,np.array(list(labels_scanned)))]
    #                 disappeared_coords_before = self.tracked_ants[np.isin(self.tracked_ants[:,2,frame-1],disappeared_labels_before),0:2,frame-1]
    #                 disappeared_coords_after = self.tracked_ants[np.isin(self.tracked_ants[:,2,frame],disappeared_labels_after),0:2,frame]
    #                 disappeared_distances_before = np.linalg.norm(disappeared_coords_before-coords, axis=1)
    #                 disappeared_distances_after = np.linalg.norm(disappeared_coords_after-coords, axis=1)
    #                 if closest_distance<MIN_DISTANCE:
    #                     disappeared_n_closest = np.sum(disappeared_distances_before<MIN_DISTANCE)+np.sum(disappeared_distances_after<MIN_DISTANCE)
    #                     if disappeared_n_closest>0:
    #                         for second_label in np.append(disappeared_labels_before,disappeared_labels_after):
    #                             if second_label == 0:
    #                                 continue
    #                             else:
    #                                 second_label_coords_before = disappeared_coords_before[disappeared_labels_before==second_label,:]
    #                                 second_label_coords_after = disappeared_coords_after[disappeared_labels_after==second_label,:]
    #                                 second_label_coords = np.concatenate((second_label_coords_before,second_label_coords_after),axis=0)
    #                                 if second_label_coords[0,0]>self.frame_size[1]*0.05 and second_label_coords[0,0]<self.frame_size[0]-self.frame_size[1]*0.05  and second_label_coords[0,1]>self.frame_size[1]*0.05 and second_label_coords[0,1]<self.frame_size[1]*0.95:
    #                                     closest_label = second_label
    #                                     idx = np.where(self.tracked_ants[:, 2, :] == label)
    #                                     self.tracked_ants[idx[0], 2, idx[1]] = closest_label
    #                                     break
    #                     else:
    #                         idx = np.where(self.tracked_ants[:,2,:]==label)
    #                         self.tracked_ants[idx[0], 2, idx[1]] = closest_label
    #                 else:
    #                     idx = np.where(self.tracked_ants[:,2,:]==label)
    #                     self.tracked_ants[idx[0], :, idx[1]] = 0
    #             else:
    #                 labels_scanned = labels_scanned.union({label})
    #     if not os.path.exists(os.path.join(self.directory, "two_vars_post_processing")):
    #         os.makedirs(os.path.join(self.directory, "two_vars_post_processing"))
    #     sio.savemat(
    #         os.path.join(os.path.join(self.directory, "two_vars_post_processing"), "tracking_data_corrected.mat"),
    #         {"tracked_blobs_matrix": self.tracked_ants})
