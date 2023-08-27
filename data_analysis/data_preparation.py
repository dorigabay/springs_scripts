import copy
import numpy as np
import pandas as pd
import os
import pickle
from scipy.signal import savgol_filter
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import itertools
import multiprocessing as mp
import pickle
# local imports:
# from data_analysis import utils
import utils
# from data_analysis.ant_tracking import AntTracking
from ant_tracking import AntTracking


class PostProcessing:
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.frame_resolution = data_dict["frame_resolution"]
        self.data_paths = []
        self.arrangements = []
        self.slicing_info = []
        for data_dir in data_dict["data_sets"].keys():
            for data_set in data_dict["data_sets"][data_dir]:
                self.data_paths.append([os.path.join(data_dir, sub_dir_name) for sub_dir_name in data_set["set_names"]])
                self.arrangements += data_set["set_arrangements"]
                self.slicing_info += data_set["set_slice_info"]
        self.data_paths_flatten = list(itertools.chain(*self.data_paths))

        self.load_data()
        self.n_ants_processing()
        self.calc_distances()
        self.repeat_values()
        self.calc_angle()
        self.create_model()
        self.calc_spring_length()
        self.calc_pulling_angle()
        self.calc_force(data_dict["calibration_model"])

    def load_data(self):
        print("loading data...")
        self.norm_size = np.array([np.nanmedian(
            np.loadtxt(os.path.join(data_path, "raw_analysis", "blue_area_sizes.csv"), delimiter=",")) for data_path in self.data_paths_flatten])
        self.ants_attached_labels = np.concatenate([np.loadtxt(
            os.path.join(data_path, "raw_analysis", "ants_attached_labels.csv"), delimiter=",") for data_path in self.data_paths_flatten], axis=0)
        self.N_ants_around_springs = np.concatenate([np.loadtxt(
            os.path.join(data_path, "raw_analysis", "N_ants_around_springs.csv"), delimiter=",") for data_path in self.data_paths_flatten], axis=0)
        self.size_ants_around_springs = np.concatenate([np.loadtxt(
            os.path.join(data_path, "raw_analysis", "size_ants_around_springs.csv"), delimiter=",") for data_path in self.data_paths_flatten], axis=0)
        self.fixed_ends_coordinates = np.concatenate([np.stack((
            np.loadtxt(os.path.join(data_path, "raw_analysis", "fixed_ends_coordinates_x.csv"), delimiter=","),
            np.loadtxt(os.path.join(data_path, "raw_analysis", "fixed_ends_coordinates_y.csv"), delimiter=",")), axis=2)
            for  data_path in self.data_paths_flatten], axis=0)
        self.free_ends_coordinates = np.concatenate([np.stack((
            np.loadtxt(os.path.join(data_path, "raw_analysis", "free_ends_coordinates_x.csv"), delimiter=","),
            np.loadtxt(os.path.join(data_path, "raw_analysis", "free_ends_coordinates_y.csv"), delimiter=",")), axis=2)
            for data_path in self.data_paths_flatten], axis=0)
        blue_part_coordinates = np.concatenate(
            [np.loadtxt(os.path.join(data_path, "raw_analysis", "blue_part_coordinates_x.csv"),
                        delimiter=",") for data_path in self.data_paths_flatten], axis=0)
        blue_part_coordinates = np.stack((blue_part_coordinates, np.concatenate(
            [np.loadtxt(os.path.join(data_path, "raw_analysis", "blue_part_coordinates_y.csv"),
                        delimiter=",") for data_path in self.data_paths_flatten], axis=0)), axis=2)
        self.object_center = blue_part_coordinates[:, 0, :]
        self.blue_tip_coordinates = blue_part_coordinates[:, -1, :]
        self.num_of_frames_per_video = np.array(
            [np.loadtxt(os.path.join(data_path, "raw_analysis", "N_ants_around_springs.csv"),
                        delimiter=",").shape[0] for data_path in self.data_paths_flatten])
        self.make_sets_idx()
        self.num_of_frames = np.sum(self.num_of_frames_per_video).astype(int)
        self.num_of_springs = self.N_ants_around_springs.shape[1]
        self.slice_data(self.slicing_info) if self.slicing_info is not None else None
        self.rearrange_data(self.arrangements) if self.arrangements is not None else None
    #     self.interpolate_data()
    #

    def make_sets_idx(self):
        self.sets_idx = []
        videos_count = 0
        for count, data_set in enumerate(self.data_paths):
            if count == 0:
                self.sets_idx.append([0, np.sum(self.num_of_frames_per_video[:len(data_set)])])
            else:
                self.sets_idx.append(
                    [self.sets_idx[-1][-1], np.sum(self.num_of_frames_per_video[:videos_count + len(data_set)])])
            videos_count += len(data_set)

    def rearrange_data(self, arrangements):
        n = self.num_of_springs
        for count, arrangement in enumerate(arrangements):
            if arrangement != None:
                start_frame = np.sum(self.num_of_frames_per_video[:count])
                end_frame = np.sum(self.num_of_frames_per_video[:count + 1])
                rearrangement = np.append(np.arange(arrangement, n), np.arange(0, arrangement))
                self.N_ants_around_springs[start_frame:end_frame, :] = self.N_ants_around_springs[start_frame:end_frame,
                                                                       rearrangement]
                self.size_ants_around_springs[start_frame:end_frame, :] = self.size_ants_around_springs[
                                                                          start_frame:end_frame, rearrangement]
                self.fixed_ends_coordinates[start_frame:end_frame, :, :] = self.fixed_ends_coordinates[
                                                                           start_frame:end_frame, rearrangement, :]
                self.free_ends_coordinates[start_frame:end_frame, :, :] = self.free_ends_coordinates[
                                                                          start_frame:end_frame, rearrangement, :]

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
            undetected_springs_for_long_time = utils.filter_continuity(
                utils.convert_bool_to_binary(np.isnan(self.fixed_ends_coordinates[start_frame:end_frame, :, 0])),
                min_size=8)
            self.N_ants_around_springs[start_frame:end_frame, :] = np.round(
                utils.interpolate_data(self.N_ants_around_springs[start_frame:end_frame, :],
                                       utils.find_cells_to_interpolate(
                                           self.N_ants_around_springs[start_frame:end_frame, :])))
            self.N_ants_around_springs[start_frame:end_frame, :][
                np.isnan(self.N_ants_around_springs[start_frame:end_frame, :])] = 0
            self.N_ants_around_springs[start_frame:end_frame, :] = self.smoothing_n_ants(
                self.N_ants_around_springs[start_frame:end_frame, :])
            self.N_ants_around_springs[start_frame:end_frame, :][undetected_springs_for_long_time] = np.nan
            all_small_attaches = np.zeros(self.N_ants_around_springs[start_frame:end_frame, :].shape)
            for n in np.unique(self.N_ants_around_springs[start_frame:end_frame, :])[1:]:
                if not np.isnan(n):
                    short_attaches = utils.filter_continuity(self.N_ants_around_springs[start_frame:end_frame, :] == n,
                                                             max_size=30)
                    all_small_attaches[short_attaches] = 1
            self.N_ants_around_springs[start_frame:end_frame, :] = \
                np.round(utils.interpolate_data(self.N_ants_around_springs[start_frame:end_frame, :],
                                                all_small_attaches.astype(bool)))
        self.rest_bool = self.N_ants_around_springs == 0

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
        self.object_blue_tip_to_fixed_end_distance = np.linalg.norm(self.fixed_ends_coordinates - blue_tip_coordinates,
                                                                    axis=2)

    def repeat_values(self):
        nest_direction = np.stack((self.object_center[:, 0], self.object_center[:, 1] - 100), axis=1)
        self.nest_direction_repeated = np.repeat(nest_direction[:, np.newaxis, :], self.free_ends_coordinates.shape[1],
                                                 axis=1)
        self.object_center_repeated = copy.copy(
            np.repeat(self.object_center[:, np.newaxis, :], self.free_ends_coordinates.shape[1], axis=1))
        self.blue_tip_coordinates_repeated = np.repeat(self.blue_tip_coordinates[:, np.newaxis, :],
                                                       self.free_ends_coordinates.shape[1], axis=1)

    def calc_angle(self):
        self.free_end_angle_to_nest = utils.calc_angle_matrix(self.free_ends_coordinates, self.object_center_repeated,
                                                              self.nest_direction_repeated) + np.pi
        self.fixed_end_angle_to_nest = utils.calc_angle_matrix(self.fixed_ends_coordinates, self.object_center_repeated,
                                                               self.nest_direction_repeated) + np.pi
        self.blue_part_angle_to_nest = utils.calc_angle_matrix(self.nest_direction_repeated,
                                                               self.object_center_repeated,
                                                               self.blue_tip_coordinates_repeated) + np.pi
        self.free_end_angle_to_blue_part = utils.calc_angle_matrix(self.blue_tip_coordinates_repeated,
                                                                   self.object_center_repeated,
                                                                   self.free_ends_coordinates) + np.pi
        self.fixed_end_angle_to_blue_part = utils.calc_angle_matrix(self.blue_tip_coordinates_repeated,
                                                                    self.object_center_repeated,
                                                                    self.fixed_ends_coordinates) + np.pi
        self.fixed_to_blue_angle_change = utils.calc_pulling_angle_matrix(self.blue_tip_coordinates_repeated,
                                                                          self.object_center_repeated,
                                                                          self.fixed_ends_coordinates)
        # self.interpolate_data()

    # def interpolate_data(self):
    #     for count, set_idx in enumerate(self.sets_idx):
    #         start_frame = set_idx[0]
    #         end_frame = set_idx[1]
    #         cells_to_interpolate = utils.find_cells_to_interpolate(self.N_ants_around_springs[start_frame:end_frame, :], min_size=5)
    #         self.free_end_angle_to_nest[start_frame:end_frame] = \
    #             utils.interpolate_data(self.free_end_angle_to_nest[start_frame:end_frame],
    #                                    cells_to_interpolate)
    #         self.fixed_end_angle_to_nest[start_frame:end_frame] = \
    #             utils.interpolate_data(self.fixed_end_angle_to_nest[start_frame:end_frame],
    #                                     cells_to_interpolate)
    #         self.blue_part_angle_to_nest[start_frame:end_frame] = \
    #             utils.interpolate_data(self.blue_part_angle_to_nest[start_frame:end_frame],
    #                                     cells_to_interpolate)
    #         self.free_end_angle_to_blue_part[start_frame:end_frame] = \
    #             utils.interpolate_data(self.free_end_angle_to_blue_part[start_frame:end_frame],
    #                                     cells_to_interpolate)
    #         self.fixed_end_angle_to_blue_part[start_frame:end_frame] = \
    #             utils.interpolate_data(self.fixed_end_angle_to_blue_part[start_frame:end_frame],
    #                                     cells_to_interpolate)
    #         self.fixed_to_blue_angle_change[start_frame:end_frame] = \
    #             utils.interpolate_data(self.fixed_to_blue_angle_change[start_frame:end_frame],
    #                                     cells_to_interpolate)


    def create_model(self):
        y_length = np.linalg.norm(self.free_ends_coordinates - self.fixed_ends_coordinates, axis=2)
        angles_to_nest = np.expand_dims(self.fixed_end_angle_to_nest, axis=2)
        fixed_end_distance = np.expand_dims(self.object_center_to_fixed_end_distance, axis=2)
        # fixed_to_tip_distance = np.expand_dims(self.object_blue_tip_to_fixed_end_distance, axis=2)
        # fixed_to_blue_angle_change = np.expand_dims(np.repeat(np.expand_dims(np.nanmedian(self.fixed_to_blue_angle_change, axis=1), axis=1), self.num_of_springs, axis=1), axis=2)
        # object_center = self.object_center_repeated
        blue_length = np.expand_dims(np.repeat(np.expand_dims(np.linalg.norm(
            self.blue_tip_coordinates - self.object_center, axis=1), axis=1), self.num_of_springs, axis=1), axis=2)
        # X = np.concatenate((np.sin(angles_to_nest), np.cos(angles_to_nest), object_center, fixed_end_distance,
        #       fixed_to_tip_distance, fixed_to_blue_angle_change, blue_length), axis=2)
        X = np.concatenate((np.sin(angles_to_nest), np.cos(angles_to_nest), blue_length, fixed_end_distance), axis=2)
        y_angle = utils.calc_pulling_angle_matrix(self.fixed_ends_coordinates, self.object_center_repeated,
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
        # fixed_to_blue_angle_change = np.expand_dims(np.repeat(np.expand_dims(np.nanmedian(self.fixed_to_blue_angle_change, axis=1), axis=1), self.num_of_springs, axis=1), axis=2)
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
        for count, set_idx in enumerate(self.sets_idx):
            start_frame = set_idx[0]
            end_frame = set_idx[1]
            self.pulling_angle[start_frame:end_frame] = \
                utils.interpolate_data(self.pulling_angle[start_frame:end_frame],
                utils.find_cells_to_interpolate(self.pulling_angle[start_frame:end_frame]))
            pulling_angle_if_rest = copy.copy(self.pulling_angle[start_frame:end_frame])
            pulling_angle_copy = copy.copy(self.pulling_angle[start_frame:end_frame])
            pulling_angle_if_rest[~self.rest_bool[start_frame:end_frame]] = np.nan
            pulling_angle_copy[self.rest_bool[start_frame:end_frame]] = np.nan
            self.pulling_angle[start_frame:end_frame] -= np.nanmedian(pulling_angle_if_rest, axis=0)

    def calc_spring_length(self):
        self.spring_length = np.linalg.norm(self.free_ends_coordinates - self.fixed_ends_coordinates, axis=2)
        pred_spring_length = self.norm_values(self.spring_length, self.models_lengths)
        self.spring_length /= pred_spring_length
        for count, set_idx in enumerate(self.sets_idx):
            start_frame = set_idx[0]
            end_frame = set_idx[1]
            self.spring_length[start_frame:end_frame] = utils.interpolate_data(
                self.spring_length[start_frame:end_frame],
                utils.find_cells_to_interpolate(self.spring_length[start_frame:end_frame]))
            self.spring_length[start_frame:end_frame] /= self.norm_size[count]
            median_spring_length_at_rest = copy.copy(self.spring_length[start_frame:end_frame])
            median_spring_length_at_rest[np.invert(self.rest_bool[start_frame:end_frame])] = np.nan
            median_spring_length_at_rest = np.nanmedian(median_spring_length_at_rest, axis=0)
            self.spring_length[start_frame:end_frame] /= median_spring_length_at_rest
        self.spring_extension = self.spring_length - 1

    def calc_force(self, model):
        self.force_direction = np.zeros((self.num_of_frames, self.num_of_springs))
        self.force_magnitude = np.zeros((self.num_of_frames, self.num_of_springs))
        self.tangential_force = np.zeros((self.num_of_frames, self.num_of_springs))
        for s in range(self.num_of_springs):
            X = np.stack((self.pulling_angle[:, s], self.spring_extension[:, s]), axis=1)
            un_nan_bool = ~np.isnan(X).any(axis=1)
            X = X[un_nan_bool]
            forces_predicted = model.predict(X)
            if self.data_dict["two_vars"]:
                self.force_direction[un_nan_bool, s] = forces_predicted[:, 0]
                self.force_magnitude[un_nan_bool, s] = forces_predicted[:, 1]
            else:
                self.tangential_force[un_nan_bool, s] = forces_predicted
        if self.data_dict["two_vars"]:
            self.tangential_force = np.sin(self.force_direction) * self.force_magnitude
        self.tangential_force *= -1

    def save_data(self):
        print(self.fixed_end_angle_to_nest)
        save_path = os.path.join(self.data_dict["output_path"])
        for count, set_paths in enumerate(self.data_paths):
            sub_dirs_names = [os.path.basename(os.path.normpath(path)) for path in set_paths]
            set_save_path = os.path.join(save_path, f"{sub_dirs_names[0]}-{sub_dirs_names[-1]}")
            os.makedirs(set_save_path, exist_ok=True)
            print("-" * 60)
            print("saving data to:", set_save_path)
            np.savez_compressed(os.path.join(set_save_path, "ants_attached_labels.npz"),
                                self.ants_attached_labels[self.sets_idx[count][0]:self.sets_idx[count][1]])
            np.savez_compressed(os.path.join(set_save_path, "N_ants_around_springs.npz"),
                                self.N_ants_around_springs[self.sets_idx[count][0]:self.sets_idx[count][1]])
            np.savez_compressed(os.path.join(set_save_path, "fixed_end_angle_to_nest.npz"),
                                self.fixed_end_angle_to_nest[self.sets_idx[count][0]:self.sets_idx[count][1]])
            np.savez_compressed(os.path.join(set_save_path, "tangential_force.npz"),
                                self.tangential_force[self.sets_idx[count][0]:self.sets_idx[count][1]])
            np.savez_compressed(os.path.join(set_save_path, "force_direction.npz"),
                                self.force_direction[self.sets_idx[count][0]:self.sets_idx[count][1]])
            np.savez_compressed(os.path.join(set_save_path, "force_magnitude.npz"),
                                self.force_magnitude[self.sets_idx[count][0]:self.sets_idx[count][1]])
            np.savez_compressed(os.path.join(set_save_path, "missing_info.npz"),
                                np.isnan(self.free_ends_coordinates[self.sets_idx[count][0]:self.sets_idx[count][1], :, 0]))
        pickle.dump(self.sets_idx, open(os.path.join(save_path, "sets_idx.pkl"), "wb"))

    def calculations(self):
        self.net_tangential_force = np.nansum(self.tangential_force, axis=1)
        self.net_tangential_force = np.array(pd.Series(self.net_tangential_force).rolling(window=5, center=True).median())
        self.net_magnitude = np.nansum(self.force_magnitude, axis=1)
        self.net_magnitude = np.array(pd.Series(self.net_magnitude).rolling(window=5, center=True).median())
        self.angular_velocity = np.nanmedian(
            utils.calc_angular_velocity(self.fixed_end_angle_to_nest, diff_spacing=20) / 20, axis=1)
        self.total_n_ants = np.nansum(self.N_ants_around_springs, axis=1)
        self.sum_pulling_angle = np.nansum(self.pulling_angle, axis=1)
        self.angle_to_nest = np.nansum(self.fixed_end_angle_to_nest, axis=1)
        self.spring_extension = np.nanmean(self.spring_length, axis=1)

    def test_correlation(self):
        self.calculations()
        corr_df = pd.DataFrame({"net_force": self.net_tangential_force, "angular_velocity": self.angular_velocity})
        corr_df = corr_df.dropna()
        correlation_score = corr_df.corr()["net_force"]["angular_velocity"]
        print(f"correlation score between net force and angular velocity: {correlation_score}")
        return correlation_score


# def glue_same_profiles(all_profiles_information, ants_attached_labels):
#     for ant in np.unique(all_profiles_information[:,0]):
#         ant_profiles = all_profiles_information[all_profiles_information[:,0]==ant]
#         for count, profile in range(ant_profiles.shape[0]):
#             if count > 0 and ant_profiles[count, 1] == ant_profiles[count-1, 1]:
#                 spring = ant_profiles[count, 1]
#                 start = ant_profiles[count, 2]
#                 end = ant_profiles[count-1, 3]+1
#                 if np.all(np.isnan(ants_attached_labels[start:end, spring-1])):
#                     ants_attached_labels[start:end, spring-1] = ant


def profile_ants_behavior(directory):
    sub_dirs_paths = [os.path.join(directory, sub_dir) for sub_dir in os.listdir(directory) if
                      os.path.isdir(os.path.join(directory, sub_dir))]
    for count, path in enumerate(sub_dirs_paths):
        # path_post_processing = os.path.join(directory, "post_processing", f"{data_set[0]}-{data_set[-1]}")
        print("Profiling ants for: ", path)
        # ants_assigned_to_springs = np.load(os.path.join(path, "ants_assigned_to_springs.npz"))["arr_0"][:, :-1].astype(np.uint8)
        ants_assigned_to_springs = np.load(os.path.join(path, "ants_assigned_to_springs_fixed.npz"))["arr_0"][:, :-1].astype(np.uint8)
        ants_attached_labels = np.load(os.path.join(path, "ants_attached_labels.npz"))["arr_0"]
        profiles = np.full(6, np.nan)  # ant, spring, start, end, precedence, sudden_appearance
        for ant in range(ants_assigned_to_springs.shape[1]):
            print(f"ant: {ant}")
            attachment = ants_assigned_to_springs[:, ant]
            events_springs = np.split(attachment, np.arange(len(attachment[1:]))[np.diff(attachment) != 0] + 1)
            events_frames = np.split(np.arange(len(attachment)),
                                     np.arange(len(attachment[1:]))[np.diff(attachment) != 0] + 1)
            precedence = 0
            for event in range(len(events_springs)):
                if events_springs[event][0] != 0 and len(events_springs[event]) > 1:
                    precedence += 1
                    start = events_frames[event][0]
                    end = events_frames[event][-1]
                    sudden_appearance = np.any(np.isnan(ants_attached_labels[start-3:start, events_springs[event][0] - 1])).astype(np.uint8)
                    profiles = np.vstack((profiles, np.array([ant + 1, events_springs[event][0], start, end, precedence, sudden_appearance])))
        profiles = profiles[1:, :]
        for count, query_data in enumerate(
                ["force_magnitude", "force_direction", "N_ants_around_springs", "fixed_end_angle_to_nest",
                 "angular_velocity", "tangential_force"]):
            if query_data == "angular_velocity":
                data = np.load(os.path.join(path, "fixed_end_angle_to_nest.npz"))["arr_0"]
                data = np.nanmedian(utils.calc_angular_velocity(data, diff_spacing=1) / 1, axis=1)
            elif query_data == "N_ants_around_springs":
                data = np.load(os.path.join(path, f"{query_data}.npz"))["arr_0"]
                all_profiles_data2 = np.full((len(profiles), 10000), np.nan)
            else:
                data = np.load(os.path.join(path, f"{query_data}.npz"))["arr_0"]
            all_profiles_data = np.full((len(profiles), 10000), np.nan)
            for profile in range(len(profiles)):
                # ant = int(profiles[profile, 0] - 1)
                spring = int(profiles[profile, 1])
                start = int(profiles[profile, 2])
                end = int(profiles[profile, 3])
                if not end - start + 1 > 10000:
                    if query_data == "angular_velocity":
                        all_profiles_data[profile, 0:end - start + 1] = data[start:end + 1]
                    elif query_data == "N_ants_around_springs":
                        all_profiles_data[profile, 0:end - start + 1] = data[start:end + 1, int(spring - 1)]
                        all_profiles_data2[profile, 0:end - start + 1] = np.nansum(data[start:end + 1], axis=1)
                    else:
                        all_profiles_data[profile, 0:end - start + 1] = data[start:end + 1, int(spring - 1)]
            if query_data == "N_ants_around_springs":
                np.savez_compressed(os.path.join(path, f"all_profiles_{query_data}_sum.npz"),
                                    all_profiles_data2)
            np.savez_compressed(os.path.join(path, f"all_profiles_{query_data}.npz"), all_profiles_data)
        np.savez_compressed(os.path.join(path, "all_profiles_information.npz"), profiles)


def multi_processing(spring_type):
    output_path = f"Z:\\Dor_Gabay\\ThesisProject\\data\\post_processed_data\\{spring_type}\\"
    data_dict = pickle.load(open(os.path.join(output_path, "data_dict.pkl"), "rb"))
    object = PostProcessing(data_dict)
    object.save_data()
    object.test_correlation()

    set_paths = []
    for data_dir in data_dict["data_sets"].keys():
        for data_set in data_dict["data_sets"][data_dir]:
            # AntTracking(([os.path.join(data_dir, set_name, "raw_analysis") for set_name in data_set["set_names"]],
            #              data_dict["output_path"]))
            set_paths.append(([os.path.join(data_dir, set_name, "raw_analysis") for set_name in data_set["set_names"]],
                              data_dict["output_path"]))
    pool = mp.Pool()
    pool.map(AntTracking, set_paths)
    pool.close()
    pool.join()

    profile_ants_behavior(data_dict["output_path"])

if __name__ == "__main__":
    # spring_type = "plus0.3"
    spring_type = "plus0.1"
    multi_processing(spring_type)

    # python data_analysis/data_preparation.py

    # spring_types = ["plus0.5", "plus0.1", "plus0"]
    # pool = mp.Pool()
    # pool.map(multi_processing, spring_types)
    # pool.close()
    # pool.join()

    # spring_type = "plus0.3"
    # output_path = f"Z:\\Dor_Gabay\\ThesisProject\\data\\post_processed_data\\{spring_type}\\"
    # os.makedirs(output_path, exist_ok=True)
    # data_dir1 = "Z:\\Dor_Gabay\\ThesisProject\\data\\analysed_with_tracking\\15.9.22\\plus0.3mm_force\\"
    # calibration_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\calibration\\calibration_{spring_type}\\"
    # calibration_model = pickle.load(open(calibration_dir + "calibration_model.pkl", "rb"))
    # data_dict = {
    #     "spring_type": spring_type,
    #     "calibration_model": calibration_model,
    #     "output_path": output_path,
    #     "two_vars": True,
    #     "frame_resolution": (1920, 1080),
    #     "data_sets":
    #         {data_dir1: [
    #             # {"set_names": [f"S52800{i}" for i in ["01", "02"]], "set_arrangements": [0, 0],
    #             #           "set_slice_info": [None, None]},
    #                      {"set_names": [f"S52800{i}" for i in ["03", "04", "05", "06", "07"]], "set_arrangements": [0, 0, 0, 0, 0],
    #                       "set_slice_info": [None, None, [30000, 68025], [0, 30000], None]},
    #                      {"set_names": [f"S52800{i}" for i in ["08", "09"]],
    #                       "set_arrangements": [19, 0],
    #                       "set_slice_info": [None, None]}
    #                      ],
    #          }
    # }
    # pickle.dump(data_dict, open(os.path.join(data_dict["output_path"], "data_dict.pkl"), "wb"))

    # spring_type = "plus0.5"
    # output_path = f"Z:\\Dor_Gabay\\ThesisProject\\data\\post_processed_data\\{spring_type}\\"
    # os.makedirs(output_path, exist_ok=True)
    # data_dir1 = "Z:\\Dor_Gabay\\ThesisProject\\data\\analysed_with_tracking\\18.9.22\\plus0.5mm_force\\"
    # calibration_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\calibration\\calibration_{spring_type}\\"
    # calibration_model = pickle.load(open(calibration_dir + "calibration_model.pkl", "rb"))
    # data_dict = {
    #     "spring_type": spring_type,
    #     "calibration_model": calibration_model,
    #     "output_path": output_path,
    #     "two_vars": False,
    #     "frame_resolution": (1920, 1080),
    #     "data_sets":
    #         {data_dir1: [{"set_names": [f"S52900{i}" for i in ["01", "02"]], "set_arrangements": [19, 19],
    #                       "set_slice_info": [None, None]}
    #                      ],
    #          }
    # }
    # pickle.dump(data_dict, open(os.path.join(data_dict["output_path"], "data_dict.pkl"), "wb"))


    # spring_type = "plus0.5"
    # output_path = f"Z:\\Dor_Gabay\\ThesisProject\\data\\post_processed_data\\{spring_type}\\"
    # os.makedirs(output_path, exist_ok=True)
    # data_dir1 = "Z:\\Dor_Gabay\\ThesisProject\\data\\analysed_with_tracking\\18.9.22\\plus0.5mm_force\\"
    # calibration_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\calibration\\calibration_{spring_type}\\"
    # calibration_model = pickle.load(open(calibration_dir + "calibration_model.pkl", "rb"))
    # data_dict = {
    #     "spring_type": spring_type,
    #     "calibration_model": calibration_model,
    #     "output_path": output_path,
    #     "two_vars": False,
    #     "frame_resolution": (1920, 1080),
    #     "data_sets":
    #         {data_dir1: [{"set_names": [f"S52900{i}" for i in ["01", "02"]], "set_arrangements": [0, 0],
    #                       "set_slice_info": [None, None]}
    #                      ],
    #          }
    # }
    # pickle.dump(data_dict, open(os.path.join(data_dict["output_path"], "data_dict.pkl"), "wb"))


    # spring_type = "plus0"
    # calibration_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\calibration\\calibration_{spring_type}\\"
    # output_path = f"Z:\\Dor_Gabay\\ThesisProject\\data\\post_processed_data\\{spring_type}\\"
    # data_dir1 = "Z:\\Dor_Gabay\\ThesisProject\\data\\analysed_with_tracking\\10.9.22\\plus0_force\\"
    # data_dir2 = "Z:\\Dor_Gabay\\ThesisProject\\data\\analysed_with_tracking\\12.9.22\\plus0_force\\"
    # calibration_model = pickle.load(open(calibration_dir + "calibration_model.pkl", "rb"))
    # data_dict = {
    #     "spring_type": spring_type,
    #     "calibration_model": calibration_model,
    #     "output_path": output_path,
    #     "two_vars": False,
    #     "frame_resolution": (1920, 1080),
    #     "data_sets":
    #         {data_dir1: [{"set_names": [f"S52000{i}" for i in ["07", "08", "09", "10"]], "set_arrangements": [0, 0, 0, 0],
    #                       "set_slice_info": [None, None, None, None]}],
    #          # data_dir2: [{"set_names": [f"S52200{i}" for i in ["03", "04", "05", "06", "07"]], "set_arrangements": [0, 0, 0, 0],
    #          #              "set_slice_info": [None, None, None, None]}]
    #          }
    # }
    # pickle.dump(data_dict, open(os.path.join(data_dict["output_path"], "data_dict.pkl"), "wb"))


    # spring_type = "plus0.1"
    # calibration_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\calibration\\calibration_{spring_type}\\"
    # output_path = f"Z:\\Dor_Gabay\\ThesisProject\\data\\post_processed_data\\{spring_type}\\"
    # data_dir1 = "Z:\\Dor_Gabay\\ThesisProject\\data\\analysed_with_tracking\\10.9.22\\plus0.1_force\\"
    # data_dir2 = "Z:\\Dor_Gabay\\ThesisProject\\data\\analysed_with_tracking\\11.9.22\\plus0.1mm_force\\"
    # calibration_model = pickle.load(open(calibration_dir + "calibration_model.pkl", "rb"))
    # data_dict = {
    #     "spring_type": spring_type,
    #     "calibration_model": calibration_model,
    #     "output_path": output_path,
    #     "two_vars": False,
    #     "frame_resolution": (1920, 1080),
    #     "data_sets":
    #         {data_dir1: [{"set_names": [f"S52000{i}" for i in ["03", "04", "05"]], "set_arrangements": [0, 0, 0],
    #                       "set_slice_info": [None, None, None]},
    #                      {"set_names": [f"S52000{i}" for i in ["06"]], "set_arrangements": [0], "set_slice_info": [None]}
    #                      ],
    #          data_dir2: [{"set_names": [f"S52100{i}" for i in ["14", "15"]], "set_arrangements": [0, 0],
    #                       "set_slice_info": [None, None]},
    #                      {"set_names": [f"S52100{i}" for i in ["16"]], "set_arrangements": [0],
    #                       "set_slice_info": [None]},
    #                      # {"set_names": [f"S52100{i}" for i in ["17", "18", "19"]], "set_arrangements": [0, 0, 0],
    #                      #  "set_slice_info": [None, None, None]},
    #                      # {"set_names": [f"S52200{i}" for i in ["01", "02", "03", "04", "05"]], "set_arrangements": [0, 0, 0, 0, 0],
    #                      #  "set_slice_info": [None, None, None, None, None]},
    #                      # {"set_names": [f"S52200{i}" for i in ["06", "07", "08"]], "set_arrangements": [0, 0, 0],
    #                      #  "set_slice_info": [None, None, None]}
    #                      ]
    #          }
    # }
    # pickle.dump(data_dict, open(os.path.join(data_dict["output_path"], "data_dict.pkl"), "wb"))


