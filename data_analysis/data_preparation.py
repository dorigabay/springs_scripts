import glob
import cv2
import os
import copy
import pickle
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import multiprocessing
from itertools import repeat
# local imports:
from data_analysis import utils
from data_analysis.ant_tracking import AntTracking


class PostProcessing:
    def __init__(self, video_dir, data_paths, output_path):
        self.video_dir = video_dir
        self.output_path = output_path
        self.data_paths = data_paths
        self.load_data()
        self.rearrange_springs_order()
        self.create_video_sets()
        self.rearrange_perspective_squares_order()
        self.correct_perspectives()
        self.n_ants_processing()
        self.repeat_values()
        self.calc_angle()
        self.create_bias_correction_models()
        self.calc_spring_length()
        self.calc_pulling_angle()
        self.calc_force()
        # self.save_data()
        self.test_correlation()
        # self.process_ant_tracking_data(restart=False)

    def load_data(self):
        print("loading data...")
        # height = np.concatenate([np.loadtxt(os.path.join(path, "perspective_squares_sizes_height.csv"), delimiter=",") for path in self.data_paths], axis=0)
        # width = np.concatenate([np.loadtxt(os.path.join(path, "perspective_squares_sizes_width.csv"), delimiter=",") for path in self.data_paths], axis=0)
        # self.perspective_squares_areas = height * width
        perspective_squares_coordinates_x = np.concatenate([np.loadtxt(os.path.join(path, "perspective_squares_coordinates_x.csv"), delimiter=",") for path in self.data_paths], axis=0)
        perspective_squares_coordinates_y = np.concatenate([np.loadtxt(os.path.join(path, "perspective_squares_coordinates_y.csv"), delimiter=",") for path in self.data_paths], axis=0)
        self.perspective_squares_coordinates = np.stack((perspective_squares_coordinates_x, perspective_squares_coordinates_y), axis=2)
        self.ants_attached_labels = np.concatenate([np.loadtxt(os.path.join(path, "ants_attached_labels.csv"), delimiter=",") for path in self.data_paths], axis=0)
        self.N_ants_around_springs = np.concatenate([np.loadtxt(os.path.join(path, "N_ants_around_springs.csv"), delimiter=",") for path in self.data_paths], axis=0)
        self.size_ants_around_springs = np.concatenate([np.loadtxt(os.path.join(path, "size_ants_around_springs.csv"), delimiter=",") for path in self.data_paths], axis=0)
        self.fixed_ends_coordinates = np.concatenate([np.stack((np.loadtxt(os.path.join(path, "fixed_ends_coordinates_x.csv"), delimiter=","),
            np.loadtxt(os.path.join(path, "fixed_ends_coordinates_y.csv"), delimiter=",")), axis=2) for  path in self.data_paths], axis=0)
        self.free_ends_coordinates = np.concatenate([np.stack((np.loadtxt(os.path.join(path, "free_ends_coordinates_x.csv"), delimiter=","),
            np.loadtxt(os.path.join(path, "free_ends_coordinates_y.csv"), delimiter=",")), axis=2) for path in self.data_paths], axis=0)
        needle_coordinates = np.concatenate([np.loadtxt(os.path.join(path, "needle_part_coordinates_x.csv"), delimiter=",") for path in self.data_paths], axis=0)
        needle_coordinates = np.stack((needle_coordinates, np.concatenate([np.loadtxt(os.path.join(path, "needle_part_coordinates_y.csv"), delimiter=",") for path in self.data_paths], axis=0)), axis=2)
        self.object_center_coordinates = needle_coordinates[:, 0, :]
        self.needle_tip_coordinates = needle_coordinates[:, -1, :]
        self.num_of_frames_per_video = np.array([np.loadtxt(os.path.join(path, "N_ants_around_springs.csv"), delimiter=",").shape[0] for path in self.data_paths])
        cap = cv2.VideoCapture(glob.glob(os.path.join(self.video_dir, "*.MP4"))[0])
        self.video_resolution = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        self.calibration_model = pickle.load(open(calibration_model_path, "rb"))

    def rearrange_springs_order(self, threshold=10):
        self.video_continuity_score = np.zeros(len(self.num_of_frames_per_video))
        for count in range(1, len(self.num_of_frames_per_video)):
            start, end = np.sum(self.num_of_frames_per_video[:count]), np.sum(self.num_of_frames_per_video[:count + 1])
            previous_free_ends_coordinates = np.nanmedian(self.fixed_ends_coordinates[start - 5:start-1, :], axis=0)
            current_free_ends_coordinates = np.nanmedian(self.fixed_ends_coordinates[start:start + 4, :], axis=0)
            num_of_springs = self.fixed_ends_coordinates.shape[1]
            arrangements_distances = np.zeros(num_of_springs)
            for arrangement in range(num_of_springs):
                rearrangement = np.append(np.arange(arrangement, num_of_springs), np.arange(0, arrangement))
                arrangements_distances[arrangement] = np.nanmedian(np.linalg.norm(previous_free_ends_coordinates - current_free_ends_coordinates[rearrangement], axis=1))
            best_arrangement = np.argmin(arrangements_distances)
            if arrangements_distances[best_arrangement] < threshold: # higher than threshold means that this movie isn't a continuation of the previous one
                rearrangement = np.append(np.arange(best_arrangement, num_of_springs), np.arange(0, best_arrangement))
                self.N_ants_around_springs[start:end, :] = self.N_ants_around_springs[start:end, rearrangement]
                self.size_ants_around_springs[start:end, :] = self.size_ants_around_springs[start:end, rearrangement]
                self.fixed_ends_coordinates[start:end, :, :] = self.fixed_ends_coordinates[start:end, rearrangement, :]
                self.free_ends_coordinates[start:end, :, :] = self.free_ends_coordinates[start:end, rearrangement, :]
            self.video_continuity_score[count] = arrangements_distances[best_arrangement]

    def create_video_sets(self):
        self.sets_frames = [[]]
        self.sets_video_paths = [[]]
        for count, score in enumerate(self.video_continuity_score):
            if score < 10:
                self.sets_video_paths[-1].append(self.data_paths[count])
                if self.sets_frames[-1] == []:
                    self.sets_frames[-1].append([0, self.num_of_frames_per_video[count]-1])
                else:
                    frame_count = self.sets_frames[-1][-1][1]+1
                    self.sets_frames[-1].append([frame_count, frame_count+self.num_of_frames_per_video[count]-1])
            else:
                frame_count = self.sets_frames[-1][-1][1]+1
                self.sets_frames.append([[frame_count, frame_count+self.num_of_frames_per_video[count]-1]])
                self.sets_video_paths.append([self.data_paths[count]])

    # def create_missing_perspective_squares(self, weight):
    #     median_area = np.median(self.perspective_squares_areas, axis=1)
    #     perspective_squares_quality = np.abs(median_area/np.nanmedian(self.perspective_squares_areas) - 1)
    #     real_squares = (perspective_squares_quality < 0.5)
    #     # if weight == 0:
    #     try:
    #         hyper_real_squares = (self.perspective_squares_squareness < 0.01)
    #         self.reference_coordinates = self.perspective_squares_coordinates[hyper_real_squares.all(axis=1)][0]
    #     except:
    #         self.reference_coordinates = self.reference_coordinates
    #     # reference_coordinates = self.perspective_squares_coordinates[real_squares.all(axis=1)][0]
    #         # print("Reference coordinates: ", self.reference_coordinates)
    #     missing_square_frames = np.any(~real_squares, axis=1) * ~np.all(~real_squares, axis=1)
    #     diff = self.perspective_squares_coordinates - self.reference_coordinates[np.newaxis, :]
    #     real_squares = np.repeat(real_squares[:, :, np.newaxis], 2, axis=2)
    #     diff[~real_squares] = np.nan
    #     median_real_squares_diff = np.nanmedian(diff, axis=1)
    #     predicted_square_coordinates = self.perspective_squares_coordinates + median_real_squares_diff[:, np.newaxis, :]
    #     predicted_square_coordinates[~missing_square_frames] = np.nan
    #     predicted_square_coordinates[real_squares] = np.nan
    #     predicted_square_coordinates[np.isnan(self.perspective_squares_coordinates)] = np.nan
    #     self.perspective_squares_coordinates[~np.isnan(predicted_square_coordinates)] = predicted_square_coordinates[~np.isnan(predicted_square_coordinates)]

    def rearrange_perspective_squares_order(self):
        x_assort = np.argsort(self.perspective_squares_coordinates[:, :, 0], axis=1)
        y_assort = np.argsort(self.perspective_squares_coordinates[:, :, 1], axis=1)
        for count, (frame_x_assort, frame_y_assort) in enumerate(zip(x_assort, y_assort)):
            if not np.any(np.isnan(self.perspective_squares_coordinates[count])):
                top_left_column = set(frame_x_assort[:2]).intersection(set(frame_y_assort[:2])).pop()
                top_right_column = set(frame_x_assort[2:]).intersection(set(frame_y_assort[:2])).pop()
                bottom_right_column = set(frame_x_assort[2:]).intersection(set(frame_y_assort[2:])).pop()
                bottom_left_column = set(frame_x_assort[:2]).intersection(set(frame_y_assort[2:])).pop()
                rearrangement = np.array([top_left_column, bottom_left_column, top_right_column, bottom_right_column])
                self.perspective_squares_coordinates[count] = self.perspective_squares_coordinates[count, rearrangement, :]

    def correct_perspectives(self):
        # median_area = np.median(self.perspective_squares_areas, axis=1)
        # perspective_squares_quality = np.abs(median_area/np.nanmedian(self.perspective_squares_areas) - 1)
        # PTMs = utils.create_projective_transform_matrix(self.perspective_squares_coordinates, perspective_squares_quality, 0.05, src_dimensions=self.video_resolution)
        PTMs = utils.create_projective_transform_matrix(self.perspective_squares_coordinates, src_dimensions=self.video_resolution)
        self.fixed_ends_coordinates = utils.apply_projective_transform(self.fixed_ends_coordinates, PTMs)
        self.free_ends_coordinates = utils.apply_projective_transform(self.free_ends_coordinates, PTMs)
        self.object_center_coordinates = utils.apply_projective_transform(self.object_center_coordinates, PTMs)
        self.needle_tip_coordinates = utils.apply_projective_transform(self.needle_tip_coordinates, PTMs)
        self.perspective_squares_coordinates = utils.apply_projective_transform(self.perspective_squares_coordinates, PTMs)

    def n_ants_processing(self):
        for count, set_idx in enumerate(self.sets_frames):
            start, end = set_idx[0][0], set_idx[-1][1]
            undetected_springs_for_long_time = utils.filter_continuity(utils.convert_bool_to_binary(np.isnan(self.fixed_ends_coordinates[start:end, :, 0])), min_size=8)
            self.N_ants_around_springs[start:end, :] = np.round(
                utils.interpolate_data(self.N_ants_around_springs[start:end, :], utils.find_cells_to_interpolate(self.N_ants_around_springs[start:end, :])))
            self.N_ants_around_springs[start:end, :][
                np.isnan(self.N_ants_around_springs[start:end, :])] = 0
            self.N_ants_around_springs[start:end, :] = self.smoothing_n_ants(
                self.N_ants_around_springs[start:end, :])
            self.N_ants_around_springs[start:end, :][undetected_springs_for_long_time] = np.nan
            all_small_attaches = np.zeros(self.N_ants_around_springs[start:end, :].shape)
            for n in np.unique(self.N_ants_around_springs[start:end, :])[1:]:
                if not np.isnan(n):
                    short_attaches = utils.filter_continuity(self.N_ants_around_springs[start:end, :] == n, max_size=30)
                    all_small_attaches[short_attaches] = 1
            self.N_ants_around_springs[start:end, :] = \
                np.round(utils.interpolate_data(self.N_ants_around_springs[start:end, :], all_small_attaches.astype(bool)))
        self.rest_bool = self.N_ants_around_springs == 0

    def smoothing_n_ants(self, array):
        for col in range(array.shape[1]):
            array[:, col] = np.abs(np.round(savgol_filter(array[:, col], 31, 2)))
        return array

    # def smoothing_fixed_ends(self, array):
    #     for col in range(array.shape[1]):
    #         x_smoothed = pd.DataFrame(array[:, col, 0]).rolling(20, center=True, min_periods=1).mean().values[:, 0]
    #         y_smoothed = pd.DataFrame(array[:, col, 1]).rolling(20, center=True, min_periods=1).mean().values[:, 0]
    #         array[:, col] = np.vstack((x_smoothed, y_smoothed)).T
    #     return array

    def repeat_values(self):
        nest_direction = np.stack((self.fixed_ends_coordinates[:, 0, 0], self.fixed_ends_coordinates[:, 0, 1]-500), axis=1)
        self.nest_direction_repeated = np.repeat(nest_direction[:, np.newaxis, :], self.fixed_ends_coordinates.shape[1], axis=1)
        object_nest_direction = np.stack((self.object_center_coordinates[:, 0], self.object_center_coordinates[:, 1] - 500), axis=1)
        self.object_nest_direction_repeated = np.repeat(object_nest_direction[:, np.newaxis, :], self.fixed_ends_coordinates.shape[1], axis=1)
        self.object_center_repeated = copy.copy(np.repeat(self.object_center_coordinates[:, np.newaxis, :], self.free_ends_coordinates.shape[1], axis=1))

    # def calc_distances(self):
    #     num_of_springs = self.fixed_ends_coordinates.shape[1]
    #     object_center = np.repeat(self.object_center_coordinates[:, np.newaxis, :], num_of_springs, axis=1)
    #     needle_tip_coordinates = np.repeat(self.needle_tip_coordinates[:, np.newaxis, :], num_of_springs, axis=1)
    #     self.needle_length = np.nanmedian(np.linalg.norm(self.needle_tip_coordinates - self.object_center_coordinates, axis=1))
    #     self.object_center_to_free_end_distance = np.linalg.norm(self.free_ends_coordinates - object_center, axis=2)
    #     self.object_center_to_fixed_end_distance = np.linalg.norm(self.fixed_ends_coordinates - object_center, axis=2)
    #     self.object_needle_tip_to_fixed_end_distance = np.linalg.norm(self.fixed_ends_coordinates - needle_tip_coordinates, axis=2)

    def calc_angle(self):
        self.object_fixed_end_angle_to_nest = utils.calc_angle_matrix(self.object_nest_direction_repeated, self.object_center_repeated, self.fixed_ends_coordinates,)+ np.pi
        self.fixed_end_angle_to_nest = utils.calc_angle_matrix(self.object_center_repeated, self.fixed_ends_coordinates, self.nest_direction_repeated) + np.pi

    def create_bias_correction_models(self):
        y_length = np.linalg.norm(self.free_ends_coordinates - self.fixed_ends_coordinates, axis=2)
        y_angle = utils.calc_pulling_angle_matrix(self.fixed_ends_coordinates, self.object_center_repeated, self.free_ends_coordinates)
        angles_to_nest = np.expand_dims(self.fixed_end_angle_to_nest, axis=2)
        X = np.concatenate((np.sin(angles_to_nest), np.cos(angles_to_nest)), axis=2)
        idx = self.rest_bool
        self.models_lengths, self.models_angles = [], []
        not_nan_idx = ~(np.isnan(y_angle) + np.isnan(X).any(axis=2)) * idx
        for col in range(y_length.shape[1]):
            X_fit = X[not_nan_idx[:, col], col]
            y_length_fit = y_length[not_nan_idx[:, col], col]
            y_angle_fit = y_angle[not_nan_idx[:, col], col]
            model_length = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
            model_angle = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
            self.models_lengths.append(model_length.fit(X_fit, y_length_fit))
            self.models_angles.append(model_angle.fit(X_fit, y_angle_fit))

    def calc_pulling_angle(self):
        self.pulling_angle = utils.calc_pulling_angle_matrix(self.fixed_ends_coordinates, self.object_center_repeated, self.free_ends_coordinates)
        # pulling_angle_prediction = utils.norm_values(self.pulling_angle, self.fixed_end_angle_to_nest, self.models_angles)
        # self.pulling_angle -= pulling_angle_prediction
        for count, set_idx in enumerate(self.sets_frames):
            start, end = set_idx[0][0], set_idx[-1][1]
            rest_pull_angle = np.copy(self.pulling_angle[start:end])
            rest_pull_angle[~self.rest_bool[start:end]] = np.nan
            self.pulling_angle[start:end] -= np.nanmedian(rest_pull_angle, axis=0)

    def calc_spring_length(self):
        self.spring_length = np.linalg.norm(self.free_ends_coordinates - self.fixed_ends_coordinates, axis=2)
        # spring_length_prediction = utils.norm_values(self.spring_length, self.fixed_end_angle_to_nest, self.models_lengths)
        # smallest_zero_for_np_float64 = np.finfo(np.float64).tiny
        # self.spring_length[spring_length_prediction != 0] /= spring_length_prediction[spring_length_prediction != 0]
        # self.spring_length[spring_length_prediction == 0] /= smallest_zero_for_np_float64

    def calc_force(self):
        self.force_direction = np.full(self.pulling_angle.shape, np.nan, dtype=np.float64)
        self.force_magnitude = np.full(self.pulling_angle.shape, np.nan, dtype=np.float64)
        for s in range(self.pulling_angle.shape[1]):
            X = np.stack((self.pulling_angle[:, s], self.spring_length[:, s]), axis=1)
            exclude_idx = np.isnan(X).any(axis=1) + np.isinf(X).any(axis=1)
            X = X[~exclude_idx]
            forces_predicted = self.calibration_model.predict(X)
            self.force_direction[~exclude_idx, s] = forces_predicted[:, 0]
            self.force_magnitude[~exclude_idx, s] = forces_predicted[:, 1]

    def save_data(self):
        for count, set_paths in enumerate(self.sets_video_paths):
            sub_dirs_names = [os.path.basename(os.path.normpath(path)) for path in set_paths]
            set_save_path = os.path.join(self.output_path, f"{sub_dirs_names[0]}-{sub_dirs_names[-1]}")
            os.makedirs(set_save_path, exist_ok=True)
            print("-" * 60)
            print("saving data to:", set_save_path)
            start, end = self.sets_frames[count][0][0], self.sets_frames[count][-1][1]
            np.savez_compressed(os.path.join(set_save_path, "needle_tip_coordinates.npz"), self.needle_tip_coordinates[start:end])
            np.savez_compressed(os.path.join(set_save_path, "object_center_coordinates.npz"), self.object_center_coordinates[start:end])
            np.savez_compressed(os.path.join(set_save_path, "fixed_ends_coordinates.npz"), self.fixed_ends_coordinates[start:end])
            np.savez_compressed(os.path.join(set_save_path, "free_ends_coordinates.npz"), self.free_ends_coordinates[start:end])
            np.savez_compressed(os.path.join(set_save_path, "perspective_squares_coordinates.npz"), self.perspective_squares_coordinates[start:end])
            np.savez_compressed(os.path.join(set_save_path, "N_ants_around_springs.npz"), self.N_ants_around_springs[start:end])
            np.savez_compressed(os.path.join(set_save_path, "size_ants_around_springs.npz"), self.size_ants_around_springs[start:end])
            np.savez_compressed(os.path.join(set_save_path, "ants_attached_labels.npz"), self.ants_attached_labels[start:end])
            np.savez_compressed(os.path.join(set_save_path, "fixed_end_angle_to_nest.npz"), self.fixed_end_angle_to_nest[start:end])
            np.savez_compressed(os.path.join(set_save_path, "object_fixed_end_angle_to_nest.npz"), self.object_fixed_end_angle_to_nest[start:end])
            np.savez_compressed(os.path.join(set_save_path, "force_direction.npz"), self.force_direction[start:end])
            np.savez_compressed(os.path.join(set_save_path, "force_magnitude.npz"), self.force_magnitude[start:end])
            np.savez_compressed(os.path.join(set_save_path, "missing_info.npz"), np.isnan(self.free_ends_coordinates[start:end, :, 0]))
        pickle.dump(self.sets_frames, open(os.path.join(self.output_path, "sets_frames.pkl"), "wb"))
        pickle.dump(self.sets_video_paths, open(os.path.join(self.output_path, "sets_video_paths.pkl"), "wb"))
        pickle.dump(self.num_of_frames_per_video, open(os.path.join(self.output_path, "num_of_frames_per_video.pkl"), "wb"))

    def test_correlation(self, sets_idx=(0, -1)):
        first_set_idx, last_set_idx = (sets_idx, sets_idx) if isinstance(sets_idx, int) else sets_idx
        start, end = self.sets_frames[first_set_idx][0][0], self.sets_frames[last_set_idx][-1][1]
        self.calculations()
        corr_df = pd.DataFrame({"net_tangential_force": self.net_tangential_force[start:end], "angular_velocity": self.angular_velocity[start:end],
                                "movement_magnitude": self.movement_magnitude[start:end], "movement_direction": self.movement_direction[start:end],
                                "net_force_magnitude": self.net_force_magnitude[start:end], "net_force_direction": self.net_force_direction[start:end]
                                })
        self.corr_df = corr_df.dropna()
        angular_velocity_correlation_score = corr_df.corr()["net_tangential_force"]["angular_velocity"]
        translation_direction_correlation_score = corr_df.corr()["net_force_direction"]["movement_direction"]
        translation_magnitude_correlation_score = corr_df.corr()["net_force_magnitude"]["movement_magnitude"]
        print(f"correlation score between net tangential force and angular velocity: {angular_velocity_correlation_score}")
        print(f"correlation score between net force direction and translation direction: {translation_direction_correlation_score}")
        print(f"correlation score between net force magnitude and translation magnitude: {translation_magnitude_correlation_score}")

    def calculations(self):
        # self.force_magnitude[~np.isnan(self.force_magnitude)*self.rest_bool] -= np.nanmean(self.force_magnitude[~np.isnan(self.force_magnitude)*self.rest_bool])
        # self.force_direction[~np.isnan(self.force_direction)*self.rest_bool] -= np.nanmean(self.force_direction[~np.isnan(self.force_direction)*self.rest_bool])
        self.calc_net_force()
        self.angular_velocity = utils.calc_angular_velocity(self.fixed_end_angle_to_nest, diff_spacing=20)/ 20
        self.angular_velocity = np.where(np.isnan(self.angular_velocity).all(axis=1), np.nan, np.nanmedian(self.angular_velocity, axis=1))
        self.movement_direction, self.movement_magnitude = utils.calc_translation_velocity(self.object_center_coordinates, spacing=40)
        self.net_force_direction = np.array(pd.Series(self.net_force_direction).rolling(window=40, center=True).median())
        self.net_force_magnitude = np.array(pd.Series(self.net_force_magnitude).rolling(window=40, center=True).median())
        self.net_tangential_force = np.array(pd.Series(self.net_tangential_force).rolling(window=5, center=True).median())

    def calc_net_force(self):
        horizontal_component = self.force_magnitude * np.cos(self.force_direction + self.fixed_end_angle_to_nest)
        vertical_component = self.force_magnitude * np.sin(self.force_direction + self.fixed_end_angle_to_nest)
        self.net_force_direction = np.arctan2(np.nansum(vertical_component, axis=1), np.nansum(horizontal_component, axis=1))
        self.net_force_magnitude = np.sqrt(np.nansum(horizontal_component, axis=1) ** 2 + np.nansum(vertical_component, axis=1) ** 2)
        self.tangential_force = np.sin(self.force_direction) * self.force_magnitude
        # self.net_tangential_force = np.nansum(self.tangential_force, axis=1)
        self.net_tangential_force = np.where(np.isnan(self.tangential_force).all(axis=1), np.nan, np.nansum(self.tangential_force, axis=1))

    def process_ant_tracking_data(self, restart=False):
        pool = multiprocessing.Pool()
        pool.starmap(AntTracking, zip(self.sets_video_paths, repeat(self.output_path), repeat(self.video_resolution), repeat(restart)))
        pool.close()
        pool.join()
        self.profile_ants_behavior()

    def profile_ants_behavior(self):
        sub_dirs_paths = [os.path.join(self.output_path, sub_dir) for sub_dir in os.listdir(self.output_path) if os.path.isdir(os.path.join(self.output_path, sub_dir))]
        for count, path in enumerate(sub_dirs_paths):
            print("\nProfiling ants for: ", path)
            ants_assigned_to_springs = np.load(os.path.join(path, "ants_assigned_to_springs_fixed.npz"))["arr_0"][:, :-1].astype(np.uint8)
            ants_attached_labels = np.load(os.path.join(path, "ants_attached_labels.npz"))["arr_0"]
            profiles = np.full(6, np.nan)  # ant, spring, start, end, precedence, sudden_appearance
            for ant in range(ants_assigned_to_springs.shape[1]):
                print("\r Ant number: ", ant, end="")
                attachment = ants_assigned_to_springs[:, ant]
                events_springs = np.split(attachment, np.arange(len(attachment[1:]))[np.diff(attachment) != 0] + 1)
                events_frames = np.split(np.arange(len(attachment)),
                                         np.arange(len(attachment[1:]))[np.diff(attachment) != 0] + 1)
                precedence = 0
                for event in range(len(events_springs)):
                    if events_springs[event][0] != 0 and len(events_springs[event]) > 1:
                        precedence += 1
                        start, end = events_frames[event][0], events_frames[event][-1]
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


if __name__ == "__main__":
    spring_type = "plus_0.1"
    date = "13.8"
    video_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\1-videos\\summer_2023\\{date}\\{spring_type}\\"
    data_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\2-video_analysis\\summer_2023\\{date}\\{spring_type}\\"
    data_paths = [os.path.join(data_dir, sub_dir) for sub_dir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, sub_dir))]
    # data_paths = data_paths[:10] + data_paths[11:13] + data_paths[14:]
    output_path = f"Z:\\Dor_Gabay\\ThesisProject\\data\\3-data_analysis\\summer_2023\\{spring_type}\\"
    calibration_model_path = os.path.join("Z:\\Dor_Gabay\\ThesisProject\\data\\2-video_analysis\\summer_2023\\calibration\\", spring_type, "calibration_model.pkl")
    self = PostProcessing(video_dir, data_paths, output_path)

