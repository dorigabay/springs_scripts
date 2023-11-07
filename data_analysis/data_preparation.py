import os

import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob

from scipy.optimize import curve_fit, minimize
# local packages:
from data_analysis import utils


SQUARENESS_THRESHOLD = 0.0005
RECTANGLE_SIMILARITY_THRESHOLD = 0.01
QUALITY_THRESHOLD = SQUARENESS_THRESHOLD * RECTANGLE_SIMILARITY_THRESHOLD


class DataPreparation:
    def __init__(self, data_paths, videos_path, n_springs=20):
        self.data_paths = data_paths
        self.videos_path = videos_path
        self.n_springs = n_springs
        self.calib_mode = True if self.n_springs == 1 else False
        self.load_data()
        self.create_missing_perspective_squares(QUALITY_THRESHOLD)
        self.rearrange_perspective_squares_order()
        self.rearrange_springs_order()
        self.create_video_sets()
        self.n_ants_processing()
        self.correct_perspectives(QUALITY_THRESHOLD)
        self.correct_object_center()
        self.calc_distances()
        self.repeat_values()
        self.calc_angle()

    def load_data(self):
        paths = self.data_paths
        perspective_squares_rectangle_similarity = np.concatenate([np.loadtxt(os.path.join(path, "perspective_squares_rectangle_similarity.csv"), delimiter=",") for path in paths], axis=0)
        perspective_squares_squareness = np.concatenate([np.loadtxt(os.path.join(path, "perspective_squares_squareness.csv"), delimiter=",") for path in paths], axis=0)
        self.perspective_squares_quality = perspective_squares_rectangle_similarity * perspective_squares_squareness
        perspective_squares_coordinates_x = np.concatenate([np.loadtxt(os.path.join(path, "perspective_squares_coordinates_x.csv"), delimiter=",") for path in paths], axis=0)
        perspective_squares_coordinates_y = np.concatenate([np.loadtxt(os.path.join(path, "perspective_squares_coordinates_y.csv"), delimiter=",") for path in paths], axis=0)
        self.perspective_squares_coordinates = np.stack((perspective_squares_coordinates_x, perspective_squares_coordinates_y), axis=2)
        self.ants_attached_labels = np.concatenate([np.loadtxt(os.path.join(path, "ants_attached_labels.csv"), delimiter=",") for path in paths], axis=0)
        self.N_ants_around_springs = np.concatenate([np.loadtxt(os.path.join(path, "N_ants_around_springs.csv"), delimiter=",") for path in paths], axis=0)
        self.size_ants_around_springs = np.concatenate([np.loadtxt(os.path.join(path, "size_ants_around_springs.csv"), delimiter=",") for path in paths], axis=0)
        self.fixed_ends_coordinates = np.concatenate([np.stack((np.loadtxt(os.path.join(path, "fixed_ends_coordinates_x.csv"), delimiter=",").reshape(-1, self.n_springs),
            np.loadtxt(os.path.join(path, "fixed_ends_coordinates_y.csv"), delimiter=",").reshape(-1, self.n_springs)), axis=2) for path in paths], axis=0)
        self.free_ends_coordinates = np.concatenate([np.stack((np.loadtxt(os.path.join(path, "free_ends_coordinates_x.csv"), delimiter=",").reshape(-1, self.n_springs),
            np.loadtxt(os.path.join(path, "free_ends_coordinates_y.csv"), delimiter=",").reshape(-1, self.n_springs)), axis=2) for path in paths], axis=0)
        needle_coordinates = np.concatenate([np.loadtxt(os.path.join(path, "needle_part_coordinates_x.csv"), delimiter=",") for path in paths], axis=0)
        needle_coordinates = np.stack((needle_coordinates, np.concatenate([np.loadtxt(os.path.join(path, "needle_part_coordinates_y.csv"), delimiter=",") for path in paths], axis=0)), axis=2)
        self.object_center_coordinates = needle_coordinates[:, 0, :]
        self.needle_tip_coordinates = needle_coordinates[:, -1, :]
        self.video_n_frames = np.array([np.loadtxt(os.path.join(path, "N_ants_around_springs.csv"), delimiter=",").shape[0] for path in paths])
        cap = cv2.VideoCapture(glob.glob(os.path.join(self.videos_path, "**", "*.MP4"), recursive=True)[0])
        self.video_resolution = np.array((int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        cap.release()
        self.missing_info = np.isnan(self.free_ends_coordinates).any(axis=2)

    def rearrange_springs_order(self, threshold=10):
        self.video_continuity_score = np.zeros(len(self.video_n_frames))
        for count in range(1, len(self.video_n_frames)):
            start, end = np.sum(self.video_n_frames[:count]), np.sum(self.video_n_frames[:count + 1])
            previous_fixed_ends_coordinates = np.nanmedian(self.fixed_ends_coordinates[start - 5:start-1, :], axis=0)
            current_fixed_ends_coordinates = np.nanmedian(self.fixed_ends_coordinates[start:start + 4, :], axis=0)
            num_of_springs = self.fixed_ends_coordinates.shape[1]
            arrangements_distances = np.zeros(num_of_springs)
            for arrangement in range(num_of_springs):
                rearrangement = np.append(np.arange(arrangement, num_of_springs), np.arange(0, arrangement))
                arrangements_distances[arrangement] = np.nanmedian(np.linalg.norm(previous_fixed_ends_coordinates - current_fixed_ends_coordinates[rearrangement], axis=1))
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
                    self.sets_frames[-1].append([0, self.video_n_frames[count] - 1])
                else:
                    frame_count = self.sets_frames[-1][-1][1]+1
                    self.sets_frames[-1].append([frame_count, frame_count + self.video_n_frames[count] - 1])
            else:
                frame_count = self.sets_frames[-1][-1][1]+1
                self.sets_frames.append([[frame_count, frame_count + self.video_n_frames[count] - 1]])
                self.sets_video_paths.append([self.data_paths[count]])

    # def rearrange_springs_order_per_frame(self, threshold=10):
    #     import itertools
    #     springs_order = np.arange(self.n_springs)
    #     for set_idx in self.sets_frames:
    #         s, e = set_idx[0][0], set_idx[-1][1] + 1
    #         for frame in range(s + 1, e):
    #             print(f"\rframe {frame}", end="")
    #             distances = np.linalg.norm(self.fixed_ends_coordinates[frame, :] - self.fixed_ends_coordinates[frame-1, :], axis=1)
    #             mismatched_springs = distances > threshold
    #             if sum(mismatched_springs) >= 2:
    #                 mismatched_springs = springs_order[mismatched_springs+np.isnan(distances)]
    #                 print(f"distances: {distances}")
    #                 permutations = list(itertools.permutations(mismatched_springs))
    #                 for permutation in permutations:
    #                     permutation = list(permutation)
    #                     print(permutation)
    #                     distances = np.linalg.norm(self.fixed_ends_coordinates[frame, permutation] - self.fixed_ends_coordinates[frame-1, mismatched_springs], axis=1)
    #                     if np.sum(distances > threshold) < 2:
    #                         self.N_ants_around_springs[frame, mismatched_springs] = self.N_ants_around_springs[frame, permutation]
    #                         self.size_ants_around_springs[frame, mismatched_springs] = self.size_ants_around_springs[frame, permutation]
    #                         self.fixed_ends_coordinates[frame, mismatched_springs] = self.fixed_ends_coordinates[frame, permutation]
    #                         self.free_ends_coordinates[frame, mismatched_springs] = self.free_ends_coordinates[frame, permutation]

    def create_missing_perspective_squares(self, quality_threshold):
        real_squares = self.perspective_squares_quality < quality_threshold
        self.perspective_squares_coordinates[~real_squares] = np.nan
        reference_coordinates = self.perspective_squares_coordinates[np.all(real_squares, axis=1)][0]
        diff = self.perspective_squares_coordinates - reference_coordinates[np.newaxis, :]
        all_nan_rows = np.isnan(diff).all(axis=2).all(axis=1)
        median_real_squares_diff = np.nanmedian(diff[~all_nan_rows], axis=1)
        predicted_square_coordinates = np.repeat(reference_coordinates[np.newaxis, :], self.perspective_squares_coordinates.shape[0], axis=0)
        predicted_square_coordinates[~all_nan_rows] += median_real_squares_diff[:, np.newaxis, :]
        self.perspective_squares_coordinates[~np.isnan(predicted_square_coordinates)] = predicted_square_coordinates[~np.isnan(predicted_square_coordinates)]
        self.perspective_squares_quality[~np.isnan(predicted_square_coordinates).any(axis=2)] = 0

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

    def correct_perspectives(self, quality_threshold):
        # first perspective correction
        PTMs = utils.create_projective_transform_matrix(self.perspective_squares_coordinates, self.perspective_squares_quality, quality_threshold, self.video_resolution)
        self.perspective_squares_coordinates = utils.apply_projective_transform(self.perspective_squares_coordinates, PTMs)
        self.fixed_ends_coordinates = utils.apply_projective_transform(self.fixed_ends_coordinates, PTMs)
        self.free_ends_coordinates = utils.apply_projective_transform(self.free_ends_coordinates, PTMs)
        self.object_center_coordinates = utils.apply_projective_transform(self.object_center_coordinates, PTMs)
        self.needle_tip_coordinates = utils.apply_projective_transform(self.needle_tip_coordinates, PTMs)
        # second perspective correction
        perspective_correction_params = self.fit_perspective_params(self.fixed_ends_coordinates.copy(), self.needle_tip_coordinates.copy(), self.free_ends_coordinates.copy())
        self.fixed_ends_coordinates = utils.project_plane_perspective(self.fixed_ends_coordinates, perspective_correction_params[[0, 2, 3]])
        needle_tip_repeated = np.copy(np.repeat(self.needle_tip_coordinates[:, np.newaxis, :], self.free_ends_coordinates.shape[1], axis=1))
        self.needle_tip_coordinates = np.nanmedian(utils.project_plane_perspective(needle_tip_repeated, perspective_correction_params[[1, 2, 3]]), axis=1)

    def fit_perspective_params(self, fixed_end_coordinates, needle_tip_coordinates, reference_coordinates):
        def loss_func(params):
            fixed_end_params = params[[0, 2, 3]]
            needle_tip_params = params[[1, 2, 3]]
            correct_fixed_end_coordinates = utils.project_plane_perspective(fixed_end_coordinates, fixed_end_params)
            correct_needle_tip_repeated = utils.project_plane_perspective(needle_tip_repeated, needle_tip_params)
            spring_length = np.linalg.norm(correct_fixed_end_coordinates - reference_coordinates, axis=2)
            spring_length_needle = np.linalg.norm(correct_needle_tip_repeated - reference_coordinates, axis=2)
            length_score = np.mean(np.nanstd(spring_length, axis=0) / np.nanmean(spring_length, axis=0))
            length_score_needle = np.mean(np.nanstd(spring_length_needle, axis=0) / np.nanmean(spring_length_needle, axis=0))
            mean_score = np.mean([length_score, length_score_needle])
            return mean_score
        needle_tip_repeated = np.copy(np.repeat(needle_tip_coordinates[:, np.newaxis, :], reference_coordinates.shape[1], axis=1))
        if np.sum(self.rest_bool) > 0:
            fixed_end_coordinates[~self.rest_bool] = np.nan
            needle_tip_repeated[~self.rest_bool] = np.nan
        x0 = np.array([0.0025, 0.0025, 3840/2, 2160/2])
        res = minimize(loss_func, x0=x0)
        print("params: ", res.x, "loss: ", res.fun)
        return res.x

    def n_ants_processing(self):
        if not self.calib_mode:
            for count, set_idx in enumerate(self.sets_frames):
                start, end = set_idx[0][0], set_idx[-1][1]
                # undetected_springs_for_long_time = utils.filter_continuity(self.missing_info[start:end, :].astype(int), min_size=8)
                interpolation_boolean = utils.filter_continuity(self.missing_info[start:end, :].astype(int), max_size=8)
                self.N_ants_around_springs[start:end, :] = np.round(utils.interpolate_data(self.N_ants_around_springs[start:end, :], interpolation_boolean))
                # self.N_ants_around_springs[start:end, :][np.isnan(self.N_ants_around_springs[start:end, :])] = 0
                self.N_ants_around_springs[start:end, :] = utils.smooth_columns(self.N_ants_around_springs[start:end, :])
                # self.N_ants_around_springs[start:end, :][undetected_springs_for_long_time] = np.nan
                # all_small_attaches = np.zeros(self.N_ants_around_springs[start:end, :].shape)
                for n in reversed(range(1, int(np.nanmax(self.N_ants_around_springs[start:end, :]))+1)):
                    short_attaches = utils.filter_continuity(self.N_ants_around_springs[start:end, :] == n, max_size=25)
                    self.N_ants_around_springs[start:end, :] = np.round(utils.interpolate_data(self.N_ants_around_springs[start:end, :], short_attaches))
                    # all_small_attaches[short_attaches] = 1

                path = "Z:\\Dor_Gabay\\ThesisProject\\data\\2-video_analysis\\summer_2023\\experiment\\plus_0.1_final\\13.8\\S5760003"
                N_ants_around_springs = np.loadtxt(os.path.join(path, "N_ants_around_springs.csv"), delimiter=",")
                interpolation_boolean = utils.filter_continuity(self.missing_info, max_size=8)
                N_ants_around_springs = np.round(utils.interpolate_data(N_ants_around_springs, interpolation_boolean))
                N_ants_around_springs = utils.smooth_columns(N_ants_around_springs)
                for n in reversed(range(1, int(np.nanmax(N_ants_around_springs)) + 1)):
                    short_attaches = utils.filter_continuity(N_ants_around_springs == n, max_size=25)
                    N_ants_around_springs = np.round(utils.interpolate_data(N_ants_around_springs, short_attaches))
            self.rest_bool = self.N_ants_around_springs == 0
        else:
            self.rest_bool = np.full((self.N_ants_around_springs.shape[0], 1), False)
            s, e = self.sets_frames[0][0][0], self.sets_frames[0][-1][1]
            self.rest_bool[s:e+1] = True

    def correct_object_center(self):
        def loss(params):
            distances = np.linalg.norm(points - params, axis=1)
            return np.nanstd(distances)
        if not self.calib_mode:
            object_center_coordinates_approximated = np.full(self.object_center_coordinates.shape, np.nan)
            for count, points in enumerate(self.fixed_ends_coordinates):
                print(f"\r{count}", end="")
                x0 = np.array([np.nanmedian(points[:, 0]), np.nanmedian(points[:, 1])])
                res = minimize(loss, x0=x0)#, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
                # print(f"\r{count}", end="")
                object_center_coordinates_approximated[count, :] = res.x
            self.object_center_coordinates = object_center_coordinates_approximated

    def calc_distances(self):
        self.spring_length = np.linalg.norm(self.free_ends_coordinates - self.fixed_ends_coordinates, axis=2)
        if not self.calib_mode:
            for count, set_idx in enumerate(self.sets_frames):
                s, e = set_idx[0][0], set_idx[-1][1] + 1
                self.spring_length[s:e] /= np.nanmedian(np.where(~self.rest_bool[s:e], np.nan, self.spring_length[s:e]), axis=0)
        else:
            self.spring_length /= np.nanmedian(np.where(~self.rest_bool, np.nan, self.spring_length), axis=0)

    def repeat_values(self):
        self.nest_direction = np.stack((self.fixed_ends_coordinates[:, :, 0], self.fixed_ends_coordinates[:, :, 1] - 500), axis=2)
        self.tip_nest_direction = np.repeat(np.stack((self.needle_tip_coordinates[:, np.newaxis, 0], self.needle_tip_coordinates[:, np.newaxis, 1] - 500), axis=2), self.n_springs, axis=1)
        self.object_center_repeated = np.copy(np.repeat(self.object_center_coordinates[:, np.newaxis, :], self.n_springs, axis=1))
        self.needle_tip_repeated = np.copy(np.repeat(self.needle_tip_coordinates[:, np.newaxis, :], self.n_springs, axis=1))
        self.object_center_repeated = np.repeat(self.object_center_coordinates[:, np.newaxis, :], self.n_springs, axis=1)
        self.object_nest_direction = np.stack((self.object_center_repeated[:, :, 0], self.object_center_repeated[:, :, 1] - 500), axis=2)

    def calc_angle(self):
        self.object_fixed_end_angle_to_nest = utils.calc_angle_matrix(self.object_nest_direction, self.object_center_repeated, self.fixed_ends_coordinates) + np.pi
        self.fixed_end_angle_to_nest = utils.calc_angle_matrix(self.object_center_repeated, self.fixed_ends_coordinates, self.nest_direction) + np.pi
        self.pulling_angle = utils.calc_pulling_angle_matrix(self.free_ends_coordinates, self.object_center_repeated, self.fixed_ends_coordinates)
        if not self.calib_mode:
            for count, set_idx in enumerate(self.sets_frames):
                s, e = set_idx[0][0], set_idx[-1][1] + 1
                self.pulling_angle[s:e] -= np.nanmedian(np.where(~self.rest_bool[s:e], np.nan, self.pulling_angle[s:e]), axis=0)
        else:
            self.pulling_angle -= np.nanmedian(np.where(~self.rest_bool, np.nan, self.pulling_angle), axis=0)
