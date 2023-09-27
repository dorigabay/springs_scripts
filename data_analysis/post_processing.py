import os
import numpy as np
import utils
import cv2
import glob
from scipy.signal import savgol_filter


SQUARENESS_THRESHOLD = 0.0005
RECTANGLE_SIMILARITY_THRESHOLD = 0.01
QUALITY_THRESHOLD = SQUARENESS_THRESHOLD * RECTANGLE_SIMILARITY_THRESHOLD


class PostProcessing:
    def __init__(self, data_paths, videos_dir, n_springs=1):
        self.n_springs = n_springs
        self.data_paths = data_paths
        self.load_data(self.data_paths, videos_dir)
        self.create_missing_perspective_squares(QUALITY_THRESHOLD)
        self.rearrange_perspective_squares_order()
        self.correct_perspectives(QUALITY_THRESHOLD)
        self.calc_distances()
        self.repeat_values()
        self.calc_angle()
        if n_springs > 1:
            self.rearrange_springs_order()
            self.create_video_sets()
            self.n_ants_processing()

    def load_data(self, paths, video_dir):
        paths = [paths] if isinstance(paths, str) else paths
        perspective_squares_rectangle_similarity = np.concatenate([np.loadtxt(os.path.join(path, "perspective_squares_rectangle_similarity.csv"), delimiter=",") for path in paths], axis=0)
        perspective_squares_squareness = np.concatenate([np.loadtxt(os.path.join(path, "perspective_squares_squareness.csv"), delimiter=",") for path in paths], axis=0)
        self.perspective_squares_quality = perspective_squares_rectangle_similarity * perspective_squares_squareness
        perspective_squares_coordinates_x = np.concatenate([np.loadtxt(os.path.join(path, "perspective_squares_coordinates_x.csv"), delimiter=",") for path in paths], axis=0)
        perspective_squares_coordinates_y = np.concatenate([np.loadtxt(os.path.join(path, "perspective_squares_coordinates_y.csv"), delimiter=",") for path in paths], axis=0)
        self.perspective_squares_coordinates = np.stack((perspective_squares_coordinates_x, perspective_squares_coordinates_y), axis=2)
        self.ants_attached_labels = np.concatenate([np.loadtxt(os.path.join(path, "ants_attached_labels.csv"), delimiter=",") for path in paths], axis=0)
        self.N_ants_around_springs = np.concatenate([np.loadtxt(os.path.join(path, "N_ants_around_springs.csv"), delimiter=",") for path in paths], axis=0)
        self.size_ants_around_springs = np.concatenate([np.loadtxt(os.path.join(path, "size_ants_around_springs.csv"), delimiter=",") for path in paths], axis=0)
        # print(np.loadtxt(os.path.join(paths[0], "fixed_ends_coordinates_x.csv"), delimiter=",").reshape(self.n_springs, -1).shape)
        self.fixed_ends_coordinates = np.concatenate([np.stack((np.loadtxt(os.path.join(path, "fixed_ends_coordinates_x.csv"), delimiter=",").reshape(-1, self.n_springs),
            np.loadtxt(os.path.join(path, "fixed_ends_coordinates_y.csv"), delimiter=",").reshape(-1, self.n_springs)), axis=2) for  path in paths], axis=0)
        self.free_ends_coordinates = np.concatenate([np.stack((np.loadtxt(os.path.join(path, "free_ends_coordinates_x.csv"), delimiter=",").reshape(-1, self.n_springs),
            np.loadtxt(os.path.join(path, "free_ends_coordinates_y.csv"), delimiter=",").reshape(-1, self.n_springs)), axis=2) for path in paths], axis=0)
        needle_coordinates = np.concatenate([np.loadtxt(os.path.join(path, "needle_part_coordinates_x.csv"), delimiter=",") for path in paths], axis=0)
        needle_coordinates = np.stack((needle_coordinates, np.concatenate([np.loadtxt(os.path.join(path, "needle_part_coordinates_y.csv"), delimiter=",") for path in paths], axis=0)), axis=2)
        self.object_center_coordinates = needle_coordinates[:, 0, :]
        self.needle_tip_coordinates = needle_coordinates[:, -1, :]
        self.num_of_frames_per_video = np.array([np.loadtxt(os.path.join(path, "N_ants_around_springs.csv"), delimiter=",").shape[0] for path in paths])
        cap = cv2.VideoCapture(glob.glob(os.path.join(video_dir, "**", "*.MP4"), recursive=True)[0])
        self.video_resolution = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

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

    def correct_perspectives(self, quality_threshold):
        PTMs = utils.create_projective_transform_matrix(self.perspective_squares_coordinates, self.perspective_squares_quality, quality_threshold, self.video_resolution)
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

    def calc_distances(self):
        object_center = np.repeat(self.object_center_coordinates[:, np.newaxis, :], self.n_springs, axis=1)
        needle_tip_coordinates = np.repeat(self.needle_tip_coordinates[:, np.newaxis, :], self.n_springs, axis=1)
        # self.needle_length = np.nanmedian(np.linalg.norm(self.needle_tip_coordinates - self.object_center_coordinates, axis=1))
        self.object_center_to_free_end_distance = np.linalg.norm(self.free_ends_coordinates - object_center, axis=2)
        self.object_center_to_fixed_end_distance = np.linalg.norm(self.fixed_ends_coordinates - object_center, axis=2)
        self.object_needle_tip_to_fixed_end_distance = np.linalg.norm(self.fixed_ends_coordinates - needle_tip_coordinates, axis=2)

    def repeat_values(self):
        nest_direction = np.stack((self.fixed_ends_coordinates[:, 0, 0], self.fixed_ends_coordinates[:, 0, 1]-500), axis=1)
        self.nest_direction_repeated = np.repeat(nest_direction[:, np.newaxis, :], self.fixed_ends_coordinates.shape[1], axis=1)
        object_nest_direction = np.stack((self.object_center_coordinates[:, 0], self.object_center_coordinates[:, 1] - 500), axis=1)
        self.object_nest_direction_repeated = np.repeat(object_nest_direction[:, np.newaxis, :], self.fixed_ends_coordinates.shape[1], axis=1)
        self.object_center_repeated = np.copy(np.repeat(self.object_center_coordinates[:, np.newaxis, :], self.free_ends_coordinates.shape[1], axis=1))

    def calc_angle(self):
        self.object_fixed_end_angle_to_nest = utils.calc_angle_matrix(self.object_nest_direction_repeated, self.object_center_repeated, self.fixed_ends_coordinates, ) + np.pi
        self.fixed_end_angle_to_nest = utils.calc_angle_matrix(self.object_center_repeated, self.fixed_ends_coordinates, self.nest_direction_repeated) + np.pi

