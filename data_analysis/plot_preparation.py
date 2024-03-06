import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
from scipy.ndimage import label
import apytl
from scipy.stats import pearsonr
# local packages:
sys.path.append(os.path.join(os.getcwd(), "data_analysis"))
import utils
# from plots import plots

matplotlib.use('TkAgg')


class AnalysisIntegration:
    def __init__(self, data_path, output_path, springs_types):
        self.data_path = data_path
        self.output_path = output_path
        self.springs_types = springs_types
        self.mm_in_pixels = 420/3840
        self.fps = 25
        self.data = {}
        self.springs_thickness = []
        for spring_type in self.springs_types:
            self.springs_thickness.append(str(float(spring_type.split("_")[1]) + 0.3) + "mm" if spring_type != "stiff" else "stiff")
            analyze = False if spring_type in ["plus_0.5", "stiff"] else True
            self.data[spring_type] = Analyser(os.path.join(data_path, spring_type), os.path.join(output_path, spring_type), analyze)


class Analyser:
    def __init__(self, dir_path, output_path, analyze=True):
        self.dir_path = dir_path
        self.output_path = output_path
        self.analyze = analyze
        # os.makedirs(self.output_path, exist_ok=True)
        # self.spring_type = spring_type
        # self.spring_thickness = str(float(spring_type.split("_")[1])+0.3)+"mm" if spring_type != "stiff" else "stiff"
        self.fps = 25
        self.paths = [os.path.join(self.dir_path, sub_dir) for sub_dir in os.listdir(self.dir_path) if os.path.isdir(os.path.join(self.dir_path, sub_dir))]
        self.load_data()
        if self.analyze:
            self.basic_calculations()
            self.extra_calculations()
            self.test_correlation()
            self.save_analysis_data()

    def load_data(self):
        self.sets_frames = pickle.load(open(os.path.join(self.dir_path, "sets_frames.pkl"), "rb"))
        self.sets_video_paths = pickle.load(open(os.path.join(self.dir_path, "sets_video_paths.pkl"), "rb"))
        self.missing_info = np.concatenate([np.load(os.path.join(path, "missing_info.npz"))['arr_0'] for path in self.paths], axis=0)
        self.object_center_coordinates = np.concatenate([np.load(os.path.join(path, "object_center_coordinates.npz"))['arr_0'] for path in self.paths], axis=0)
        self.object_center_coordinates[self.object_center_coordinates > np.nanpercentile(self.object_center_coordinates, 99.9)] = np.nan
        self.object_center_coordinates[self.object_center_coordinates < np.nanpercentile(self.object_center_coordinates, 0.1)] = np.nan
        self.needle_tip_coordinates = np.concatenate([np.load(os.path.join(path, "needle_tip_coordinates.npz"))['arr_0'] for path in self.paths], axis=0)
        self.N_ants_around_springs = np.concatenate([np.load(os.path.join(path, "N_ants_around_springs.npz"))['arr_0'] for path in self.paths], axis=0)
        self.rest_bool = self.N_ants_around_springs == 0
        self.fixed_end_angle_to_nest = np.concatenate([np.load(os.path.join(path, "fixed_end_angle_to_nest.npz"))['arr_0'] for path in self.paths], axis=0)
        if self.analyze:
            self.fixed_end_angle_to_wall = np.concatenate([np.load(os.path.join(path, "fixed_end_angle_to_wall.npz"))['arr_0'] for path in self.paths], axis=0)
            self.force_direction = np.concatenate([np.load(os.path.join(path, "force_direction.npz"))['arr_0'] for path in self.paths], axis=0)
            self.force_magnitude = np.concatenate([np.load(os.path.join(path, "force_magnitude.npz"))['arr_0'] for path in self.paths], axis=0)
            self.fixed_ends_coordinates = np.concatenate([np.load(os.path.join(path, "fixed_ends_coordinates.npz"))['arr_0'] for path in self.paths], axis=0)
            self.free_ends_coordinates = np.concatenate([np.load(os.path.join(path, "free_ends_coordinates.npz"))['arr_0'] for path in self.paths], axis=0)
            self.n_springs = self.fixed_ends_coordinates.shape[1]

        # keep_start = self.sets_frames[1][0][0]
        # keep_end = self.sets_frames[1][-1][-1]
        # self.missing_info[:keep_start, :] = True
        # self.missing_info[keep_end:, :] = True
        # self.object_center_coordinates[:keep_start, :] = np.nan
        # self.object_center_coordinates[keep_end:, :] = np.nan
        # self.needle_tip_coordinates[:keep_start, :] = np.nan
        # self.needle_tip_coordinates[keep_end:, :] = np.nan
        # self.N_ants_around_springs[:keep_start, :] = np.nan
        # self.N_ants_around_springs[keep_end:, :] = np.nan
        # self.fixed_end_angle_to_nest[:keep_start, :] = np.nan
        # self.fixed_end_angle_to_nest[keep_end:, :] = np.nan
        # self.fixed_end_angle_to_wall[:keep_start, :] = np.nan
        # self.fixed_end_angle_to_wall[keep_end:, :] = np.nan
        # self.force_direction[:keep_start, :] = np.nan
        # self.force_direction[keep_end:, :] = np.nan
        # self.force_magnitude[:keep_start, :] = np.nan
        # self.force_magnitude[keep_end:, :] = np.nan
        # self.fixed_ends_coordinates[:keep_start, :, :] = np.nan
        # self.fixed_ends_coordinates[keep_end:, :, :] = np.nan
        # self.free_ends_coordinates[:keep_start, :, :] = np.nan
        # self.free_ends_coordinates[keep_end:, :, :] = np.nan

        # self.ant_profiles = np.concatenate([np.load(os.path.join(path, "ant_profiles.npz"))['arr_0'] for path in self.paths], axis=0)
        # self.profiles_precedence = self.ant_profiles[:, 4]
        # self.profiles_ant_labels = self.ant_profiles[:, 0]

    def basic_calculations(self):
        self.force_direction = utils.interpolate_columns(self.force_direction)
        self.force_magnitude = utils.interpolate_columns(self.force_magnitude)
        remove_info = self.force_magnitude < 0
        # self.force_magnitude[self.force_magnitude > np.nanpercentile(self.force_magnitude, 99.9)] = np.nan
        self.force_magnitude -= np.mean(self.force_magnitude[self.N_ants_around_springs == 0], axis=0)
        # self.force_direction[remove_info] = np.nan
        self.negative_magnitude = self.force_magnitude < -np.percentile(np.abs(self.force_magnitude), 20)
        self.tangential_force = np.sin(self.force_direction) * self.force_magnitude
        self.angular_velocity = utils.calc_angular_velocity(self.fixed_end_angle_to_wall, window=4) * -1
        self.angular_velocity26 = utils.calc_angular_velocity(self.fixed_end_angle_to_wall, window=26) * -1

    def extra_calculations(self, window_size=10):
        horizontal_component = self.force_magnitude * np.cos(self.force_direction + self.fixed_end_angle_to_nest)
        horizontal_component = utils.interpolate_columns(horizontal_component)
        vertical_component = self.force_magnitude * np.sin(self.force_direction + self.fixed_end_angle_to_nest)
        vertical_component = utils.interpolate_columns(vertical_component)
        self.net_force_direction = np.arctan2(np.nansum(vertical_component, axis=1), np.nansum(horizontal_component, axis=1))
        self.net_force_magnitude = np.sqrt(np.nansum(horizontal_component, axis=1) ** 2 + np.nansum(vertical_component, axis=1) ** 2)
        self.net_force_direction = np.array(pd.Series(self.net_force_direction).rolling(window=window_size, center=True).median())
        self.net_force_magnitude = np.array(pd.Series(self.net_force_magnitude).rolling(window=window_size, center=True).median())
        all_nans = np.isnan(self.tangential_force).all(axis=1)
        self.net_tangential_force = np.nansum(self.tangential_force, axis=1)
        self.net_tangential_force[all_nans] = np.nan
        # self.net_tangential_force = np.array(pd.Series(self.net_tangential_force).rolling(window=window_size, center=True).median())
        self.momentum_direction, self.momentum_magnitude = utils.calc_translation_velocity(self.object_center_coordinates, window=window_size)
        self.momentum_direction = np.round(utils.interpolate_columns(self.momentum_direction), 4)
        self.momentum_magnitude = np.round(utils.interpolate_columns(self.momentum_magnitude), 4)
        self.total_n_ants = np.where(np.isnan(self.N_ants_around_springs).all(axis=1), np.nan, np.nansum(self.N_ants_around_springs, axis=1))
        self.discrete_angular_velocity, self.velocity_change = utils.discretize_angular_velocity(self.angular_velocity, self.sets_frames)
        self.calculate_synchronization()
        self.n_ants_labeling()
        # self.profile_ants_based_on_tracking()
        # self.n_ants_profiler()

    def calculate_auto_correlation(self, max_lag=75):
        single_ant_labeled = self.N_ants_labeled[0]
        unique_labels, label_counts = np.unique(single_ant_labeled.flatten(), return_counts=True)
        unique_labels = unique_labels[np.logical_and(label_counts > 150, label_counts < 1000)]
        print("\rFound {} number of profile, which fits the conditions.".format(len(unique_labels)))
        self.auto_correlation = np.full((len(unique_labels), max_lag), np.nan)
        data = self.force_magnitude / np.mean(self.force_magnitude[single_ant_labeled != 0])
        for profile_count, profile_label in enumerate(unique_labels):
            apytl.Bar().drawbar(profile_count, len(unique_labels), fill='*')
            profile_data = data[single_ant_labeled == profile_label]
            for window in range(1, max_lag + 1):
                autocorr, _ = pearsonr(profile_data[:-window], profile_data[window:])
                self.auto_correlation[profile_count, window - 1] = autocorr

    def n_ants_labeling(self):
        n_ants_dilated = utils.column_dilation(self.N_ants_around_springs)
        self.unique_n_ants = np.unique(n_ants_dilated)[1:-1].astype(np.int64)
        self.N_ants_labeled = np.full(np.append(len(self.unique_n_ants),  self.N_ants_around_springs.shape), np.nan)
        for n_count, n in enumerate(self.unique_n_ants):
            labeled = label(n_ants_dilated == n)[0]
            self.N_ants_labeled[n_count] = labeled[:, list(range(0, labeled.shape[1], 2))]

    def n_ants_profiler(self):
        angular_velocity_6 = utils.calc_angular_velocity(self.fixed_end_angle_to_wall, window=6)
        discrete_angular_velocity_6, _ = utils.discretize_angular_velocity(angular_velocity_6, self.sets_frames)
        angular_velocity_6 = np.repeat(np.expand_dims(angular_velocity_6, axis=1), self.n_springs, axis=1)
        discrete_angular_velocity_6 = np.repeat(np.expand_dims(discrete_angular_velocity_6, axis=1), self.n_springs, axis=1)
        angular_velocity_26 = utils.calc_angular_velocity(self.fixed_end_angle_to_wall, window=26)
        discrete_angular_velocity_26, _ = utils.discretize_angular_velocity(angular_velocity_26, self.sets_frames)
        angular_velocity_26 = np.repeat(np.expand_dims(angular_velocity_26, axis=1), self.n_springs, axis=1)
        discrete_angular_velocity_26 = np.repeat(np.expand_dims(discrete_angular_velocity_26, axis=1), self.n_springs, axis=1)
        unique_n_ants = self.unique_n_ants
        labels_sum = int(np.sum([np.max(self.N_ants_labeled[n]) for n in range(len(unique_n_ants))]))
        _, counts = np.unique(self.N_ants_labeled[0].flatten(), return_counts=True)
        max_length = int(np.max(counts[1:]))
        self.profiles_frames_idx = np.full((labels_sum, max_length, len(unique_n_ants)), np.nan)
        self.profiles_magnitude = np.full((labels_sum, max_length, len(unique_n_ants)), np.nan)
        self.profiles_directions = np.full((labels_sum, max_length, len(unique_n_ants)), np.nan)
        self.profiles_ants_sum = np.full((labels_sum, max_length, len(unique_n_ants)), np.nan)
        self.profiles_velocity_6 = np.full((labels_sum, max_length, len(unique_n_ants)), np.nan)
        self.profiles_velocity_26 = np.full((labels_sum, max_length, len(unique_n_ants)), np.nan)
        self.profiles_discrete_velocity_6 = np.full((labels_sum, max_length, len(unique_n_ants)), np.nan)
        self.profiles_discrete_velocity_26 = np.full((labels_sum, max_length, len(unique_n_ants)), np.nan)
        self.profiles_attachments = np.full(self.N_ants_labeled[0].shape, False)
        self.profiles_detachments = np.full(self.N_ants_labeled[0].shape, False)
        iteration_count = 0
        for n_count, n in enumerate(unique_n_ants):
            n_ants_labeled = self.N_ants_labeled[n_count]
            profile_labels, labels_counts = np.unique(n_ants_labeled.flatten(), return_counts=True)
            if n == 1:
                self.single_ant_profile_labels = profile_labels[1:]
                self.single_ant_profile_counts = labels_counts[1:]
            for profile_count, profile_label in enumerate(profile_labels[1:]):
                apytl.Bar().drawbar(iteration_count, labels_sum, fill='*')
                iteration_count += 1
                frames, spring = np.where(n_ants_labeled == profile_label)
                if n == 1:
                    idx = np.where(n_ants_labeled == profile_label)
                    last_frame = n_ants_labeled.shape[0] - 1
                    if idx[0][0] != 0 \
                            and self.N_ants_around_springs[idx[0][0] - 1, idx[1][0]] == 0 \
                            and not self.missing_info[idx[0][0] - 1, idx[1][0]]:
                        self.profiles_attachments[idx[0][0], idx[1][0]] = True
                    if idx[0][-1] != last_frame \
                            and self.N_ants_around_springs[idx[0][-1] + 1, idx[1][-1]] == 0 \
                            and not self.missing_info[idx[0][-1] + 1, idx[1][-1]]:
                        self.profiles_detachments[idx[0][-1], idx[1][-1]] = True
                if len(frames) > 0:
                    self.profiles_frames_idx[profile_count, :len(frames), n_count] = frames
                    self.profiles_magnitude[profile_count, :len(frames), n_count] = self.force_magnitude[frames, spring]
                    self.profiles_directions[profile_count, :len(frames), n_count] = self.force_direction[frames, spring]
                    self.profiles_ants_sum[profile_count, :len(frames), n_count] = np.sum(self.N_ants_around_springs[frames], axis=1)
                    self.profiles_velocity_6[profile_count, :len(frames), n_count] = angular_velocity_6[frames, spring]
                    self.profiles_velocity_26[profile_count, :len(frames), n_count] = angular_velocity_26[frames, spring]
                    self.profiles_discrete_velocity_6[profile_count, :len(frames), n_count] = discrete_angular_velocity_6[frames, spring]
                    self.profiles_discrete_velocity_26[profile_count, :len(frames), n_count] = discrete_angular_velocity_26[frames, spring]

    def ants_attachments_detachments(self):
        single_ant_labeled = self.N_ants_labeled[0]
        self.single_ant_profile_labels = np.unique(single_ant_labeled)[1:]
        self.single_ant_profile_labels, self.single_ant_profile_counts = np.unique(single_ant_labeled.flatten(), return_counts=True)
        self.single_ant_profile_labels = self.single_ant_profile_labels[1:]
        self.single_ant_profile_counts = self.single_ant_profile_counts[1:]
        self.profiles_attachments = np.full(single_ant_labeled.shape, False)
        self.profiles_detachments = np.full(single_ant_labeled.shape, False)
        last_frame = single_ant_labeled.shape[0] - 1
        for profile_count, profile_label in enumerate(self.single_ant_profile_labels):
            apytl.Bar().drawbar(profile_count, len(self.single_ant_profile_labels), fill='*')
            idx = np.where(single_ant_labeled == profile_label)
            if idx[0][0] != 0 \
                    and self.N_ants_around_springs[idx[0][0]-1, idx[1][0]] == 0 \
                    and not self.missing_info[idx[0][0]-1, idx[1][0]]:
                self.profiles_attachments[idx[0][0], idx[1][0]] = True
            if idx[0][-1] != last_frame \
                    and self.N_ants_around_springs[idx[0][-1]+1, idx[1][-1]] == 0 \
                    and not self.missing_info[idx[0][-1]+1, idx[1][-1]]:
                self.profiles_detachments[idx[0][-1], idx[1][-1]] = True

    def profiler(self, data):
        profiled_data = np.full((self.ant_profiles.shape[0], self.longest_profile), np.nan)
        for profile in range(len(self.ant_profiles)):
            spring = int(self.ant_profiles[profile, 1])
            start = int(self.ant_profiles[profile, 2])
            end = int(self.ant_profiles[profile, 3])
            if not end - start + 1 > self.longest_profile:
                if len(data.shape) == 1:
                    profiled_data[profile, 0:end - start + 1] = data[start:end + 1]
                elif len(data.shape) == 2:
                    profiled_data[profile, 0:end - start + 1] = data[start:end + 1, int(spring - 1)]
        return profiled_data

    def calculate_synchronization(self):
        self.kuramoto_score = np.full(self.force_direction.shape[0], np.nan, dtype=np.float64)
        force_direction_max = np.nanpercentile(np.abs(self.force_direction), 99)
        for row in range(self.force_direction.shape[0]):
            row_direction = self.force_direction[row, :][self.N_ants_around_springs[row] > 0].copy()
            row_direction = (row_direction / force_direction_max) * np.pi
            if len(row_direction) > 1:
                self.kuramoto_score[row] = utils.kuramoto_order_parameter(row_direction)[0]

    def profiler_check(self):
        profiled_check = np.full((self.ant_profiles.shape[0], 5), False)
        for profile in range(len(self.ant_profiles)):
            spring = int(self.ant_profiles[profile, 1])
            start = int(self.ant_profiles[profile, 2])
            end = int(self.ant_profiles[profile, 3])
            sudden_appearance = np.any(self.missing_info[start-3:start, spring-1])
            not_single_ant = not self.N_ants_around_springs[start, spring-1] == 1
            ants_before = np.any(self.N_ants_around_springs[start-3:start, spring-1] != 0)
            sudden_disappearance = np.any(self.missing_info[end+1:end+4, spring-1])
            ants_after = np.any(self.N_ants_around_springs[end+1:end+4, spring-1] != 0)
            profiled_check[profile, 0] = sudden_appearance
            profiled_check[profile, 1] = not_single_ant
            profiled_check[profile, 2] = ants_before
            profiled_check[profile, 3] = sudden_disappearance
            profiled_check[profile, 4] = ants_after
        return profiled_check
    #
    def profile_ants_based_on_tracking(self):
        self.profiled_check = self.profiler_check()
        self.longest_profile = np.max(self.ant_profiles[:, 3] - self.ant_profiles[:, 2]).astype(int)
        self.longest_profile = 12000 if self.longest_profile > 12000 else self.longest_profile
        self.profiled_N_ants_around_springs = self.profiler(self.N_ants_around_springs)
        self.profiled_N_ants_around_springs[self.profiled_N_ants_around_springs == 0] = 1
        self.profiled_N_ants_around_springs_sum = self.profiler(self.total_n_ants)
        self.profiled_fixed_end_angle_to_nest = self.profiler(self.fixed_end_angle_to_nest)
        self.profiled_force_direction = self.profiler(self.force_direction)
        self.profiled_force_magnitude = self.profiler(self.force_magnitude)
        self.profiled_negative_magnitude = self.profiled_force_magnitude < 0
        self.profiled_angular_velocity = self.profiler(self.angular_velocity)
        self.profiled_tangential_force = self.profiler(self.tangential_force)
        self.profiled_net_tangential_force = self.profiler(self.net_tangential_force)
        self.profiled_discrete_angular_velocity = self.profiler(self.discrete_angular_velocity)
        reverse_arg_sort_columns = np.full(self.profiled_N_ants_around_springs.shape, 0)
        nans = np.isnan(self.profiled_N_ants_around_springs)
        nans_sum = np.sum(nans, axis=1)
        for row in range(self.profiled_N_ants_around_springs.shape[0]):
            reverse_arg_sort_columns[row, nans_sum[row]:] = np.where(~nans[row])[0]
            reverse_arg_sort_columns[row, :nans_sum[row]] = np.where(nans[row])[0]
        reverse_arg_sort_rows = np.repeat(np.expand_dims(np.arange(self.profiled_N_ants_around_springs.shape[0]), axis=1), self.profiled_N_ants_around_springs.shape[1], axis=1)
        self.reverse_argsort = (reverse_arg_sort_rows, reverse_arg_sort_columns)

    def test_correlation(self):
        df = pd.DataFrame({"net_tangential_force": self.net_tangential_force, "angular_velocity": self.angular_velocity})
        corr_angular_velocity = df.dropna().corr()["net_tangential_force"]["angular_velocity"]
        print(f"Pearson of net tangential force and angular velocity: {np.round(corr_angular_velocity, 3)}")
        idx = np.where(np.abs(self.discrete_angular_velocity) >= 0.5)
        df = pd.DataFrame({"net_tangential_force": self.net_tangential_force[idx], "angular_velocity": self.angular_velocity[idx]})
        corr_angular_velocity_at_movement = df.dropna().corr()["net_tangential_force"]["angular_velocity"]
        print(f"Pearson of net tangential force and angular velocity at movement: {np.round(corr_angular_velocity_at_movement, 3)}")
        df = pd.DataFrame({"net_momentum": self.momentum_magnitude, "net_force": self.net_force_magnitude})
        corr_translational_velocity = df.dropna().corr()["net_momentum"]["net_force"]
        print(f"correlation score between net force and net momentum: {corr_translational_velocity}")
        idx = np.where(self.momentum_magnitude >= 1)
        df = pd.DataFrame({"net_momentum": self.momentum_magnitude[idx], "net_force": self.net_force_magnitude[idx]})
        corr_trans_velocity_at_movement = df.dropna().corr()["net_momentum"]["net_force"]
        print(f"correlation score between net force and net momentum at movement: {corr_trans_velocity_at_movement}")
        output_path = os.path.join(self.output_path, "Figure 2")
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "correlation.txt"), "w") as f:
            f.write(f"Pearson of net tangential force and angular velocity: {np.round(corr_angular_velocity, 3)}\n")
            f.write(f"Pearson of net tangential force and angular velocity at movement: {np.round(corr_angular_velocity_at_movement, 3)}\n")
            f.write(f"correlation score between net force and net momentum: {corr_translational_velocity}\n")
            f.write(f"correlation score between net force and net momentum at no angular movement: {corr_trans_velocity_at_movement}\n")

    def save_analysis_data(self):
        output_path = os.path.join(self.output_path, "results_data")
        os.makedirs(output_path, exist_ok=True)
        np.savez_compressed(os.path.join(output_path, "angular_velocity.npz"), self.angular_velocity)
        np.savez_compressed(os.path.join(output_path, "angular_velocity26.npz"), self.angular_velocity26)
        np.savez_compressed(os.path.join(output_path, "tangential_force.npz"), self.tangential_force)
        np.savez_compressed(os.path.join(output_path, "N_ants_around_springs.npz"), self.N_ants_around_springs)
        np.savez_compressed(os.path.join(output_path, "missing_info.npz"), self.missing_info)


if __name__ == "__main__":
    spring_types = ["plus_0.0", "plus_0.1", "plus_0.2", "plus_0.3", "plus_0.5", "stiff"]
    data_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\3-data_analysis\\summer_2023\\experiment\\"
    output_dir = f"Z:\\Dor_Gabay\\ThesisProject\\results\\summer_2023\\"
    self = AnalysisIntegration(data_dir, output_dir, spring_types)

