import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
from scipy.ndimage import label

# import for plots:
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import savgol_filter

# local packages:
from data_analysis import utils
from data_analysis import plots
matplotlib.use('TkAgg')


class Analyser:
    def __init__(self, dir_path, output_path, spring_type):
        self.dir_path = dir_path
        self.output_path = output_path
        self.spring_type = spring_type
        self.fps = 25
        self.paths = [os.path.join(self.dir_path, sub_dir) for sub_dir in os.listdir(self.dir_path) if os.path.isdir(os.path.join(self.dir_path, sub_dir))]
        self.load_data()
        self.calculations()
        # self.test_correlation()
        # self.create_plots()

    def load_data(self):
        self.sets_frames = pickle.load(open(os.path.join(self.dir_path, "sets_frames.pkl"), "rb"))
        self.sets_video_paths = pickle.load(open(os.path.join(self.dir_path, "sets_video_paths.pkl"), "rb"))
        self.missing_info = np.concatenate([np.load(os.path.join(path, "missing_info.npz"))['arr_0'] for path in self.paths], axis=0)
        self.object_center_coordinates = np.concatenate([np.load(os.path.join(path, "object_center_coordinates.npz"))['arr_0'] for path in self.paths], axis=0)
        self.needle_tip_coordinates = np.concatenate([np.load(os.path.join(path, "needle_tip_coordinates.npz"))['arr_0'] for path in self.paths], axis=0)
        self.N_ants_around_springs = np.concatenate([np.load(os.path.join(path, "N_ants_around_springs.npz"))['arr_0'] for path in self.paths], axis=0)
        self.rest_bool = self.N_ants_around_springs == 0
        self.fixed_end_angle_to_nest = np.concatenate([np.load(os.path.join(path, "fixed_end_angle_to_nest.npz"))['arr_0'] for path in self.paths], axis=0)
        self.force_direction = np.concatenate([np.load(os.path.join(path, "force_direction.npz"))['arr_0'] for path in self.paths], axis=0)
        self.force_magnitude = np.concatenate([np.load(os.path.join(path, "force_magnitude.npz"))['arr_0'] for path in self.paths], axis=0)
        self.fixed_ends_coordinates = np.concatenate([np.load(os.path.join(path, "fixed_ends_coordinates.npz"))['arr_0'] for path in self.paths], axis=0)
        self.free_ends_coordinates = np.concatenate([np.load(os.path.join(path, "free_ends_coordinates.npz"))['arr_0'] for path in self.paths], axis=0)
        self.ant_profiles = np.concatenate([np.load(os.path.join(path, "ant_profiles.npz"))['arr_0'] for path in self.paths], axis=0)
        self.profiles_precedence = self.ant_profiles[:, 4]
        self.profiles_ant_labels = self.ant_profiles[:, 0]

    def calculations(self, window_size=10):
        # translational force:
        self.force_direction = utils.interpolate_columns(self.force_direction)
        self.force_magnitude = utils.interpolate_columns(self.force_magnitude)
        horizontal_component = self.force_magnitude * np.cos(self.force_direction + self.fixed_end_angle_to_nest)
        horizontal_component = utils.interpolate_columns(horizontal_component)
        vertical_component = self.force_magnitude * np.sin(self.force_direction + self.fixed_end_angle_to_nest)
        vertical_component = utils.interpolate_columns(vertical_component)
        self.net_force_direction = np.arctan2(np.nansum(vertical_component, axis=1), np.nansum(horizontal_component, axis=1))
        self.net_force_magnitude = np.sqrt(np.nansum(horizontal_component, axis=1) ** 2 + np.nansum(vertical_component, axis=1) ** 2)
        self.net_force_direction = np.array(pd.Series(self.net_force_direction).rolling(window=window_size, center=True).median())
        self.net_force_magnitude = np.array(pd.Series(self.net_force_magnitude).rolling(window=window_size, center=True).median())
        # tangential force:
        self.tangential_force = utils.interpolate_columns(np.sin(self.force_direction) * self.force_magnitude)
        self.net_tangential_force = np.where(np.isnan(self.tangential_force).all(axis=1), np.nan, np.nansum(self.tangential_force, axis=1))
        self.net_tangential_force = np.array(pd.Series(self.net_tangential_force).rolling(window=window_size, center=True).median())
        # angular velocity:
        angular_velocity = utils.calc_angular_velocity(self.fixed_end_angle_to_nest, diff_spacing=4) / 4
        angular_velocity = np.where(np.isnan(angular_velocity).all(axis=1), np.nan, np.nanmedian(angular_velocity, axis=1))
        outlier_threshold = np.nanpercentile(np.abs(angular_velocity), 99)
        angular_velocity[np.abs(angular_velocity) > outlier_threshold] = np.nan
        self.angular_velocity = angular_velocity
        # self.angular_velocity = utils.calc_angular_velocity(self.fixed_end_angle_to_nest, diff_spacing=window_size) / window_size
        # self.angular_velocity = utils.interpolate_data_columns(self.angular_velocity)
        # self.angular_velocity = np.where(np.isnan(self.angular_velocity).all(axis=1), np.nan, np.nanmedian(self.angular_velocity, axis=1))
        # self.angular_velocity = np.array(pd.Series(self.angular_velocity).rolling(window=window_size, center=True).median())
        # self.angular_velocity = np.round(self.angular_velocity, 4) * -1
        # translational velocity:
        self.momentum_direction, self.momentum_magnitude = utils.calc_translation_velocity(self.object_center_coordinates, spacing=window_size)
        self.momentum_direction = np.round(utils.interpolate_columns(self.momentum_direction), 4)
        self.momentum_magnitude = np.round(utils.interpolate_columns(self.momentum_magnitude), 4)
        # ants:
        self.total_n_ants = np.where(np.isnan(self.N_ants_around_springs).all(axis=1), np.nan, np.nansum(self.N_ants_around_springs, axis=1))
        self.discrete_angular_velocity, self.velocity_change = self.discretize_angular_velocity()
        self.calculate_kuramoto()
        # self.profile_ants_behavior()
        # self.ants_profiling_analysis()
        # self.find_profiles_beginnings()
        self.profile_ants_based_on_springs()

    def profile_ants_based_on_springs(self):
        n_ants_dilated = utils.column_dilation(self.N_ants_around_springs)
        self.unique_n_ants = np.unique(n_ants_dilated)[1:-1].astype(np.int64)
        self.N_ants_labeled = np.full(np.append(len(self.unique_n_ants),  self.N_ants_around_springs.shape), np.nan)
        for n_count, n in enumerate(self.unique_n_ants):
            labeled = label(n_ants_dilated == n)[0]
            self.N_ants_labeled[n_count] = labeled[:, list(range(0, labeled.shape[1], 2))]

    def discretize_angular_velocity(self):
        angular_velocity = utils.calc_angular_velocity(self.fixed_end_angle_to_nest, diff_spacing=4) / 4
        angular_velocity = np.where(np.isnan(angular_velocity).all(axis=1), np.nan, np.nanmedian(angular_velocity, axis=1))
        outlier_threshold = np.nanpercentile(np.abs(angular_velocity), 99)
        angular_velocity[np.abs(angular_velocity) > outlier_threshold] = np.nan
        discrete_velocity_vectors = []
        threshold_stop = np.nanpercentile(np.abs(angular_velocity), 50)
        threshold_very_slow = np.nanpercentile(np.abs(angular_velocity), 67.5)
        threshold_slow = np.nanpercentile(np.abs(angular_velocity), 75)
        threshold_medium = np.nanpercentile(np.abs(angular_velocity), 87.5)

        for count, set_idx in enumerate(self.sets_frames):
            # set_idx = self.sets_frames[0]
            s, e = set_idx[0][0], set_idx[-1][1]+1
            set_angular_velocity = angular_velocity[s:e]
            set_angular_velocity = utils.interpolate_columns(set_angular_velocity)
            stop = np.abs(set_angular_velocity) < threshold_stop
            very_slow = np.all([threshold_stop <= np.abs(set_angular_velocity), np.abs(set_angular_velocity) < threshold_very_slow], axis=0)
            slow = np.all([threshold_very_slow <= np.abs(set_angular_velocity), np.abs(set_angular_velocity) < threshold_slow], axis=0)
            medium = np.all([threshold_slow <= np.abs(set_angular_velocity), np.abs(set_angular_velocity) < threshold_medium], axis=0)
            fast = np.abs(set_angular_velocity) >= threshold_medium
            signed_angular_velocity = set_angular_velocity.copy()
            signed_angular_velocity[stop] = 0
            signed_angular_velocity[very_slow] = np.sign(signed_angular_velocity[very_slow]) * 0.25
            signed_angular_velocity[slow] = np.sign(signed_angular_velocity[slow]) * 0.5
            signed_angular_velocity[medium] = np.sign(signed_angular_velocity[medium]) * 0.75
            signed_angular_velocity[fast] = np.sign(signed_angular_velocity[fast]) * 1
            velocity_labels = [1, 0.75, 0.5, 0.25, 0]
            # for i in velocity_labels[:-1]:
            # for j in range(3):
            for i in velocity_labels:
                small_parts = utils.filter_continuity_vector(np.abs(signed_angular_velocity) == i, max_size=5)
                signed_angular_velocity = utils.interpolate_columns(signed_angular_velocity, small_parts)
            small_parts = utils.filter_continuity_vector(~np.isin(np.abs(signed_angular_velocity), [1, 0.75, 0.5, 0.25, 0]), max_size=5)
            signed_angular_velocity = utils.interpolate_columns(signed_angular_velocity, small_parts)
            for velocity_count, velocity in enumerate(velocity_labels[1:]):
                out_of_label = (signed_angular_velocity > velocity) * (signed_angular_velocity < velocity_labels[velocity_count])
                signed_angular_velocity[out_of_label] = velocity
                out_of_label_minus = (signed_angular_velocity < -velocity) * (signed_angular_velocity > -velocity_labels[velocity_count])
                signed_angular_velocity[out_of_label_minus] = -velocity
            for velocity_count, velocity in enumerate(velocity_labels[:-1]):
                small_parts_plus = utils.filter_continuity_vector(signed_angular_velocity == velocity, max_size=5)
                signed_angular_velocity[small_parts_plus] = velocity_labels[velocity_count+1]
                small_parts_minus = utils.filter_continuity_vector(signed_angular_velocity == -velocity, max_size=5)
                signed_angular_velocity[small_parts_minus] = -velocity_labels[velocity_count+1]
            # import matplotlib.pyplot as plt
            # change = np.abs(np.diff(signed_angular_velocity)) > 0.5
            # if count == 4:
            #     plt.clf()
            #     plt.plot(np.arange(len(set_angular_velocity)), signed_angular_velocity)
            #     x_change = np.arange(len(set_angular_velocity))[1:][change]
            #     y_change = signed_angular_velocity[1:][change]
            #     plt.scatter(x_change, y_change, c='r')
            #     plt.show()
            discrete_velocity_vectors.append(signed_angular_velocity)
        discrete_velocity = np.concatenate(discrete_velocity_vectors)
        change = np.abs(np.diff(discrete_velocity)) > 0.5
        change = np.concatenate([[False], change])
        return discrete_velocity * -1, change

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
        # self.alignment_percentage = np.full(self.force_direction.shape[0], np.nan)
        self.kuramoto_score = np.full(self.force_direction.shape[0], np.nan, dtype=np.float64)
        force_direction_max = np.nanpercentile(np.abs(self.force_direction), 99)
        for row in range(self.force_direction.shape[0]):
            row_direction = self.force_direction[row, :][self.N_ants_around_springs[row] > 0].copy()
            row_direction = (row_direction / force_direction_max) * np.pi
            if len(row_direction) > 1:
                self.kuramoto_score[row] = utils.kuramoto_order_parameter(row_direction)[0]
                # signs_sum = [np.sum(np.sign(row_direction) == 1), np.sum(np.sign(row_direction) == -1)]
                # self.alignment_percentage[row] = signs_sum[np.argmax(signs_sum)] / np.sum(signs_sum)

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

    def profile_ants_behavior(self):
        self.profiled_check = self.profiler_check()
        self.longest_profile = np.max(self.ant_profiles[:, 3] - self.ant_profiles[:, 2]).astype(int)
        self.longest_profile = 12000 if self.longest_profile > 12000 else self.longest_profile
        self.profiled_N_ants_around_springs = self.profiler(self.N_ants_around_springs)
        self.profiled_N_ants_around_springs[self.profiled_N_ants_around_springs == 0] = 1
        self.profiled_N_ants_around_springs_sum = self.profiler(self.total_n_ants)
        self.profiled_fixed_end_angle_to_nest = self.profiler(self.fixed_end_angle_to_nest)
        self.profiled_force_direction = self.profiler(self.force_direction)
        self.profiled_force_magnitude = self.profiler(self.force_magnitude)
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

    # def calc_ant_replacement_rate(self):
    #     n_changes = np.nansum(np.abs(np.diff(self.N_ants_around_springs, axis=0)), axis=1)
    #     sum_of_changes = np.diff(np.nansum(self.N_ants_around_springs, axis=1))
    #     cancelling_number = (n_changes - np.abs(sum_of_changes))/2
    #     added_ants = np.copy(cancelling_number)
    #     added_ants[sum_of_changes > 0] += np.abs(sum_of_changes)[sum_of_changes > 0]
    #     removed_ants = np.copy(cancelling_number)
    #     removed_ants[sum_of_changes < 0] += np.abs(sum_of_changes)[sum_of_changes < 0]
    #     self.n_replacments_per_frame = n_changes

    def ants_profiling(self, beginning_length=250):
        self.attaching_single_ant_profiles = np.full((len(self.profiles_precedence), self.longest_profile), False)
        self.detaching_single_ant_profiles = np.full((len(self.profiles_precedence), self.longest_profile), False)
        self.profiles_beginnings = np.full(self.force_magnitude.shape, False)
        arranged = np.arange(self.longest_profile)
        reversed_profiled_N_ants_around_springs = self.profiled_N_ants_around_springs[self.reverse_argsort]
        for profile in range(len(self.profiles_precedence)):
            spring = int(self.ant_profiles[profile, 1])-1
            start = int(self.ant_profiles[profile, 2])
            end = int(self.ant_profiles[profile, 3])
            if not np.any(self.profiled_check[profile, :3]):
                first_n_ants_change = arranged[:-1][np.diff(self.profiled_N_ants_around_springs[profile, :]) != 0][0]
                self.attaching_single_ant_profiles[profile, 0:first_n_ants_change+1] = True
            if not np.any(self.profiled_check[profile, -2:]) and reversed_profiled_N_ants_around_springs[profile, -1] == 1:
                last_n_ants_change = arranged[1:][np.diff(reversed_profiled_N_ants_around_springs[profile, :]) != 0][-1]
                self.detaching_single_ant_profiles[profile, last_n_ants_change:] = True
            if end - start > beginning_length:
                self.profiles_beginnings[start:start + beginning_length, spring] = True
            else:
                self.profiles_beginnings[start:end, spring] = True
        self.middle_events = utils.filter_continuity((self.profiled_N_ants_around_springs == 1).transpose(), min_size=200).transpose()


    def test_correlation(self, sets_idx=(0, -1)):
        first_set_idx, last_set_idx = (sets_idx, sets_idx) if isinstance(sets_idx, int) else sets_idx
        start, end = self.sets_frames[first_set_idx][0][0], self.sets_frames[last_set_idx][-1][1]
        corr_df = pd.DataFrame({"net_tangential_force": self.net_tangential_force[start:end], "angular_velocity": self.angular_velocity[start:end],
                                "net_momentum": self.momentum_magnitude[start:end],# * np.sin(self.momentum_direction[start:end]),
                                "net_force": self.net_force_magnitude[start:end] #* np.sin(self.net_force_direction[start:end]),
                                })
        self.corr_df = corr_df.dropna()
        angular_velocity_correlation_score = corr_df.corr()["net_tangential_force"]["angular_velocity"]
        translation_correlation_score = corr_df.corr()["net_momentum"]["net_force"]
        print(f"correlation score between net tangential force and angular velocity: {angular_velocity_correlation_score}")
        print(f"correlation score between net force and net momentum: {translation_correlation_score}")
        # s, e = self.sets_frames[first_set_idx][0][0], self.sets_frames[first_set_idx][-1][1]  # start and end of the first video
        # plots.plot_correlation(self, start=s, end=e, output_path=os.path.join(self.output_path, "correlation"))

    def create_plots(self):
        # plots.plot_ant_profiles(self, output_dir=os.path.join(self.output_path, "profiles"), window_size=11, profile_size=200)
        # plots.draw_single_profiles(self, os.path.join(self.output_path, "single_profiles_S5760003"), profile_min_length=200,
        #                            start=self.sets_frames[0][0][0], end=self.sets_frames[0][0][1])
        plots.draw_single_profiles(self, output_path=os.path.join(self.output_path, "single_profiles_S5760003"), profile_min_length=200, start=self.sets_frames[0][0][0],
                             end=self.sets_frames[0][0][1])
        plots.plot_alignment(self, os.path.join(self.output_path, "alignment"), profile_size=200)

    def save_analysis_data(self, spring_type):
        output_path = os.path.join("Z:\\Dor_Gabay\\ThesisProject\\results\\", "analysis_data", spring_type)
        os.makedirs(output_path, exist_ok=True)
        np.savez_compressed(os.path.join(output_path, "angular_velocity.npz"), self.angular_velocity)
        np.savez_compressed(os.path.join(output_path, "tangential_force.npz"), self.tangential_force)
        np.savez_compressed(os.path.join(output_path, "N_ants_around_springs.npz"), self.N_ants_around_springs)
        np.savez_compressed(os.path.join(output_path, "missing_info.npz"), self.missing_info)


if __name__ == "__main__":
    spring_type = "plus_0.1"
    data_analysis_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\3-data_analysis\\summer_2023\\experiment\\{spring_type}\\"
    # data_analysis_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\3-data_analysis\\summer_2023\\experiment\\{spring_type}_final_final\\"
    output_dir = f"Z:\\Dor_Gabay\\ThesisProject\\results\\summer_2023\\{spring_type}\\"
    # output_dir = f"Z:\\Dor_Gabay\\ThesisProject\\results\\summer_2023\\{spring_type}_final_final\\"
    self = Analyser(data_analysis_dir, output_dir, spring_type)

