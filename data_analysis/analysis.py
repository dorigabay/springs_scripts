import os
import pickle
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit, minimize
matplotlib.use('TkAgg')  # or 'Qt5Agg' or other interactive backend
import matplotlib.pyplot as plt
# local packages:
from data_analysis import utils
# from data_analysis import plots


# class DataAnalyserAllSprings:
#     def __init__(self, sub_dirs_paths):
#         self.sub_dirs_paths = sub_dirs_paths
#         self.spring_types = [os.path.basename(path).split("_")[0] for path in self.sub_dirs_paths]
#         self.main_dir_path = os.path.split(self.sub_dirs_paths[0])[0]
#         self.load_data()
#         self.calculations()
#
#     def load_data(self):
#         print("Loading data...")
#         self.angular_velocity = [np.load(os.path.join(path, "angular_velocity.npz"))['arr_0'] for path in self.sub_dirs_paths]
#         self.tangential_force = [np.load(os.path.join(path, "tangential_force.npz"))['arr_0'] for path in self.sub_dirs_paths]
#         self.N_ants_around_springs = [np.load(os.path.join(path, "N_ants_around_springs.npz"))['arr_0'] for path in self.sub_dirs_paths]
#         self.missing_info = [np.load(os.path.join(path, "missing_info.npz"))['arr_0'] for path in self.sub_dirs_paths]
#         print("Data loaded.")
#
#     def calculations(self):
#         self.net_tangential_force = [np.copy(self.tangential_force[i]) for i in range(len(self.tangential_force))]
#         for i in range(len(self.net_tangential_force)):
#             self.net_tangential_force[i][self.missing_info[i]] = np.nan
#         self.net_tangential_force = [np.nansum(self.net_tangential_force[i], axis=1) for i in range(len(self.net_tangential_force))]
#         self.sum_N_ants = [np.copy(self.N_ants_around_springs[i]) for i in range(len(self.N_ants_around_springs))]
#         for i in range(len(self.sum_N_ants)):
#             self.sum_N_ants[i][self.missing_info[i]] = np.nan
#         self.sum_N_ants = [np.nansum(self.sum_N_ants[i], axis=1) for i in range(len(self.sum_N_ants))]
#
#         self.data = pd.DataFrame([
#             np.concatenate(self.angular_velocity, axis=0),
#             np.concatenate(self.net_tangential_force, axis=0),
#             np.concatenate(self.sum_N_ants, axis=0),
#             np.repeat(self.spring_types, [len(self.angular_velocity[i]) for i in range(len(self.angular_velocity))])
#         ])
#         self.data = self.data.transpose()
#         self.data.columns = ["angular_velocity", "net_tangential_force", "sum_N_ants", "spring_type"]


class AnalyserPerSpring:
    def __init__(self, dir_path, spring_type):
        self.dir_path = dir_path
        self.spring_type = spring_type
        self.paths = [os.path.join(self.dir_path, sub_dir) for sub_dir in os.listdir(self.dir_path) if os.path.isdir(os.path.join(self.dir_path, sub_dir))]
        # self.paths = self.paths[:1]
        self.load_data()
        self.calculations()
        # self.calib_plots()
        self.angle_to_nest_bias()
        self.test_correlation()
        # macro_scale_output_dir = os.path.join("Z:\\Dor_Gabay\\ThesisProject\\results\\graphs\\", self.spring_type, "macro_scale_results")
        # os.makedirs(macro_scale_output_dir, exist_ok=True)
        # ant_profiles_output_dir = os.path.join("Z:\\Dor_Gabay\\ThesisProject\\results\\graphs\\",self.spring_type, "ant_profiles_results")
        # os.makedirs(ant_profiles_output_dir, exist_ok=True)
        # plots.plot_ant_profiles(self, title=self.spring_type, output_dir=ant_profiles_output_dir)
        # plots.plot_query_distribution_to_angular_velocity(self, start=0, end=None, window_size=50, title=self.spring_type, output_dir=macro_scale_output_dir)
        # self.save_analysis_data(self.spring_type)

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
        # self.ant_profiles = np.concatenate([np.load(os.path.join(path, "ant_profiles.npz"))['arr_0'] for path in self.paths], axis=0)
        # self.profiles_precedence = self.ant_profiles[:, 4]
        # self.profiles_ant_labels = self.ant_profiles[:, 0]

    def calculations(self):
        # for count, set_frames in enumerate(self.sets_frames):
        #     s, e = set_frames[0][0], set_frames[-1][1]
        #     self.force_magnitude[s:e] -= np.nanmedian(np.where(~self.rest_bool[s:e], np.nan, self.force_magnitude[s:e]), axis=0)
        #     self.force_direction[s:e] -= np.nanmedian(np.where(~self.rest_bool[s:e], np.nan, self.force_direction[s:e]), axis=0)
        self.calc_net_force()
        self.angular_velocity = utils.calc_angular_velocity(self.fixed_end_angle_to_nest, diff_spacing=50) / 50
        self.angular_velocity = np.where(np.isnan(self.angular_velocity).all(axis=1), np.nan, np.nanmedian(self.angular_velocity, axis=1))
        self.angular_velocity = np.round(self.angular_velocity, 4)
        self.momentum_direction, self.momentum_magnitude = utils.calc_translation_velocity(self.object_center_coordinates, spacing=50)
        self.momentum_direction = np.round(self.momentum_direction, 4)
        self.momentum_magnitude = np.round(self.momentum_magnitude, 4)
        self.net_force_direction = np.array(pd.Series(self.net_force_direction).rolling(window=5, center=True).median())
        self.net_force_magnitude = np.array(pd.Series(self.net_force_magnitude).rolling(window=5, center=True).median())
        self.net_tangential_force = np.array(pd.Series(self.net_tangential_force).rolling(window=5, center=True).median())

        # self.total_n_ants = np.where(np.isnan(self.N_ants_around_springs).all(axis=1), np.nan, np.nansum(self.N_ants_around_springs, axis=1))
        # self.profile_ants_behavior()
        # self.ants_profiling_analysis()

    def calc_net_force(self):
        horizontal_component = self.force_magnitude * np.cos(self.force_direction + self.fixed_end_angle_to_nest)
        vertical_component = self.force_magnitude * np.sin(self.force_direction + self.fixed_end_angle_to_nest)
        self.net_force_direction = np.arctan2(np.nansum(vertical_component, axis=1), np.nansum(horizontal_component, axis=1))
        self.net_force_magnitude = np.sqrt(np.nansum(horizontal_component, axis=1) ** 2 + np.nansum(vertical_component, axis=1) ** 2)
        self.tangential_force = np.sin(self.force_direction) * self.force_magnitude
        self.net_tangential_force = np.where(np.isnan(self.tangential_force).all(axis=1), np.nan, np.nansum(self.tangential_force, axis=1))
        # self.net_tangential_force = np.sum(self.tangential_force, axis=1)

    def profile_ants_behavior(self):
        self.longest_profile = np.max(self.ant_profiles[:, 3] - self.ant_profiles[:, 2]).astype(int)
        self.longest_profile = 12000 if self.longest_profile > 12000 else self.longest_profile
        def profiler(data):
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
        self.profiled_N_ants_around_springs = profiler(self.N_ants_around_springs)
        self.profiled_N_ants_around_springs_sum = profiler(self.total_n_ants)
        self.profiled_fixed_end_angle_to_nest = profiler(self.fixed_end_angle_to_nest)
        self.profiled_force_direction = profiler(self.force_direction)
        self.profiled_force_magnitude = profiler(self.force_magnitude)
        self.profiled_angular_velocity = profiler(self.angular_velocity)
        self.profiled_tangential_force = profiler(self.tangential_force)

    def find_direction_change(self):
        direction_change = []
        for set_idx in self.sets_frames:
            set_fixed_end_angle = self.fixed_end_angle_to_nest[set_idx[0]:set_idx[1]]
            set_angular_velocity = np.nanmedian(utils.calc_angular_velocity(set_fixed_end_angle, diff_spacing=20)/20, axis=1)
            rolling_median = pd.Series(set_angular_velocity).interpolate(method='linear')
            rolling_median = rolling_median.rolling(window=3000, min_periods=1).median()
            rolling_sum = pd.Series(np.abs(set_angular_velocity)).interpolate(method='linear')
            rolling_sum = rolling_sum.rolling(window=3000, min_periods=1).sum()
            object_moves = rolling_sum > 1
            sign_change = np.append(np.diff(np.sign(rolling_median)), 0)
            sign_change_idx = np.arange(len(set_angular_velocity))[(sign_change != 0) * object_moves]
            direction_change.append(sign_change_idx+set_idx[0])
        return np.array(direction_change)

    def calc_ant_replacement_rate(self):
        n_changes = np.nansum(np.abs(np.diff(self.N_ants_around_springs,axis=0)), axis=1)
        sum_of_changes = np.diff(np.nansum(self.N_ants_around_springs, axis=1))
        cancelling_number = (n_changes - np.abs(sum_of_changes))/2
        added_ants = np.copy(cancelling_number)
        added_ants[sum_of_changes > 0] += np.abs(sum_of_changes)[sum_of_changes > 0]
        removed_ants = np.copy(cancelling_number)
        removed_ants[sum_of_changes < 0] += np.abs(sum_of_changes)[sum_of_changes < 0]
        self.n_replacments_per_frame = n_changes

    def calc_pulling_direction_change(self):
        pass

    def ants_profiling_analysis(self):
        """
        creates a boolean array for the profiles that start with one ant, until another ant joins the spring.
        On top of that, it chooses only profiles that had information before attachment,
        to avoid bias of suddenly appearing springs.
        """
        self.single_ant_profiles = np.full((len(self.profiles_precedence), self.longest_profile), False)
        arranged = np.arange(self.longest_profile)
        for profile in range(len(self.profiles_precedence)):
            if self.profiled_N_ants_around_springs[profile, 0] == 1:
                if self.ant_profiles[profile, 5] == 0:
                    first_n_ants_change = arranged[:-1][np.diff(self.profiled_N_ants_around_springs[profile,:]) != 0][0]
                    self.single_ant_profiles[profile,0:first_n_ants_change+1] = True

    def test_correlation(self, sets_idx=(0, -1)):
        first_set_idx, last_set_idx = (sets_idx, sets_idx) if isinstance(sets_idx, int) else sets_idx
        start, end = self.sets_frames[first_set_idx][0][0], self.sets_frames[last_set_idx][-1][1]
        corr_df = pd.DataFrame({"net_tangential_force": self.net_tangential_force[start:end], "angular_velocity": self.angular_velocity[start:end],
                                "momentum_magnitude": self.momentum_magnitude[start:end], "momentum_direction": self.momentum_direction[start:end],
                                "net_force_magnitude": self.net_force_magnitude[start:end], "net_force_direction": self.net_force_direction[start:end]
                                })
        self.corr_df = corr_df.dropna()
        angular_velocity_correlation_score = corr_df.corr()["net_tangential_force"]["angular_velocity"]
        translation_direction_correlation_score = corr_df.corr()["net_force_direction"]["momentum_direction"]
        translation_magnitude_correlation_score = corr_df.corr()["net_force_magnitude"]["momentum_magnitude"]
        print(f"correlation score between net tangential force and angular velocity: {angular_velocity_correlation_score}")
        print(f"correlation score between net force direction and translation direction: {translation_direction_correlation_score}")
        print(f"correlation score between net force magnitude and translation magnitude: {translation_magnitude_correlation_score}")

    def save_analysis_data(self, spring_type):
        output_path = os.path.join("Z:\\Dor_Gabay\\ThesisProject\\results\\", "analysis_data", spring_type)
        os.makedirs(output_path, exist_ok=True)
        np.savez_compressed(os.path.join(output_path, "angular_velocity.npz"), self.angular_velocity)
        np.savez_compressed(os.path.join(output_path, "tangential_force.npz"), self.tangential_force)
        np.savez_compressed(os.path.join(output_path, "N_ants_around_springs.npz"), self.N_ants_around_springs)
        np.savez_compressed(os.path.join(output_path, "missing_info.npz"), self.missing_info)

    def draw_fitted_sine(self, explained, explaining):
        nan_idx = np.isnan(explained)+np.isnan(explaining)
        sort_idx = np.argsort(explaining[~nan_idx])
        explained = explained[~nan_idx][sort_idx]
        explaining = explaining[~nan_idx][sort_idx]
        p0 = [np.max(explained) - np.min(explained), 1, 0, np.mean(explained)]
        params = curve_fit(utils.sine_function, explaining, explained, p0=p0)[0]
        fitted_sine = utils.sine_function(explaining, *params)
        return explaining, fitted_sine

    def calib_plots(self):
        # first video start and end:
        print(f"first video start: {self.sets_frames[0][0]}")
        s, e = self.sets_frames[0][0][0], self.sets_frames[0][0][1]
        self.needle_tip_repeated = np.copy(np.repeat(self.needle_tip_coordinates[:, np.newaxis, :], self.force_magnitude.shape[1], axis=1))
        self.object_center_repeated = np.copy(np.repeat(self.object_center_coordinates[:, np.newaxis, :], self.force_magnitude.shape[1], axis=1))
        needle_length = np.linalg.norm(self.needle_tip_repeated[s:e] - self.object_center_repeated[s:e], axis=2)
        spring_length = np.linalg.norm(self.fixed_ends_coordinates[s:e] - self.free_ends_coordinates[s:e], axis=2)/needle_length
        angles_to_nest = self.fixed_end_angle_to_nest[s:e] - np.pi
        zero_angle_idx = np.logical_and(angles_to_nest < 0.3, angles_to_nest > -0.3)
        # plot boxplots of spring length at zero angle
        plt.figure()
        plt.boxplot(spring_length[zero_angle_idx])
        plt.title("spring length at zero angle")
        output_path = os.path.join("Z:\\Dor_Gabay\\ThesisProject\\results\\tests\\", self.spring_type)
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, "spring_length_at_zero_angle.png"))
        plt.show()

    def angle_to_nest_bias(self):
        video_resolution = np.array([2160, 3840])
        frames_second_video = self.sets_frames[0][0]
        s, e = frames_second_video
        angles_to_nest = self.fixed_end_angle_to_nest#[~self.rest_bool]
        force_magnitude = self.force_magnitude#[~self.rest_bool]
        self.needle_tip_repeated = np.copy(np.repeat(self.needle_tip_coordinates[:, np.newaxis, :], self.force_magnitude.shape[1], axis=1))
        self.object_center_repeated = np.copy(np.repeat(self.object_center_coordinates[:, np.newaxis, :], self.force_magnitude.shape[1], axis=1))
        needle_length = np.linalg.norm(self.needle_tip_repeated - self.fixed_ends_coordinates, axis=2)
        pulling_angle = utils.calc_pulling_angle_matrix(self.fixed_ends_coordinates, self.needle_tip_repeated, self.free_ends_coordinates)
        pulling_angle -= np.nanmedian(pulling_angle)
        spring_length = np.linalg.norm(self.fixed_ends_coordinates - self.free_ends_coordinates, axis=2)
        spring_length[~self.rest_bool] = np.nan
        pulling_angle[~self.rest_bool] = np.nan
        needle_length[~self.rest_bool] = np.nan
        angles_to_nest[~self.rest_bool] = np.nan
        plt.scatter(angles_to_nest[s:e, 0], spring_length[s:e, 0], s=1, alpha=0.5)
        plt.show()
        plt.clf()
        plt.scatter(angles_to_nest[s:e, 0], pulling_angle[s:e, 0], s=1, alpha=0.5)
        plt.show()
        plt.clf()
        plt.scatter(angles_to_nest[s:e, 0], needle_length[s:e, 0], s=1, alpha=0.5)
        plt.show()


        # plt.figure()
        # def position(rest=False):
        #     rest = self.rest_bool if rest else ~self.rest_bool
        #     position_from_center = self.object_center_coordinates - video_resolution / 2
        #     position_from_center = np.expand_dims(position_from_center, axis=1).repeat(20, axis=1)[rest]
        #     smallest_x_distances = position_from_center[:, 0] < np.nanpercentile(position_from_center[:, 0], 20)
        #     smallest_y_distances = position_from_center[:, 1] < np.nanpercentile(position_from_center[:, 1], 20)
        #     largest_x_distances = position_from_center[:, 0] > np.nanpercentile(position_from_center[:, 0], 80)
        #     largest_y_distances = position_from_center[:, 1] > np.nanpercentile(position_from_center[:, 1], 80)
        #     medium_x_distances = np.logical_and(position_from_center[:, 0] > np.nanpercentile(position_from_center[:, 0], 40),
        #                                         position_from_center[:, 0] < np.nanpercentile(position_from_center[:, 0], 60))
        #     medium_y_distances = np.logical_and(position_from_center[:, 1] > np.nanpercentile(position_from_center[:, 1], 40),
        #                                         position_from_center[:, 1] < np.nanpercentile(position_from_center[:, 1], 60))
        #     upper_left = np.logical_and(smallest_x_distances, smallest_y_distances)
        #     upper_right = np.logical_and(largest_x_distances, smallest_y_distances)
        #     lower_left = np.logical_and(smallest_x_distances, largest_y_distances)
        #     lower_right = np.logical_and(largest_x_distances, largest_y_distances)
        #     middle = np.logical_and(medium_x_distances, medium_y_distances)
        #     return upper_left, upper_right, lower_left, lower_right, middle
        #
        # upper_left, upper_right, lower_left, lower_right, middle = position()
        # rest_upper_left, rest_upper_right, rest_lower_left, rest_lower_right, rest_middle = position(rest=True)
        # # fig, axs = plt.subplots(3, 3, figsize=(20, 10))
        # # axs[0, 0].scatter(angles_to_nest[~self.rest_bool][upper_left], force_magnitude[~self.rest_bool][upper_left], s=1, alpha=0.1)
        # # axs[0, 0].plot(*self.draw_fitted_sine(force_magnitude[~self.rest_bool][upper_left], angles_to_nest[~self.rest_bool][upper_left]), c="r", linestyle='dashed')
        # # axs[0, 0].plot(*self.draw_fitted_sine(force_magnitude[self.rest_bool][rest_upper_left], angles_to_nest[self.rest_bool][rest_upper_left]), c="g", linestyle='dashed')
        # # axs[0, 0].title.set_text("upper left")
        # # axs[0, 1].scatter(angles_to_nest[self.rest_bool][rest_upper_left], needle_length[self.rest_bool][rest_upper_left], s=1, alpha=0.1)
        # # axs[0, 1].plot(*self.draw_fitted_sine(needle_length[self.rest_bool][rest_upper_left], angles_to_nest[self.rest_bool][rest_upper_left]), c="b", linestyle='dashed')
        #
        # # axs[0, 1].title.set_text("upper left")
        # # axs[0, 2].scatter(angles_to_nest[~self.rest_bool][upper_right], force_magnitude[~self.rest_bool][upper_right], s=1, alpha=0.1, c="g")
        # # axs[0, 2].plot(*self.draw_fitted_sine(force_magnitude[~self.rest_bool][upper_right], angles_to_nest[~self.rest_bool][upper_right]), c="r", linestyle='dashed')
        # # axs[0, 2].plot(*self.draw_fitted_sine(force_magnitude[self.rest_bool][rest_upper_right], angles_to_nest[self.rest_bool][rest_upper_right]), c="g", linestyle='dashed')
        # # axs[0, 2].title.set_text("upper right")
        # # axs[1, 2].scatter(angles_to_nest[self.rest_bool][rest_upper_right], needle_length[self.rest_bool][rest_upper_right], s=1, alpha=0.1, c="g")
        # # axs[1, 2].plot(*self.draw_fitted_sine(needle_length[self.rest_bool][rest_upper_right], angles_to_nest[self.rest_bool][rest_upper_right]), c="b", linestyle='dashed')
        # # axs[1, 2].title.set_text("upper right")
        # # axs[2, 0].scatter(angles_to_nest[~self.rest_bool][lower_left], force_magnitude[~self.rest_bool][lower_left], s=1, alpha=0.1, c="m")
        # # axs[2, 0].plot(*self.draw_fitted_sine(force_magnitude[~self.rest_bool][lower_left], angles_to_nest[~self.rest_bool][lower_left]), c="r", linestyle='dashed')
        # # axs[2, 0].plot(*self.draw_fitted_sine(force_magnitude[self.rest_bool][rest_lower_left], angles_to_nest[self.rest_bool][rest_lower_left]), c="g", linestyle='dashed')
        # # axs[2, 0].title.set_text("lower left")
        # # axs[1, 0].scatter(angles_to_nest[self.rest_bool][rest_lower_left], needle_length[self.rest_bool][rest_lower_left], s=1, alpha=0.1, c="m")
        # # axs[1, 0].plot(*self.draw_fitted_sine(needle_length[self.rest_bool][rest_lower_left], angles_to_nest[self.rest_bool][rest_lower_left]), c="b", linestyle='dashed')
        # # axs[1, 0].title.set_text("lower left")
        # # axs[2, 2].scatter(angles_to_nest[~self.rest_bool][lower_right], force_magnitude[~self.rest_bool][lower_right], s=1, alpha=0.1, c="c")
        # # axs[2, 2].plot(*self.draw_fitted_sine(force_magnitude[~self.rest_bool][lower_right], angles_to_nest[~self.rest_bool][lower_right]), c="r", linestyle='dashed')
        # # axs[2, 2].plot(*self.draw_fitted_sine(force_magnitude[self.rest_bool][rest_lower_right], angles_to_nest[self.rest_bool][rest_lower_right]), c="g", linestyle='dashed')
        # # axs[2, 2].title.set_text("lower right")
        # # axs[2, 1].scatter(angles_to_nest[self.rest_bool][rest_lower_right], needle_length[self.rest_bool][rest_lower_right], s=1, alpha=0.1, c="c")
        # # axs[2, 1].plot(*self.draw_fitted_sine(needle_length[self.rest_bool][rest_lower_right], angles_to_nest[self.rest_bool][rest_lower_right]), c="b", linestyle='dashed')
        # # axs[2, 1].title.set_text("lower right")
        # # axs[1, 1].scatter(angles_to_nest[~self.rest_bool][middle], force_magnitude[~self.rest_bool][middle], s=1, alpha=0.1)
        # # axs[1, 1].plot(*self.draw_fitted_sine(force_magnitude[~self.rest_bool][middle], angles_to_nest[~self.rest_bool][middle]), c="r", linestyle='dashed')
        # # axs[1, 1].plot(*self.draw_fitted_sine(force_magnitude[self.rest_bool][rest_middle], angles_to_nest[self.rest_bool][rest_middle]), c="g", linestyle='dashed')
        # # axs[1, 1].title.set_text("middle")
        # # axs[1, 0].legend(["fitted sine", "rest", "needle length"])
        # # plt.show()
        # #create box plot. in one plot make a box for: rest_upper_left, rest_upper_right, rest_lower_left, rest_lower_right, rest_middle. the data is spring length at rest.
        # plt.figure()
        # plt.boxplot(spring_length[rest_upper_left][~np.isnan(spring_length[rest_upper_left])], positions=[1], widths=0.5)
        # plt.boxplot(spring_length[rest_upper_right][~np.isnan(spring_length[rest_upper_right])], positions=[2], widths=0.5)
        # plt.boxplot(spring_length[rest_lower_left][~np.isnan(spring_length[rest_lower_left])], positions=[3], widths=0.5)
        # plt.boxplot(spring_length[rest_lower_right][~np.isnan(spring_length[rest_lower_right])], positions=[4], widths=0.5)
        # plt.boxplot(spring_length[rest_middle][~np.isnan(spring_length[rest_middle])], positions=[5], widths=0.5)
        # plt.xticks([1, 2, 3, 4, 5], ["upper left", "upper right", "lower left", "lower right", "middle"])
        # plt.title("spring length at zero angle")
        # output_path = os.path.join("Z:\\Dor_Gabay\\ThesisProject\\results\\tests\\", self.spring_type)
        # os.makedirs(output_path, exist_ok=True)
        # plt.savefig(os.path.join(output_path, "spring_length_at_zero_angle_experiment.png"))
        # plt.show()

if __name__ == "__main__":
    spring_type = "plus_0.1"
    data_analysis_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\3-data_analysis\\summer_2023\\calibration\\{spring_type}\\"
    self = AnalyserPerSpring(data_analysis_dir, spring_type)
    # self.angle_to_nest_bias()

