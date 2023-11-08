import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
# local packages:
from data_analysis import utils
from data_analysis import plots


class Analyser:
    def __init__(self, dir_path, output_path, spring_type):
        self.dir_path = dir_path
        self.output_path = output_path
        self.spring_type = spring_type
        self.fps = 25
        self.paths = [os.path.join(self.dir_path, sub_dir) for sub_dir in os.listdir(self.dir_path) if os.path.isdir(os.path.join(self.dir_path, sub_dir))]
        self.load_data()
        self.calculations()
        self.find_direction_changes()
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
        self.force_direction = utils.interpolate_data(self.force_direction)
        self.force_magnitude = utils.interpolate_data(self.force_magnitude)
        horizontal_component = self.force_magnitude * np.cos(self.force_direction + self.fixed_end_angle_to_nest)
        horizontal_component = utils.interpolate_data(horizontal_component)
        vertical_component = self.force_magnitude * np.sin(self.force_direction + self.fixed_end_angle_to_nest)
        vertical_component = utils.interpolate_data(vertical_component)
        self.net_force_direction = np.arctan2(np.nansum(vertical_component, axis=1), np.nansum(horizontal_component, axis=1))
        self.net_force_magnitude = np.sqrt(np.nansum(horizontal_component, axis=1) ** 2 + np.nansum(vertical_component, axis=1) ** 2)
        self.net_force_direction = np.array(pd.Series(self.net_force_direction).rolling(window=window_size, center=True).median())
        self.net_force_magnitude = np.array(pd.Series(self.net_force_magnitude).rolling(window=window_size, center=True).median())
        # tangential force:
        self.tangential_force = utils.interpolate_data(np.sin(self.force_direction) * self.force_magnitude)
        self.net_tangential_force = np.where(np.isnan(self.tangential_force).all(axis=1), np.nan, np.nansum(self.tangential_force, axis=1))
        self.net_tangential_force = np.array(pd.Series(self.net_tangential_force).rolling(window=window_size, center=True).median())
        # angular velocity:
        self.angular_velocity = utils.calc_angular_velocity(self.fixed_end_angle_to_nest, diff_spacing=window_size) / window_size
        self.angular_velocity = utils.interpolate_data(self.angular_velocity)
        self.angular_velocity = np.where(np.isnan(self.angular_velocity).all(axis=1), np.nan, np.nanmedian(self.angular_velocity, axis=1))
        self.angular_velocity = np.array(pd.Series(self.angular_velocity).rolling(window=window_size, center=True).median())
        self.angular_velocity = np.round(self.angular_velocity, 4) * -1
        # translational velocity:
        self.momentum_direction, self.momentum_magnitude = utils.calc_translation_velocity(self.object_center_coordinates, spacing=window_size)
        self.momentum_direction = np.round(utils.interpolate_data(self.momentum_direction), 4)
        self.momentum_magnitude = np.round(utils.interpolate_data(self.momentum_magnitude), 4)
        # ants:
        self.total_n_ants = np.where(np.isnan(self.N_ants_around_springs).all(axis=1), np.nan, np.nansum(self.N_ants_around_springs, axis=1))
        self.profile_ants_behavior()
        self.ants_profiling_analysis()

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
        # FIXME: profiled_N_ants_around_springs_sum needs to according to profiled_N_ants_around_springs
        self.profiled_N_ants_around_springs_sum = self.profiler(self.total_n_ants)
        self.profiled_fixed_end_angle_to_nest = self.profiler(self.fixed_end_angle_to_nest)
        self.profiled_force_direction = self.profiler(self.force_direction)
        self.profiled_force_magnitude = self.profiler(self.force_magnitude)
        self.profiled_angular_velocity = self.profiler(self.angular_velocity)
        self.profiled_tangential_force = self.profiler(self.tangential_force)
        reverse_arg_sort_columns = np.full(self.profiled_N_ants_around_springs.shape, 0)
        nans = np.isnan(self.profiled_N_ants_around_springs)
        nans_sum = np.sum(nans, axis=1)
        for row in range(self.profiled_N_ants_around_springs.shape[0]):
            reverse_arg_sort_columns[row, nans_sum[row]:] = np.where(~nans[row])[0]
            reverse_arg_sort_columns[row, :nans_sum[row]] = np.where(nans[row])[0]
        reverse_arg_sort_rows = np.repeat(np.expand_dims(np.arange(self.profiled_N_ants_around_springs.shape[0]), axis=1), self.profiled_N_ants_around_springs.shape[1], axis=1)
        self.reverse_argsort = (reverse_arg_sort_rows, reverse_arg_sort_columns)

    def find_direction_changes(self):
        direction_change = []
        for set_idx in self.sets_frames:
            s, e = set_idx[0][0], set_idx[-1][1]
            import matplotlib.pyplot as plt
            angular_velocity = utils.calc_angular_velocity(self.fixed_end_angle_to_nest, diff_spacing=80) / 80
            angular_velocity = np.where(np.isnan(angular_velocity).all(axis=1), np.nan, np.nanmedian(angular_velocity, axis=1))
            angular_velocity = np.array(pd.Series(angular_velocity).rolling(window=30, center=True).median())
            angular_velocity = utils.interpolate_data(angular_velocity)
            upper = np.nanpercentile(angular_velocity, 70)
            lower = np.nanpercentile(angular_velocity, 30)
            angular_velocity[(angular_velocity < upper) * (angular_velocity > lower)] = 0
            plt.plot(np.arange(len(angular_velocity)), np.sign(angular_velocity))
            plt.show()
            # translational_velocity = np.sin(self.momentum_direction) * self.momentum_magnitude
            self.momentum_direction, self.momentum_magnitude = utils.calc_translation_velocity(self.object_center_coordinates, spacing=10)
            self.momentum_magnitude = np.array(pd.Series(self.momentum_magnitude).rolling(window=20, center=True).median())
            momentum_magnitude = self.momentum_magnitude.copy()
            upper = np.nanpercentile(momentum_magnitude, 95)
            # lower = np.nanpercentile(translational_velocity, 1.5)
            momentum_magnitude[momentum_magnitude < upper] = 0
            plt.plot(np.arange(len(momentum_magnitude)), np.sign(momentum_magnitude))
            plt.show()

            # set_fixed_end_angle = self.fixed_end_angle_to_nest[s:e]
            # set_angular_velocity = np.nanmedian(utils.calc_angular_velocity(set_fixed_end_angle, diff_spacing=20)/20, axis=1)
            # rolling_median = pd.Series(set_angular_velocity).interpolate(method='linear')
            # rolling_median = rolling_median.rolling(window=3000, min_periods=1).median()
            # rolling_sum = pd.Series(np.abs(set_angular_velocity)).interpolate(method='linear')
            # rolling_sum = rolling_sum.rolling(window=3000, min_periods=1).sum()
            # object_moves = rolling_sum > 1
            # sign_change = np.append(np.diff(np.sign(rolling_median)), 0)
            # sign_change_idx = np.arange(len(set_angular_velocity))[(sign_change != 0) * object_moves]
            # direction_change.append(sign_change_idx+s)
        # print(direction_change)
        return np.array(direction_change)

    def calc_ant_replacement_rate(self):
        n_changes = np.nansum(np.abs(np.diff(self.N_ants_around_springs, axis=0)), axis=1)
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
        self.attaching_single_ant_profiles = np.full((len(self.profiles_precedence), self.longest_profile), False)
        self.detaching_single_ant_profiles = np.full((len(self.profiles_precedence), self.longest_profile), False)
        arranged = np.arange(self.longest_profile)
        reversed_profiled_N_ants_around_springs = self.profiled_N_ants_around_springs[self.reverse_argsort]
        for profile in range(len(self.profiles_precedence)):
            if not np.any(self.profiled_check[profile, :3]):
                first_n_ants_change = arranged[:-1][np.diff(self.profiled_N_ants_around_springs[profile, :]) != 0][0]
                self.attaching_single_ant_profiles[profile, 0:first_n_ants_change+1] = True
            if not np.any(self.profiled_check[profile, -2:]) and reversed_profiled_N_ants_around_springs[profile, -1] == 1:
                last_n_ants_change = arranged[1:][np.diff(reversed_profiled_N_ants_around_springs[profile, :]) != 0][-1]
                self.detaching_single_ant_profiles[profile, last_n_ants_change:] = True

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
        # plots.draw_single_profiles(self, os.path.join(self.output_path, "detaching_single_profiles_S5760003"), profile_min_length=200, examples_number=200,
        plots.draw_single_profiles(self, os.path.join(self.output_path, "attaching_single_profiles_S5760003"), profile_min_length=200, examples_number=200,
                                   start=self.sets_frames[0][0][0], end=self.sets_frames[0][0][1])
        # plots.draw_single_profiles(self, os.path.join(self.output_path, "single_profiles_S5760003_1"), profile_min_length=200, examples_number=200, start=self.sets_frames[0][0][0], end=self.sets_frames[0][0][1])

    def save_analysis_data(self, spring_type):
        output_path = os.path.join("Z:\\Dor_Gabay\\ThesisProject\\results\\", "analysis_data", spring_type)
        os.makedirs(output_path, exist_ok=True)
        np.savez_compressed(os.path.join(output_path, "angular_velocity.npz"), self.angular_velocity)
        np.savez_compressed(os.path.join(output_path, "tangential_force.npz"), self.tangential_force)
        np.savez_compressed(os.path.join(output_path, "N_ants_around_springs.npz"), self.N_ants_around_springs)
        np.savez_compressed(os.path.join(output_path, "missing_info.npz"), self.missing_info)


if __name__ == "__main__":
    spring_type = "plus_0.1"
    # data_analysis_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\3-data_analysis\\summer_2023\\experiment\\{spring_type}\\"
    data_analysis_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\3-data_analysis\\summer_2023\\experiment\\{spring_type}_final\\"
    output_dir = f"Z:\\Dor_Gabay\\ThesisProject\\results\\summer_2023\\{spring_type}_final\\"
    self = Analyser(data_analysis_dir, output_dir, spring_type)

