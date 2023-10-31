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
        self.test_correlation()
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

    def profile_ants_behavior(self):
        def profiler_check():
            profiled_check = np.full((self.ant_profiles.shape[0], 2), False)
            for profile in range(len(self.ant_profiles)):
                spring = int(self.ant_profiles[profile, 1])
                start = int(self.ant_profiles[profile, 2])
                end = int(self.ant_profiles[profile, 3])
                sudden_appearance = np.any(self.missing_info[start-3:start, spring-1])
                ants_before = np.any(self.N_ants_around_springs[start-3:start, spring-1] != 0)
                profiled_check[profile, 0] = sudden_appearance
                profiled_check[profile, 1] = ants_before
            return profiled_check
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

        self.profiled_check = profiler_check()
        self.longest_profile = np.max(self.ant_profiles[:, 3] - self.ant_profiles[:, 2]).astype(int)
        self.longest_profile = 12000 if self.longest_profile > 12000 else self.longest_profile
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
            if not np.any(self.profiled_check[profile, :]):
                first_n_ants_change = arranged[:-1][np.diff(self.profiled_N_ants_around_springs[profile, :]) != 0][0]
                self.single_ant_profiles[profile, 0:first_n_ants_change+1] = True
            # if self.profiled_N_ants_around_springs[profile, 0] == 1:
            #     if self.ant_profiles[profile, 5] == 0:

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
        plots.plot_ant_profiles(self, output_dir=os.path.join(self.output_path, "profiles"), window_size=11, profile_size=200)
        plots.draw_single_profiles(self, os.path.join(self.output_path, "single_profiles_S5760011"), profile_min_length=200, examples_number=200, start=self.sets_frames[1][-1][0], end=self.sets_frames[1][-1][1])

    def save_analysis_data(self, spring_type):
        output_path = os.path.join("Z:\\Dor_Gabay\\ThesisProject\\results\\", "analysis_data", spring_type)
        os.makedirs(output_path, exist_ok=True)
        np.savez_compressed(os.path.join(output_path, "angular_velocity.npz"), self.angular_velocity)
        np.savez_compressed(os.path.join(output_path, "tangential_force.npz"), self.tangential_force)
        np.savez_compressed(os.path.join(output_path, "N_ants_around_springs.npz"), self.N_ants_around_springs)
        np.savez_compressed(os.path.join(output_path, "missing_info.npz"), self.missing_info)


if __name__ == "__main__":
    spring_type = "plus_0.1"
    # data_analysis_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\3-data_analysis\\summer_2023\\calibration\\{spring_type}\\"
    data_analysis_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\3-data_analysis\\summer_2023\\experiment\\{spring_type}\\"
    output_dir = f"Z:\\Dor_Gabay\\ThesisProject\\results\\summer_2023\\{spring_type}\\"
    self = Analyser(data_analysis_dir, output_dir, spring_type)

