import pickle

from data_analysis import utils
from data_analysis import plots
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


class DataAnalyser():
    def __init__(self, sub_dirs_paths):
        self.sub_dirs_paths = sub_dirs_paths
        self.dir_path = os.path.split(self.sub_dirs_paths[0])[0]
        # self.sub_dirs_sets_names = [f"{data_set[0]}-{data_set[-1]}" for data_set in sub_dirs_set]
        # self.video_name = os.path.basename(self.directory)
        self.load_data()
        self.calculations()

    def load_data(self):
        print("loading data...")
        # self.sets_frames = np.array([np.load(os.path.join(path, "N_ants_around_springs.npz"))['arr_0'].shape[0] for path in self.sub_dirs_paths])
        self.sets_idx = pickle.load(open(os.path.join(self.dir_path, "sets_idx.pkl"), "rb"))
        self.N_ants_around_springs = np.concatenate([np.load(os.path.join(path, "N_ants_around_springs.npz"))['arr_0'] for path in self.sub_dirs_paths], axis=0)
        self.fixed_end_angle_to_nest = np.concatenate([np.load(os.path.join(path, "fixed_end_angle_to_nest.npz"))['arr_0'] for path in self.sub_dirs_paths], axis=0)
        self.force_direction = np.concatenate([np.load(os.path.join(path, "force_direction.npz"))['arr_0'] for path in self.sub_dirs_paths], axis=0)
        self.force_magnitude = np.concatenate([np.load(os.path.join(path, "force_magnitude.npz"))['arr_0'] for path in self.sub_dirs_paths], axis=0)* 1000

        if os.path.exists(os.path.join(self.dir_path, "tangential_force.npz")):
            self.tangential_force = np.concatenate([np.load(os.path.join(path, "tangential_force.npz"))['arr_0'] for path in self.sub_dirs_paths], axis=0)* 1000
            self.all_profiles_tangential_force = np.concatenate([np.load(os.path.join(path, "all_profiles_tangential_force.npz"))['arr_0'] for path in self.sub_dirs_paths], axis=0)* 1000
        else:
            self.tangential_force = np.concatenate([np.load(os.path.join(path, "angular_force.npz"))['arr_0'] for path in self.sub_dirs_paths], axis=0)* 1000
            self.all_profiles_tangential_force = np.concatenate([np.load(os.path.join(path, "all_profiles_angular_force.npz"))['arr_0'] for path in self.sub_dirs_paths], axis=0)* 1000
        profiles_number_by_sets = [np.load(os.path.join(path, "all_profiles_force_magnitude.npz"))['arr_0'].shape[0] for path in self.sub_dirs_paths]
        self.profiles_start_frame = np.concatenate([np.repeat(set_idx[0], profiles_number) for set_idx, profiles_number in zip(self.sets_idx, profiles_number_by_sets)])
        self.all_profiles_force_magnitude = np.concatenate([np.load(os.path.join(path, "all_profiles_force_magnitude.npz"))['arr_0'] for path in self.sub_dirs_paths], axis=0)* 1000
        self.all_profiles_force_direction = np.concatenate([np.load(os.path.join(path, "all_profiles_force_direction.npz"))['arr_0'] for path in self.sub_dirs_paths], axis=0)
        self.all_profiles_N_ants_around_springs = np.concatenate([np.load(os.path.join(path, "all_profiles_N_ants_around_springs.npz"))['arr_0'] for path in self.sub_dirs_paths], axis=0)
        self.all_profiles_N_ants_around_springs_sum = np.concatenate([np.load(os.path.join(path, "all_profiles_N_ants_around_springs_sum.npz"))['arr_0'] for path in self.sub_dirs_paths], axis=0)
        self.all_profiles_fixed_end_angle_to_nest = np.concatenate([np.load(os.path.join(path, "all_profiles_fixed_end_angle_to_nest.npz"))['arr_0'] for path in self.sub_dirs_paths], axis=0)
        self.all_profiles_angular_velocity = np.concatenate([np.load(os.path.join(path, "all_profiles_angular_velocity.npz"))['arr_0'] for path in self.sub_dirs_paths], axis=0)
        self.all_profiles_information = np.concatenate([np.load(os.path.join(path, "all_profiles_information.npz"))['arr_0'] for path in self.sub_dirs_paths], axis=0)
        self.all_profiles_precedence = self.all_profiles_information[:, 4]
        self.all_profiles_ant_labels = self.all_profiles_information[:, 0]
        print("Data loaded successfully")

    def find_direction_change(self):
        direction_change = []
        for set_idx in self.sets_idx:
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
        print(direction_change)
        return np.array(direction_change)

    def calculations(self):
        # print(self.fixed_end_angle_to_nest)
        self.angular_velocity = np.nanmedian(utils.calc_angular_velocity(self.fixed_end_angle_to_nest, diff_spacing=20)/20, axis=1)
        # self.angular_force = self.force_magnitude * np.sin(self.force_direction)
        self.net_tangential_force = np.nansum(self.tangential_force, axis=1)
        self.net_tangential_force = np.array(pd.Series(self.net_tangential_force).rolling(window=5,center=True).median())
        self.net_magnitude = np.nansum(self.force_magnitude, axis=1)
        self.net_magnitude = np.array(pd.Series(self.net_magnitude).rolling(window=5,center=True).median())
        self.total_n_ants = np.nansum(self.N_ants_around_springs, axis=1)
        self.ants_profiling_analysis()
        # self.find_direction_change()

    def ants_profiling_analysis(self):
        """
        creates a boolean array for the profiles that start with one ant, until another ant joins the spring.
        On top of that, it chooses only profiles that had information before attachment,
        to avoid bias of suddenly appearing springs.
        """
        self.single_ant_profiles = np.full((len(self.all_profiles_precedence), 10000), False)
        arranged = np.arange(10000)
        for profile in range(len(self.all_profiles_precedence)):
            if self.all_profiles_N_ants_around_springs[profile, 0] == 1:
                if self.all_profiles_information[profile, 5] == 0:
                    first_n_ants_change = arranged[:-1][np.diff(self.all_profiles_N_ants_around_springs[profile,:]) != 0][0]
                    self.single_ant_profiles[profile,0:first_n_ants_change+1] = True
                    # unique = np.unique(self.all_profiles_N_ants_around_springs[profile,0:first_n_ants_change+1])
                    # if len(unique) > 1:
                    #     print(unique)


if __name__ == "__main__":
    # spring_types = ["plus0", "plus0.1", "plus0.3", "plus0.5"]
    spring_types = ["plus0.5"]
    # analysed = {}
    for spring_type in spring_types:
        directory = f"Z:\\Dor_Gabay\\ThesisProject\\data\\post_processed_data\\{spring_type}\\"
        sub_dirs_paths = [os.path.join(directory, sub_dir) for sub_dir in os.listdir(directory) if os.path.isdir(os.path.join(directory, sub_dir))]
        analysed = DataAnalyser(sub_dirs_paths)

        macro_scale_output_dir = os.path.join("Z:\\Dor_Gabay\\ThesisProject\\results", spring_type, "macro_scale_results")
        os.makedirs(macro_scale_output_dir, exist_ok=True)
        ant_profiles_output_dir = os.path.join("Z:\\Dor_Gabay\\ThesisProject\\results",spring_type, "ant_profiles_results")
        os.makedirs(ant_profiles_output_dir, exist_ok=True)

        # plots.draw_single_profiles(analysed, ant_profiles_output_dir, profile_min_length=200)
        # plots.plot_ant_profiles(analysed, title=spring_type, output_dir=ant_profiles_output_dir)
        # plots.plot_query_distribution_to_angular_velocity(analysed, start=0, end=None, window_size=50, title=spring_type, output_dir=macro_scale_output_dir)
        # plots.plot_mean_angular_velocity_to_attachment_time(analysed, window_size=50, title=spring_type, output_dir=ant_profiles_output_dir)

