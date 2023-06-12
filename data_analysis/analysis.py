from data_analysis import utils
# from data_analysis import plots
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt


class DataAnalyser():
    def __init__(self,directory, sub_dirs_set):
        self.directory = directory
        self.sub_dirs_sets_names = [f"{data_set[0]}-{data_set[-1]}" for data_set in sub_dirs_set]
        # self.video_name = os.path.basename(self.directory)
        self.load_data()
        self.calculations()

    def load_data(self):
        print("loading data from directory: ", self.directory)
        directory = os.path.join(self.directory, "post_processing")
        self.N_ants_around_springs = np.concatenate([np.load(os.path.join(directory, set_sub_dir, "N_ants_around_springs.npz"))['arr_0'] for set_sub_dir in self.sub_dirs_sets_names], axis=0)
        self.fixed_end_angle_to_nest = np.concatenate([np.load(os.path.join(directory, set_sub_dir, "fixed_end_angle_to_nest.npz"))['arr_0'] for set_sub_dir in self.sub_dirs_sets_names], axis=0)
        self.force_direction = np.concatenate([np.load(os.path.join(directory, set_sub_dir, "force_direction.npz"))['arr_0'] for set_sub_dir in self.sub_dirs_sets_names], axis=0)
        self.force_magnitude = np.concatenate([np.load(os.path.join(directory, set_sub_dir, "force_magnitude.npz"))['arr_0'] for set_sub_dir in self.sub_dirs_sets_names], axis=0)
        self.force_magnitude = self.force_magnitude * 1000
        # self.ants_assigned_to_springs = np.load(os.path.join(directory, "ants_assigned_to_springs.npz"))['arr_0']
        self.all_profiles_force_magnitude = np.concatenate([np.load(os.path.join(directory, set_sub_dir, "all_profiles_force_magnitude.npz"))['arr_0'] for set_sub_dir in self.sub_dirs_sets_names], axis=0)
        self.all_profiles_force_magnitude = self.all_profiles_force_magnitude * 1000
        self.all_profiles_force_direction = np.concatenate([np.load(os.path.join(directory, set_sub_dir, "all_profiles_force_direction.npz"))['arr_0'] for set_sub_dir in self.sub_dirs_sets_names], axis=0)
        self.all_profiles_N_ants_around_springs = np.concatenate([np.load(os.path.join(directory, set_sub_dir, "all_profiles_N_ants_around_springs.npz"))['arr_0'] for set_sub_dir in self.sub_dirs_sets_names], axis=0)
        self.all_profiles_N_ants_around_springs_sum = np.concatenate([np.load(os.path.join(directory, set_sub_dir, "all_profiles_N_ants_around_springs_sum.npz"))['arr_0'] for set_sub_dir in self.sub_dirs_sets_names], axis=0)
        self.all_profiles_fixed_end_angle_to_nest = np.concatenate([np.load(os.path.join(directory, set_sub_dir, "all_profiles_fixed_end_angle_to_nest.npz"))['arr_0'] for set_sub_dir in self.sub_dirs_sets_names], axis=0)
        self.all_profiles_angular_velocity = np.concatenate([np.load(os.path.join(directory, set_sub_dir, "all_profiles_angular_velocity.npz"))['arr_0'] for set_sub_dir in self.sub_dirs_sets_names], axis=0)
        self.all_profiles_information = np.concatenate([np.load(os.path.join(directory, set_sub_dir, "all_profiles_information.npz"))['arr_0'] for set_sub_dir in self.sub_dirs_sets_names], axis=0)
        self.all_profiles_precedence = self.all_profiles_information[:, 4]
        self.all_profiles_ant_labels = self.all_profiles_information[:, 0]
        print("Data loaded successfully")

    def calculations(self):
        # self.ant_profiling()
        self.angular_velocity = np.nanmedian(utils.calc_angular_velocity(self.fixed_end_angle_to_nest, diff_spacing=20)/20, axis=1)
        self.total_force = self.force_magnitude * np.sin(self.force_direction)
        self.net_force = np.nansum(self.total_force, axis=1)
        self.net_force = np.array(pd.Series(self.net_force).rolling(window=5,center=True).median())
        self.net_magnitude = np.nansum(self.force_magnitude, axis=1)
        self.net_magnitude = np.array(pd.Series(self.net_magnitude).rolling(window=5,center=True).median())
        # self.total_n_ants = np.nansum(self.N_ants_around_springs, axis=1)
        self.ants_profiling_analysis()

    def ants_profiling_analysis(self):
        self.profiles_start_with_one_ant = np.full((len(self.all_profiles_precedence), 10000), False)
        aranged = np.arange(10000)
        for profile in range(len(self.all_profiles_precedence)):
            if self.all_profiles_N_ants_around_springs[profile,0] == 1:
                first_n_ants_change = aranged[:-1][np.diff(self.all_profiles_N_ants_around_springs[profile,:]) != 0][0]
                self.profiles_start_with_one_ant[profile,0:first_n_ants_change+1] = True
                unique = np.unique(self.all_profiles_N_ants_around_springs[profile,0:first_n_ants_change+1])
                if len(unique) > 1:
                    print(unique)


if __name__ == "__main__":
    spring_type = "plus_0.3mm"
    directory = "Z:\\Dor_Gabay\\ThesisProject\\data\\analysed_with_tracking\\15.9.22\\plus0.3mm_force\\"
    sub_dirs_set = [[f"S528000{i}" for i in [3, 4, 5, 6, 7]], [f"S528000{i}" for i in [8, 9]]]
    analysed = DataAnalyser(directory, sub_dirs_set)

    output_dir = os.path.join("Z:\\Dor_Gabay\\ThesisProject\\results", spring_type, "macro_scale_results")
    # os.makedirs(output_dir, exist_ok=True)
    # plots.plot_query_distribution_to_angular_velocity(analysed, start=0, end=None, window_size=50, title=spring_type, output_dir=output_dir)

    output_dir = os.path.join("Z:\\Dor_Gabay\\ThesisProject\\results",spring_type, "ant_profiles_results")
    os.makedirs(output_dir, exist_ok=True)
    plots.plot_mean_angular_velocity_to_attachment_time(analysed, window_size=50, title=spring_type, output_dir=output_dir)
    # plots.plot_ant_profiles(analysed, window_size=1, title=spring_type, output_dir=output_dir)
    # plots.plot_overall_behavior(analysed, start=0, end=None, window_size=50, title=spring_type, output_dir=output_dir)

