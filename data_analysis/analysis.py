from data_analysis import plots,utils
import numpy as np
import os


class DataAnalyser(object):
    def __init__(self,directory):

        self.directory = directory
        self.video_name = os.path.basename(self.directory)
        self.load_data(self.directory)
        self.calculations()

    def load_data(self,directory):
        directory = os.path.join(directory, "post_processed_data")
        self.N_ants_around_springs = np.loadtxt(os.path.join(directory, "N_ants_around_springs.csv"), delimiter=",")
        self.fixed_end_angle_to_nest = np.loadtxt(os.path.join(directory, "fixed_end_angle_to_nest.csv"), delimiter=",")
        self.force_direction = np.loadtxt(os.path.join(directory, "force_direction.csv"), delimiter=",")
        self.force_magnitude = np.loadtxt(os.path.join(directory, "force_magnitude.csv"), delimiter=",")
        # self.pulling_angle = np.loadtxt(os.path.join(directory, "pulling_angle.csv"), delimiter=",")
        # self.spring_length = np.loadtxt(os.path.join(directory, "spring_length.csv"), delimiter=",")
        # self.force = np.loadtxt(os.path.join(directory, "force.csv"), delimiter=",")

    def calculations(self):
        # self.velocity = np.nanmedian(utils.calc_angular_velocity(self.fixed_end_angle_to_nest, diff_spacing=1), axis=1)
        self.velocity_spaced = np.nanmedian(utils.calc_angular_velocity(self.fixed_end_fixed_coordinates_angle_to_nest, diff_spacing=20), axis=1)
        self.total_n_ants = np.nansum(self.N_ants_around_springs, axis=1)
        self.pulling_angle = np.nansum(self.pulling_angle, axis=1)
        self.angle_to_nest = np.nansum(self.fixed_end_fixed_coordinates_angle_to_nest, axis=1)
        self.spring_extension = np.nanmean(self.spring_length, axis=1)
        self.total_force = np.sin(self.force_direction) * self.force_magnitude
        self.net_force = np.nansum(self.force, axis=1)
        # self.net_force = np.nansum(self.total_force,axis=1)

if __name__ == "__main__":
    # %matplotlib qt
    # dir = "Z:\\Dor_Gabay\\ThesisProject\\data\\test3\\15.9.22\\plus0.3mm_force\\S5280006\\"
    dir = "Z:\\Dor_Gabay\\ThesisProject\\data\\exp_collected_data\\15.9.22\\plus0.3mm_force\\S5280006\\"
    video_name = "S5280006"
    analysed = DataAnalyser(dir)
    # from data_analysis.plots import plot_overall_behavior
    plots.plot_overall_behavior(analysed, start=0, end=None, window_size=50, title=video_name, output_dir=dir)
    # plots.plot_pulling_angle_over_angle_to_nest(analysed, start=0, end=None)

