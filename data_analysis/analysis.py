import pickle
from data_analysis import data_preparation,plots,utils
import numpy as np
import pandas as pd
import os
import copy

class DataAnalyser:
    def __init__(self,directory):
        self.directory = "Z:\\Dor_Gabay\\ThesisProject\\data\\test3\\15.9.22\\plus0.3mm_force\\S5280006\\"
        self.video_name = os.path.basename(self.directory)
        self.load_data(self.directory)
        self.calculations()

    def load_data(self,directory):
        directory = os.path.join(directory, "post_processed_data/")
        self.N_ants_around_springs = np.loadtxt(f"{directory}N_ants_around_springs.csv", delimiter=",")
        self.spring_length = np.loadtxt(f"{directory}spring_length.csv", delimiter=",")
        self.fixed_end_angle_to_nest = np.loadtxt(f"{directory}fixed_end_angle_to_nest.csv", delimiter=",")
        self.pulling_angle = np.loadtxt(f"{directory}pulling_angle.csv", delimiter=",")

    def calculations(self):
        self.velocity = np.nanmedian(utils.calc_angular_velocity(self.fixed_end_angle_to_nest, diff_spacing=1), axis=1)
        self.velocity_spaced = np.nanmedian(utils.calc_angular_velocity(self.fixed_end_angle_to_nest, diff_spacing=20), axis=1)
        self.total_n_ants = np.nansum(self.N_ants_around_springs, axis=1)
        self.pulling_angle = np.nansum(self.pulling_angle, axis=1)
        self.spring_extension = np.nanmean(self.spring_length, axis=1)

# def analyse_data(data_prepared):
#     # analyse the data, and return the results
#     analysed = {}
#     for spring_type in data_prepared:
#         analysed[spring_type] = {}
#         for video in data_prepared[spring_type]:
#             analysed[spring_type][video] = {}
#             data = copy.deepcopy(data_prepared[spring_type][video])
#             data = correct_bias(data)
#             total_n_ants = np.nansum(data.N_ants_around_springs, axis=1)
#             velocity = np.nanmedian(utils.calc_angular_velocity(data.angle_to_nest,diff_spacing=1), axis=1)
#             velocity_spaced = np.nanmedian(utils.calc_angular_velocity(data.angle_to_nest,diff_spacing=20), axis=1)
#             # pulling_angle = np.nansum(data.angle_to_blue_part, axis=1)
#             # pulling_angle = np.nanmean(data.angle_to_blue_part, axis=1)
#             pulling_angle = np.nanmean(data.pulling_angle, axis=1)
#             # pulling_angle_sign = np.nansum(np.sign(data.angle_to_blue_part), axis=1)
#             spring_extension = np.nansum(data.spring_length, axis=1)
#             # extension_mul_sin_angle = np.nanmean(data.springs_length_processed * np.sin(data.pulling_angle), axis=1)
#             # logritmic_pulling_angle = ((np.abs(np.sin(data.pulling_angle))+1)**1)*np.sign(data.pulling_angle)
#             # extension_mul_sin_angle_log = np.nansum(np.abs(data.springs_length_processed) * logritmic_pulling_angle, axis=1)
#             # springs_wrenching = np.nansum(data.springs_wrenching, axis=1)
#             # extension_mul_sin_angle = data.springs_length * np.sin(data.angle_to_blue_part)
#             # force_applied = np.nansum(data.wrenching_force+extension_mul_sin_angle, axis=1)
#             #bullian vector for if there is a least one nan in the row, in the pulling angle
#             # nan_in_pulling_angle = np.any(np.isnan(data.angle_to_blue_part), axis=1)
#             df = pd.DataFrame({"total_n_ants": total_n_ants, "velocity": velocity, "velocity_spaced": velocity_spaced,
#                                "pulling_angle": pulling_angle, "springs_extension": spring_extension})
#             analysed[spring_type][video]["overall_behaviour"] = df
#             # analysed[spring_type][video]["extension_mul_sin_angle_matrix"] = np.abs(data.springs_length_processed) * np.sin(data.pulling_angle) * np.sign(data.springs_length_processed)
#             # insistence length - defined as the number of frames in which the ant is pulling the spring to the same direction.
#             # the length is calculated for each ant separately, and the direction is defined by the sign of the pulling angle for the last 25 frames
#             # insistence_length = np.zeros(data.N_ants_around_springs.shape)
#             # for ant in range(data.N_ants_around_springs.shape[1]):
#             #     insistence_length[:,ant] = get_insisence_length(data.pulling_angle[:,ant])
#             # df = pd.DataFrame({"insistence_length": np.nanmean(insistence_length, axis=1)})
#             # analysed[spring_type][video]["insistence_length"] = df
#             analysed[spring_type][video]["corrected_data"] = data
#     return analysed
#
# def get_persistence_length(pulling_angle,number_of_ants_array, window_size=25):
#     single_ant_events = pulling_angle[number_of_ants_array== 1]
#     #moving average over the pulling angle, for window size of 25 frames, over columns
#     moving_average_angle = np.convolve(single_ant_events, np.ones((window_size,))/window_size, mode='valid')
#     moving_mean_angle_sign = np.sign(moving_average_angle)
#     dilated = utils.column_dilation(moving_mean_angle_sign)
#     labeled, n_labels = utils.label(dilated)
#     # get the size of the labels
#     label_sizes = np.bincount(labeled.ravel())
#     eroded = utils.column_erosion(labeled)
#     # insistence length for the mean angle of rows
#     mean_pulling_angle = np.nanmean(pulling_angle, axis=1)
#     return eroded, label_sizes
#
# def median_by_bins(vec1, vec2):
#     bin_edges = np.linspace(vec1.min(), vec1.max(), 101)
#     bin_indices = np.digitize(vec1, bin_edges)
#     bin_medians = np.array([np.nanmedian(vec2[bin_indices == i]) for i in range(1, bin_edges.size)])
#     median_of_medians = np.nanmedian(bin_medians)
#     return median_of_medians
#
# def correct_bias(video_data):
#     video_data = copy.deepcopy(video_data)
#     rest_bool = video_data.N_ants_around_springs == 0
#     angle_to_nest_rest = copy.copy(video_data.angle_to_nest)
#     angle_to_nest_rest[np.invert(rest_bool)] = np.nan
#
#     rest_spring_length = copy.copy(video_data.spring_length)
#     rest_spring_length[np.invert(rest_bool)] = np.nan
#     pulling_angle_rest = copy.copy(video_data.pulling_angle)
#     pulling_angle_rest[np.invert(rest_bool)] = np.nan
#
#     blue_length_change = video_data.blue_length#/np.nanmean(video_data.blue_length)
#     df = pd.DataFrame({"blue_length_change": blue_length_change, "angle_to_nest": video_data.angle_to_nest[:,0]}).dropna()
#     #plot blue length change vs angle to nest
#     # import matplotlib.pyplot as plt
#     # plt.clf()
#     # plt.scatter(df["angle_to_nest"],df["blue_length_change"])
#     # plt.show()
#
#     blue_bias_equation = utils.deduce_bias_equation(df["angle_to_nest"], df["blue_length_change"])
#     # blue_bias_equation = utils.deduce_bias_equation(video_data.angle_to_nest[:,0], blue_length_change)
#     norm_length = utils.normalize(
#         video_data.spring_length,
#         video_data.angle_to_nest, blue_bias_equation)
#     rest_spring_length = copy.copy(norm_length)
#     rest_spring_length[np.invert(rest_bool)] = np.nan
#     rest_spring_length_median = np.nanmedian(rest_spring_length, axis=0)
#     video_data.spring_length = norm_length-rest_spring_length_median
#     # # plot the angle to the nest for the rest ants
#     # import matplotlib.pyplot as plt
#     # plt.figure()
#     # # plt.scatter(angle_to_nest_rest,pulling_angle_rest)
#     # plt.scatter(video_data.angle_to_nest[:,0],video_data.blue_length)
#     # plt.show()
#     for spring in range(20):
#         # springs lengths normalization
#         # rest_length_spring = rest_spring_length[:,spring]
#         # angle_to_nest_rest_spring = angle_to_nest_rest[:, spring]
#         # df = pd.DataFrame({"rest_length_spring": rest_length_spring, "angles_to_nest_rest": angle_to_nest_rest_spring}).dropna()
#         # bias_equation = utils.deduce_bias_equation(df["angles_to_nest_rest"], df["rest_length_spring"])
#         # norm_length = utils.normalize(
#         #     video_data.spring_length[:,spring],
#         #     video_data.angle_to_nest[:,spring], bias_equation)
#         # video_data.spring_length[:,spring] = norm_length
#
#         # springs angles normalization
#         pulling_angle_rest_spring = pulling_angle_rest[:, spring]
#         angles_to_nest_rest_spring = angle_to_nest_rest[:, spring]
#         df = pd.DataFrame({"pulling_angle_rest": pulling_angle_rest_spring,
#                            "angles_to_nest_rest": angles_to_nest_rest_spring}).dropna()
#         bias_equation = utils.deduce_bias_equation(df['angles_to_nest_rest'], df['pulling_angle_rest'])
#         norm_angle = utils.normalize(
#             video_data.pulling_angle[:,spring],
#             video_data.angle_to_nest[:,spring], bias_equation)
#         video_data.pulling_angle[:, spring] = norm_angle
#     # # springs wrenching force calculation
#     # pulling_angles = copy.copy(data_prepared_copy[spring_type][video].pulling_angle).flatten()
#     # lengths = copy.copy(data_prepared_copy[spring_type][video].springs_length_processed).flatten()
#     # pulling_angles[np.isnan(pulling_angles)] = 0
#     # lengths[np.isnan(lengths)] = 0
#     # max_length = np.median(np.sort(lengths)[-100:])
#     # max_pulling_angle = np.median(np.sort(pulling_angles)[-100:])
#     # wrenching_force = (data_prepared_copy[spring_type][video].pulling_angle/max_pulling_angle) * max_length
#     # data_prepared_copy[spring_type][video].wrenching_force = wrenching_force
#     return video_data
#
#
# def save_data(analysed, output_dir):
#
#     for spring_type in analysed:
#         for video in analysed[spring_type]:
#             output_dir = os.path.join(output_dir, spring_type, video,"normalized_data")
#             # create a folder for the data
#             if not os.path.exists(output_dir):
#                 os.makedirs(output_dir)
#             # save the data
#             arrays = [analysed[spring_type][video]["corrected_data"].angle_to_blue_part,
#                    analysed[spring_type][video]["corrected_data"].angle_to_nest,
#                    analysed[spring_type][video]["corrected_data"].blue_length,
#                    analysed[spring_type][video]["corrected_data"].spring_length,
#                       analysed[spring_type][video]["corrected_data"].pulling_angle,]
#             names = ["angle_to_blue_part", "angle_to_nest", "blue_length", "spring_length", "pulling_angle"]
#             for array,name in zip(arrays,names):
#                 #save numpy array as csv
#                 np.savetxt(os.path.join(output_dir, name+ ".csv"), array, delimiter=",")

if __name__ == "__main__":
    dir = "Z:/Dor_Gabay/ThesisProject/data/test3/"
    analysed = DataAnalyser(dir)
    from data_analysis.plots import plot_overall_behavior
    plot_overall_behavior(analysed, start=0, end=None, window_size=50)

