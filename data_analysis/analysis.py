import pickle
from data_analysis import data_preparation,plots,utils
import numpy as np
import pandas as pd
import os
import copy

# def get_outliers(array, threshold=1):
#     # get the outliers from an array
#     # outliers are defined as values that are more than percent_threshold away from the median
#     outliers = np.where(np.abs(array) > 0.01)[0]
#     print("outliers: ", outliers)
#     return outliers
#
# def remove_outliers(data_analysed, threshold=1):
#     # get the outliers frame numbers from the concatenated data,
#     # and remove them from the data_prepared, so that they won't be used in the analysis
#     for spring_type in data_analysed:
#         for video in data_analysed[spring_type]:
#             velocity_data = data_analysed[spring_type][video]["velocity"]
#             outliers = get_outliers(velocity_data)
#             data_analysed[spring_type][video].iloc[outliers,:] = np.nan
#     return data_analysed


def analyse_data(data_prepared):
    # analyse the data, and return the results
    analysed = {}
    for spring_type in data_prepared:
        analysed[spring_type] = {}
        for video in data_prepared[spring_type]:
            analysed[spring_type][video] = {}
            data = copy.deepcopy(data_prepared[spring_type][video])
            total_n_ants = np.nansum(data.N_ants_around_springs, axis=1)
            velocity = np.nanmedian(utils.calc_angular_velocity(data.angles_to_nest,diff_spacing=1), axis=1)
            velocity_spaced = np.nanmedian(utils.calc_angular_velocity(data.angles_to_nest,diff_spacing=20), axis=1)
            pulling_angle = np.nansum(data.pulling_angle, axis=1)
            pulling_angle_sign = np.nansum(np.sign(data.pulling_angle), axis=1)
            springs_extension = np.nansum(data.springs_length_processed, axis=1)
            # extension_mul_sin_angle = np.nanmean(data.springs_length_processed * np.sin(data.pulling_angle), axis=1)
            # logritmic_pulling_angle = ((np.abs(np.sin(data.pulling_angle))+1)**1)*np.sign(data.pulling_angle)
            # extension_mul_sin_angle_log = np.nansum(np.abs(data.springs_length_processed) * logritmic_pulling_angle, axis=1)
            # springs_wrenching = np.nansum(data.springs_wrenching, axis=1)
            extension_mul_sin_angle = data.springs_length_processed * np.sin(data.pulling_angle)
            force_applied = np.nansum(data.wrenching_force+extension_mul_sin_angle, axis=1)
            #bullian vector for if there is a least one nan in the row, in the pulling angle
            nan_in_pulling_angle = np.any(np.isnan(data.pulling_angle), axis=1)
            df = pd.DataFrame({"total_n_ants": total_n_ants, "velocity": velocity, "velocity_spaced": velocity_spaced,
                               "pulling_angle": pulling_angle, "springs_extension": springs_extension,
                               "pulling_angle_sign": pulling_angle_sign,"force_applied": force_applied})
            analysed[spring_type][video]["overall_behaviour"] = df
            analysed[spring_type][video]["extension_mul_sin_angle_matrix"] = np.abs(data.springs_length_processed) * np.sin(data.pulling_angle) * np.sign(data.springs_length_processed)
            # insistence length - defined as the number of frames in which the ant is pulling the spring to the same direction.
            # the length is calculated for each ant separately, and the direction is defined by the sign of the pulling angle for the last 25 frames
            # insistence_length = np.zeros(data.N_ants_around_springs.shape)
            # for ant in range(data.N_ants_around_springs.shape[1]):
            #     insistence_length[:,ant] = get_insisence_length(data.pulling_angle[:,ant])
            # df = pd.DataFrame({"insistence_length": np.nanmean(insistence_length, axis=1)})
            # analysed[spring_type][video]["insistence_length"] = df

    return analysed

def get_insisence_length(pulling_angle,number_of_ants_array, window_size=25):
    single_ant_events = pulling_angle[number_of_ants_array== 1]
    #moving average over the pulling angle, for window size of 25 frames, over columns
    moving_average_angle = np.convolve(single_ant_events, np.ones((window_size,))/window_size, mode='valid')
    moving_mean_angle_sign = np.sign(moving_average_angle)
    dilated = utils.column_dilation(moving_mean_angle_sign)
    labeled, n_labels = utils.label(dilated)
    # get the size of the labels
    label_sizes = np.bincount(labeled.ravel())
    eroded = utils.column_erosion(labeled)
    # insistence length for the mean angle of rows
    mean_pulling_angle = np.nanmean(pulling_angle, axis=1)
    return eroded, label_sizes

def median_by_bins(vec1, vec2):
    bin_edges = np.linspace(vec1.min(), vec1.max(), 101)
    bin_indices = np.digitize(vec1, bin_edges)
    bin_medians = np.array([np.nanmedian(vec2[bin_indices == i]) for i in range(1, bin_edges.size)])
    median_of_medians = np.nanmedian(bin_medians)
    return median_of_medians

def correct_lengths_bias(data_prepared):
    data_prepared_copy = copy.deepcopy(data_prepared)
    for spring_type in data_prepared:
        for video in data_prepared[spring_type]:
            data = data_prepared[spring_type][video]
            rest_bool = data.N_ants_around_springs == 0
            angles_to_nest_rest = copy.copy(data.angles_to_nest)
            angles_to_nest_rest[np.invert(rest_bool)] = np.nan

            rest_springs_length = copy.copy(data.springs_length_processed)
            rest_springs_length[np.invert(rest_bool)] = np.nan
            pulling_angle_rest = copy.copy(data.pulling_angle)
            pulling_angle_rest[np.invert(rest_bool)] = np.nan

            data_prepared_copy[spring_type][video].wrenching_force = np.zeros(data.springs_length_processed.shape)
            for spring in range(20):

                # springs lengths normalization
                rest_length_spring = rest_springs_length[:,spring]
                angles_to_nest_rest_spring = angles_to_nest_rest[:, spring]
                df = pd.DataFrame({"rest_length_spring": rest_length_spring, "angles_to_nest_rest": angles_to_nest_rest_spring}).dropna()
                bias_equation = utils.deduce_bias_equation(df["angles_to_nest_rest"], df["rest_length_spring"])
                norm_length = utils.normalize(
                    data_prepared_copy[spring_type][video].springs_length_processed[:,spring],
                    data_prepared_copy[spring_type][video].angles_to_nest[:,spring], bias_equation)
                data_prepared_copy[spring_type][video].springs_length_processed[:,spring] = norm_length

                # springs angles normalization
                pulling_angle_rest_spring = pulling_angle_rest[:, spring]
                angles_to_nest_rest_spring = angles_to_nest_rest[:, spring]
                df = pd.DataFrame({"pulling_angle_rest": pulling_angle_rest_spring,
                                   "angles_to_nest_rest": angles_to_nest_rest_spring}).dropna()
                # median = median_by_bins(df['angles_to_nest_rest'], df['pulling_angle_rest'])
                bias_equation = utils.deduce_bias_equation(df['angles_to_nest_rest'], df['pulling_angle_rest'])
                norm_angle = utils.normalize(
                    data_prepared_copy[spring_type][video].pulling_angle[:,spring],
                    data_prepared_copy[spring_type][video].angles_to_nest[:,spring], bias_equation)
                data_prepared_copy[spring_type][video].pulling_angle[:, spring] = norm_angle

            # springs wrenching force calculation
            pulling_angles = copy.copy(data_prepared_copy[spring_type][video].pulling_angle).flatten()
            lengths = copy.copy(data_prepared_copy[spring_type][video].springs_length_processed).flatten()
            pulling_angles[np.isnan(pulling_angles)] = 0
            lengths[np.isnan(lengths)] = 0
            max_length = np.median(np.sort(lengths)[-100:])
            max_pulling_angle = np.median(np.sort(pulling_angles)[-100:])
            wrenching_force = (data_prepared_copy[spring_type][video].pulling_angle/max_pulling_angle) * max_length
            data_prepared_copy[spring_type][video].wrenching_force = wrenching_force
    return data_prepared_copy


def save_data(analysed, prepared, path):
    # create a folder for the data
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path,"analysed_data.pkl"), 'wb') as f:
        pickle.dump(analysed, f)
    with open(os.path.join(path,"prepared_data.pkl"), 'wb') as f:
        pickle.dump(prepared, f)

if __name__ == "__main__":
    # spring_types_directories = (utils.iter_folder("Z:/Dor_Gabay/ThesisProject/data/test6/"))
    # data_prepared = data_preparation.prepare_multiple(spring_types_directories)
    data_prepared_normalized = correct_lengths_bias(data_prepared)
    analysed_data = analyse_data(data_prepared_normalized)
    # save_data(analysed_data,data_prepared_normalized, "Z:/Dor_Gabay/ThesisProject/data/test6/pickle_files/")

