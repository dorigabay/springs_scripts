# from data_analysis import utils
from data_analysis import data_preparation,plots,utils
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_outliers(array, threshold=1):
    # get the outliers from an array
    # outliers are defined as values that are more than percent_threshold away from the median
    outliers = np.where(np.abs(array) > 0.01)[0]
    print("outliers: ", outliers)
    return outliers

def remove_outliers(data_analysed, threshold=1):
    # get the outliers frame numbers from the concatenated data,
    # and remove them from the data_prepared, so that they won't be used in the analysis
    for spring_type in data_analysed:
        for video in data_analysed[spring_type]:
            velocity_data = data_analysed[spring_type][video]["velocity"]
            outliers = get_outliers(velocity_data)
            data_analysed[spring_type][video].iloc[outliers,:] = np.nan
    return data_analysed

def analyse_data(data_prepared):
    # analyse the data, and return the results
    analysed = {}
    for spring_type in data_prepared:
        analysed[spring_type] = {}
        for video in data_prepared[spring_type]:
            analysed[spring_type][video] = {}
            data = data_prepared[spring_type][video]
            total_n_ants = np.nansum(data.N_ants_around_springs, axis=1)
            velocity = np.nanmedian(utils.calc_angular_velocity(data.springs_angles_to_nest,diff_spacing=1), axis=1)
            velocity_spaced = np.nanmedian(utils.calc_angular_velocity(data.springs_angles_to_nest,diff_spacing=10), axis=1)
            pulling_angle = np.nanmean(data.springs_angles_to_object, axis=1)
            df = pd.DataFrame({"total_n_ants": total_n_ants, "velocity": velocity, "velocity_spaced": velocity_spaced, "pulling_angle": pulling_angle})
            analysed[spring_type][video] = df
    return analysed

def concat_data(data_analysed):
    # concatenate the data from all the videos
    data_concatenated = {}
    for spring_type in data_analysed:
        data_concatenated[spring_type] = pd.concat(data_analysed[spring_type])
        data_concatenated[spring_type].reset_index(inplace=True)
    return data_concatenated
# def concatenate_data(data_prepared):
#     # concatenate all the data from the different videos, by spring type
#     concatenated = {}
#     for spring_type in data_prepared:
#         concatenated[spring_type] = pd.DataFrame()
#         for count, video_data in enumerate(data_prepared[spring_type]):
#             data = data_prepared[spring_type][video_data]
#             # parameters to concatenate:
#             total_n_ants = np.nansum(data.N_ants_around_springs, axis=1)
#             velocity = np.nanmedian(utils.calc_angular_velocity(data.springs_angles_to_nest,diff_spacing=1), axis=1)
#             velocity_spaced = np.nanmedian(utils.calc_angular_velocity(data.springs_angles_to_nest,diff_spacing=10), axis=1)
#             # velocity = (np.nanmedian(data.angular_velocity_nest, axis=1))
#             # velocity_spaced = (np.nanmedian(data.angular_velocity_nest_spaced, axis=1))
#             pulling_angle = np.mean(data.springs_angles_to_object, axis=1)
#             if count == 0:
#                 concatenated[spring_type] = pd.DataFrame({"total_n_ants": total_n_ants,"velocity":velocity,
#                     "velocity_spaced": velocity_spaced,"pulling_angle": pulling_angle})
#             else:
#                 concatenated[spring_type] = pd.concat([concatenated[spring_type],pd.DataFrame({"total_n_ants": total_n_ants,"velocity":velocity,
#                     "velocity_spaced": velocity_spaced,"pulling_angle": pulling_angle})])
#             if count == len(data_prepared[spring_type])-1:
#                 concatenated[spring_type] = concatenated[spring_type].reset_index()
#     return concatenated


spring_types_directories = (utils.iter_folder("Z:/Dor_Gabay/ThesisProject/data/videos_analysis_data2/"))
data_prepared = data_preparation.prepare_multiple(spring_types_directories)
analysed_data = analyse_data(data_prepared)

# analysed_data = remove_outliers(analysed_data,2)
concatenated_data = concat_data(analysed_data)

df = pd.DataFrame(analysed_data["plus0_force"]["S5200009"])
plots.plot_velocity_moving_average(df, start=0, end=None)