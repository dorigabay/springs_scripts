from data_analysis import utils
from data_analysis import data_preparation
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# def sum_n_lines(array,n_lines=10):
#     summed_array = np.abs(array.copy()[list(range(0,array.shape[0],n_lines)),:])
#     for n in range(1,n_lines):
#         add = np.abs(array[list(range(n,array.shape[0],n_lines)),:])
#         summed_array[0:add.shape[0],:] += add
#     return summed_array



def concatenate_data(data_prepared):
    # concatenate all the data from the different videos, by spring type
    concatenated = {}
    for spring_type in data_prepared:
        concatenated[spring_type] = pd.DataFrame()
        for count, video_data in enumerate(data_prepared[spring_type]):
            data = data_prepared[spring_type][video_data]
            # parameters to concatenate:
            total_n_ants = np.nansum(data.N_ants_around_springs, axis=1)
            velocity = np.nanmedian(utils.calc_angular_velocity(data.springs_angles_to_nest,diff_spacing=1), axis=1)
            velocity_spaced = np.nanmedian(utils.calc_angular_velocity(data.springs_angles_to_nest,diff_spacing=10), axis=1)
            # velocity = (np.nanmedian(data.angular_velocity_nest, axis=1))
            # velocity_spaced = (np.nanmedian(data.angular_velocity_nest_spaced, axis=1))
            pulling_angle = np.mean(data.springs_angles_to_object, axis=1)
            if count == 0:
                concatenated[spring_type] = pd.DataFrame({"total_n_ants": total_n_ants,"velocity":velocity,
                    "velocity_spaced": velocity_spaced,"pulling_angle": pulling_angle})
            else:
                concatenated[spring_type] = pd.concat([concatenated[spring_type],pd.DataFrame({"total_n_ants": total_n_ants,"velocity":velocity,
                    "velocity_spaced": velocity_spaced,"pulling_angle": pulling_angle})])
            if count == len(data_prepared[spring_type])-1:
                concatenated[spring_type] = concatenated[spring_type].reset_index()
    return concatenated

# spring_types_directories = (utils.iter_folder("Z:/Dor_Gabay/ThesisProject/data/videos_analysis_data/18.9.22/"))
# data_prepared = data_preparation.prepare_multiple(spring_types_directories)
concatenated = concatenate_data(data_prepared)

dat = concatenated["plus0.5mm_force"]

def scatter_plot_with_moving_average(values, window_size):
  # Compute the moving average of the values
  moving_averages = np.convolve(values, np.ones((window_size,))/window_size, mode='valid')

  # Generate x values for the plot
  x_values = range(len(moving_averages))

  # Create the scatter plot
  plt.scatter(x_values, values[window_size-1:])

  # Add the moving average line
  plt.plot(x_values, moving_averages)

  # Show the plot
  plt.show()

def plot_velocity_moving_average(concatenated, spring_type, start=0, end=1000, window_size=500):
    plt.close()
    plt.clf()
    df = concatenated[spring_type]

    fig, ax1 = plt.subplots()
    ax1.plot(df["velocity"][start:end], color="red")
    ax1.set_xlabel("time (frames)")
    ax1.set_ylabel("velocity (rad/s)", color="red")
    ax1.tick_params(axis="y", labelcolor="red")

    ax2 = ax1.twinx()
    x = np.linspace(start, end, end - start)
    y = df["total_n_ants"][start:end]
    moving_averages = np.convolve(y, np.ones((window_size,)) / window_size, mode='valid')
    # ax2.scatter(x[window_size - 1:], y[window_size - 1:], color="blue", alpha=0.3)
    ax2.plot(x[window_size - 1:], moving_averages, color="blue")
    ax2.set_ylabel("total_n_ants", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    fig.tight_layout()
    plt.pause(1)


plot_velocity_moving_average(concatenated, "plus0.5mm_force", start=0, end=22000)


