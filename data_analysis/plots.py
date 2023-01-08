import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# def plot_velocity_moving_average(concatenated, spring_type,video_name, start=0, end=None, window_size=500):
#     plt.close()
#     plt.clf()
#     df = concatenated[spring_type][video_name]
#     if end is None:
#         end = df.shape[0]
#     fig, ax1 = plt.subplots()
#     ax1.plot(df["velocity"][start:end], color="red")
#     ax1.set_xlabel("time (frames)")
#     ax1.set_ylabel("velocity (rad/s)", color="red")
#     ax1.tick_params(axis="y", labelcolor="red")
#
#     ax2 = ax1.twinx()
#     x = np.linspace(start, end, end - start)
#     y = df["total_n_ants"][start:end]
#     moving_averages = np.convolve(y, np.ones((window_size,)) / window_size, mode='valid')
#     # ax2.scatter(x[window_size - 1:], y[window_size - 1:], color="blue", alpha=0.3)
#     ax2.plot(x[window_size - 1:], moving_averages, color="blue")
#     ax2.set_ylabel("total_n_ants", color="blue")
#     ax2.tick_params(axis="y", labelcolor="blue")
#     fig.title = "velocity and total_n_ants over time"
#     fig.tight_layout()
#     plt.pause(1)



def plot_velocity_moving_average(df, start=0, end=None, window_size=200):
    # plt.close()
    plt.clf()
    # df = concatenated[spring_type][video_name]
    if end is None:
        end = df.shape[0]
    fig, ax1 = plt.subplots()
    ax1.plot(df["velocity"][start:end], color="red")
    ax1.set_xlabel("time (frames)")
    ax1.set_ylabel("velocity (rad/s)", color="red")
    ax1.tick_params(axis="y", labelcolor="red")

    ax2 = ax1.twinx()
    x = np.linspace(start, end, end - start)
    y = df["total_n_ants"][start:end]
    y[np.isnan(y)] = 0
    moving_averages = np.convolve(y, np.ones((window_size,)) / window_size, mode='valid')
    # ax2.scatter(x[window_size - 1:], y[window_size - 1:], color="blue", alpha=0.3)
    ax2.plot(x[window_size - 1:], moving_averages, color="blue")
    ax2.set_ylabel("total_n_ants", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")
    fig.title = "velocity and total_n_ants over time"
    fig.tight_layout()
    plt.pause(1)

if __name__ == "__main__":
    # %matplotlib qt
    df = pd.DataFrame(analysed_data["plus0_force"]["S5200009"])
    # for key in concatenated_data.keys():
    #     df = pd.DataFrame(concatenated_data[key])
    #     plot_velocity_moving_average(df, start=0, end=None, window_size=200)
    # df = pd.DataFrame(concatenated_data["plus0.1_force"])
    plot_velocity_moving_average(df, start=0, end=None)


