import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_overall_behavior(analysed, start=0, end=None, window_size=1, title=""):
    title = analysed.video_name
    # plt.close()
    plt.clf()
    # df = concatenated[spring_type][video_name]
    if end is None:
        end = analysed.pulling_angle.shape[0]
    # set figure and title
    fig, ax1 = plt.subplots()
    # set a color by RGB
    red = np.array((239, 59, 46))/255
    purple = np.array((112, 64, 215))/255
    blue = np.array((86,64,213))/255
    green = np.array((93, 191, 71))/255
    purple_brown = np.array((153,86,107))/255

    velocity_color = red
    total_n_ants_color = purple
    pulling_angle_color = blue
    springs_extension_color = green
    springs_extension_normed_color = purple_brown

    # plot velocity
    # y = df["velocity"][start:end]
    y = analysed.velocity_spaced[start:end]
    x = np.linspace(start, end, end - start)
    # plot the moving median of the velocity
    moving_median = pd.Series(y).rolling(window_size).median()
    # moving_averages = np.convolve(y, np.ones((window_size,)) / window_size, mode='valid')
    # ax1.plot(df["velocity"][start:end], color="red")
    # ax1.plot(df["velocity_spaced"][start:end], color="red")
    ax1.plot(x, moving_median, label="moving median",color=velocity_color)
    # ax1.plot(x[window_size - 1:], moving_averages, color="red")
    ax1.set_xlabel("time (frames)")
    ax1.set_ylabel("velocity (rad/ 10_frames)", color=velocity_color)
    ax1.tick_params(axis="y", labelcolor=velocity_color)


    # plot total number of ants holding the object
    # fig.suptitle(f"velocity (moving median) VS total_number_of_ants (moving mean) (movie:{title})")
    # ax2 = ax1.twinx()
    # y = df["total_n_ants"][start:end]
    # moving_averages = np.convolve(y, np.ones((window_size,)) / window_size, mode='valid')
    # ax2.plot(x[window_size - 1:], moving_averages, color=total_n_ants_color)
    # ax2.set_ylabel("total_n_ants", color=total_n_ants_color)
    # ax2.tick_params(axis="y", labelcolor=total_n_ants_color)

    # plot the mean pulling angles of the springs
    fig.suptitle(f"angular_velocity (moving median) VS mean_springs_pulling_angles (moving mean) (movie:{title})")
    ax2 = ax1.twinx()
    y = analysed.pulling_angle[start:end]
    moving_averages = np.convolve(y, np.ones((window_size,)) / window_size, mode='valid')
    ax2.plot(x[window_size - 1:], moving_averages, color=pulling_angle_color)
    ax2.set_ylabel("pulling_angle (rad/ frame)", color=pulling_angle_color)
    ax2.tick_params(axis="y", labelcolor=pulling_angle_color)

    #plot the mean pulling forces of the springs
    # fig.suptitle(f"angular_velocity (moving median) VS mean_springs_extension (moving mean) (movie:{title})")
    # ax2 = ax1.twinx()
    # y = df["springs_extension"][start:end]
    # moving_averages = np.convolve(y, np.ones((window_size,)) / window_size, mode='valid')
    # ax2.plot(x[window_size - 1:], moving_averages, color=springs_extension_color)
    # ax2.set_ylabel("springs_extension (number of pixels)", color=springs_extension_color)
    # ax2.tick_params(axis="y", labelcolor=springs_extension_color)

    # plot the mean extension of the springs times cos pulling angle
    # fig.suptitle(f"angular_velocity (moving median) VS mean_springs_extension_normed_to_angle (moving mean) (movie:{title})")
    # ax2 = ax1.twinx()
    # y = df["extension_mul_sin_angle"][start:end]
    # moving_averages = np.convolve(y, np.ones((window_size,)) / window_size, mode='valid')
    # ax2.plot(x[window_size - 1:], moving_averages, color=springs_extension_normed_color)
    # ax2.set_ylabel("extension_normed_to_pulling_angle (rad/ frame)", color=springs_extension_normed_color)
    # ax2.tick_params(axis="y", labelcolor=springs_extension_normed_color)

    # plot the sum of forces of the springs
    # fig.suptitle(f"angular_velocity (moving median) VS force_applied (moving mean) (movie:{title})")
    # ax2 = ax1.twinx()
    # y = df["force_applied"][start:end]
    # moving_averages = np.convolve(y, np.ones((window_size,)) / window_size, mode='valid')
    # ax2.plot(x[window_size - 1:], moving_averages, color=springs_extension_normed_color)
    # ax2.set_ylabel("extension_normed_to_pulling_angle (rad/ frame)", color=springs_extension_normed_color)
    # ax2.tick_params(axis="y", labelcolor=springs_extension_normed_color)


    # ax2 = ax1.twinx()
    # y = df["spring12"][start:end]
    # moving_averages = np.convolve(y, np.ones((window_size,)) / window_size, mode='valid')
    # ax2.plot(x[window_size - 1:], moving_averages, color="green")

    # plot the mean extension of the springs times cos pulling angle
    # fig.suptitle(f"angular_velocity (moving median) VS sum_pulling_angle_sign (moving mean) (movie:{title})")
    # ax2 = ax1.twinx()
    # y = df["pulling_angle_sign"][start:end]
    # moving_averages = np.convolve(y, np.ones((window_size,)) / window_size, mode='valid')
    # ax2.plot(x[window_size - 1:], moving_averages, color=springs_extension_normed_color)
    # ax2.set_ylabel("pulling_angle_sign (rad/ frame)", color=springs_extension_normed_color)
    # ax2.tick_params(axis="y", labelcolor=springs_extension_normed_color)


    #add a line at y=0
    ax2.axhline(y=0, color="black", linestyle="--")
    ax1.axhline(y=0, color="black", linestyle="--")


    # limit the x axis to 0-30000
    ax1.set_xlim(0, 30000)
    ax2.set_xlim(0, 30000)

    # plot a line - red if True, green if False in df["nan_in_pulling_angle"], put the line at y=0
    # put the folowing plot on top of all the other plots
    # ax3 = ax1.twinx()
    # color = df["nan_in_pulling_angle"][start:end].astype(int)
    # y = np.linspace(0, 0, len(x))
    # ax3.scatter(x, y, c=color, cmap="RdYlGn", s=10)

    fig.tight_layout()
    plt.pause(1)

# def correlation_rest_lengths_over_nest_direction(df, start=0, end=None, title=""):
#     # plot a correlation matrix of the rest lengths for angles to nest direction
#     plt.close()
#     plt.clf()
#     if end is None:
#         end = df.shape[0]
#     df = df[start:end]
#     corr_matrix = df.corr()
#     corr_matrix = corr_matrix[corr_matrix.columns[0:8]]
#     corr_matrix = corr_matrix.loc[corr_matrix.index[0:8]]
#     sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
#     plt.title("correlation matrix of rest lengths for angles to nest direction"+title)
#     plt.pause(1)

def correlation_rest_lengths_over_nest_direction(df, start=0, end=None, title=""):
    # plot scatter plots of the rest lengths for angles to nest direction
    plt.close()
    plt.clf()
    if end is None:
        end = df.shape[0]
    df = df[start:end]
    plt.scatter(df["rest_length"], df["rest_angle"])
    # fig.suptitle("correlation matrix of rest lengths for angles to nest direction"+title)
    # axs.scatter(df["rest_length"], df["rest_length"])
    plt.pause(1)

def histogram_pulling_angles(array):
    # plot a histogram of the pulling angles
    plt.clf()
    array = array.flatten()
    plt.hist(array, bins=10)
    plt.title("pulling angle histogram")
    plt.pause(1)



if __name__ == "__main__":
    # %matplotlib qt
    pass
