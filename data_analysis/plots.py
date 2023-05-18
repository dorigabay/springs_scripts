import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_overall_behavior(analysed, start=0, end=None, window_size=1, title="", output_dir=None):
    plt.clf()
    if end is None:
        end = analysed.pulling_angle.shape[0]
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

    x = np.linspace(start, end, end - start)
    angular_velocity = analysed.angular_velocity[start:end]
    angular_velocity = pd.Series(angular_velocity).rolling(window_size).median()
    ax1.plot(x, angular_velocity, label="moving median",color=velocity_color)
    ax1.set_xlabel("time (frames)")
    ax1.set_ylabel("velocity (rad/ 10_frames)", color=velocity_color)
    ax1.tick_params(axis="y", labelcolor=velocity_color)

    # plot the mean pulling angles of the springs
    fig.suptitle(f"angular_velocity (moving median) VS net_force (moving mean) (movie:{title})")
    ax2 = ax1.twinx()
    # net_force = analysed.net_force[start:end]
    net_force = analysed.net_magnitude[start:end]
    net_force = pd.Series(net_force).rolling(window_size).median()
    # moving_averages = np.convolve(y, np.ones((window_size,)) / window_size, mode='valid')
    ax2.plot(x, net_force, color=pulling_angle_color)
    # ax2.plot(x[window_size - 1:], net_force_moving_median, color=pulling_angle_color)
    ax2.set_ylabel("net_force (rad/ frame)", color=pulling_angle_color)
    ax2.tick_params(axis="y", labelcolor=pulling_angle_color)

    # add a line at y=0
    ax2.axhline(y=0, color="black", linestyle="--")
    ax1.axhline(y=0, color="black", linestyle="--")

    # set the y axis to have y=0 at the same place for both plots
    ax1.set_ylim(np.nanmax(angular_velocity) * -1.1, np.nanmax(angular_velocity) * 1.1)
    ax2.set_ylim(np.nanmax(analysed.net_force[start:end]) * -1.1, np.nanmax(analysed.net_force[start:end]) * 1.1)

    fig.tight_layout()
    #fig size should be 1920x1080
    fig.set_size_inches(19.2, 10.8)
    # plt.show()
    if output_dir is not None:
        import os
        print("Saving figure to path: ", os.path.join(output_dir, f"angular_velocity VS mean_springs_extension (movie {title}).png"))
        plt.savefig(os.path.join(output_dir, f"{title}.png"))

def plot_pulling_angle_over_angle_to_nest(analysed, start=0, end=None, title=""):
    # plot a scatter plot of the pulling angle over the angle to nest direction
    plt.close()
    plt.clf()
    if end is None:
        end = analysed.pulling_angle.shape[0]
    plt.scatter(analysed.pulling_angle[start:end], analysed.angle_to_nest[start:end])
    plt.title("pulling angle VS angle to nest direction")
    plt.pause(1)

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
