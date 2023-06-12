import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy import stats


def define_colors(labels_num=5):
    # set a color by RGB
    red = np.array((239, 59, 46))/255
    purple = np.array((112, 64, 215))/255
    blue = np.array((86,64,213))/255
    green = np.array((93, 191, 71))/255
    purple_brown = np.array((153,86,107))/255
    colors = [red, purple, blue, green, purple_brown]
    return colors[:labels_num+1]


def define_output_dir(output_dir):
    if output_dir is not None:
        output_dir = os.path.join(output_dir, "plots")
        os.makedirs(output_dir, exist_ok=True)
    else:
        raise ValueError("Please provide an output directory")
    return output_dir


def plot_angular_velocity_distribution(analysed, start=0, end=None, window_size=1, title="", output_dir=None):
    plt.clf()
    if end is None:
        end = analysed.angular_velocity.shape[0]

    x = np.abs(analysed.angular_velocity[start:end])
    x_nans = np.isnan(x)
    x = x[~x_nans]
    sns.displot(x, kind="kde", fill=True, cmap="mako")
    plt.title(f"angular velocity distribution (movie:{title})")
    plt.xlabel("angular velocity (rad/ 20_frames)")
    plt.ylabel("density")
    # leave enough margin for the title
    plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
    # save the figure
    output_dir = define_output_dir(output_dir)
    print("Saving figure to path: ", os.path.join(output_dir, f"angular velocity distribution (movie {title}).png"))
    plt.savefig(os.path.join(output_dir, f"angular velocity distribution (movie {title}).png"))

# plot_angular_velocity_distribution(analysed, start=0, end=None, window_size=50, title=video_name, output_dir=directory)


def plot_ant_profiles(analysed, window_size=1, title="", output_dir=None):
    # output_dir = define_output_dir(output_dir)

    #plot number of cases
    plt.clf()
    y = np.sum(analysed.profiles_start_with_one_ant, axis=0)
    y = y[y>50]
    x = np.arange(0, y.shape[0], 1)
    plt.plot(x, y, color="purple")
    plt.ylabel(f"number of profiles")
    plt.xlabel("frame")
    print("Saving figure to path: ", os.path.join(output_dir, f"number of profiles (movie {title}).png"))
    plt.savefig(os.path.join(output_dir, f"number of profiles (movie {title}).png"))

    force_magnitude_copy = np.copy(analysed.all_profiles_force_magnitude)
    force = np.abs(analysed.all_profiles_force_magnitude * np.sin(analysed.all_profiles_force_direction))
    for y_ori, y_title in zip([force_magnitude_copy, force],["force magnitude (mN)", "angular force (mN)"]):
        for attachment in ["first attachment", "all but first attachment"]:
            if attachment == "first attachment":
                bool = np.copy(analysed.profiles_start_with_one_ant)
                bool[analysed.all_profiles_precedence != 1, :] = False
                y = np.copy(y_ori)
            else:
                bool = np.copy(analysed.profiles_start_with_one_ant)
                bool[analysed.all_profiles_precedence == 1, :] = False
                y = np.copy(y_ori)
            bool[~bool[:, 200], :] = False
            bool[:, 201:] = False
            y[~bool] = np.nan
            y[y==0] = np.nan
            y_mean = np.nanmean(y, axis=0)
            y_SEM_upper = y_mean + np.nanstd(y, axis=0) / np.sqrt(np.sum(~np.isnan(y), axis=0))
            y_SEM_lower = y_mean - np.nanstd(y, axis=0) / np.sqrt(np.sum(~np.isnan(y), axis=0))
            y_not_nan = ~np.isnan(y_mean)
            y_mean,y_SEM_upper,y_SEM_lower = y_mean[y_not_nan],y_SEM_upper[y_not_nan],y_SEM_lower[y_not_nan]
            x = np.arange(0, y_mean.shape[0], 1)
            plt.clf()
            plt.plot(x, y_mean, color="purple")
            plt.fill_between(x, y_SEM_lower, y_SEM_upper, alpha=0.5, color="orange")
            plt.title(f"ant {y_title} profiles - {attachment} (movie:{title})")
            plt.xlabel("frame")
            plt.ylabel(f"{y_title} (mN)")
            plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
            print("Saving figure to path: ", os.path.join(output_dir, f"ant {y_title} profiles - {attachment} (movie {title}).png"))
            plt.savefig(os.path.join(output_dir, f"ant {y_title} profiles - {attachment} (movie {title}).png"))
# plot_ant_profiles(analysed, window_size=50, title=spring_type, output_dir=output_dir)

# def plot_profiles_length_distrubution(analysed, window_size=1, title="", output_dir=None):
#     for attachment in ["first attachment", "all but first attachment"]:
#         if attachment == "first attachment":
#             bool = np.copy(analysed.profiles_start_with_one_ant)
#             bool[analysed.all_profiles_precedence != 1, :] = False
#         else:
#             bool = np.copy(analysed.profiles_start_with_one_ant)
#             bool[analysed.all_profiles_precedence == 1, :] = False
#
#
#         x[~(bool.any(axis=1))] = np.nan
#         x = x[~np.isnan(x)]
#
#         quantile = np.quantile(x, 0.90)
#         x = x[x<quantile]
#
#
#         y = np.sum(~np.isnan(analysed.all_profiles_angular_velocity), axis=1).astype(float)
#         x = analysed.all_profiles_precedence
#
#         plt.plot(y=analysed.all_profiles_precedence, x=np.sum(~np.isnan(analysed.all_profiles_angular_velocity), axis=1).astype(float), color="purple")
#
#         #plot distribution without kde
#         plt.clf()
#         sns.displot(x, kind="hist", bins=100, kde=False)
#         plt.title(f"ant angular velocity profiles length distribution - {attachment} (movie:{title})")
#         plt.xlabel("number of frames")
#         plt.ylabel("density")
#         plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
#         print("Saving figure to path: ", os.path.join(output_dir, f"ant angular velocity profiles length distribution - {attachment} (movie {title}).png"))
#         plt.savefig(os.path.join(output_dir, f"ant angular velocity profiles length distribution - {attachment} (movie {title}).png"))
# # plot_profiles_length_distrubution(analysed, title=spring_type, output_dir=output_dir)


def plot_query_information_to_attachment_length(analysed, start=0, end=None, window_size=1, title="", output_dir=None):
    x_velocity = np.abs(analysed.all_profiles_angular_velocity)
    x_N_ants = analysed.all_profiles_N_ants_around_springs_sum
    for x_ori, x_name in zip([x_velocity, x_N_ants], ["profile velocity mean", "mean number of ants"]):
        for attachment in ["1st", "2st&higher"]:
            if attachment == "1st":
                bool = np.copy(analysed.profiles_start_with_one_ant)
                bool[analysed.all_profiles_precedence != 1, :] = False
                x = np.copy(x_ori)
            else:
                bool = np.copy(analysed.profiles_start_with_one_ant)
                bool[analysed.all_profiles_precedence == 1, :] = False
                x = np.copy(x_ori)

            x[~bool] = np.nan
            y = np.sum(~np.isnan(x), axis=1).astype(float)
            x = np.nanmean(x, axis=1)
            argsort_y = np.argsort(y)
            x = x[argsort_y]
            y = y[argsort_y]

            quantile = np.quantile(y, 0.99)
            under_quantile = y < quantile
            x = x[under_quantile]
            y = y[under_quantile]
            quantile = np.nanquantile(x, 0.99)
            under_quantile = x < quantile
            x = x[under_quantile]
            y = y[under_quantile]
            #plot
            plt.clf()
            plt.scatter(x, y/50, color="purple", s=1)
            plt.title(f"{x_name} to attachment time- {attachment}")
            plt.xlabel(f"{x_name}")
            plt.ylabel(f"persistence time (s)")
            plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
            print("Saving figure to path: ", os.path.join(output_dir, f"{x_name} to attachment time - {attachment} (spring: {title}).png"))
            plt.savefig(os.path.join(output_dir, f"{x_name} to attachment time - {attachment} (spring {title}).png"))
plot_query_information_to_attachment_length(analysed, window_size=50, title=spring_type, output_dir=output_dir)


def plot_query_distribution_to_angular_velocity(analysed, start=0, end=None, window_size=1, title="", output_dir=None):
    if end is None:
        end = analysed.angular_velocity.shape[0]
    # output_dir = define_output_dir(output_dir)

    single_ant = analysed.N_ants_around_springs[start:end] == 1
    more_than_one = analysed.N_ants_around_springs[start:end] > 1
    x = np.abs(analysed.angular_velocity[start:end])
    x_nans = np.isnan(x)
    x = x[~x_nans]
    quantile = np.quantile(x, 0.99)
    upper_quantile = x < quantile
    x = x[upper_quantile]
    x_binned = np.linspace(np.nanmin(x), np.nanmax(x), 100)

    y_angular_force = np.abs(analysed.force_magnitude[start:end] * np.sin(analysed.force_direction))
    y_force_magnitude = np.abs(analysed.force_magnitude[start:end])
    for y_ori, name in zip([y_angular_force, y_force_magnitude],
                            ["mean angular force", "mean force magnitude"]):
        for bool, label in zip([single_ant, more_than_one], ["single ant", "more than one ant"]):
            y = np.copy(y_ori)
            y[~bool] = np.nan
            y = y[~x_nans]
            y = y[upper_quantile]

            y_mean = np.nanmean(y, axis=1)
            y_SEM_upper = y_mean + np.nanstd(y, axis=1) / np.sqrt(np.sum(~np.isnan(y), axis=1))
            y_SEM_lower = y_mean - np.nanstd(y, axis=1) / np.sqrt(np.sum(~np.isnan(y), axis=1))
            y_mean_binned = np.zeros((len(x_binned),))
            y_SEM_upper_bined = np.zeros((len(x_binned),))
            y_SEM_lower_bined = np.zeros((len(x_binned),))
            for i in range(len(x_binned) - 1):
                y_mean_binned[i] = np.nanmean(y[(x >= x_binned[i]) & (x < x_binned[i + 1])])
                y_SEM_upper_bined[i] = np.nanmean(y_SEM_upper[(x >= x_binned[i]) & (x < x_binned[i + 1])])
                y_SEM_lower_bined[i] = np.nanmean(y_SEM_lower[(x >= x_binned[i]) & (x < x_binned[i + 1])])
            plt.clf()
            plt.plot(x_binned[:-1], y_mean_binned[:-1])
            plt.fill_between(x_binned[:-1], y_SEM_lower_bined[:-1], y_SEM_upper_bined[:-1], alpha=0.5, color="orange")
            plt.title(f"{name} to angular velocity (movie:{title})")
            plt.xlabel("angular velocity (rad/ 20_frames)")
            plt.ylabel(f"{name} (mN)")
            plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
            print("Saving figure to path: ", os.path.join(output_dir, f"{name}, {label} to angular velocity (movie {title}).png"))
            plt.savefig(os.path.join(output_dir, f"{name}, {label} to angular velocity (movie {title}).png"))

            lower_bins = y[(x <= x_binned[30])].flatten()
            upper_bins = y[(x >= x_binned[70])].flatten()
            lower_bins = lower_bins[~np.isnan(lower_bins)]
            upper_bins = upper_bins[~np.isnan(upper_bins)]
            t, p = stats.ttest_ind(lower_bins, upper_bins)
            plt.clf()
            sns.barplot(x=["low angular velocity", "high angular velocity"], y=[np.mean(lower_bins), np.mean(upper_bins)])
            plt.xlabel("")
            plt.text(0.5, 0.5, f"p = {p}")
            plt.xticks([0, 1], ["low angular velocity", "high angular velocity"])
            plt.ylabel(f"{name} (mN)")
            plt.title(f"{name} to angular velocity (movie:{title}{p})")
            plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
            print("Saving figure to path: ", os.path.join(output_dir, f"{name}, {label} to angular velocity - bar plot (movie {title}).png"))
            plt.savefig(os.path.join(output_dir, f"{name}, {label} to angular velocity - bar plot (movie {title}).png"))
output_dir = os.path.join("Z:\\Dor_Gabay\\ThesisProject\\results",spring_type, "macro_scale_results")
plot_query_distribution_to_angular_velocity(analysed, start=0, end=None, window_size=50, title=spring_type, output_dir=output_dir)


def plot_overall_behavior(analysed, start=0, end=None, window_size=1, title="", output_dir=None):
    plt.clf()
    if end is None:
        end = analysed.pulling_angle.shape[0]
    fig, ax1 = plt.subplots()
    velocity_color, _, pulling_angle_color, _, _ = define_colors()

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
    net_force = analysed.net_force[start:end]
    # net_force = analysed.net_magnitude[start:end]
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
    # ax2.set_ylim(np.nanmax(analysed.net_force[start:end]) * -1.1, np.nanmax(analysed.net_force[start:end]) * 1.1)
    ax2.set_ylim(-0.001, 0.001)

    fig.tight_layout()
    #fig size should be 1920x1080
    fig.set_size_inches(19.2, 10.8)
    output_dir = define_output_dir(output_dir)
    print("Saving figure to path: ", os.path.join(output_dir, f"angular_velocity VS mean_springs_extension (movie {title}).png"))
    plt.savefig(os.path.join(output_dir, f"{title}.png"))


if __name__ == "__main__":
    # %matplotlib qt
    pass
