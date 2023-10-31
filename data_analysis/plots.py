import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy import stats
from scipy.signal import savgol_filter
import copy
import re


def define_colors(labels_num=5):
    # set a color by RGB
    # red, purple, blue, green, purple_brown
    red = np.array((239, 59, 46))/255
    purple = np.array((112, 64, 215))/255
    blue = np.array((86, 64, 213))/255
    green = np.array((93, 191, 71))/255
    purple_brown = np.array((153, 86, 107))/255
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


def plot_ant_profiles(analysed, output_dir, window_size=11, title="", profile_size=200):
    os.makedirs(output_dir, exist_ok=True)
    #plot number of cases
    plt.clf()
    y = np.sum(analysed.single_ant_profiles, axis=0)
    y = y[y > analysed.fps]
    x = np.arange(0, y.shape[0], 1)/analysed.fps
    plt.plot(x, y, color="purple")
    plt.ylabel(f"number of profiles")
    plt.xlabel("seconds")
    print("Saving figure to path: ", os.path.join(output_dir, f"number of profiles.png"))
    plt.savefig(os.path.join(output_dir, f"number of profiles.png"))

    force_magnitude_copy = np.abs(analysed.profiled_force_magnitude)
    tangential_force = np.abs(analysed.profiled_tangential_force)
    for y_ori, y_title in zip([force_magnitude_copy, tangential_force], ["force magnitude", "tangential force"]):
    # for y_ori, y_title in zip([tangential_force], ["tangential force"]):
        for attachment in ["first attachment", "all but first attachment", "all attachments"]:
            if attachment == "first attachment":
                bool = np.copy(analysed.single_ant_profiles)
                bool[analysed.profiles_precedence != 1, :] = False
                y = np.copy(y_ori)
            elif attachment == "all but first attachment":
                bool = np.copy(analysed.single_ant_profiles)
                bool[analysed.profiles_precedence == 1, :] = False
                y = np.copy(y_ori)
            else:
                bool = np.copy(analysed.single_ant_profiles)
                y = np.copy(y_ori)
            bool[~bool[:, profile_size], :] = False
            bool[:, profile_size+1:] = False
            y[~bool] = np.nan
            y[y == 0] = np.nan
            y_mean = np.nanmean(y, axis=0)
            y_SEM_upper = y_mean + np.nanstd(y, axis=0) / np.sqrt(np.sum(~np.isnan(y), axis=0))
            y_SEM_lower = y_mean - np.nanstd(y, axis=0) / np.sqrt(np.sum(~np.isnan(y), axis=0))
            y_not_nan = ~np.isnan(y_mean)
            y_mean,y_SEM_upper,y_SEM_lower = y_mean[y_not_nan],y_SEM_upper[y_not_nan],y_SEM_lower[y_not_nan]
            x = np.arange(0, y_mean.shape[0], 1)/analysed.fps
            # print(y_mean.shape, y_SEM_upper.shape, y_SEM_lower.shape, x.shape)
            plt.clf()
            plt.plot(x, savgol_filter(y_mean, window_size, 3), color="purple")
            plt.fill_between(x, savgol_filter(y_SEM_lower, window_size, 3),
                             savgol_filter(y_SEM_upper, window_size, 3), alpha=0.5, color="orange")
            plt.title(f"ant {y_title} profiles")
            plt.xlabel("seconds")
            plt.ylabel(f"{y_title} (mN)")
            plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
            print("Saving figure to path: ", os.path.join(output_dir, f"ant {y_title} profiles - {attachment}.png"))
            plt.savefig(os.path.join(output_dir, f"ant {y_title} profiles - {attachment}.png"))
# plot_ant_profiles(self, output_dir=os.path.join(output_dir,"profiles"), window_size=11, profile_size=200)


def draw_single_profiles(analysed, output_path, profile_min_length=200, examples_number=-1, start=0, end=None):
    profiles_idx = analysed.ant_profiles[:, 2] >= start
    profiles_idx *= analysed.ant_profiles[:, 3] < end if end is not None else np.ones_like(profiles_idx)
    occasions = ((analysed.ant_profiles[:, 3] - analysed.ant_profiles[:, 2]) > profile_min_length) * ~np.any(analysed.profiled_check, axis=1)
    occasions = occasions * profiles_idx
    print("Number of found profiles: ", np.sum(occasions))
    profiles = np.arange(len(occasions))[occasions]
    os.makedirs(output_path, exist_ok=True)
    examples_number = np.sum(occasions) if examples_number == -1 else examples_number
    # profiles = np.random.choice(profiles, size=examples_number, replace=False)
    for i in profiles:
        angular_force = analysed.profiled_tangential_force[i, :]
        force_magnitude = analysed.profiled_force_magnitude[i, :]
        info = analysed.ant_profiles[i, :]
        x = np.arange(np.sum(~np.isnan(force_magnitude)))
        y = force_magnitude[~np.isnan(force_magnitude)]
        plt.clf()
        plt.plot(x, y, color="purple")
        plt.xlim(0, profile_min_length)
        plt.title(f"spring: {info[1]}, start: {info[2]-start}, end: {info[3]-start}, precedence: {info[4]}")
        plt.savefig(os.path.join(output_path, f"profile_{i}.png"))
# draw_single_profiles(self, os.path.join(self.output_path, "single_profiles_S5870005"), profile_min_length=200, start=self.sets_frames[-1][3][0], end=self.sets_frames[-1][3][1])
# draw_single_profiles(self, os.path.join(self.output_path, "single_profiles_S5760011"), profile_min_length=200, examples_number=200, start_from=self.sets_frames[1][-1][0])
# draw_single_profiles(self, os.path.join(self.output_path, "single_profiles"), profile_min_length=200, examples_number=200, start_from=self.sets_frames[1][-1][0])

def angle_to_nest_bias(self):
    import matplotlib.pyplot as plt
    angles_to_nest = self.fixed_end_angle_to_nest[self.rest_bool]
    # distance_from_center = np.linalg.norm((self.object_center_coordinates-self.video_resolution/2), axis=1)
    force_magnitude = self.force_magnitude[self.rest_bool]
    force_direction = self.force_direction[self.rest_bool]
    # plot dot plot of angle to nest vs force magnitude
    # plt.clf()
    plt.scatter(angles_to_nest, force_magnitude, s=1, c=force_direction)
    plt.xlabel("angle to nest")
    plt.ylabel("force magnitude")
    plt.title("angle to nest vs force magnitude")
    plt.show()
    # plot dot plot of angle to nest vs force direction
    # plt.clf()
    # plt.scatter(angles_to_nest, force_direction, s=1, c=force_magnitude)
    # plt.xlabel("angle to nest")
    # plt.ylabel("force direction")
    # plt.title("angle to nest vs force direction")
    # plt.colorbar()
    # plt.show()
# angle_to_nest_bias(self)




# def plot_profiles_length_distrubution(analysed, window_size=1, title="", output_dir=None):
#     for attachment in ["first attachment", "all but first attachment"]:
#         if attachment == "first attachment":
#             bool = np.copy(analysed.single_ant_profiles)
#             bool[analysed.all_profiles_precedence != 1, :] = False
#         else:
#             bool = np.copy(analysed.single_ant_profiles)
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
#         print("Saving figure to path: ", os.path.join(output_dir, f"ant angular velocity profiles length distribution - {attachment}.png"))
#         plt.savefig(os.path.join(output_dir, f"ant angular velocity profiles length distribution - {attachment} (movie {title}).png"))
# # plot_profiles_length_distrubution(analysed, title=spring_type, output_dir=output_dir)

def plot_springs_bar_plot_comparison(analysed, window_size=5, title="", output_dir=None):
    data = copy.copy(analysed.calib_data)
    data["angular_velocity"] = np.abs(data["angular_velocity"])
    plt.clf()
    sns.barplot(x="spring_type", y="angular_velocity", data=data, edgecolor=".5", facecolor=(0, 0, 0, 0))
    plt.xlabel("")
    # add labels to each bar
    plt.xticks([0, 1,2,3], [f"stiffness: {float(re.split('[a-z]',spring_type)[-1])}" for spring_type in analysed.spring_types])
    # plt.xticks([0, 1], [f"stiffness: {float(re.split('[a-z]',spring_type)[-1])}" for spring_type in analysed.spring_types])
    plt.ylabel(f"angular velocity (rad/s)")
    plt.title(f"angular velocity to spring stiffness")
    plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
    print("Saving figure to path: ", os.path.join(output_dir, f"angular velocity to spring stiffness - bar plot.png"))
    plt.savefig(os.path.join(output_dir, f"angular velocity to spring stiffness - bar plot.png"))
# plot_springs_bar_plot_comparison(self, output_dir=output_path)

def plot_springs_comparison(analysed, window_size=5, title="", output_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    data = copy.copy(analysed.calib_data)

    # data["angular_velocity"] = np.abs(data["angular_velocity"])
    # data.loc[data.loc[:,"sum_N_ants"]<=1, "angular_velocity"] = np.nan
    # plt.clf()
    # sns.lineplot(data=data, x="sum_N_ants", y="angular_velocity", hue="spring_type")
    # plt.title(f"angular velocity to number of carrying ants")
    # plt.xlabel("number of ants")
    # plt.ylabel("angular velocity (rad/frame)")
    # plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
    # print("Saving figure to path: ", os.path.join(output_dir, f"angular velocity to number of carrying ants.png"))
    # plt.savefig(os.path.join(output_dir, f"angular velocity to number of carrying ants.png"))

    data["angular_velocity"] = np.abs(data["angular_velocity"])
    data["net_tangential_force"] = np.abs(data["net_tangential_force"])
    for x_name, x_query in zip(["number of ants"], ["sum_N_ants"]):
        for y_name, y_query in zip(["angular velocity (rad/frame)", "net tangential force mN"], ["angular_velocity", "net_tangential_force"]):
            x = [np.unique(data.loc[data.loc[:,"spring_type"]==spring_type, x_query])[2:] for spring_type in np.unique(data.loc[:, "spring_type"])]
            y = [np.zeros(len(x[i])) for i in range(len(np.unique(data.loc[:, "spring_type"])))]
            y_SEM = [np.zeros(len(x[i])) for i in range(len(np.unique(data.loc[:, "spring_type"])))]
            y_SEM_upper = [np.zeros(len(x[i])) for i in range(len(np.unique(data.loc[:, "spring_type"])))]
            y_SEM_lower = [np.zeros(len(x[i])) for i in range(len(np.unique(data.loc[:, "spring_type"])))]
            for i, spring_type in enumerate(np.unique(data.loc[:, "spring_type"])):
                for j, N_ants in enumerate(x[i]):
                    y[i][j] = np.nanmean(data.loc[(data.loc[:, "spring_type"]==spring_type) & (data.loc[:, x_query]==N_ants), y_query])
                    y_SEM[i][j] = np.nanstd(data.loc[(data.loc[:, "spring_type"]==spring_type) & (data.loc[:, x_query]==N_ants), y_query])/np.sqrt(len(data.loc[(data.loc[:, "spring_type"]==spring_type) & (data.loc[:, x_query]==N_ants), y_query]))
                    y_SEM_upper[i][j] = y[i][j] + y_SEM[i][j]
                    y_SEM_lower[i][j] = y[i][j] - y_SEM[i][j]
            plt.clf()
            for i, spring_type, color in zip(range(len(x)), np.unique(data.loc[:, "spring_type"]), ["purple", "orange", "green", "lightblue"]):
                y[i], y_SEM_upper[i], y_SEM_lower[i] = savgol_filter(y[i].astype(float), window_size, 3),\
                    savgol_filter(y_SEM_upper[i].astype(float), window_size, 3),\
                    savgol_filter(y_SEM_lower[i].astype(float), window_size, 3)
                plt.plot(x[i].astype(float), y[i].astype(float), color=color, label=f"stiffness: {float(re.split('[a-z]',spring_type)[-1])}")
                plt.fill_between(x[i].astype(float), y_SEM_lower[i].astype(float), y_SEM_upper[i].astype(float), color=color, alpha=0.2)
            plt.legend()
            plt.title(f"{y_name} to {x_name}")
            plt.xlabel(f"{x_name}")
            # plt.ylabel("angular velocity (rad/frame)")
            plt.ylabel(f"{y_name})")
            plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
            print("Saving figure to path: ", os.path.join(output_dir, f"{y_query} to {x_query}.png"))
            plt.savefig(os.path.join(output_dir, f"{y_query} to {x_query}.png"))
# plot_springs_comparison(self, output_dir=output_path)


def plot_query_information_to_attachment_length(analysed, start=0, end=None, window_size=1, title="", output_dir=None):
    x_velocity = np.abs(analysed.all_profiles_angular_velocity)
    x_N_ants = analysed.all_profiles_N_ants_around_springs_sum
    for x_ori, x_name in zip([x_velocity, x_N_ants], ["profile velocity mean", "mean number of ants"]):
        for attachment in ["1st", "2st&higher", "all"]:
            if attachment == "1st":
                bool = np.copy(analysed.single_ant_profiles)
                bool[analysed.all_profiles_precedence != 1, :] = False
                x = np.copy(x_ori)
            elif attachment == "2st&higher":
                bool = np.copy(analysed.single_ant_profiles)
                bool[analysed.all_profiles_precedence == 1, :] = False
                x = np.copy(x_ori)
            else:
                bool = np.copy(analysed.single_ant_profiles)
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
            plt.title(f"{x_name} to attachment time")
            plt.xlabel(f"{x_name}")
            plt.ylabel(f"persistence time (s)")
            plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
            print("Saving figure to path: ", os.path.join(output_dir, f"{x_name} to attachment time - {attachment} (spring: {title}).png"))
            plt.savefig(os.path.join(output_dir, f"{x_name} to attachment time - {attachment} (spring {title}).png"))
# plot_query_information_to_attachment_length(analysed, window_size=50, title=spring_type, output_dir=ant_profiles_output_dir)


def plot_replacements_rate(analysed, start=0, end=None, window_size=10, title="", output_dir=None):
    if end is None:
        end = analysed.angular_velocity.shape[0]

    x = np.abs(analysed.angular_velocity[start:end])[:-1]
    x_nans = np.isnan(x)
    x = x[~x_nans]
    under_quantile = x < np.quantile(x, 0.99)
    x = x[under_quantile]
    x_binned = np.linspace(np.nanmin(x), np.nanmax(x), 100)

    y_replacements_rate = analysed.n_replacments_per_frame[start:end]
    for y_ori, name in zip([y_replacements_rate], ["number of changes per frame"]):
        y = np.copy(y_ori)
        y = y[~x_nans]
        y = y[under_quantile]
        y_mean_binned = np.zeros((len(x_binned),))
        y_SEM_upper_bined = np.zeros((len(x_binned),))
        y_SEM_lower_bined = np.zeros((len(x_binned),))
        for i in range(len(x_binned) - 1):

            y_mean_binned[i] = np.nanmean(y[(x >= x_binned[i]) & (x < x_binned[i + 1])])
            y_SEM_upper_bined[i] = np.nanmean(y[(x >= x_binned[i]) & (x < x_binned[i + 1])]) + \
                                    np.nanstd(y[(x >= x_binned[i]) & (x < x_binned[i + 1])]) / np.sqrt(np.sum((x >= x_binned[i]) & (x < x_binned[i + 1])))
            y_SEM_lower_bined[i] = np.nanmean(y[(x >= x_binned[i]) & (x < x_binned[i + 1])]) - \
                                    np.nanstd(y[(x >= x_binned[i]) & (x < x_binned[i + 1])]) / np.sqrt(np.sum((x >= x_binned[i]) & (x < x_binned[i + 1])))
        plt.clf()
        plt.plot(x_binned[:-1], savgol_filter(y_mean_binned[:-1], window_size, 3), color="purple")
        plt.fill_between(x_binned[:-1], savgol_filter(y_SEM_lower_bined[:-1], window_size, 3),
                         savgol_filter(y_SEM_upper_bined[:-1], window_size, 3), color="purple", alpha=0.2)
        plt.title(f"{name} to angular velocity")
        plt.xlabel(f"angular velocity (rad/s)")
        plt.ylabel(f"{name}")
        plt.legend()
        plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
        print("Saving figure to path: ", os.path.join(output_dir, f"{name} to angular velocity (spring: {title}).png"))
        plt.savefig(os.path.join(output_dir, f"{name} to angular velocity (spring {title}).png"))
# plot_replacements_rate(analysed, window_size=10, title=spring_type, output_dir=macro_scale_output_dir)


def plot_agreement_with_the_group(analysed, start=0, end=None, window_size=1, title="", output_dir=None):
    single_ant = analysed.all_profiles_N_ants_around_springs[start:end] == 1
    more_than_one = analysed.all_profiles_N_ants_around_springs[start:end] > 1
    #
    # x = np.abs(analysed.angular_velocity[start:end])
    # x_nans = np.isnan(x)
    # x = x[~x_nans]
    # quantile = np.quantile(x, 0.99)
    # upper_quantile = x < quantile
    # x = x[upper_quantile]
    # x_binned = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    x = analysed.all_profiles_angular_velocity[start:end]
    y_tangential_force = analysed.all_profiles_tangential_force[start:end]
    for ori, y_title in zip([(y_tangential_force, x)],["tangential force"]):
        for bool in [single_ant, more_than_one]:
            y = np.copy(ori[0])
            y[~bool] = np.nan
            x = np.copy(ori[1])
            x[~bool] = np.nan

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


def plot_query_distribution_to_angular_velocity(analysed, start=0, end=None, window_size=1, title="", output_dir=None):
    if end is None:
        end = analysed.angular_velocity.shape[0]

    single_ant = analysed.N_ants_around_springs[start:end] == 1
    more_than_one = analysed.N_ants_around_springs[start:end] > 1
    all_springs = np.ones((analysed.N_ants_around_springs[start:end].shape[0],analysed.N_ants_around_springs[start:end].shape[1])).astype("bool")
    all_ants = analysed.N_ants_around_springs[start:end] >= 1
    no_ants = analysed.N_ants_around_springs[start:end] == 0
    # x = analysed.angular_velocity[start:end]
    x = np.abs(analysed.angular_velocity[start:end])
    x_nans = np.isnan(x)
    x = x[~x_nans]
    # quantile = (x < np.quantile(x, 0.99)) & (x > np.quantile(x, 0.01))
    quantile = (x < np.quantile(x, 0.99))
    x = x[quantile]
    x_binned = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    y_N_ants = analysed.N_ants_around_springs[start:end]
    y_N_ants = y_N_ants[~x_nans]
    y_N_ants = y_N_ants[quantile]
    N_ants_mean = np.nansum(y_N_ants, axis=1)

    y_tangential_force = analysed.tangential_force[start:end]
    y_force_magnitude = np.abs(analysed.force_magnitude[start:end])
    for y_ori, name in zip([np.abs(y_tangential_force), y_force_magnitude, y_tangential_force, y_tangential_force],
                            ["mean tangential force", "mean force magnitude", "net tangential force", "contradicting force"]):
        for bool, label in zip([single_ant, more_than_one, all_ants, no_ants, all_springs], ["single ant", "more than one ant", "all_ants", "no ants", "all springs"]):
            y = np.copy(y_ori)
            if name == "contradicting force":
                disagreements = (np.sign(y) * np.sign(analysed.angular_velocity[start:end])[:, np.newaxis]) < 0
                y[disagreements] = np.nan
                y = np.abs(y)
            y[~bool] = np.nan
            y = y[~x_nans]
            y = y[quantile]
            # if (name == "mean tangential force" and label in ["single ant", "more than one ant", "all_ants", "no ants"])\
            #     or (name == "mean force magnitude" and label in ["single ant"])\
            #     or (name == "net tangential force" and label in ["all_ants", "no ants", "all springs"])\
            #     or (name == "mean disagreement with the group" and label in ["single ant", "more than one ant", "all_ants", "no ants"]):
            #     continue
            if name == "contradicting force" and label in ["single ant", "more than one ant", "all_ants", "no ants"]:
                if name == "net tangential force":
                    y = np.abs(np.nansum(y, axis=1))
                y_mean_binned = np.zeros((len(x_binned),))
                N_ants_mean_binned = np.zeros((len(x_binned),))
                y_SEM_upper_bined = np.zeros((len(x_binned),))
                y_SEM_lower_bined = np.zeros((len(x_binned),))
                for i in range(len(x_binned) - 1):
                    y_mean_binned[i] = np.nanmean(y[(x >= x_binned[i]) & (x < x_binned[i + 1])])
                    N_ants_mean_binned[i] = np.nanmean(N_ants_mean[(x >= x_binned[i]) & (x < x_binned[i + 1])])
                    y_SEM_upper_bined[i] = np.nanmean(y[(x >= x_binned[i]) & (x < x_binned[i + 1])]) + \
                                           np.nanstd(y[(x >= x_binned[i]) & (x < x_binned[i + 1])]) / np.sqrt(np.sum((x >= x_binned[i]) & (x < x_binned[i + 1])))
                    y_SEM_lower_bined[i] = np.nanmean(y[(x >= x_binned[i]) & (x < x_binned[i + 1])]) - \
                                           np.nanstd(y[(x >= x_binned[i]) & (x < x_binned[i + 1])]) / np.sqrt(np.sum((x >= x_binned[i]) & (x < x_binned[i + 1])))

                plt.clf()
                fig, ax1 = plt.subplots()
                ax1.plot(x_binned[:-1], savgol_filter(y_mean_binned[:-1], 10, 3), color="purple")
                ax1.fill_between(x_binned[:-1], savgol_filter(y_SEM_lower_bined[:-1],10,3), savgol_filter(y_SEM_upper_bined[:-1],10,3), alpha=0.5, color="orange")
                ax1.set_xlabel("angular velocity (rad/ frame)")
                ax1.set_ylabel(f"{name} (mN)", color="purple")
                ax1.tick_params(axis='y')
                if name == "mean tangential force":
                    ax2 = ax1.twinx()
                    ax2.plot(x_binned[:-1], N_ants_mean_binned[:-1], color="black", alpha=0.5)
                    ax2.set_ylabel("number of ants", color="black", alpha=0.5)

                plt.title(f"{name} to angular velocity")
                plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
                print("Saving figure to path: ", os.path.join(output_dir, f"{name}, {label} to angular velocity (movie {title}).png"))
                plt.savefig(os.path.join(output_dir, f"{name}, {label} to angular velocity (movie {title}).png"))

                #creat bar and violin plots
                lower_bins = y[(x <= x_binned[30])].flatten()
                upper_bins = y[(x >= x_binned[70])].flatten()
                lower_bins = lower_bins[~np.isnan(lower_bins)]
                upper_bins = upper_bins[~np.isnan(upper_bins)]
                t, p = stats.ttest_ind(lower_bins, upper_bins)
                print(f"p value for {name} to angular velocity (movie:{title}): {p}")
                df = pd.DataFrame({"anglar velocity": ["low angular velocity"] * len(lower_bins) + ["high angular velocity"] * len(upper_bins),
                                     f"{name} (mN)": np.concatenate([lower_bins, upper_bins])})
                plt.clf()
                sns.barplot(x="anglar velocity", y=f"{name} (mN)", data=df, edgecolor=".5", facecolor=(0, 0, 0, 0))
                plt.xlabel("")
                plt.text(0.5, 0.5, f"p = {p}")
                plt.xticks([0, 1], ["low angular velocity", "high angular velocity"])
                plt.ylabel(f"{name} (mN)")
                plt.title(f"{name} to angular velocity")
                plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
                print("Saving figure to path: ", os.path.join(output_dir, f"{name}, {label} to angular velocity - bar plot (movie {title}).png"))
                plt.savefig(os.path.join(output_dir, f"{name}, {label} to angular velocity - bar plot (movie {title}).png"))
                # #violin plot
                # plt.clf()
                # sns.violinplot(x="anglar velocity", y=f"{name} (mN)", data=df, edgecolor=".5", facecolor=(0, 0, 0, 0))
                # plt.xlabel("")
                # plt.text(0.5, 0.5, f"p = {p}")
                # plt.xticks([0, 1], ["low angular velocity", "high angular velocity"])
                # plt.ylabel(f"{name} (mN)")
                # plt.title(f"{name} to angular velocity")
                # plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
                # print("Saving figure to path: ", os.path.join(output_dir, f"{name}, {label} to angular velocity - violin plot (movie {title}).png"))
                # plt.savefig(os.path.join(output_dir, f"{name}, {label} to angular velocity - violin plot (movie {title}).png"))
# plot_query_distribution_to_angular_velocity(analysed, start=0, end=None, window_size=50, title=spring_type, output_dir=macro_scale_output_dir)


def plot_correlation(analysed, start=0, end=None, output_path=None):
    # output_dir = define_output_dir(output_dir)
    os.makedirs(output_path, exist_ok=True)
    title = "angular velocity to tangential force correlation"
    end = analysed.pulling_angle.shape[0] if end is None else end
    red, purple, blue, green, purple_brown = define_colors()
    plt.clf()
    fig, ax1 = plt.subplots()
    fig.suptitle(title)
    time = np.linspace(start, end, end - start) / analysed.fps
    # plot the angular velocity
    angular_velocity = analysed.angular_velocity[start:end] * analysed.fps
    ax1.plot(time, angular_velocity, color=red)
    ax1.set_xlabel("time (sec)")
    ax1.set_ylabel("angular velocity (rad/ sec)", color=red)
    ax1.tick_params(axis="y", labelcolor=red)
    # plot the net tangential force
    ax2 = ax1.twinx()
    net_tangential_force = analysed.net_tangential_force[start:end]
    ax2.plot(time, net_tangential_force, color=purple)
    ax2.set_ylabel("net_tangential_force (mN)", color=purple)
    ax2.tick_params(axis="y", labelcolor=purple)

    ticks1 = ax1.get_yaxis().get_majorticklocs()
    ticks2 = ax2.get_yaxis().get_majorticklocs()
    y_min = min(min(ticks1), min(ticks2))
    y_max = max(max(ticks1), max(ticks2))
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)

    # draw the zero line
    ax1.axhline(y=0, color="black", linestyle="--")
    ax2.axhline(y=0, color="black", linestyle="--")

    fig.tight_layout()
    fig.set_size_inches(19.2, 10.8)
    plt.show()
    print("Saving figure to path: ", os.path.join(output_path, title+".png"))
    plt.savefig(os.path.join(output_path, title+".png"))

