import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import label
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import itertools
import apytl
import sys
from scipy.stats import pearsonr
import apytl
# local imports:
sys.path.append(os.path.join(os.getcwd(), "data_analysis"))
import utils


def show_colors(palette, n=5):
    palette = sns.color_palette(palette, n)
    sns.palplot(palette)
    plt.xlabel('Palette Index')
    plt.ylabel('Color')
    plt.title('Seaborn Color Palette')
    plt.show()

# used functions:
def distribution_of_new_joiners_over_nest_direction(self, output_path, bins=50, sample_size=1800):
    new_joiners_frames, new_joiners_springs = np.where(np.diff(self.N_ants_around_springs, axis=0) == 1)
    new_joiners_frames += 1
    data_angles = self.fixed_end_angle_to_nest[new_joiners_frames, new_joiners_springs].copy()
    plt.clf()
    ax = plt.subplot(111, projection='polar')
    angles = np.linspace(0, 2 * np.pi, bins)
    r, theta = np.histogram(data_angles[~np.isnan(data_angles)], bins=angles)
    r = np.append(r, r[0])
    theta[-1] = 0
    ax.plot(theta, r)
    # ax.set_theta_offset(np.pi)
    ax.fill_between(theta, 0, r, alpha=0.2)
    ax.set_yticklabels([])
    ax.grid(True)
    plt.title(f"Distribution of new joiners over nest direction")
    plt.savefig(os.path.join(output_path, f"Distribution of new joiners over nest direction.png"))

    # plt.clf()
    # up_quarter_idx = np.logical_and(data_angles >= (np.pi / 4), data_angles < (3 * np.pi / 4))  # degrees 45-135
    # left_quarter_idx = np.logical_and(data_angles >= (3 * np.pi / 4), data_angles < (5 * np.pi / 4))  # degrees 135-225
    # down_quarter_idx = np.logical_and(data_angles >= (5 * np.pi / 4), data_angles < (7 * np.pi / 4))  # degrees 225-315
    # right_quarter_idx = np.logical_or(data_angles >= (7 * np.pi / 4), data_angles < (np.pi / 4))  # degrees 315-45
    # up_quarter_histogram = data_angles[up_quarter_idx]
    # left_quarter_histogram = data_angles[left_quarter_idx]
    # down_quarter_histogram = data_angles[down_quarter_idx]
    # right_quarter_histogram = data_angles[right_quarter_idx]
    # # print(up_quarter_histogram.shape, left_quarter_histogram.shape, down_quarter_histogram.shape, right_quarter_histogram.shape)
    # df = pd.DataFrame({"Left quarter": left_quarter_histogram.shape, "Right quarter": right_quarter_histogram.shape,
    #                    "Up quarter": up_quarter_histogram.shape, "Down quarter": down_quarter_histogram.shape})
    # sns.barplot(data=df, palette="rocket", ci="sd")
    # plt.title(f"New joiners over nest direction")
    # plt.xlabel("Nest direction")
    # plt.ylabel("Number of new joiners")
    # # utils.draw_significant_stars(df)
    # # draw_significant_stars(df)
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_path, f"New joiners over nest direction.png"))
# distribution_of_new_joiners_over_nest_direction(self, os.path.join(self.output_path, "nest_direction_distributions"), bins=50)





def distributions_with_binned_angle_to_nest(self, output_path, bins=50, sample_size=50000):
    translational_velocity_copy = np.copy(translational_velocity)
    angle_to_nest,  to, analyse_type = self.fixed_end_angle_to_nest.copy(), "nest", "mean"
    # angle_to_nest,  to, analyse_type = self.fixed_end_angle_to_nest.copy(), "nest", "high"
    # velocities = [[0, 0.25], [0.75, 1]]
    velocities = [0, 0.25, 0.5, 0.75, 1]
    discrete_velocity = self.discrete_angular_velocity.copy()
    values_idx = np.where(np.isin(np.abs(discrete_velocity),  np.array(velocities).flatten()))[0]
    if len(values_idx) > sample_size:
        values_idx = np.random.choice(values_idx, size=sample_size, replace=False)
    discrete_velocity = np.full(self.discrete_angular_velocity.shape, np.nan)
    discrete_velocity[values_idx] = self.discrete_angular_velocity[values_idx]

    # _, discrete_velocity = utils.calc_translation_velocity(self.object_center_coordinates, window=4)
    # discrete_velocity[discrete_velocity > np.nanpercentile(discrete_velocity, 99)] = np.nan
    # fast_idx = discrete_velocity > np.nanpercentile(discrete_velocity, 75)
    # mid_idx = np.logical_and(discrete_velocity > np.nanpercentile(discrete_velocity, 25), discrete_velocity < np.nanpercentile(discrete_velocity, 75))
    # stop_idx = discrete_velocity < np.nanpercentile(discrete_velocity, 25)
    # discrete_velocity[fast_idx] = 1
    # discrete_velocity[mid_idx] = 0.5
    # discrete_velocity[stop_idx] = 0
    # discrete_velocity[np.abs(self.discrete_angular_velocity) >= 0.25] = 0.5

    angles = np.linspace(0, 2 * np.pi, bins)
    for data_title, types_data in zip(["force magnitude", "tangential force"], [self.force_magnitude, self.tangential_force]):
        types_data = types_data.copy()
        if data_title == "force magnitude":
            types_data[types_data < 0] = np.nan
        types_data = np.abs(types_data)
        types_data[types_data > np.nanpercentile(types_data, 99)] = np.nan
        upper_percentile = np.nanpercentile(types_data, 100, axis=0)
        lower_percentile = np.nanpercentile(types_data, 90, axis=0)
        force_bool_idx = np.logical_and(types_data > lower_percentile, types_data < upper_percentile)
        strong_pulls_data = np.full((bins - 1, len(velocities)), np.nan, dtype=np.float64)
        mean_data = np.full((bins - 1, len(velocities)), np.nan, dtype=np.float64)
        all_pulls_data = np.full((bins - 1, len(velocities)), np.nan, dtype=np.float64)
        for angle_count in range(bins - 1):
            angle_bool_idx = np.logical_and(angle_to_nest >= angles[angle_count], angle_to_nest < angles[angle_count + 1])
            angle_bool_idx[~(self.N_ants_around_springs > 0)] = False
            for velocity_count, velocity in enumerate(velocities):
                velocity_bool_idx = angle_bool_idx.copy()
                velocity_bool_idx[~np.isin(np.abs(discrete_velocity), velocity)] = False
                mean_data[angle_count, velocity_count] = np.nanmean(np.abs(types_data[velocity_bool_idx]))
                strong_pulls_data[angle_count, velocity_count] = np.sum(force_bool_idx[velocity_bool_idx])
                all_pulls_data[angle_count, velocity_count] = np.sum(velocity_bool_idx)
        data = strong_pulls_data / all_pulls_data  # normalizing
        data = np.append(data, data[0, :].reshape(1, -1), axis=0)
        data = np.flip(data, axis=0)
        if analyse_type == "mean":
            data = np.append(mean_data, mean_data[0, :].reshape(1, -1), axis=0)
            data = np.flip(data, axis=0)

        plt.clf()
        theta = angles[:-1]
        theta = np.append(theta, theta[0])
        theta = np.flip(theta)
        ax = plt.subplot(111, projection='polar')
        ax.set_title(f"Distribution over velocities - high {data_title}")
        for col in range(len(velocities)):
            r = data[:, col]
            ax.plot(theta, r, label=f"{velocities[col]}")
            if col != 0:
                ax.fill_between(theta, r, data[:, col - 1], alpha=0.2, color="red")
        if to == "wall":
            ax.set_theta_offset(np.pi / 2)
        ax.legend(title="Velocity", loc=(1, 0.8))
        ax.set_yticklabels([])
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"Angle to {to} - {analyse_type} {data_title}.png"))

        plt.clf()
        up_quarter_idx = np.logical_and(angle_to_nest >= (np.pi / 4), angle_to_nest < (3 * np.pi / 4))  # degrees 45-135
        left_quarter_idx = np.logical_and(angle_to_nest >= (3 * np.pi / 4), angle_to_nest < (5 * np.pi / 4))  # degrees 135-225
        down_quarter_idx = np.logical_and(angle_to_nest >= (5 * np.pi / 4), angle_to_nest < (7 * np.pi / 4))  # degrees 225-315
        right_quarter_idx = np.logical_or(angle_to_nest >= (7 * np.pi / 4), angle_to_nest < (np.pi / 4))  # degrees 315-45
        right_half_idx = np.logical_or(angle_to_nest >= (3 * np.pi / 2), angle_to_nest < (np.pi / 2))  # degrees 270-90
        left_half_idx = np.logical_and(angle_to_nest >= (np.pi / 2), angle_to_nest < (3 * np.pi / 2))  # degrees 90-270
        up_quarter_histogram = np.random.choice(types_data[up_quarter_idx], size=sample_size, replace=False)
        left_quarter_histogram = np.random.choice(types_data[left_quarter_idx], size=sample_size, replace=False)
        down_quarter_histogram = np.random.choice(types_data[down_quarter_idx], size=sample_size, replace=False)
        right_quarter_histogram = np.random.choice(types_data[right_quarter_idx], size=sample_size, replace=False)
        print(up_quarter_histogram.shape, left_quarter_histogram.shape, down_quarter_histogram.shape, right_quarter_histogram.shape)
        df = pd.DataFrame({"Left quarter": left_quarter_histogram, "Right quarter": right_quarter_histogram,
                           "Up quarter": up_quarter_histogram, "Down quarter": down_quarter_histogram})
        left_half_data = np.random.choice(types_data[left_half_idx], size=sample_size, replace=False)
        right_half_data = np.random.choice(types_data[right_half_idx], size=sample_size, replace=False)
        # df = pd.DataFrame({"Left half": left_half_data, "Right half": right_half_data})
        sns.barplot(data=df, palette="rocket")
        plt.title(f"Strong pulls over nest direction - high {data_title}")
        plt.xlabel("Nest direction")
        plt.ylabel("Number of strong pulls")
        utils.draw_significant_stars(df)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"Bar plot Angle to {to} - {analyse_type} {data_title}.png"))
# distributions_with_binned_angle_to_nest(self, os.path.join(self.output_path), bins=50)


def plot_alignment_simpler(self, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    single_ant_bool = self.profiled_N_ants_around_springs == 1
    angular_velocity = self.profiled_discrete_angular_velocity.copy()
    force_direction = np.sin(self.profiled_force_direction)
    force_direction_threshold = np.nanpercentile(np.abs(force_direction), 60)
    force_direction[np.abs(force_direction) < force_direction_threshold] = 0
    plt.clf()
    force_direction_copy = force_direction.copy()
    force_direction_copy[~single_ant_bool] = 0
    alignment = np.sign(force_direction_copy) * np.sign(angular_velocity)
    total_alignment = np.sum(alignment == 1, axis=1)
    total_events = np.sum(np.isin(alignment, [-1, 1]), axis=1)
    percentage = total_alignment / total_events
    percentage[total_events < 20] = np.nan
    fig, axs = plt.subplots(1, figsize=(12, 6))
    fig.suptitle(f"Alignment percentage")
    sns.histplot(percentage, color="gold", ax=axs, bins=20)
    axs.set_xlabel('Alignment (%)')
    plt.tight_layout()
    print("Saving figure to path: ", os.path.join(output_dir, f"Ants alignment.png"))
    plt.savefig(os.path.join(output_dir, f"Ants alignment.png"))
# plot_alignment_simpler(self, os.path.join(self.output_path, "alignment"))


def plot_alignment_kuramoto(self, output_dir, sample_size=50000):
    velocities = [1, 0.75, 0.5, 0.25]
    alignments = np.full((sample_size, 4), np.nan)
    for velocity_count, velocity in enumerate(velocities):
        for frame_count, idx in enumerate(np.where(np.abs(self.discrete_angular_velocity) == velocity)[0]):
            if frame_count == sample_size - 1:
                break
            else:
                directions = self.force_direction[idx][self.N_ants_around_springs[idx] > 0]
                alignments[frame_count, velocity_count] = utils.kuramoto_order_parameter(directions)[0]
    df = pd.DataFrame(alignments, columns=velocities)
    plt.clf()
    sns.barplot(data=df, palette="mako")
    plt.title("Ants alignment using Kuramoto order parameter")
    plt.xlabel("Discrete velocities")
    plt.ylabel("Alignment (Kuramoto order parameter)")
    # plt.ylim(0.9875, 0.995)
    plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
    plt.show()
    print("Saving figure to path: ", os.path.join(output_dir, f"Ants alignment - Kuramoto.png"))
    plt.savefig(os.path.join(output_dir, f"Ants alignment - Kuramoto.png"))
# plot_alignment(self, os.path.join(self.output_path, "alignment"), sample_size=50000)





def plot_single_profiles(self, output_path, sample_size=2000):
    os.makedirs(output_path, exist_ok=True)
    for profile in range(sample_size):
        plt.clf()
        ant = int(self.ant_profiles[profile, 0])
        spring = int(self.ant_profiles[profile, 1])
        start = int(self.ant_profiles[profile, 2])
        end = int(self.ant_profiles[profile, 3])
        length = end - start
        if length > 200:
            time_array = np.arange(length)
            velocity = self.discrete_angular_velocity[start:end]
            plt.scatter(self.profiled_fixed_end_angle_to_nest[profile, :length], self.profiled_force_magnitude[profile, :length], c=time_array, cmap="coolwarm")
            plt.legend()
            plt.title(f"ant: {ant}, spring: {spring}, start: {start}, end: {end}")
            plt.xlabel("Direction to nest (degrees)")
            plt.ylabel("Force magnitude (mN)")
            # limit x between 0 and 2pi
            plt.xlim(0, 2 * np.pi)
            plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
            print("Saving figure to path: ", os.path.join(output_path, f"{profile} profile.png"))
            plt.savefig(os.path.join(output_path, f"{profile} profile.png"))
# plot_single_profiles(self, os.path.join(self.output_path,"single_profiles_direction_to_nest"))


def draw_single_profiles(self, output_path, profile_min_length=200, start=0, end=None):
    # output_path = os.path.join(self.output_path, "single_profiles_S5760003")
    # start = self.sets_frames[0][0][0]
    # end = self.sets_frames[0][0][1]
    profile_min_length = 200
    profiles_idx = self.ant_profiles[:, 2] >= start
    profiles_idx *= self.ant_profiles[:, 3] < end if end is not None else np.ones_like(profiles_idx)
    tangential_force = self.profiled_tangential_force
    tangential_force[self.profiled_N_ants_around_springs != 1] = np.nan
    force_magnitude = self.profiled_force_magnitude
    force_magnitude[self.profiled_N_ants_around_springs != 1] = np.nan
    for profile_type in ["detaching", "attaching", "entire_profile"]:
        # profile_type = "entire_profile"
        type_output_path = os.path.join(output_path, profile_type)
        os.makedirs(os.path.join(type_output_path), exist_ok=True)
        if profile_type == "attaching":
            profile_bool = self.attaching_single_ant_profiles
        elif profile_type == "detaching":
            profile_bool = np.copy(self.detaching_single_ant_profiles)
        else:
            profile_bool = self.profiled_N_ants_around_springs == 1
            # profile_bool[~np.any(analysed.profiled_N_ants_around_springs == 1), :] = False
        occasions = (np.sum(profile_bool, axis=1) >= profile_min_length) * profiles_idx
        print(f"Saving {np.sum(occasions)} {profile_type} profiles to path: \n", type_output_path)
        profiles = np.arange(len(occasions))[occasions]
        if len(profiles) > 200:
            # randomly select 200 profiles
            profiles = np.random.choice(profiles, 200, replace=False)
        force_magnitude_type = force_magnitude[self.reverse_argsort] if profile_type == "detaching" else force_magnitude
        tangential_force_type = tangential_force[self.reverse_argsort] if profile_type == "detaching" else tangential_force
        for i in profiles:
            force_magnitude_i = force_magnitude_type[i, :]
            tangential_force_i = tangential_force_type[i, :]
            if profile_type == "attaching":
                force_magnitude_i = force_magnitude_i[:profile_min_length]
                tangential_force_i = tangential_force_i[:profile_min_length]
            elif profile_type == "detaching":
                force_magnitude_i = force_magnitude_i[-profile_min_length:]
                tangential_force_i = tangential_force_i[-profile_min_length:]
            else:
                force_magnitude_i = force_magnitude_i
                tangential_force_i = tangential_force_i
            info = self.ant_profiles[i, :]
            x = np.arange(np.sum(~np.isnan(force_magnitude_i)))
            magnitude_y = force_magnitude_i[~np.isnan(force_magnitude_i)]
            tangential_y = tangential_force_i[~np.isnan(tangential_force_i)]
            plt.clf()
            # create two plot in one figure
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            sns.lineplot(x=x, y=magnitude_y, ax=axs[0], color="purple")
            sns.lineplot(x=x, y=tangential_y, ax=axs[1], color="purple")
            plt.title(f"spring: {info[1]}, start: {info[2] - start}, end: {info[3] - start}, precedence: {info[4]}")
            plt.savefig(os.path.join(type_output_path, f"profile_{i}.png"))
# draw_single_profiles(self, output_path=os.path.join(self.output_path, "single_profiles_S5760003"), profile_min_length=200, start=self.sets_frames[0][0][0], end=self.sets_frames[0][0][1])








def NAntsProfiles_angular_velocity_over_N_ants(self, output_path, sample_size=100000):
    n_ants_velocities = np.full((sample_size, len(self.unique_n_ants)), np.nan)
    for n_count, n in enumerate(self.unique_n_ants):
        single_ant_labeled = self.N_ants_labeled[n_count]
        velocity = np.repeat(np.expand_dims(self.angular_velocity, axis=1), single_ant_labeled.shape[1], axis=1)
        velocities = velocity[single_ant_labeled != 0]
        if len(velocities) > sample_size:
            velocities = velocities[np.random.choice(np.arange(len(velocities)), size=sample_size, replace=False)]
        n_ants_velocities[:len(velocities), n_count] = velocities
    plt.clf()
    df = pd.DataFrame(n_ants_velocities, columns=self.unique_n_ants)
    sns.barplot(df, palette="rocket")
    plt.title(f"Angular velocity over N ants")
    plt.xlabel("Number of ants around the spring")
    plt.ylabel("Angular velocity (rad/s)")
    plt.show()
    plt.savefig(os.path.join(output_path, f"Angular velocity over N ants.png"))
# NAntsProfiles_angular_velocity_over_N_ants(self, os.path.join(self.output_path, "macro_analysis"))










# tested functions:
def check_velocity_changes(analysed, output_dir, window_size=5):
    moving_starts = analysed.velocity_change.copy()
    moving_idx = np.abs(analysed.discrete_angular_velocity[:-1][analysed.velocity_change[1:]]) > 0.25
    moving_starts[analysed.velocity_change][moving_idx] = False
    moving_starts = np.arange(len(analysed.velocity_change))[moving_starts]
    moving_stops = analysed.velocity_change.copy()
    stop_idx = np.abs(analysed.discrete_angular_velocity[:-1][analysed.velocity_change[1:]]) <= 0.25
    moving_stops[analysed.velocity_change][stop_idx] = False
    moving_stops = np.arange(len(analysed.velocity_change))[moving_stops]
    no_change = np.arange(len(analysed.velocity_change))[~analysed.velocity_change]
    random = np.random.choice(no_change, len(moving_starts), replace=False)

    magnitude = analysed.force_magnitude.copy()
    magnitude[analysed.N_ants_around_springs == 0] = np.nan
    tangential = analysed.tangential_force.copy()
    tangential[analysed.N_ants_around_springs == 0] = np.nan
    # direction = np.sign(analysed.force_direction)
    # direction[analysed.N_ants_around_springs != 1] = np.nan

    n_springs = analysed.force_magnitude.shape[1]
    # magnitude_data = np.full((np.sum(analysed.velocity_change), 3), np.nan, dtype=np.float64)
    before_tangential_data = np.full((np.sum(analysed.velocity_change) * n_springs, 3), np.nan, dtype=np.float64)
    after_tangential_data = np.full((np.sum(analysed.velocity_change) * n_springs, 3), np.nan, dtype=np.float64)
    # tangential_data = np.full((np.sum(analysed.velocity_change) * n_springs, 3), np.nan, dtype=np.float64)
    # ants_number_data = np.full((np.sum(analysed.velocity_change), 3), np.nan, dtype=np.float64)
    # alignment_data = np.full((np.sum(analysed.velocity_change), 3), np.nan, dtype=np.float64)
    for data_count, data in enumerate([moving_starts, moving_stops, random]):
        for count, i in enumerate(data):
            before_tangential_data[count, data_count] = np.nanmean(np.abs(tangential[i - window_size:i]))
            after_tangential_data[count, data_count] = np.nanmean(np.abs(tangential[i:i + window_size]))

            # magnitude_data[count, data_count] = np.nanmean(np.abs(magnitude[i - window_size:i])) - np.nanmean(np.abs(magnitude[i:i + window_size]))
            # magnitude_data[count*n_springs:count*n_springs+n_springs, data_count] = \
            #     np.nanmean(magnitude[i:i + window_size], axis=0) - np.nanmean(magnitude[i - window_size:i], axis=0)
            # tangential_data[count*n_springs:count*n_springs+n_springs, data_count] = \
            # tangential_data[count, data_count] = np.nanmean(np.abs(tangential[i:i + window_size])) - np.nanmean(np.abs(tangential[i - window_size:i]))
            # ants_number_data[count, data_count] = np.abs(np.nanmean(np.sum(analysed.N_ants_around_springs[i - window_size:i], axis=1)) -
            #                                              np.nanmean(np.sum(analysed.N_ants_around_springs[i:i + window_size], axis=1)))
            # direction_before = direction[i - window_size:i]
            # direction_after = direction[i:i + window_size]
            # alignment_data[count, 0] = np.abs(np.sum(direction_before == 1) / np.sum(np.isin(direction_before, [-1, 1])) - 0.5)
            # alignment_data[count, 1] = np.abs(np.sum(direction_after == 1) / np.sum(np.isin(direction_after, [-1, 1])) - 0.5)
    labels = ["Movement starts", "Movement stops", "Random"]
    plt.clf()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # df = pd.DataFrame(magnitude_data, columns=labels)
    # sns.barplot(df, palette="mako", ax=axes[0])
    # sns.boxplot(df, palette="mako", ax=axes[0], showfliers=False)
    # axes[0].set_title("Ants change in force magnitude after velocity change")
    # axes[0].set_ylabel("Mean force magnitude change")

    df = pd.DataFrame({"before": before_tangential_data[:, 0], "after": after_tangential_data[:, 0]})
    sns.barplot(df, palette="mako", ax=axes[0])
    axes[0].set_title("starts")
    axes[0].set_ylabel("Mean tangential force change")

    df = pd.DataFrame({"before": before_tangential_data[:, 1], "after": after_tangential_data[:, 1]})
    sns.barplot(df, palette="mako", ax=axes[1])
    axes[1].set_title("stops")
    axes[1].set_ylabel("Mean tangential force change")

    df = pd.DataFrame({"before": before_tangential_data[:, 2], "after": after_tangential_data[:, 2]})
    sns.barplot(df, palette="mako", ax=axes[2])
    axes[2].set_title("random")
    axes[2].set_ylabel("Mean tangential force change")

    # df = pd.DataFrame(tangential_data, columns=labels)
    # sns.barplot(df, palette="mako", ax=axes[1])
    # axes[1].set_title("Ants change in tangential force after velocity change")
    # axes[1].set_ylabel("Mean tangential force change")

    # df = pd.DataFrame(ants_number_data, columns=labels)
    # sns.barplot(df, palette="mako", ax=axes[2])
    # axes[2].set_title("Ants change in alignment after velocity change")
    # axes[2].set_ylabel("Alignment change")

    plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
    plt.tight_layout()
    plt.show()
    # plt.savefig(os.path.join(output_dir, f"Ants change in force after velocity change.png"))
# check_velocity_changes(self, self.output_path,  window_size=20)


def kuramoto_dynamics(analysed):
    direction = analysed.force_direction
    kuramoto_direction = np.full(direction.shape[0], np.nan, dtype=np.float64)
    for row in range(direction.shape[0]):
        row_direction = direction[row, :][analysed.N_ants_around_springs[row] > 0].copy()
        row_direction = row_direction[~np.isnan(row_direction)]
        if len(row_direction) > 1:
            kuramoto_direction[row] = utils.kuramoto_order_parameter(row_direction)[0]
    kuramoto_direction[kuramoto_direction < 0.85] = np.nan
    velocity = analysed.discrete_angular_velocity.copy()
    velocity = velocity / np.nanmax(np.abs(velocity)) * 0.05 + 1
    plt.clf()
    sns.lineplot(x=np.arange(0, kuramoto_direction.shape[0]), y=velocity, color="green", size=0.2, alpha=0.3)
    sns.lineplot(x=np.arange(0, kuramoto_direction.shape[0]), y=kuramoto_direction, color="orange", size=0.2, alpha=0.5)
    # remove legend
    plt.legend([], [], frameon=False)
    plt.show()
# kuramoto_dynamics(self)


def present_trajectories(analysed):
    signed_direction = analysed.force_direction.copy()
    direction_threshold = np.nanpercentile(np.abs(signed_direction), 40)
    signed_direction[np.abs(signed_direction) < direction_threshold] = 0
    signed_direction = np.sign(signed_direction)
    red_idx = signed_direction == 1
    blue_idx = signed_direction == -1
    white_idx = signed_direction == 0
    black_idx = np.isnan(signed_direction)
    arranged_time = [np.arange(signed_direction.shape[0]) for i in range(signed_direction.shape[1])]
    arranged_time = np.vstack(arranged_time).transpose()
    magnitude = analysed.force_magnitude.copy()
    magnitude = magnitude / np.nanmax(np.abs(magnitude)) + np.arange(1, signed_direction.shape[1] + 1, 1)
    tangential_force = analysed.tangential_force.copy()
    tangential_force = tangential_force / np.nanmax(np.abs(tangential_force)) + np.arange(1, signed_direction.shape[1] + 1, 1)
    angular_velocity = analysed.discrete_angular_velocity.copy() / 4
    angular_velocity = np.repeat(np.expand_dims(angular_velocity, axis=1), signed_direction.shape[1], axis=1) + np.arange(1, signed_direction.shape[1] + 1, 1)
    plt.clf()
    for line in range(angular_velocity.shape[1]):
        sns.lineplot(x=arranged_time[:, line], y=angular_velocity[:, line], color='green', alpha=0.3)
        plt.axhline(y=line + 1, color='gray', alpha=0.7)
    # sns.scatterplot(x=arranged_time[blue_idx], y=tangential_force[blue_idx], color='blue', s=1)
    sns.scatterplot(x=arranged_time[blue_idx], y=magnitude[blue_idx], color='blue', s=1)
    # sns.scatterplot(x=arranged_time[red_idx], y=tangential_force[red_idx], color='red', s=1)
    sns.scatterplot(x=arranged_time[red_idx], y=magnitude[red_idx], color='red', s=1)
    # sns.scatterplot(x=arranged_time[white_idx], y=tangential_force[white_idx], color='white', s=1)
    # sns.scatterplot(x=arranged_time[black_idx], y=tangential_force[black_idx], color='black', s=1)
    plt.yticks(np.arange(1, signed_direction.shape[1] + 1, 1))
    frame = analysed.sets_frames[0][0][-1] + 820
    plt.axvline(x=frame, color='black', alpha=0.5)
    plt.show()
# present_trajectories(self)





def profiles_direction_preference(analysed, sample_size=2000):
    direction_percentile = np.nanpercentile(analysed.force_direction, 50)
    velocities = [1, 0.75, 0.5, 0.25, 0]
    direction_preference = np.full((sample_size, len(velocities)), np.nan)
    for profile in range(sample_size):
        spring = int(analysed.ant_profiles[profile, 1]) - 1
        start = int(analysed.ant_profiles[profile, 2])
        end = int(analysed.ant_profiles[profile, 3])
        direction = analysed.force_direction[start:end, spring].copy()
        direction[direction < direction_percentile] = np.nan
        direction[analysed.N_ants_around_springs[start:end, spring] != 1] = np.nan
        signed_direction = np.sign(direction)
        for velocity_count, velocity in enumerate(velocities):
            signed_direction_copy = signed_direction.copy()
            signed_direction_copy[analysed.discrete_angular_velocity[start:end] != velocity] = np.nan
            unique, counts = np.unique(signed_direction_copy, return_counts=True)
            direction_preference[profile, velocity_count] = counts[np.argmax(counts)] / np.sum(counts)
    plt.clf()
    fig, axs = plt.subplots(1, len(velocities), figsize=(25, 5))
    for i in range(len(velocities)):
        sns.histplot(direction_preference[:, i], ax=axs[i], bins=20)
        # put mean and std on plot
        mean = np.nanmean(direction_preference[:, i])
        std = np.nanstd(direction_preference[:, i])
        axs[i].axvline(mean, color="red")
        plt.title(f"Direction preference for velocity {velocities[i]}")
        plt.xlabel("Direction preference")
        plt.ylabel("Frequency")
    plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
    print("Saving figure to path: ", os.path.join(analysed.output_path, "direction_preference.png"))
    plt.savefig(os.path.join(analysed.output_path, "direction_preference.png"))
# profiles_direction_preference(self, sample_size=14000)


# unused functions:
def distributions_with_binned_angle_to_nest2(self, output_path, bins=50, sample_size=50000):
    # new_joiners_frames, new_joiners_springs = np.where(np.diff(self.N_ants_around_springs, axis=0) == 1)
    # before_zero = self.N_ants_around_springs[new_joiners_frames, new_joiners_springs]
    # new_joiners_frames = new_joiners_frames[before_zero == 0]
    # new_joiners_springs = new_joiners_springs[before_zero == 0]
    # new_joiners_frames += 1
    #
    angles = np.linspace(0, 2 * np.pi, bins)

    # for data_title, types_data in zip(["force magnitude", "tangential force"], [self.force_magnitude, self.tangential_force]):
    #     data = np.full((bins, sample_size), np.nan)
    #     for profile_count, profile in enumerate(zip(new_joiners_frames, new_joiners_springs):
    #         if np.all(self.N_ants_around_springs[profile[0]:profile[0]+50, profile[1]]==1):
    #             data[]
    leaders_bool_idx = self.profiles_beginnings * (self.N_ants_around_springs == 1)
    non_leaders_bool_idx = ~self.profiles_beginnings * (self.N_ants_around_springs == 1)
    leader_labels = ["non-leaders", "leaders"]
    leaders_bools = [non_leaders_bool_idx, leaders_bool_idx]
    for data_title, types_data in zip(["force magnitude", "tangential force"], [self.force_magnitude, self.tangential_force]):
        percentile = np.nanpercentile(types_data, 90, axis=0)
        force_bool_idx = types_data > percentile
        strong_pulls_data = np.full((len(angles) - 1, 2), np.nan, dtype=np.float64)
        all_pulls_data = np.full((len(angles) - 1, 2), np.nan, dtype=np.float64)
        for angle_count in range(len((angles[:-1]))):
            angle_bool_idx = np.logical_and(self.fixed_end_angle_to_nest >= angles[angle_count], self.fixed_end_angle_to_nest < angles[angle_count + 1])
            angle_bool_idx[~np.isin(self.discrete_angular_velocity, [1, 0.75, 0.5]), :] = False
            for leaders_count, leaders_bool in enumerate(leaders_bools):
                final_bool_idx = angle_bool_idx.copy()
                final_bool_idx[~leaders_bool] = False
                strong_pulls_data[angle_count, leaders_count] = np.sum(force_bool_idx[final_bool_idx])
                all_pulls_data[angle_count, leaders_count] = np.sum(final_bool_idx)
        plt.clf()
        theta = angles[:-1]
        theta = np.append(theta, theta[0])
        data_copy = strong_pulls_data / all_pulls_data
        data_copy = np.append(data_copy, data_copy[0, :].reshape(1, -1), axis=0)
        ax = plt.subplot(111, projection='polar')
        ax.set_title(f"Distribution over nest direction of {data_title} for leaders and non-leaders")
        for col in range(len(leader_labels)):
            r = data_copy[:, col]
            if col != 0:
                r += data_copy[:, col - 1]
            ax.plot(theta, r, label=f"{leader_labels[col]}")
            if col != 0:
                ax.fill_between(theta, r, data_copy[:, col - 1], alpha=0.2)
            else:
                ax.fill_between(theta, r, 0, alpha=0.2)
        ax.set_theta_offset(np.pi)
        ax.legend(title="", loc='upper right')
        ax.set_yticklabels([])
        ax.grid(True)
        ax.set_rorigin(-0.1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"Distribution over nest direction of {data_title} for leaders and nonleaders.png"))
# distributions_with_binned_angle_to_nest2(self, os.path.join(self.output_path, "nest_direction_distributions"), bins=50)

# def NAntsProfiles_tangential_force(self, output_path, length=200, sample_size=5000):
#     length, sample_size = 250, 5000
#     profiles = np.unique(self.N_ants_labeled[0])[1:]
#     mean = np.full((len(profiles), 2), np.nan)
#     std = np.full((len(profiles), 2), np.nan)
#     percentile = np.full((len(profiles), 2), np.nan)
#     # tangential_force = np.abs(self.tangential_force)
#     tangential_force = np.abs(self.force_magnitude)
#     for profile_count, profile in enumerate(profiles):
#         # print(f"Profile {profile_count} out of {len(profiles)}\r", end="")
#         apytl.Bar().drawbar(profile_count, len(profiles), fill='*')
#         idx_bool = self.N_ants_labeled[0] == profile
#         if np.sum(idx_bool) >= length:
#             idx = np.where(idx_bool)
#             start, end, spring = idx[0][0], idx[0][-1], idx[1][0]
#             profile_tangential = tangential_force[start:end + 1, spring]
#             if start != 0 and self.N_ants_around_springs[start - 1, spring] == 0:
#                 mean[profile_count, 0] = np.mean(profile_tangential[100:length])
#                 std[profile_count, 0] = np.std(profile_tangential[100:length])
#                 percentile[profile_count, 0] = np.percentile(profile_tangential[100:length], 90)
#             if (end - start) >= (length + 25 * 7 + 150) and end + 1 != tangential_force.shape[0] and self.N_ants_around_springs[end + 1, spring] == 0:
#                 mean[profile_count, 1] = np.mean(profile_tangential[length:length + 150])
#                 std[profile_count, 1] = np.std(profile_tangential[length:length + 150])
#                 percentile[profile_count, 1] = np.percentile(profile_tangential[length:length + 150], 90)
#     plt.clf()
#     fig, axs = plt.subplots(1, 3, figsize=(18, 5))
#     for count, (title, data) in enumerate(zip(["Mean", "Std", "Strongest"], [mean, std, percentile])):
#         df = pd.DataFrame(data, columns=["Start", "End"])
#         sns.barplot(data=df, palette="rocket", ax=axs[count])
#         axs[count].set_title(f"{title} tangential force at the start and end of the profile")
#         axs[count].set_xlabel("Profile position")
#         axs[count].set_ylabel("Tangential force")
#         utils.draw_significant_stars(df, axs=axs[count])
#     plt.show()
# NAntsProfiles_tangential_force(self, os.path.join(self.output_path, ""))


# def plot_colors():
#     colors = sns.color_palette("rocket")
#     #put all colors in one plot, with their names:
#     plt.figure(figsize=(12, 2))
#     for i, color in enumerate(colors):
#         plt.subplot(1, len(colors), i+1)
#         plt.xticks([])
#         plt.yticks([])
#         plt.title(color)
#         plt.axhline(i, color=color, linewidth=10)
#     plt.show()
#
#
# def define_colors(labels_num=5):
#     # set a color by RGB
#     # red, purple, blue, green, purple_brown
#     red = np.array((239, 59, 46))/255
#     purple = np.array((112, 64, 215))/255
#     blue = np.array((86, 64, 213))/255
#     green = np.array((93, 191, 71))/255
#     purple_brown = np.array((153, 86, 107))/255
#     colors = [red, purple, blue, green, purple_brown]
#     return colors[:labels_num+1]
#
#
# def define_output_dir(output_dir):
#     if output_dir is not None:
#         output_dir = os.path.join(output_dir, "plots")
#         os.makedirs(output_dir, exist_ok=True)
#     else:
#         raise ValueError("Please provide an output directory")
#     return output_dir
#
#
# def plot_angular_velocity_distribution(analysed, start=0, end=None, window_size=1, title="", output_dir=None):
#     plt.clf()
#     if end is None:
#         end = analysed.angular_velocity.shape[0]
#
#     x = np.abs(analysed.angular_velocity[start:end])
#     x_nans = np.isnan(x)
#     x = x[~x_nans]
#     sns.displot(x, kind="kde", fill=True, cmap="mako")
#     plt.title(f"angular velocity distribution (movie:{title})")
#     plt.xlabel("angular velocity (rad/ 20_frames)")
#     plt.ylabel("density")
#     # leave enough margin for the title
#     plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
#     # save the figure
#     output_dir = define_output_dir(output_dir)
#     print("Saving figure to path: ", os.path.join(output_dir, f"angular velocity distribution (movie {title}).png"))
#     plt.savefig(os.path.join(output_dir, f"angular velocity distribution (movie {title}).png"))
# # plot_angular_velocity_distribution(analysed, start=0, end=None, window_size=50, title=video_name, output_dir=directory)
#
#
# def ant_decisiveness(analysed, output_dir, profile_size, sample_size=5000):
#     os.makedirs(output_dir, exist_ok=True)
#     leading_bool = analysed.attaching_single_ant_profiles.copy()
#     leading_bool[~leading_bool[:, profile_size-1], :] = False
#     leading_bool[:, profile_size:] = False
#
#     after_leading_bool = analysed.middle_events.copy()
#     after_leading_bool[:, :profile_size] = False
#     idx = np.where(after_leading_bool)
#     line_idx = np.arange(len(idx[0]))
#     for line in np.unique(idx[0]):
#         where_in_line = idx[0] == line
#         first_in_line = line_idx[where_in_line][0]
#         after_leading_bool[line, first_in_line + profile_size:] = False
#
#     angular_velocity = analysed.profiled_discrete_angular_velocity.copy()
#     velocities = [1, 0.75, 0.5, 0.25, 0]
#     velocities = [1, 0.5, 0]
#     # force_thresholds = [np.nanpercentile(np.abs(analysed.profiled_tangential_force), percent) for percent in [0, 50, 60, 70, 80, 90]] + [np.nanmax(np.abs(analysed.profiled_tangential_force))]
#     # force_threshold = np.nanpercentile(np.abs(analysed.profiled_tangential_force), 80)
#     profiled_force_direction = analysed.profiled_force_direction.copy()
#     force_direction_threshold = np.nanpercentile(np.abs(profiled_force_direction), 60)
#     profiled_force_direction[np.abs(profiled_force_direction) < force_direction_threshold] = 0
#     for data_bool, title in zip([leading_bool, after_leading_bool], ["while leading", "after leading"]):
#         # data_bool, title = leading_bool, "while leading"
#         results = np.full((sample_size, len(velocities)-1), np.nan, dtype=np.float64)
#         # data_bool_copy = data_bool.copy()
#         # data_bool_copy[np.abs(analysed.profiled_force_direction) < force_direction_threshold] = False
#         # tangential_force = analysed.profiled_tangential_force.copy()
#         # tangential_force[(np.abs(analysed.profiled_tangential_force) < force_threshold)] = 0  # + (np.abs(analysed.profiled_tangential_force) > force_thresholds[count+1])] = 0
#         # tangential_force[~data_bool] = 0
#         for count_velocity, velocity in enumerate(velocities):
#             if count_velocity != len(velocities) - 1:
#                 idx_velocity = (np.abs(angular_velocity) > velocity) * (np.abs(angular_velocity) <= velocities[count_velocity + 1])
#                 # idx_velocity = (np.abs(angular_velocity) <= 1) * (np.abs(angular_velocity) > 0.75)
#             # else:  # all velocities above 0
#             #     idx_velocity = np.abs(angular_velocity) >= velocity
#                 singed_direction = np.sign(profiled_force_direction)
#                 singed_direction[idx_velocity] = 0
#                 idx_not_zero = np.where(np.isin(singed_direction, [-1, 1]))
#                 unique_idx = np.unique(idx_not_zero[0])
#                 # randomly take 10000 events
#                 if len(unique_idx) > sample_size:
#                     unique_idx = np.random.choice(unique_idx, sample_size, replace=False)
#                 for count, row in enumerate(unique_idx):
#                     # length = len(idx_not_zero[1][idx_not_zero[0] == row])
#                     n_switching = np.sum(np.diff(singed_direction[row, idx_not_zero[1][idx_not_zero[0] == row]]) != 0)
#                     results[count, count_velocity] = n_switching
#
#         df = pd.DataFrame(results, columns=velocities[:-1])
#         plt.clf()
#         # sns.violinplot(data=df, palette="mako")
#         sns.barplot(data=df, palette="mako")
#         plt.title(f"Ants decisiveness - {title}")
#         plt.xlabel("Velocities")
#         plt.ylabel("Number of direction switching events")
#         # leave enough margin for the title
#         plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
#         plt.show()
#         print("Saving figure to path: ", os.path.join(output_dir, f"Ants decisiveness - {title} 2 velocities.png"))
#         plt.savefig(os.path.join(output_dir, f"Ants decisiveness - {title} 2 velocities.png"))
# # ant_decisiveness(self, output_dir, profile_size=200, sample_size=5000)
#
#
# def velocity_influence_on_force_magnitude_change_rate(analysed, output_path, sample_size=10000):
#     velocities = [1, 0.75, 0.5, 0.25, 0]
#     sample_size = 100000
#     profile_N_ants = analysed.profiled_N_ants_around_springs.copy()
#     angular_velocity = analysed.profiled_discrete_angular_velocity.copy()
#     angular_velocity = np.abs(angular_velocity[:, 1:])
#     profiled_magnitude = analysed.profiled_force_magnitude.copy()
#     profiled_magnitude[~(profile_N_ants == 1)] = np.nan
#     profiled_magnitude[:, :25] = np.nan
#     profiled_tangential_force = analysed.profiled_tangential_force
#     profiled_tangential_force[~(profile_N_ants == 1)] = np.nan
#     profiled_tangential_force[:, :25] = np.nan
#     tangential_force_percentile = np.nanpercentile(np.abs(profiled_tangential_force), 60)
#     # profiled_tangential_force[np.abs(profiled_tangential_force) < tangential_force_percentile] = 0
#     tangential_fluctuations = np.abs(np.diff(profiled_tangential_force, axis=1))
#     magnitude_fluctuations = np.abs(np.diff(profiled_magnitude, axis=1))
#     data = np.full((sample_size, len(velocities)-1,), np.nan, dtype=np.float64)
#     for count, velocity in enumerate(velocities[:-1]):
#         idx_velocity = (angular_velocity <= velocity) * (angular_velocity > velocities[count+1])
#         vector = magnitude_fluctuations[idx_velocity].flatten()
#         #rake only the first 10000 events
#         if len(vector) > sample_size:
#             vector = np.random.choice(vector, sample_size, replace=False)
#         data[:sample_size, count] = vector
#     df = pd.DataFrame(data, columns=velocities[:-1])
#     plt.clf()
#     # sns.violinplot(data=df, palette="mako")
#     # sns.boxplot(data=df, palette="mako")
#     sns.barplot(data=df, palette="mako")
#     # lmit the y axis to 0.5
#     # plt.ylim(0, 0.01)
#     plt.title("Ants tangential force fluctuations")
#     plt.xlabel("Velocities")
#     plt.ylabel("Tangential force fluctuations")
#     # leave enough margin for the title
#     plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
#     plt.show()
#     # plt.savefig(os.path.join(output_path, f"Ants tangential force fluctuations.png"))
# # velocity_influence_on_force_magnitude_change_rate(self, os.path.join(self.output_path), sample_size=10000)
#
#
# def velocity_influence_on_force_magnitude(analysed, output_dir, profile_size=200):
#     os.makedirs(output_dir, exist_ok=True)
#     leading_bool = analysed.attaching_single_ant_profiles.copy()
#     leading_bool[~leading_bool[:, profile_size-1], :] = False
#     leading_bool[:, profile_size:] = False
#
#     after_leading_bool = analysed.middle_events.copy()
#     after_leading_bool[:, :profile_size] = False
#     idx = np.where(after_leading_bool)
#     line_idx = np.arange(len(idx[0]))
#     for line in np.unique(idx[0]):
#         where_in_line = idx[0] == line
#         first_in_line = line_idx[where_in_line][0]
#         after_leading_bool[line, first_in_line + profile_size:] = False
#
#     force_magnitude = analysed.profiled_force_magnitude.copy()
#     angular_velocity = analysed.profiled_discrete_angular_velocity.copy()
#     velocities = [1, 0.75, 0.5, 0.25, 0]
#     for data_bool, title in zip([leading_bool, after_leading_bool], ["while leading", "after leading"]):
#         # force_magnitude_vs_velocity = np.full((1, 4), 0, dtype=np.float64)
#         force_magnitude_vs_velocity = np.full((10000, 4), 0, dtype=np.float64)
#         for count_velocity, velocity in enumerate(velocities):
#             if count_velocity != len(velocities) - 1:
#                 data_bool_copy = data_bool.copy()
#                 data_bool_copy[(np.abs(angular_velocity) > velocity) * (np.abs(angular_velocity) <= velocities[count_velocity + 1])] = False
#                 # idx = (np.abs(angular_velocity[data_bool]) <= velocity) * (np.abs(angular_velocity[data_bool]) > velocities[count_velocity + 1])
#                 # force_magnitude_vector = force_magnitude[data_bool][idx]
#                 force_magnitude_vector = force_magnitude[data_bool_copy]
#                 force_magnitude_vector = force_magnitude_vector[~np.isnan(force_magnitude_vector)]
#                 if len(force_magnitude_vector) > 10000:
#                     np.random.seed(42)
#                     force_magnitude_vector = np.random.choice(force_magnitude_vector, 10000, replace=False)
#                 force_magnitude_vs_velocity[:len(force_magnitude_vector), count_velocity] = force_magnitude_vector
#                 # force_magnitude_vs_velocity[0, count_velocity] = np.median(force_magnitude_vector)
#         df = pd.DataFrame(force_magnitude_vs_velocity, columns=["very fast", "fast", "slow", "very slow"])
#         plt.clf()
#         plt.figure(figsize=(7, 5))
#         color_palette = sns.color_palette("mako", as_cmap=True)
#         sns.barplot(data=df, palette='Blues')
#         plt.title(f"Force magnitude over discrete angular velocity - {title}")
#         plt.ylabel("Force magnitude")
#         print("Saving figure to path: ", os.path.join(output_dir, f"Force magnitude over discrete angular velocity - {title}.png"))
#         plt.savefig(os.path.join(output_dir, f"Force magnitude over discrete angular velocity - {title}.png"))
#         # plt.clf()
#         # plt.figure(figsize=(7, 5))
#         # sns.violinplot(data=df, palette='Blues')
#         # plt.title(f"Force magnitude over discrete angular velocity - {title}")
#         # plt.ylabel("Force magnitude")
#         # print("Saving figure to path: ", os.path.join(output_dir, f"Force magnitude over discrete angular velocity - {title} - violinplot.png"))
#         # plt.savefig(os.path.join(output_dir, f"Force magnitude over discrete angular velocity - {title} - violinplot.png"))
# # velocity_influence_on_force_magnitude(self, output_dir, profile_size=200)
#
#
# def velocity_influence_on_alignment(analysed, output_dir, profile_size=200):
#     os.makedirs(output_dir, exist_ok=True)
#     leading_bool = analysed.attaching_single_ant_profiles.copy()
#     leading_bool[~leading_bool[:, profile_size-1], :] = False
#     leading_bool[:, profile_size:] = False
#
#     after_leading_bool = analysed.middle_events.copy()
#     after_leading_bool[:, :profile_size] = False
#     idx = np.where(after_leading_bool)
#     line_idx = np.arange(len(idx[0]))
#     for line in np.unique(idx[0]):
#         where_in_line = idx[0] == line
#         first_in_line = line_idx[where_in_line][0]
#         after_leading_bool[line, first_in_line + profile_size:] = False
#     angular_velocity = analysed.profiled_discrete_angular_velocity.copy()
#     percentiles = [0, 20, 40, 60, 80]
#     velocities = [1, 0.75, 0.5, 0.25, 0]
#
#     # force_thresholds = [np.nanpercentile(np.abs(analysed.profiled_tangential_force), percent) for percent in percentiles] + [np.nanmax(np.abs(analysed.profiled_tangential_force))]
#     force_thresholds = [np.nanpercentile(np.abs(analysed.profiled_force_magnitude), percent) for percent in percentiles] + [np.nanmax(np.abs(analysed.profiled_force_magnitude))]
#     force_direction = np.sin(analysed.profiled_force_direction)
#     force_direction_threshold = np.nanpercentile(np.abs(force_direction), 80)
#     force_direction[np.abs(force_direction) < force_direction_threshold] = 0
#     for data_bool, title in zip([leading_bool, after_leading_bool], ["while leading", "after leading"]):
#         results = np.full((5, 4, 60000), 0, dtype=np.float64)
#         for count, force_threshold in enumerate(force_thresholds):
#             if count != len(force_thresholds) - 1:
#                 # force_magnitude = analysed.profiled_force_magnitude.copy()
#                 # tangential_force = analysed.profiled_tangential_force.copy()
#                 # tangential_force[(np.abs(analysed.profiled_tangential_force) < force_threshold) + (np.abs(analysed.profiled_tangential_force) > force_thresholds[count+1])] = 0
#                 # data_bool_copy[np.abs(analysed.profiled_force_direction) < force_direction_threshold] = False
#                 # force_magnitude[(np.abs(analysed.profiled_force_magnitude) < force_threshold) + (np.abs(analysed.profiled_force_magnitude) > force_thresholds[count+1])] = 0
#                 # alignment = np.sign(tangential_force[data_bool_copy]) * np.sign(angular_velocity[data_bool_copy])
#                 # alignment_per_velocity = np.full((4, len(alignment)), np.nan)
#                 data_bool_copy = data_bool.copy()
#                 data_bool_copy[(np.abs(analysed.profiled_force_magnitude) < force_threshold) + (np.abs(analysed.profiled_force_magnitude) > force_thresholds[count+1])] = False
#                 alignment = np.sign(force_direction[data_bool_copy]) * np.sign(angular_velocity[data_bool_copy])
#                 alignment_per_velocity = np.full((4, 60000), np.nan)
#                 for count_velocity, velocity in enumerate(velocities):
#                     if count_velocity != len(velocities) - 1:
#                         idx = (np.abs(angular_velocity[data_bool_copy]) <= velocity) * (np.abs(angular_velocity[data_bool_copy]) > velocities[count_velocity + 1])
#                         length = len(angular_velocity[data_bool_copy][idx])
#                         alignment_per_velocity[count_velocity, :length] = alignment[idx]
#                         # results[count, count_velocity, :length] = alignment[idx]
#                 # percentage = np.sum(alignment_per_velocity == 1, axis=1)/np.sum(np.isin(alignment_per_velocity, [-1, 1]), axis=1)
#                 # results[count, :, :] = percentage
#                 results[count, :, :] = alignment_per_velocity
#         results_percentage = np.sum(results == 1, axis=2) / np.sum(np.isin(results, [-1, 1]), axis=2)
#         error = np.sqrt(results_percentage * (1 - results_percentage) / np.sum(np.isin(results, [-1, 1]), axis=2))
#         results_percentage = results_percentage.transpose()
#         error = error.transpose()
#         # df = pd.DataFrame(results_percentage, columns=[str(i) for i in velocities[:-1]], index=[f"{percentiles[i]}%-{percentiles[i+1]}%" for i in range(len(percentiles)-1)] + [f"{percentiles[-1]}%-100%"])
#         df = pd.DataFrame(results_percentage, columns=[f"{percentiles[i]}%-{percentiles[i+1]}%" for i in range(len(percentiles)-1)] + [f"{percentiles[-1]}%-100%"], index=[str(i) for i in velocities[:-1]])
#         plt.clf()
#         plt.figure(figsize=(10, 5))
#         for i, velocity in enumerate(velocities[:-1]):
#             plt.plot(df.columns, df.iloc[i], marker='o', color=f"C{i}", label=f"{velocity:.2f}")
#             plt.fill_between(df.columns, df.iloc[i] - error[i], df.iloc[i] + error[i], alpha=0.3)
#         plt.title(f"Ants alignment - {title}")
#         plt.legend(title="Velocity")
#         plt.ylabel("Alignment (%)")
#         plt.xlabel("Force magnitude range (percentile)")
#         plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
#         print("Saving figure to path: ", os.path.join(output_dir, f"Ants alignment (velocity and magnitude) - {title}.png"))
#         plt.savefig(os.path.join(output_dir, f"Ants alignment (velocity and magnitude) - {title}.png"))
#         plt.show()
# # velocity_influence_on_alignment(self, os.path.join(self.output_path, "alignment"), profile_size=200)
#
#
# def check_switching(analysed, margin=20, sample_size=10000):
#     results = np.full((4, sample_size), np.nan, dtype=np.float64)
#     angular_velocity = analysed.discrete_angular_velocity
#     signed_direction = np.sign(np.sin(analysed.force_direction))
#     velocities = [1, 0.75, 0.5, 0.25, 0]
#     for velocity_count, velocity in enumerate(velocities[:-1]):
#         idx_velocity = np.where((np.abs(angular_velocity) <= velocity) * (np.abs(angular_velocity) > velocities[velocity_count + 1]))[0]
#         if len(idx_velocity) > sample_size:
#             idx_velocity = np.random.choice(idx_velocity, sample_size)
#         for moment_count, moment in enumerate(idx_velocity):
#             direction_before = np.sum(signed_direction[moment - margin:moment] == 1, axis=0) / np.sum(np.isin(signed_direction[moment - margin:moment], [-1, 1]), axis=0)
#             direction_after = np.sum(signed_direction[moment:moment + margin] == 1, axis=0) / np.sum(np.isin(signed_direction[moment:moment + margin], [-1, 1]), axis=0)
#             switching_percent = np.sum(np.abs(direction_before - direction_after) >= 0.5) / 20
#             results[velocity_count, moment_count] = switching_percent
#     df = pd.DataFrame(results.transpose(), columns=[str(i) for i in velocities[:-1]])
#     plt.clf()
#     plt.figure(figsize=(10, 5))
#     # create boxplot
#     df.boxplot()
#     plt.title(f"Switching percentage")
#     plt.ylabel("Switching percentage (%)")
#     plt.xlabel("Velocity")
#     plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
#     plt.show()
#     print("Saving figure to path: ", os.path.join(analysed.output_path, f"Switching percentage.png"))
#     plt.savefig(os.path.join(analysed.output_path, f"Switching percentage.png"))
# # check_switching(self, margin=10)
#
#
# def distribution_of_ants_correlation(analysed, sample_size=1000):
#     angular_velocity = utils.calc_angular_velocity(analysed.fixed_end_angle_to_nest, diff_spacing=20) / 20
#     angular_velocity = np.where(np.isnan(angular_velocity).all(axis=1), np.nan, np.nanmedian(angular_velocity, axis=1))
#     outlier_threshold = np.nanpercentile(np.abs(angular_velocity), 99)
#     angular_velocity[np.abs(angular_velocity) > outlier_threshold] = np.nan
#     profiles_correlation = np.full((sample_size), np.nan, dtype=np.float64)
#     for profile in range(1000):
#         spring = int(analysed.ant_profiles[profile, 1])
#         start = int(analysed.ant_profiles[profile, 2])
#         end = int(analysed.ant_profiles[profile, 3])
#         magnitude = np.abs(analysed.force_magnitude[start:end, spring-1])
#         tangential_force = analysed.tangential_force[start:end, spring-1]
#         velocity = angular_velocity[start:end]
#         # correlation = np.corrcoef(velocity[~np.isnan(velocity)], magnitude[~np.isnan(velocity)])[0, 1]
#         correlation = np.corrcoef(velocity[~np.isnan(velocity)], tangential_force[~np.isnan(velocity)])[0, 1]
#         # kendall correlation
#         # correlation = stats.kendalltau(velocity[~np.isnan(velocity)], magnitude[~np.isnan(velocity)])[0]
#         # spearman correlation
#         # correlation = stats.spearmanr(velocity[~np.isnan(velocity)], magnitude[~np.isnan(velocity)])[0]
#         profiles_correlation[profile] = correlation
#     plt.clf()
#     plt.figure(figsize=(10, 5))
#     plt.hist(profiles_correlation, bins=50)
#     plt.title(f"Correlation between velocity and force magnitude")
#     plt.ylabel("Frequency")
#     plt.xlabel("Correlation")
#     plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
#     plt.show()
#     print("Saving figure to path: ", os.path.join(analysed.output_path, f"Correlation between velocity and force magnitude.png"))
#     plt.savefig(os.path.join(analysed.output_path, f"Correlation between velocity and force magnitude.png"))
# # distribution_of_ants_correlation(self, sample_size=100000)
#
#
# def distributions_over_angle_to_nest(analysed):
#     force_magnitude_percentile = np.nanpercentile(analysed.force_magnitude, 90, axis=0)
#     tangential_force_percentile = np.nanpercentile(analysed.tangential_force, 90, axis=0)
#     high_force_idx = analysed.force_magnitude > force_magnitude_percentile
#     high_tangential_idx = analysed.tangential_force > tangential_force_percentile
#     velocities = [1, 0.75, 0.5, 0.25, 0]
#     colors = sns.color_palette("rocket")
#     for title_data, data_idx in zip(["High force magnitude", "High tangential force"], [high_force_idx, high_tangential_idx]):
#         plt.clf()
#         fig, axs = plt.subplots(1, len(velocities) + 1, figsize=(20, 5))
#         fig.suptitle(f'Angle to nest distribution for {title_data}')
#         for count, velocity in enumerate(velocities):
#             angle_to_nest = analysed.fixed_end_angle_to_nest.copy()
#             angle_to_nest[analysed.discrete_angular_velocity != velocity] = np.nan
#             sns.histplot(angle_to_nest[data_idx], bins=50, ax=axs[count], color=colors[count])
#             axs[count].set_title(f"Velocity: {velocity}")
#             if count != 0:
#                 axs[count].set_ylabel("")
#         angle_to_nest = analysed.fixed_end_angle_to_nest.copy()
#         sns.histplot(angle_to_nest[data_idx], bins=50, ax=axs[-1], color="black")
#         axs[-1].set_title(f"All velocities")
#         plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
#         plt.tight_layout()
#         print("Saving figure to path: ", os.path.join(analysed.output_path, f"Angle to nest distribution for {title_data}.png"))
#         plt.savefig(os.path.join(analysed.output_path, f"Angle to nest distribution for {title_data}.png"))
# # distributions_over_angle_to_nest(self)
#
#

#
# def plot_alignment(analysed, output_dir, profile_size=200):
#     os.makedirs(output_dir, exist_ok=True)
#     leading_bool = analysed.attaching_single_ant_profiles.copy()
#     leading_bool[~leading_bool[:, profile_size-1], :] = False
#     leading_bool[:, profile_size:] = False
#
#     after_leading_bool = analysed.middle_events.copy()
#     after_leading_bool[:, :profile_size] = False
#     idx = np.where(after_leading_bool)
#     line_idx = np.arange(len(idx[0]))
#     for line in np.unique(idx[0]):
#         where_in_line = idx[0] == line
#         first_in_line = line_idx[where_in_line][0]
#         after_leading_bool[line, first_in_line + profile_size:] = False
#
#     percentiles = [0, 50, 100]
#     # force_titles = [f"Force range (percentile): {percentiles[i]}-{percentiles[i + 1]}" for i in range(len(percentiles) - 1)]
#     velocity_titles = ["slow", "fast"]
#     force_titles = ["Low force magnitude", "High force magnitude"]
#     leading_titles = ["While leading", "After leading"]
#     colors = np.array([["skyblue", "olive"], ["gold", "teal"]])  #, "red"]
#     # force_thresholds = [np.nanpercentile(np.abs(analysed.profiled_force_magnitude), percent) for percent in percentiles] + [np.nanmax(np.abs(analysed.profiled_force_magnitude))]
#     angular_velocity = analysed.profiled_discrete_angular_velocity.copy()
#     velocity_bools = [(angular_velocity <= 0.5)*(angular_velocity != 0), (angular_velocity > 0.5)]
#
#     force_direction = np.sin(analysed.profiled_force_direction)
#     force_direction_threshold = np.nanpercentile(np.abs(force_direction), 60)
#     force_direction[np.abs(force_direction) < force_direction_threshold] = 0
#
#     force_threshold = np.nanpercentile(np.abs(analysed.profiled_force_magnitude), 50)
#     force_bools = [(np.abs(analysed.profiled_force_magnitude) < force_threshold), (np.abs(analysed.profiled_force_magnitude) >= force_threshold)]
#     for velocity_bool, velocity_title in zip(velocity_bools, velocity_titles):
#         plt.clf()
#         fig, axs = plt.subplots(2, 2, figsize=(12, 8))
#         fig.suptitle(f"Alignment percentage - {velocity_title} object velocity")
#         fig.text(0.5, 0.01, 'Alignment (%)', ha='center')
#         for data_count, data_bool in enumerate([leading_bool, after_leading_bool]):
#             for force_count, force_bool in enumerate(force_bools):
#                 force_direction_copy = force_direction.copy()
#                 force_direction_copy[~force_bool] = 0
#                 force_direction_copy[~velocity_bool] = 0
#                 force_direction_copy[~data_bool] = 0
#                 alignment = np.sign(force_direction_copy) * np.sign(angular_velocity)
#                 total_alignment = np.sum(alignment == 1, axis=1)
#                 total_events = np.sum(np.isin(alignment, [-1, 1]), axis=1)
#                 percentage = total_alignment / total_events
#                 percentage = percentage[total_events > 20]
#                 sns.histplot(percentage, color=colors[data_count, force_count], ax=axs[data_count, force_count], bins=10)  # ,stat='percent')
#                 axs[data_count, force_count].set_xlabel('')
#                 axs[data_count, force_count].set_ylabel('')
#                 if force_count == 0:
#                     axs[data_count, force_count].set_ylabel(leading_titles[data_count]+f"\nCount")
#                 if data_count == 0:
#                     axs[data_count, force_count].title.set_text(force_titles[force_count])
#         plt.tight_layout()
#         print("Saving figure to path: ", os.path.join(output_dir, f"ants alignment - {velocity_title}.png"))
#         plt.savefig(os.path.join(output_dir, f"ants alignment - {velocity_title}.png"))
# # plot_alignment(self, os.path.join(self.output_path, "alignment"), profile_size=200)
#
#
# def angle_to_nest_bias(self):
#     import matplotlib.pyplot as plt
#     angles_to_nest = self.fixed_end_angle_to_nest[self.rest_bool]
#     # distance_from_center = np.linalg.norm((self.object_center_coordinates-self.video_resolution/2), axis=1)
#     force_magnitude = self.force_magnitude[self.rest_bool]
#     force_direction = self.force_direction[self.rest_bool]
#     # plot dot plot of angle to nest vs force magnitude
#     # plt.clf()
#     plt.scatter(angles_to_nest, force_magnitude, s=1, c=force_direction)
#     plt.xlabel("angle to nest")
#     plt.ylabel("force magnitude")
#     plt.title("angle to nest vs force magnitude")
#     plt.show()
#     # plot dot plot of angle to nest vs force direction
#     # plt.clf()
#     # plt.scatter(angles_to_nest, force_direction, s=1, c=force_magnitude)
#     # plt.xlabel("angle to nest")
#     # plt.ylabel("force direction")
#     # plt.title("angle to nest vs force direction")
#     # plt.colorbar()
#     # plt.show()
# # angle_to_nest_bias(self)
#
#
# def plot_springs_bar_plot_comparison(analysed, window_size=5, title="", output_dir=None):
#     data = copy.copy(analysed.calib_data)
#     data["angular_velocity"] = np.abs(data["angular_velocity"])
#     plt.clf()
#     sns.barplot(x="spring_type", y="angular_velocity", data=data, edgecolor=".5", facecolor=(0, 0, 0, 0))
#     plt.xlabel("")
#     # add labels to each bar
#     plt.xticks([0, 1,2,3], [f"stiffness: {float(re.split('[a-z]',spring_type)[-1])}" for spring_type in analysed.spring_types])
#     # plt.xticks([0, 1], [f"stiffness: {float(re.split('[a-z]',spring_type)[-1])}" for spring_type in analysed.spring_types])
#     plt.ylabel(f"angular velocity (rad/s)")
#     plt.title(f"angular velocity to spring stiffness")
#     plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
#     print("Saving figure to path: ", os.path.join(output_dir, f"angular velocity to spring stiffness - bar plot.png"))
#     plt.savefig(os.path.join(output_dir, f"angular velocity to spring stiffness - bar plot.png"))
# # plot_springs_bar_plot_comparison(self, output_dir=output_path)
#
#
# def plot_springs_comparison(analysed, window_size=5, title="", output_dir=None):
#     os.makedirs(output_dir, exist_ok=True)
#     data = copy.copy(analysed.calib_data)
#
#     # data["angular_velocity"] = np.abs(data["angular_velocity"])
#     # data.loc[data.loc[:,"sum_N_ants"]<=1, "angular_velocity"] = np.nan
#     # plt.clf()
#     # sns.lineplot(data=data, x="sum_N_ants", y="angular_velocity", hue="spring_type")
#     # plt.title(f"angular velocity to number of carrying ants")
#     # plt.xlabel("number of ants")
#     # plt.ylabel("angular velocity (rad/frame)")
#     # plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
#     # print("Saving figure to path: ", os.path.join(output_dir, f"angular velocity to number of carrying ants.png"))
#     # plt.savefig(os.path.join(output_dir, f"angular velocity to number of carrying ants.png"))
#
#     data["angular_velocity"] = np.abs(data["angular_velocity"])
#     data["net_tangential_force"] = np.abs(data["net_tangential_force"])
#     for x_name, x_query in zip(["number of ants"], ["sum_N_ants"]):
#         for y_name, y_query in zip(["angular velocity (rad/frame)", "net tangential force mN"], ["angular_velocity", "net_tangential_force"]):
#             x = [np.unique(data.loc[data.loc[:,"spring_type"]==spring_type, x_query])[2:] for spring_type in np.unique(data.loc[:, "spring_type"])]
#             y = [np.zeros(len(x[i])) for i in range(len(np.unique(data.loc[:, "spring_type"])))]
#             y_SEM = [np.zeros(len(x[i])) for i in range(len(np.unique(data.loc[:, "spring_type"])))]
#             y_SEM_upper = [np.zeros(len(x[i])) for i in range(len(np.unique(data.loc[:, "spring_type"])))]
#             y_SEM_lower = [np.zeros(len(x[i])) for i in range(len(np.unique(data.loc[:, "spring_type"])))]
#             for i, spring_type in enumerate(np.unique(data.loc[:, "spring_type"])):
#                 for j, N_ants in enumerate(x[i]):
#                     y[i][j] = np.nanmean(data.loc[(data.loc[:, "spring_type"]==spring_type) & (data.loc[:, x_query]==N_ants), y_query])
#                     y_SEM[i][j] = np.nanstd(data.loc[(data.loc[:, "spring_type"]==spring_type) & (data.loc[:, x_query]==N_ants), y_query])/np.sqrt(len(data.loc[(data.loc[:, "spring_type"]==spring_type) & (data.loc[:, x_query]==N_ants), y_query]))
#                     y_SEM_upper[i][j] = y[i][j] + y_SEM[i][j]
#                     y_SEM_lower[i][j] = y[i][j] - y_SEM[i][j]
#             plt.clf()
#             for i, spring_type, color in zip(range(len(x)), np.unique(data.loc[:, "spring_type"]), ["purple", "orange", "green", "lightblue"]):
#                 y[i], y_SEM_upper[i], y_SEM_lower[i] = savgol_filter(y[i].astype(float), window_size, 3),\
#                     savgol_filter(y_SEM_upper[i].astype(float), window_size, 3),\
#                     savgol_filter(y_SEM_lower[i].astype(float), window_size, 3)
#                 plt.plot(x[i].astype(float), y[i].astype(float), color=color, label=f"stiffness: {float(re.split('[a-z]',spring_type)[-1])}")
#                 plt.fill_between(x[i].astype(float), y_SEM_lower[i].astype(float), y_SEM_upper[i].astype(float), color=color, alpha=0.2)
#             plt.legend()
#             plt.title(f"{y_name} to {x_name}")
#             plt.xlabel(f"{x_name}")
#             # plt.ylabel("angular velocity (rad/frame)")
#             plt.ylabel(f"{y_name})")
#             plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
#             print("Saving figure to path: ", os.path.join(output_dir, f"{y_query} to {x_query}.png"))
#             plt.savefig(os.path.join(output_dir, f"{y_query} to {x_query}.png"))
# # plot_springs_comparison(self, output_dir=output_path)
#
#
# def plot_query_information_to_attachment_length(analysed, start=0, end=None, window_size=1, title="", output_dir=None):
#     x_velocity = np.abs(analysed.all_profiles_angular_velocity)
#     x_N_ants = analysed.all_profiles_N_ants_around_springs_sum
#     for x_ori, x_name in zip([x_velocity, x_N_ants], ["profile velocity mean", "mean number of ants"]):
#         for attachment in ["1st", "2st&higher", "all"]:
#             if attachment == "1st":
#                 bool = np.copy(analysed.single_ant_profiles)
#                 bool[analysed.all_profiles_precedence != 1, :] = False
#                 x = np.copy(x_ori)
#             elif attachment == "2st&higher":
#                 bool = np.copy(analysed.single_ant_profiles)
#                 bool[analysed.all_profiles_precedence == 1, :] = False
#                 x = np.copy(x_ori)
#             else:
#                 bool = np.copy(analysed.single_ant_profiles)
#                 x = np.copy(x_ori)
#
#             x[~bool] = np.nan
#             y = np.sum(~np.isnan(x), axis=1).astype(float)
#             x = np.nanmean(x, axis=1)
#             argsort_y = np.argsort(y)
#             x = x[argsort_y]
#             y = y[argsort_y]
#
#             quantile = np.quantile(y, 0.99)
#             under_quantile = y < quantile
#             x = x[under_quantile]
#             y = y[under_quantile]
#             quantile = np.nanquantile(x, 0.99)
#             under_quantile = x < quantile
#             x = x[under_quantile]
#             y = y[under_quantile]
#             #plot
#             plt.clf()
#             plt.scatter(x, y/50, color="purple", s=1)
#             plt.title(f"{x_name} to attachment time")
#             plt.xlabel(f"{x_name}")
#             plt.ylabel(f"persistence time (s)")
#             plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
#             print("Saving figure to path: ", os.path.join(output_dir, f"{x_name} to attachment time - {attachment} (spring: {title}).png"))
#             plt.savefig(os.path.join(output_dir, f"{x_name} to attachment time - {attachment} (spring {title}).png"))
# # plot_query_information_to_attachment_length(analysed, window_size=50, title=spring_type, output_dir=ant_profiles_output_dir)
#
#
# def plot_replacements_rate(analysed, start=0, end=None, window_size=10, title="", output_dir=None):
#     if end is None:
#         end = analysed.angular_velocity.shape[0]
#
#     x = np.abs(analysed.angular_velocity[start:end])[:-1]
#     x_nans = np.isnan(x)
#     x = x[~x_nans]
#     under_quantile = x < np.quantile(x, 0.99)
#     x = x[under_quantile]
#     x_binned = np.linspace(np.nanmin(x), np.nanmax(x), 100)
#
#     y_replacements_rate = analysed.n_replacments_per_frame[start:end]
#     for y_ori, name in zip([y_replacements_rate], ["number of changes per frame"]):
#         y = np.copy(y_ori)
#         y = y[~x_nans]
#         y = y[under_quantile]
#         y_mean_binned = np.zeros((len(x_binned),))
#         y_SEM_upper_bined = np.zeros((len(x_binned),))
#         y_SEM_lower_bined = np.zeros((len(x_binned),))
#         for i in range(len(x_binned) - 1):
#
#             y_mean_binned[i] = np.nanmean(y[(x >= x_binned[i]) & (x < x_binned[i + 1])])
#             y_SEM_upper_bined[i] = np.nanmean(y[(x >= x_binned[i]) & (x < x_binned[i + 1])]) + \
#                                     np.nanstd(y[(x >= x_binned[i]) & (x < x_binned[i + 1])]) / np.sqrt(np.sum((x >= x_binned[i]) & (x < x_binned[i + 1])))
#             y_SEM_lower_bined[i] = np.nanmean(y[(x >= x_binned[i]) & (x < x_binned[i + 1])]) - \
#                                     np.nanstd(y[(x >= x_binned[i]) & (x < x_binned[i + 1])]) / np.sqrt(np.sum((x >= x_binned[i]) & (x < x_binned[i + 1])))
#         plt.clf()
#         plt.plot(x_binned[:-1], savgol_filter(y_mean_binned[:-1], window_size, 3), color="purple")
#         plt.fill_between(x_binned[:-1], savgol_filter(y_SEM_lower_bined[:-1], window_size, 3),
#                          savgol_filter(y_SEM_upper_bined[:-1], window_size, 3), color="purple", alpha=0.2)
#         plt.title(f"{name} to angular velocity")
#         plt.xlabel(f"angular velocity (rad/s)")
#         plt.ylabel(f"{name}")
#         plt.legend()
#         plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
#         print("Saving figure to path: ", os.path.join(output_dir, f"{name} to angular velocity (spring: {title}).png"))
#         plt.savefig(os.path.join(output_dir, f"{name} to angular velocity (spring {title}).png"))
# # plot_replacements_rate(analysed, window_size=10, title=spring_type, output_dir=macro_scale_output_dir)
#
#
# def plot_agreement_with_the_group(analysed, start=0, end=None, window_size=1, title="", output_dir=None):
#     single_ant = analysed.all_profiles_N_ants_around_springs[start:end] == 1
#     more_than_one = analysed.all_profiles_N_ants_around_springs[start:end] > 1
#     #
#     # x = np.abs(analysed.angular_velocity[start:end])
#     # x_nans = np.isnan(x)
#     # x = x[~x_nans]
#     # quantile = np.quantile(x, 0.99)
#     # upper_quantile = x < quantile
#     # x = x[upper_quantile]
#     # x_binned = np.linspace(np.nanmin(x), np.nanmax(x), 100)
#     x = analysed.all_profiles_angular_velocity[start:end]
#     y_tangential_force = analysed.all_profiles_tangential_force[start:end]
#     for ori, y_title in zip([(y_tangential_force, x)],["tangential force"]):
#         for bool in [single_ant, more_than_one]:
#             y = np.copy(ori[0])
#             y[~bool] = np.nan
#             x = np.copy(ori[1])
#             x[~bool] = np.nan
#
#             y[y==0] = np.nan
#             y_mean = np.nanmean(y, axis=0)
#             y_SEM_upper = y_mean + np.nanstd(y, axis=0) / np.sqrt(np.sum(~np.isnan(y), axis=0))
#             y_SEM_lower = y_mean - np.nanstd(y, axis=0) / np.sqrt(np.sum(~np.isnan(y), axis=0))
#             y_not_nan = ~np.isnan(y_mean)
#             y_mean,y_SEM_upper,y_SEM_lower = y_mean[y_not_nan],y_SEM_upper[y_not_nan],y_SEM_lower[y_not_nan]
#             x = np.arange(0, y_mean.shape[0], 1)
#             plt.clf()
#             plt.plot(x, y_mean, color="purple")
#             plt.fill_between(x, y_SEM_lower, y_SEM_upper, alpha=0.5, color="orange")
#
#
# def plot_query_distribution_to_angular_velocity(analysed, start=0, end=None, window_size=1, title="", output_dir=None):
#     if end is None:
#         end = analysed.angular_velocity.shape[0]
#
#     single_ant = analysed.N_ants_around_springs[start:end] == 1
#     more_than_one = analysed.N_ants_around_springs[start:end] > 1
#     all_springs = np.ones((analysed.N_ants_around_springs[start:end].shape[0],analysed.N_ants_around_springs[start:end].shape[1])).astype("bool")
#     all_ants = analysed.N_ants_around_springs[start:end] >= 1
#     no_ants = analysed.N_ants_around_springs[start:end] == 0
#     # x = analysed.angular_velocity[start:end]
#     x = np.abs(analysed.angular_velocity[start:end])
#     x_nans = np.isnan(x)
#     x = x[~x_nans]
#     # quantile = (x < np.quantile(x, 0.99)) & (x > np.quantile(x, 0.01))
#     quantile = (x < np.quantile(x, 0.99))
#     x = x[quantile]
#     x_binned = np.linspace(np.nanmin(x), np.nanmax(x), 100)
#     y_N_ants = analysed.N_ants_around_springs[start:end]
#     y_N_ants = y_N_ants[~x_nans]
#     y_N_ants = y_N_ants[quantile]
#     N_ants_mean = np.nansum(y_N_ants, axis=1)
#
#     y_tangential_force = analysed.tangential_force[start:end]
#     y_force_magnitude = np.abs(analysed.force_magnitude[start:end])
#     for y_ori, name in zip([np.abs(y_tangential_force), y_force_magnitude, y_tangential_force, y_tangential_force],
#                             ["mean tangential force", "mean force magnitude", "net tangential force", "contradicting force"]):
#         for bool, label in zip([single_ant, more_than_one, all_ants, no_ants, all_springs], ["single ant", "more than one ant", "all_ants", "no ants", "all springs"]):
#             y = np.copy(y_ori)
#             if name == "contradicting force":
#                 disagreements = (np.sign(y) * np.sign(analysed.angular_velocity[start:end])[:, np.newaxis]) < 0
#                 y[disagreements] = np.nan
#                 y = np.abs(y)
#             y[~bool] = np.nan
#             y = y[~x_nans]
#             y = y[quantile]
#             # if (name == "mean tangential force" and label in ["single ant", "more than one ant", "all_ants", "no ants"])\
#             #     or (name == "mean force magnitude" and label in ["single ant"])\
#             #     or (name == "net tangential force" and label in ["all_ants", "no ants", "all springs"])\
#             #     or (name == "mean disagreement with the group" and label in ["single ant", "more than one ant", "all_ants", "no ants"]):
#             #     continue
#             if name == "contradicting force" and label in ["single ant", "more than one ant", "all_ants", "no ants"]:
#                 if name == "net tangential force":
#                     y = np.abs(np.nansum(y, axis=1))
#                 y_mean_binned = np.zeros((len(x_binned),))
#                 N_ants_mean_binned = np.zeros((len(x_binned),))
#                 y_SEM_upper_bined = np.zeros((len(x_binned),))
#                 y_SEM_lower_bined = np.zeros((len(x_binned),))
#                 for i in range(len(x_binned) - 1):
#                     y_mean_binned[i] = np.nanmean(y[(x >= x_binned[i]) & (x < x_binned[i + 1])])
#                     N_ants_mean_binned[i] = np.nanmean(N_ants_mean[(x >= x_binned[i]) & (x < x_binned[i + 1])])
#                     y_SEM_upper_bined[i] = np.nanmean(y[(x >= x_binned[i]) & (x < x_binned[i + 1])]) + \
#                                            np.nanstd(y[(x >= x_binned[i]) & (x < x_binned[i + 1])]) / np.sqrt(np.sum((x >= x_binned[i]) & (x < x_binned[i + 1])))
#                     y_SEM_lower_bined[i] = np.nanmean(y[(x >= x_binned[i]) & (x < x_binned[i + 1])]) - \
#                                            np.nanstd(y[(x >= x_binned[i]) & (x < x_binned[i + 1])]) / np.sqrt(np.sum((x >= x_binned[i]) & (x < x_binned[i + 1])))
#
#                 plt.clf()
#                 fig, ax = plt.subplots()
#                 ax1.plot(x_binned[:-1], savgol_filter(y_mean_binned[:-1], 10, 3), color="purple")
#                 ax1.fill_between(x_binned[:-1], savgol_filter(y_SEM_lower_bined[:-1],10,3), savgol_filter(y_SEM_upper_bined[:-1],10,3), alpha=0.5, color="orange")
#                 ax1.set_xlabel("angular velocity (rad/ frame)")
#                 ax1.set_ylabel(f"{name} (mN)", color="purple")
#                 ax1.tick_params(axis='y')
#                 if name == "mean tangential force":
#                     ax2 = ax1.twinx()
#                     ax2.plot(x_binned[:-1], N_ants_mean_binned[:-1], color="black", alpha=0.5)
#                     ax2.set_ylabel("number of ants", color="black", alpha=0.5)
#
#                 plt.title(f"{name} to angular velocity")
#                 plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
#                 print("Saving figure to path: ", os.path.join(output_dir, f"{name}, {label} to angular velocity (movie {title}).png"))
#                 plt.savefig(os.path.join(output_dir, f"{name}, {label} to angular velocity (movie {title}).png"))
#
#                 #creat bar and violin plots
#                 lower_bins = y[(x <= x_binned[30])].flatten()
#                 upper_bins = y[(x >= x_binned[70])].flatten()
#                 lower_bins = lower_bins[~np.isnan(lower_bins)]
#                 upper_bins = upper_bins[~np.isnan(upper_bins)]
#                 t, p = stats.ttest_ind(lower_bins, upper_bins)
#                 print(f"p value for {name} to angular velocity (movie:{title}): {p}")
#                 df = pd.DataFrame({"anglar velocity": ["low angular velocity"] * len(lower_bins) + ["high angular velocity"] * len(upper_bins),
#                                      f"{name} (mN)": np.concatenate([lower_bins, upper_bins])})
#                 plt.clf()
#                 sns.barplot(x="anglar velocity", y=f"{name} (mN)", data=df, edgecolor=".5", facecolor=(0, 0, 0, 0))
#                 plt.xlabel("")
#                 plt.text(0.5, 0.5, f"p = {p}")
#                 plt.xticks([0, 1], ["low angular velocity", "high angular velocity"])
#                 plt.ylabel(f"{name} (mN)")
#                 plt.title(f"{name} to angular velocity")
#                 plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
#                 print("Saving figure to path: ", os.path.join(output_dir, f"{name}, {label} to angular velocity - bar plot (movie {title}).png"))
#                 plt.savefig(os.path.join(output_dir, f"{name}, {label} to angular velocity - bar plot (movie {title}).png"))
#                 # #violin plot
#                 # plt.clf()
#                 # sns.violinplot(x="anglar velocity", y=f"{name} (mN)", data=df, edgecolor=".5", facecolor=(0, 0, 0, 0))
#                 # plt.xlabel("")
#                 # plt.text(0.5, 0.5, f"p = {p}")
#                 # plt.xticks([0, 1], ["low angular velocity", "high angular velocity"])
#                 # plt.ylabel(f"{name} (mN)")
#                 # plt.title(f"{name} to angular velocity")
#                 # plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
#                 # print("Saving figure to path: ", os.path.join(output_dir, f"{name}, {label} to angular velocity - violin plot (movie {title}).png"))
#                 # plt.savefig(os.path.join(output_dir, f"{name}, {label} to angular velocity - violin plot (movie {title}).png"))
# # plot_query_distribution_to_angular_velocity(analysed, start=0, end=None, window_size=50, title=spring_type, output_dir=macro_scale_output_dir)




# Figure 1:
# def calibration_error(path, fig_size=np.array([68, 37.5]), font_size=6):
#     # path = "Z:\\Dor_Gabay\\ThesisProject\\data\\2-video_analysis\\summer_2023\\calibration\\plus_0.1"
#     y_prediction = np.load(os.path.join(path, "y_test_prediction.npy"))
#     y_true = np.load(os.path.join(path, "y_test.npy"))
#     data_direction = pd.DataFrame(np.concatenate((y_prediction[:, :1], y_true[:, :1]), axis=1), columns=["Prediction", "True"])
#     direction_bins = np.histogram(data_direction["True"], bins=10)[1]
#     direction_x_values = direction_bins[:-1] + (direction_bins[1:] - direction_bins[:-1]) / 2
#     direction_means = np.array([])
#     direction_std = np.array([])
#     for lower, upper in zip(direction_bins[:-1], direction_bins[1:]):
#         idx = (data_direction["True"] >= lower) * (data_direction["True"] < upper)
#         direction_means = np.append(direction_means, np.mean(data_direction["Prediction"][idx]))
#         direction_std = np.append(direction_std, np.std(data_direction["Prediction"][idx]))
#     data_magnitude = pd.DataFrame(np.concatenate((y_prediction[:, 1:], y_true[:, 1:]), axis=1), columns=["Prediction", "True"])
#     magnitude_x_values = np.unique(data_magnitude["True"])
#     magnitude_means = np.stack([np.mean(data_magnitude["Prediction"][data_magnitude["True"] == i]) for i in magnitude_x_values])
#     magnitude_std = np.stack([np.std(data_magnitude["Prediction"][data_magnitude["True"] == i]) for i in magnitude_x_values])
#
#     for data, means, stds, x_values, x_label, y_label, title in zip([data_direction, data_magnitude],
#                                            [direction_means, magnitude_means],
#                                            [direction_std, magnitude_std],
#                                            [direction_x_values, magnitude_x_values],
#                                            [r"Force direction $(rad)$", r"Force magnitude $(mN)$"],
#                                            ["Predicted direction $(rad)$", r"Predicted magnitude $(mN)$"],
#                                            ["Direction calibration error", "Magnitude calibration error"]):
#         plt.clf()
#         colors = sns.color_palette("rocket")
#         fig, ax = plt.subplots()
#         fig.set_size_inches(fig_size[0] * 0.0393701, fig_size[1] * 0.0393701)
#         sns.scatterplot(data=data, x="True", y="Prediction", color="gray", alpha=0.5, s=5)
#         sns.scatterplot(x=x_values, y=means, color=colors[2], s=25)
#         plt.fill_between(x_values, means - stds, means + stds, alpha=0.2, color=colors[-2])
#         plt.xlabel(x_label)
#         plt.ylabel(y_label)
#         plt.xlim([-0.1, np.max(data["True"]) * 1.1])
#         plt.ylim([-0.1, np.max(data["True"]) * 1.1])
#         plt.plot([0, np.max(data["True"])], [0, np.max(data["True"])], linestyle="--", color="black", alpha=0.6)
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
#                      ax.get_xticklabels() + ax.get_yticklabels()):
#             item.set_fontsize(font_size)
#         plt.tight_layout()
#         plt.savefig(os.path.join(path, f"{title}.svg"))
# calibration_error(os.path.normpath("Z:\\Dor_Gabay\\ThesisProject\\results\\summer_2023\\plus_0.1\\calibration_figures\\"))





# Figure 2:
# def velocity_over_average_force(self, output_path, sample_size=10000, window=26, bins=20, fig_size=(75, 50), font_size=6):
#     def func(x, a, b):
#         return a * x + b
#     angular_velocity = utils.calc_angular_velocity(self.fixed_end_angle_to_wall, window=window, percentile_threshold=98)
#     discrete_angular_velocity, _ = utils.discretize_angular_velocity(angular_velocity, self.sets_frames)
#     angular_velocity = np.abs(angular_velocity)
#     tangential_force = self.tangential_force.copy()
#     # tangential_force = np.array([gaussian_filter1d(column, sigma=window) for column in tangential_force.T]).T
#     tangential_force[np.abs(tangential_force) > np.nanpercentile(np.abs(tangential_force), 98)] = np.nan
#     tangential_force[self.force_magnitude < 0] = np.nan
#     tangential_force[np.sum(np.isnan(tangential_force), axis=1) > 0] = np.nan
#     nan_idx = np.all(np.isnan(tangential_force), axis=1)
#     net_tangential_force = np.nansum(tangential_force, axis=1)
#     gaussian_filter1d(net_tangential_force, sigma=window, output=net_tangential_force)
#     net_tangential_force[nan_idx] = np.nan
#
#     # net_tangential_force[np.sum(self.N_ants_around_springs, axis=1) > 10] = np.nan
#     net_tangential_force = np.abs(net_tangential_force)
#     velocity_bins = np.histogram(np.abs(angular_velocity)[~np.isnan(angular_velocity)], bins=bins)[1]#[2:]
#     y_binned = np.full((sample_size, len(velocity_bins) - 1), np.nan)
#     for count, (lower, upper) in enumerate(zip(velocity_bins[:-1], velocity_bins[1:])):
#         idx = np.logical_and(angular_velocity > lower, angular_velocity <= upper)
#         y = net_tangential_force[idx]
#         if len(y) > sample_size:
#             y = np.random.choice(y, sample_size, replace=False)
#         y_binned[:len(y), count] = y
#
#     plt.clf()
#     fig, ax = plt.subplots()
#     fig.set_size_inches(fig_size[0] * 0.0393701, fig_size[1] * 0.0393701)
#     y_mean = np.nanmean(y_binned, axis=0)
#     y_SEM_upper = y_mean + np.nanstd(y_binned, axis=0) / np.sqrt(np.sum(~np.isnan(y_binned), axis=0))
#     y_SEM_lower = y_mean - np.nanstd(y_binned, axis=0) / np.sqrt(np.sum(~np.isnan(y_binned), axis=0))
#     df = pd.DataFrame({"y": y_mean, "x": (velocity_bins[:-1] + velocity_bins[1:]) / 2})
#     df["x"] = df["x"] * (180 / np.pi) * (self.fps)
#     # fit a linear function only for x values between 0-15
#     x = df["x"].values
#     y = df["y"].values
#     y = y[(x > 0) & (x < 15)]
#     x = x[(x > 0) & (x < 15)]
#     df_line_fit = pd.DataFrame({"x": x, "y": y})
#     popt = curve_fit(func, df_line_fit["x"], df_line_fit["y"], p0=[1, 1])
#     df_line_fit["y_fit"] = func(df_line_fit["x"], *popt[0])
#     plt.fill_between(df["x"], y_SEM_lower, y_SEM_upper, alpha=0.5, color="purple")
#     sns.lineplot(data=df, x="x", y="y", color="purple")
#     if window > 10:
#         sns.lineplot(data=df_line_fit, x="x", y="y_fit", color=sns.color_palette("rocket")[1], linestyle="--")
#     r_squared = 1 - (np.sum((df_line_fit["y"] - df_line_fit["y_fit"]) ** 2) / np.sum((df_line_fit["y"] - np.mean(df_line_fit["y"])) ** 2))
#     plt.ylabel(r"Net tangential force $(mN)$", fontsize=font_size)
#     plt.xlabel(r"Angular velocity ($deg. s^{-1}$)", fontsize=font_size)
#     ax.tick_params(axis='both', which='major', labelsize=font_size)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_path, f"Velocity over average force, window {window}, R squared {np.round(r_squared, 3)}.svg"))
# # velocity_over_average_force(self, os.path.join(self.output_path, "macro_analysis"))
#
#
# def plot_correlation(self, output_path,  sets_idx=(1, 1), window=26, font_size=6, fig_size=np.array([150, 50])):
#     start, end = self.sets_frames[sets_idx[0]][0][0], self.sets_frames[sets_idx[1]][-1][1]
#     # output_dir = define_output_dir(output_dir)
#     os.makedirs(output_path, exist_ok=True)
#     title = "angular velocity to tangential force correlation"
#     plt.clf()
#     colors = sns.color_palette("rocket", 2)
#     fig, ax = plt.subplots()
#     fig.set_size_inches(fig_size[0] * 0.0393701, fig_size[1] * 0.0393701)
#     time = np.linspace(start, end, end - start) / self.fps
#
#     angular_velocity = utils.calc_angular_velocity(self.fixed_end_angle_to_wall, window=window, remove_outliers=True) * -1
#     angular_velocity = utils.interpolate_columns(angular_velocity)
#     angular_velocity = angular_velocity[start:end] * self.fps
#     gaussian_filter1d(angular_velocity, sigma=window, output=angular_velocity)
#     ax.plot(time, angular_velocity, color=colors[0])
#     ax.set_xlabel("Time $(s)$", fontsize=font_size)
#     ax.set_ylabel("Angular velocity $(deg. s^-1)$", color=colors[0], fontsize=font_size)
#     #set axes label size:
#     ax.tick_params(axis="y", labelcolor=colors[0])
#     ax.tick_params(axis='both', which='major', labelsize=font_size)
#
#
#     ax2 = ax.twinx()
#     net_tangential_force = self.net_tangential_force.copy()[start:end]
#     gaussian_filter1d(net_tangential_force, sigma=window, output=net_tangential_force)
#     ax2.plot(time, net_tangential_force, color=colors[1])
#     ax2.set_ylabel("Net tangential force $(mN)$", color=colors[1], fontsize=font_size)
#     ax2.tick_params(axis="y", labelcolor=colors[1])
#     ax2.tick_params(axis='both', which='major', labelsize=font_size)
#
#     ticks1 = ax.get_yaxis().get_majorticklocs()
#     ticks2 = ax2.get_yaxis().get_majorticklocs()
#     y_min = min(min(ticks1), min(ticks2))
#     y_max = max(max(ticks1), max(ticks2))
#     ax.set_ylim(y_min, y_max)
#     ax2.set_ylim(y_min, y_max)
#     # ax.set_xlim(0, 1000)
#     # ax2.set_xlim(0, 1000)
#     ax.axhline(y=0, color="black", linestyle="--")
#     ax2.axhline(y=0, color="black", linestyle="--")
#     ax.spines['top'].set_visible(False)
#     ax2.spines['top'].set_visible(False)
#     fig.tight_layout()
#     # plt.savefig(os.path.join(output_path, "Angular velocity to tangential force correlation.png"))
#     plt.savefig(os.path.join(output_path, "Angular velocity to tangential force correlation.svg"), format="svg")
# # plot_correlation(self, output_path=os.path.join(self.output_path, "correlation"))
#
#
# def plot_auto_correlation(self, output_path, fig_size=(75, 50), font_size=6):
#     self.calculate_auto_correlation()
#     data = self.auto_correlation.copy()
#     lags = np.arange(1, self.auto_correlation.shape[1]+1)
#     plt.clf()
#     colors = sns.color_palette("rocket", 3)
#     fig, ax = plt.subplots()
#     fig.set_size_inches(fig_size[0] * 0.0393701, fig_size[1] * 0.0393701)
#     means = np.mean(data, axis=0)
#     stds = np.std(data, axis=0)
#     popt = curve_fit(utils.exponential_decay, lags, means, p0=[1, 1])
#     plt.fill_between(lags, means - stds, means + stds, alpha=0.3, color=colors[2])
#     sns.scatterplot(x=lags, y=means, color=colors[1], s=30)
#     sns.lineplot(x=lags, y=utils.exponential_decay(lags, *popt[0]), color=colors[0], linestyle="--")
#     plt.ylabel(r"Auto correlation")
#     plt.xlabel(r"Window size")
#     plt.ylim([-0.1, 1.1])
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
#         item.set_fontsize(font_size)
#     plt.tight_layout()
#     plt.show()
#     plt.savefig(os.path.join(output_path, f"Auto correlation.svg"))
# # plot_auto_correlation(self, os.path.join(self.output_path, "macro_analysis"))





# Figure 3:
# def plot_ant_profiles(self, output_dir, window_size=11, profile_size=200):
#     # plt.clf()
#     # y = np.sum(self.attaching_single_ant_profiles, axis=0)
#     # y = y[y > self.fps]
#     # x = np.arange(0, y.shape[0], 1) / self.fps
#     # plt.plot(x, y, color="purple")
#     # plt.ylabel(f"number of profiles")
#     # plt.xlabel("seconds")
#     # print("Saving figure to path: ", os.path.join(output_dir, f"number of profiles.png"))
#     # plt.savefig(os.path.join(output_dir, f"number of profiles.png"))
#
#     force_magnitude = np.abs(self.profiled_force_magnitude)
#     force_magnitude[self.profiled_negative_magnitude < 0] = np.nan
#     tangential_force = np.abs(self.profiled_tangential_force)
#     tangential_force[self.profiled_negative_magnitude] = np.nan
#     single_ant_labeled = self.N_ants_labeled[0]
#     profile_labels = np.unique(single_ant_labeled)[1:]
#     profiles_attachments = np.full(single_ant_labeled.shape, False)
#     profiles_detachments = np.full(single_ant_labeled.shape, False)
#     for profile_count, profile_label in enumerate(profile_labels):
#         idx = np.where(single_ant_labeled == profile_label)
#         profiles_attachments[idx[0][0], idx[1][0]] = True
#         profiles_detachments[idx[0][-1], idx[1][-1]] = True
#
#     n_ants = self.N_ants_around_springs[1:].copy()
#     n_ants[profiles_attachments[:-1] != 0] = np.nan
#     attachments = np.where(n_ants == 0)
#     attachments = attachments[0] + 1, attachments[1]
#     dettachments = np.where(self.N_ants_around_springs[1:][single_ant_labeled[:-1] == 0] != 0)
#
#     for y_ori, y_title in zip([force_magnitude, tangential_force], ["force magnitude", "tangential force"]):
#         precedence_bool = np.ones(self.ant_profiles.shape[0], dtype=bool)
#         attaching_bool = self.attaching_single_ant_profiles * precedence_bool[:, np.newaxis]
#         detaching_bool = self.detaching_single_ant_profiles * precedence_bool[:, np.newaxis]
#         single_ant_bool = (self.profiled_N_ants_around_springs == 1) * precedence_bool[:, np.newaxis]
#         attaching_bool[~attaching_bool[:, profile_size], :] = False
#         detaching_bool[~detaching_bool[:, -profile_size - 1], :] = False
#         attaching_bool[:, profile_size + 1:] = False
#         detaching_bool[:, :-profile_size] = False
#         single_ant_bool[:, :profile_size] = False
#         single_ant_bool[:, profile_size + 500:] = False
#         attaching_y = np.copy(y_ori)
#         detaching_y = np.copy(y_ori)[self.reverse_argsort]
#         single_ant_y = np.copy(y_ori)
#         attaching_y[~attaching_bool] = np.nan
#         detaching_y[~detaching_bool] = np.nan
#         single_ant_y[~single_ant_bool] = np.nan
#         for y, name in zip([attaching_y, detaching_y, single_ant_y], ["attaching", "detaching", "middle"]):
#             y_mean = np.nanmean(y, axis=0)
#             y_SEM_upper = y_mean + np.nanstd(y, axis=0) / np.sqrt(np.sum(~np.isnan(y), axis=0))
#             y_SEM_lower = y_mean - np.nanstd(y, axis=0) / np.sqrt(np.sum(~np.isnan(y), axis=0))
#             y_not_nan = ~np.isnan(y_mean)
#             y_mean, y_SEM_upper, y_SEM_lower = y_mean[y_not_nan], y_SEM_upper[y_not_nan], y_SEM_lower[y_not_nan]
#             x = np.arange(0, y_mean.shape[0], 1) / self.fps
#             plt.clf()
#             plt.plot(x, savgol_filter(y_mean, window_size, 3), color="purple")
#             # plt.plot(x, y_mean, color="purple")
#             # plt.fill_between(x, savgol_filter(y_SEM_lower, window_size, 3), savgol_filter(y_SEM_upper, window_size, 3), alpha=0.5, color="orange")
#             plt.fill_between(x, y_SEM_lower, y_SEM_upper, alpha=0.5, color="orange")
#             plt.title(f"ant {y_title} profiles")
#             plt.xlabel("seconds")
#             plt.ylabel(f"{y_title} (mN)")
#             plt.subplots_adjust(top=0.9, bottom=0.15, left=0.15, right=0.9, hspace=0.5)
#             print("Saving figure to path: ", os.path.join(output_dir, f"{name} ant {y_title} profiles.png"))
#             plt.savefig(os.path.join(output_dir, f"{name} ant {y_title} profiles.png"))
# # plot_ant_profiles(self, output_dir=os.path.join(output_dir,"Figure 3"), window_size=11, profile_size=200)

# def profiles_distribution(analysed, output_dir):
#     n_profiles = analysed.ant_profiles.shape[0]
#     tangential_percentile = np.nanpercentile(analysed.profiled_tangential_force, 60)
#     data = np.full((n_profiles, 8), np.nan)
#     for count in range(n_profiles):
#         profile_duration = analysed.ant_profiles[count, 3] - analysed.ant_profiles[count, 2]
#         if not profile_duration < 50:
#             profile_N_ants = analysed.profiled_N_ants_around_springs[count]
#             profile_magnitude = analysed.profiled_force_magnitude[count]
#             profile_magnitude[~(profile_N_ants == 1)] = np.nan
#             profile_magnitude[:25] = np.nan
#             profile_tangential_force = analysed.profiled_tangential_force[count]
#             profile_tangential_force[~(profile_N_ants == 1)] = np.nan
#             profile_tangential_force[:25] = np.nan
#             profile_tangential_force[profile_tangential_force < tangential_percentile] = np.nan
#             tangential_fluctuations = np.abs(np.diff(profile_tangential_force[~np.isnan(profile_tangential_force)]))
#             magnitude_fluctuations = np.abs(np.diff(profile_magnitude[~np.isnan(profile_magnitude)]))
#             profile_percentile = np.nanpercentile(profile_magnitude, 95)
#             data[count, 0] = profile_percentile
#             data[count, 1] = np.nanmean(profile_magnitude)
#             data[count, 2] = np.nanstd(profile_magnitude) / np.nanmean(profile_magnitude)
#             data[count, 3] = len(profile_magnitude[~np.isnan(profile_magnitude)])
#             data[count, 4] = np.nanmean(tangential_fluctuations)
#             data[count, 5] = np.nanmean(magnitude_fluctuations)
#             data[count, 6] = np.nanstd(profile_tangential_force) / np.nanmean(profile_tangential_force)
#             data[count, 7] = np.sum(np.diff(np.sign(profile_tangential_force[~np.isnan(profile_tangential_force)])) != 0) / len(profile_magnitude[~np.isnan(profile_magnitude)])
#     # remove outliers from each column
#     for column in range(data.shape[1]):
#         lower_percentile = np.nanpercentile(data[:, column], 5)
#         upper_percentile = np.nanpercentile(data[:, column], 95)
#         data[:, column][data[:, column] < lower_percentile] = np.nan
#         data[:, column][data[:, column] > upper_percentile] = np.nan
#
#     colors = ['orange', 'pink', 'cyan', 'green', 'blue', 'red', 'purple']
#     plt.clf()
#     fig, ax = plt.subplots(1, 9, figsize=(30, 5))
#     sns.histplot(data[:, 0], ax=ax[0], bins=100, color=colors[0])
#     ax[0].set_xlabel("Force magnitude (95%)")
#     sns.scatterplot(x=data[:, 1], y=data[:, 2], ax=ax[1], color=colors[1], s=3)
#     sns.regplot(x=data[:, 1], y=data[:, 2], ax=ax[1], color="red", scatter=False)
#     ax[1].set_xlabel("Mean force magnitude")
#     ax[1].set_ylabel("Force magnitude variation (std/mean)")
#     sns.scatterplot(x=data[:, 1], y=data[:, 3], ax=ax[2], color=colors[2], s=3)
#     ax[2].set_xlabel("Mean force magnitude")
#     ax[2].set_ylabel("Profile duration")
#     sns.scatterplot(x=data[:, 3], y=data[:, 2], ax=ax[3], color=colors[3], s=3)
#     ax[3].set_xlabel("Profile duration")
#     ax[3].set_ylabel("Force magnitude variation (std/mean)")
#     sns.scatterplot(x=data[:, 1], y=data[:, 4], ax=ax[4], color=colors[4], s=3)
#     ax[4].set_xlabel("Mean force magnitude")
#     ax[4].set_ylabel("Tangential force difference")
#     sns.scatterplot(x=data[:, 1], y=data[:, 5], ax=ax[5], color=colors[5], s=3)
#     ax[5].set_xlabel("Mean force magnitude")
#     ax[5].set_ylabel("Force magnitude difference")
#     sns.scatterplot(x=data[:, 3], y=data[:, 4], ax=ax[6], color=colors[2], s=3)
#     ax[6].set_xlabel("Profile duration")
#     ax[6].set_ylabel("Tangential force difference")
#     sns.scatterplot(x=data[:, 1], y=data[:, 6], ax=ax[7], color=colors[2], s=3)
#     ax[7].set_xlabel("Mean force magnitude")
#     ax[7].set_ylabel("Tangential force variation (std/mean)")
#     sns.scatterplot(x=data[:, 1], y=data[:, 7], ax=ax[8], color=colors[2], s=3)
#     ax[8].set_xlabel("Mean force magnitude")
#     ax[8].set_ylabel("Direction switch rate")
#     plt.show()
#     plt.tight_layout()
#     print("Saving figure to path: ", os.path.join(output_dir, "Profiles distribution.png"))
#     plt.savefig(os.path.join(output_dir, "Profiles distribution.png"))
# # profiles_distribution(self, self.output_path)