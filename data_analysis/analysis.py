from data_analysis import utils
# from data_analysis import plots
import numpy as np
import pandas as pd
import os


class DataAnalyser():
    def __init__(self,directory):
        self.directory = directory
        self.video_name = os.path.basename(self.directory)
        self.load_data(self.directory)
        self.calculations()

    def load_data(self,directory):
        print("loading data from directory: ", directory)
        directory = os.path.join(directory, "two_vars_post_processing")
        self.N_ants_around_springs = np.load(os.path.join(directory, "N_ants_around_springs.npz"))['arr_0']
        self.fixed_end_angle_to_nest = np.load(os.path.join(directory, "fixed_end_angle_to_nest.npz"))['arr_0']
        self.force_direction = np.load(os.path.join(directory, "force_direction.npz"))['arr_0']
        self.force_magnitude = np.load(os.path.join(directory, "force_magnitude.npz"))['arr_0']
        # self.ants_assigned_to_springs = np.load(os.path.join(directory, "ants_assigned_to_springs.npz"))['arr_0']
        # self.all_profiles_force_magnitude = np.load(os.path.join(directory, "all_profiles_force_magnitude.npz"))['arr_0']
        # self.all_profiles_force_direction = np.load(os.path.join(directory, "all_profiles_force_direction.npz"))['arr_0']
        # self.all_profiles_ants_number = np.load(os.path.join(directory, "all_profiles_ants_number.npz"))['arr_0']
        # self.all_profiles_angle_to_nest = np.load(os.path.join(directory, "all_profiles_angle_to_nest.npz"))['arr_0']
        # self.all_profiles_precedence = np.load(os.path.join(directory, "all_profiles_precedence.npz"))['arr_0']
        print("Data loaded successfully")

    def calculations(self):
        # self.ant_profiling()
        self.angular_velocity = np.nanmedian(utils.calc_angular_velocity(self.fixed_end_angle_to_nest, diff_spacing=20)/20, axis=1)
        self.total_force = self.force_magnitude * self.force_direction
        self.net_force = np.nansum(self.total_force, axis=1)
        self.net_force = np.array(pd.Series(self.net_force).rolling(window=5,center=True).median())
        self.net_magnitude = np.nansum(self.force_magnitude, axis=1)
        self.net_magnitude = np.array(pd.Series(self.net_magnitude).rolling(window=5,center=True).median())
        self.total_n_ants = np.nansum(self.N_ants_around_springs, axis=1)
        # self.ants_profiling_analysis()


    def ants_profiling_analysis(self):
        self.profiles_start_with_one_ant = np.full((len(self.all_profiles_precedence), 10000), False)
        aranged = np.arange(10000)
        for profile in range(len(self.all_profiles_precedence)):
            if self.all_profiles_ants_number[profile,0] == 1:
                first_n_ants_change = aranged[:-1][np.diff(self.all_profiles_ants_number[profile,:]) != 0][0]
                self.profiles_start_with_one_ant[profile,0:first_n_ants_change+1] = True



if __name__ == "__main__":
    # %matplotlib qt
    # dir = "Z:\\Dor_Gabay\\ThesisProject\\data\\test3\\15.9.22\\plus0.3mm_force\\S5280006\\"
    video_name = "S5280008"
    directory = f"Z:\\Dor_Gabay\\ThesisProject\\data\\analysed_with_tracking\\15.9.22\\plus0.3mm_force\\{video_name}\\"
    analysed = DataAnalyser(directory)
    # from data_analysis.plots import plot_overall_behavior
    # output_dir = os.path.join(directory, "plots")
    # plots.plot_force_distribution_to_angular_velocity(analysed, start=0, end=None, window_size=50, title=video_name, output_dir=directory)
    # plots.plot_overall_behavior(analysed, start=0, end=None, window_size=50, title=video_name, output_dir=output_dir)
    # plots.plot_pulling_angle_over_angle_to_nest(analysed, start=0, end=None)

        # def remove_artificial_cases(self):
        #     print("Removing artificial cases...")
        #     unreal_ants_attachments = np.full(self.N_ants_around_springs.shape, np.nan)
        #     switch_attachments = np.full(self.N_ants_around_springs.shape, np.nan)
        #     unreal_detachments = np.full(self.N_ants_around_springs.shape, np.nan)
        #     for ant in range(self.ants_assigned_to_springs.shape[1]):
        #         attachment = self.ants_assigned_to_springs[:,ant]
        #         if not np.all(attachment == 0):
        #             first_attachment = np.where(attachment != 0)[0][0]
        #             for count, spring in enumerate(np.unique(attachment)[1:]):
        #                 frames = np.where(attachment==spring)[0]
        #                 sum_assign = np.sum(self.ants_assigned_to_springs[frames[0]:frames[-1]+3,:] == spring, axis=1)
        #                 real_sum = self.N_ants_around_springs[frames[0]:frames[-1]+3,int(spring-1)]
        #                 if frames[0] == first_attachment and np.all(sum_assign[0:2] != real_sum[0:2]):
        #                     unreal_ants_attachments[frames,int(spring-1)] = ant
        #                 elif np.all(sum_assign[0:2] != real_sum[0:2]):
        #                     switch_attachments[frames,int(spring-1)] = ant
        #                 else:
        #                     if np.all(sum_assign[-1] != real_sum[-1]):
        #                         unreal_detachments[frames[-1], int(spring-1)] = ant
        #     for frame in range(self.ants_assigned_to_springs.shape[0]):
        #         detach_ants = np.unique(unreal_detachments[frame,:])
        #         for detach_ant in detach_ants[~np.isnan(detach_ants)]:
        #             spring = np.where(unreal_detachments[frame,:] == detach_ant)[0][0]
        #             assign_ant = unreal_ants_attachments[frame,spring]
        #             if not np.isnan(assign_ant):
        #                 self.ants_assigned_to_springs[unreal_ants_attachments[:,spring]==assign_ant,int(detach_ant)] = spring+1
        #                 self.ants_assigned_to_springs[unreal_ants_attachments[:,spring]==assign_ant,int(assign_ant)] = 0
        #         switch_ants = np.unique(switch_attachments[frame, :])
        #         for switch_ant in switch_ants[~np.isnan(switch_ants)]:
        #             spring = np.where(switch_attachments[frame,:] == switch_ant)[0][0]
        #             switch_from_spring = np.where(switch_attachments[frame-1,:] == switch_ant)[0]
        #             switch_from_spring = switch_from_spring[switch_from_spring != spring]
        #             if len(switch_from_spring) > 0:
        #                 switch_frames = np.where(switch_attachments[:,spring] == switch_ant)[0]
        #                 self.ants_assigned_to_springs[switch_frames, int(switch_ant)] = switch_from_spring[0]
        #     self.ants_assigned_to_springs = self.ants_assigned_to_springs[:,:-1]
        #
        # def ant_profiling(self):
        #     self.remove_artificial_cases()
        #     print("Profiling ants...")
        #     self.ants_assigned_to_springs = self.ants_assigned_to_springs[:, :-1]
        #     profiles = np.full(5, np.nan) # ant, spring, start, end, precedence
        #     for ant in range(self.ants_assigned_to_springs.shape[1]):
        #         attachment = self.ants_assigned_to_springs[:, ant]
        #         events_springs = np.split(attachment, np.arange(len(attachment[1:]))[np.diff(attachment) != 0] + 1)
        #         events_frames = np.split(np.arange(len(attachment)), np.arange(len(attachment[1:]))[np.diff(attachment) != 0] + 1)
        #         precedence = 0
        #         for event in range(len(events_springs)):
        #             if events_springs[event][0]!=0 and len(events_springs[event]) > 1:
        #                 precedence += 1
        #                 start = events_frames[event][0]
        #                 end = events_frames[event][-1]
        #                 profiles = np.vstack((profiles, np.array([ant+1, events_springs[event][0], start, end, precedence])))
        #     profiles = profiles[1:, :]
        #     self.all_profiles_magnitude = np.full((len(profiles), 10000), np.nan)
        #     self.all_profiles_direction = np.full((len(profiles), 10000), np.nan)
        #     self.all_profiles_ants_number = np.full((len(profiles), 10000), np.nan)
        #     self.all_profiles_angle_to_nest = np.full((len(profiles), 10000), np.nan)
        #     self.all_profiles_precedence = profiles[:,4]
        #     for profile in range(len(profiles)):
        #         ant = int(profiles[profile,0]-1)
        #         spring = int(profiles[profile,1])
        #         start = int(profiles[profile,2])
        #         end = int(profiles[profile,3])
        #         if not end-start+1 > 10000:
        #             self.all_profiles_magnitude[profile,0:end-start+1] = self.force_magnitude[start:end+1,int(spring-1)]
        #             self.all_profiles_direction[profile,0:end-start+1] = self.force_direction[start:end+1,int(spring-1)]
        #             self.all_profiles_ants_number[profile,0:end-start+1] = self.N_ants_around_springs[start:end+1,int(spring-1)]
        #             self.all_profiles_angle_to_nest[profile,0:end-start+1] = self.fixed_end_angle_to_nest[start:end+1,int(spring-1)]
        #         else:
        #             self.all_profiles_precedence[profile] = np.nan
        # #     #save
        #     output_dir = os.path.join(self.directory,"two_vars_post_processing", "profiles")
        #     os.makedirs(output_dir, exist_ok=True)
        #     np.savez_compressed(os.path.join(output_dir, "all_profiles_magnitude.npz"), self.all_profiles_magnitude)
        #     np.savez_compressed(os.path.join(output_dir, "all_profiles_direction.npz"), self.all_profiles_direction)
        #     np.savez_compressed(os.path.join(output_dir, "all_profiles_ants_number.npz"), self.all_profiles_ants_number)
        #     np.savez_compressed(os.path.join(output_dir, "all_profiles_angle_to_nest.npz"), self.all_profiles_angle_to_nest)
        #     np.savez_compressed(os.path.join(output_dir, "all_profiles_precedence.npz"), self.all_profiles_precedence)
