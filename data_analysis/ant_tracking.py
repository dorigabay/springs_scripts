import os
import numpy as np
import pickle
import scipy.io as sio
from scipy.ndimage import binary_closing
import subprocess
from multiprocessing import Pool
from itertools import repeat
# local imports:
from data_analysis import utils


class AntTracking:
    def __init__(self, data_paths, output_path, set_frames, frame_resolution=(2160, 3840), restart=False):
        self.data_paths = data_paths
        self.restart = restart
        self.frame_size = frame_resolution
        self.set_frames = set_frames
        self.sub_dirs_names = [os.path.basename(os.path.split(os.path.normpath(path))[-1]) for path in data_paths]
        self.output_path = os.path.join(output_path, f"{self.sub_dirs_names[0]}-{self.sub_dirs_names[-1]}")
        print(f"Tracking ants for {self.sub_dirs_names[0]}-{self.sub_dirs_names[-1]}")
        self.first_tracking_calculation()
        self.correct_tracked_ants()
        self.assign_ants_to_springs()
        self.connect_occasions()
        self.create_ant_profiles()
        print(f"\rFinished tracking ants for {self.sub_dirs_names[0]}-{self.sub_dirs_names[-1]}")

    def first_tracking_calculation(self):
        if self.restart or not os.path.exists(os.path.join(self.output_path, "tracking_data.mat")):
            ants_centers_x = np.concatenate([np.loadtxt(os.path.join(path, "ants_centers_x.csv"), delimiter=",") for path in self.data_paths], axis=0)
            ants_centers_y = np.concatenate([np.loadtxt(os.path.join(path, "ants_centers_y.csv"), delimiter=",") for path in self.data_paths], axis=0)
            ants_centers = np.stack((ants_centers_x, ants_centers_y), axis=2)
            ants_centers_mat = np.zeros((ants_centers.shape[0], 1), dtype=np.object)
            for i in range(ants_centers.shape[0]):
                ants_centers_mat[i, 0] = ants_centers[i, :, :]
            os.makedirs(self.output_path, exist_ok=True)
            if self.restart:
                print(f"removing old tracking data from {self.output_path}")
                os.system(f"del {self.output_path}\\ants_centers.mat")
            print(f"Saving ants centers data (Stage 1/7)... ({self.sub_dirs_names[0]}-{self.sub_dirs_names[-1]})")
            sio.savemat(os.path.join(self.output_path, "ants_centers.mat"), {"ants_centers": ants_centers_mat})
            matlab_script_path = "Z:\\Dor_Gabay\\ThesisProject\\scripts\\munkres_tracker\\"
            os.chdir(matlab_script_path)
            os.system("PATH=$PATH:'C:\Program Files\MATLAB\R2022a\bin'")
            execution_string = f"matlab -r ""ants_tracking('" + self.output_path + "\\')"""
            print(f"Running matlab script for ants tracking (Stage 2/7)... ({self.sub_dirs_names[0]}-{self.sub_dirs_names[-1]})")
            run_output = subprocess.run(execution_string, shell=True, capture_output=True)
            print(run_output.stdout.decode("utf-8"))

    def correct_tracked_ants(self):
        """
        This function removes all new labels that were create inside the frame, and not on the boundries (10% of the frame)
        """
        if self.restart or not os.path.exists(os.path.join(self.output_path, "tracking_data_corrected.mat")):
            utils.wait_for_existance(self.output_path, "tracking_data.mat")
            print(f"Correcting tracked ants (Stage 3/7)... ({self.sub_dirs_names[0]}-{self.sub_dirs_names[-1]})")
            tracked_ants = sio.loadmat(os.path.join(self.output_path, "tracking_data.mat"))["tracked_blobs_matrix"]
            num_of_frames = tracked_ants.shape[2]

            frames_arrange = np.repeat(np.arange(tracked_ants.shape[2])[np.newaxis, :], tracked_ants.shape[0], axis=0)
            ants_arrange = np.repeat(np.arange(tracked_ants.shape[0])[np.newaxis, :], tracked_ants.shape[2], axis=0).T
            tracked_ants[np.isnan(tracked_ants)] = 0

            appeared_labels = np.array([])
            appeared_labels_idx = []
            appeared_last_coords = np.array([]).reshape(0, 2)
            disappeared_labels = []
            disappeared_labels_partners = []
            disappeared_labels_partners_relative_distance = []
            disappeared_labels_partners_idx = []
            for frame in range(1,num_of_frames-1):
                print("\r frame: ", frame, end="")
                if frame+20>=num_of_frames:
                    break
                if frame >= 11: min_frame = frame - 10
                else: min_frame = 1
                max_frame = frame + 15000
                labels_in_window = set(tracked_ants[:,2,min_frame:frame+20].flatten())
                labels_scanned_for_appearing = set(tracked_ants[:,2,min_frame-1].flatten())
                labels_now = set(tracked_ants[:, 2, frame])
                labels_before = set(tracked_ants[:, 2, frame - 1])
                appearing_labels = labels_in_window.difference(labels_scanned_for_appearing)
                disappearing_labels = labels_before.difference(labels_now)

                appeared_not_collected = appearing_labels.difference(set(appeared_labels))
                for label in appeared_not_collected:
                    label_idx = np.isin(tracked_ants[:, 2, min_frame:max_frame], label)
                    label_idx = (ants_arrange[:, min_frame:max_frame][label_idx],frames_arrange[:, min_frame:max_frame][label_idx])
                    coords = tracked_ants[label_idx[0], 0:2, label_idx[1]]
                    first_appear = np.argmin(label_idx[1])
                    if self.test_if_on_boundaries(coords[first_appear, :]):
                        appeared_labels = np.append(appeared_labels,label)
                        median_coords = coords[first_appear,:].reshape(1,2)
                        appeared_last_coords = np.concatenate((appeared_last_coords, median_coords), axis=0)
                        appeared_labels_idx.append(label_idx)

                currently_appearing = np.array([x for x in range(len(appeared_labels)) if appeared_labels[x] in appearing_labels])
                if len(currently_appearing)>0:
                    currently_appearing_labels = appeared_labels[currently_appearing]
                    currently_appearing_idx = [appeared_labels_idx[x] for x in currently_appearing]
                    currently_appearing_last_coords = appeared_last_coords[currently_appearing]
                    for label in disappearing_labels:
                        coords = tracked_ants[tracked_ants[:, 2, frame-1]==label, 0:2, frame-1].flatten()
                        if self.test_if_on_boundaries(coords):
                            if currently_appearing_last_coords.shape[0] > 0:
                                distances = np.linalg.norm(currently_appearing_last_coords - coords, axis=1)
                                closest_appearing_label_argmin = np.argsort(distances)[0]
                                if label==currently_appearing_labels[closest_appearing_label_argmin] \
                                        and len(distances)>1:
                                    closest_appearing_label_argmin = np.argsort(distances)[1]
                                partner_label = currently_appearing_labels[closest_appearing_label_argmin]
                                distance_closest = distances[closest_appearing_label_argmin]
                                idx_closest_appearing_label = currently_appearing_idx[closest_appearing_label_argmin]
                                if distance_closest<300:
                                    disappeared_labels.append(label)
                                    disappeared_labels_partners.append(partner_label)
                                    disappeared_labels_partners_relative_distance.append(distance_closest)
                                    disappeared_labels_partners_idx.append(idx_closest_appearing_label)

            for i in range(len(disappeared_labels)):
                idx = disappeared_labels_partners_idx[i]
                tracked_ants[idx[0], 2, idx[1]] = disappeared_labels[i]
                if disappeared_labels_partners[i] in disappeared_labels:
                    disappeared_labels[disappeared_labels.index(disappeared_labels_partners[i])] = disappeared_labels[i]
                duplicated_idxs = np.arange(len(disappeared_labels_partners))[np.array(disappeared_labels_partners)==disappeared_labels_partners[i]]
                duplicated_labels = [disappeared_labels[x] for x in duplicated_idxs]
                disappeared_labels = [np.min(duplicated_labels) if x in duplicated_labels else x for x in disappeared_labels]

            for unique_label in np.unique(tracked_ants[:, 2, :]):
                occurrences = np.count_nonzero(tracked_ants[:, 2, :] == unique_label)
                if occurrences < 20:
                    idx = np.where(tracked_ants[:, 2, :] == unique_label)
                    tracked_ants[idx[0], 2, idx[1]] = 0
            print(f"\nSaving corrected tracking data (Stage 3/7)... ({self.sub_dirs_names[0]}-{self.sub_dirs_names[-1]})")
            sio.savemat(os.path.join(self.output_path, "tracking_data_corrected.mat"), {"tracked_blobs_matrix": tracked_ants})

    def test_if_on_boundaries(self, coordinates):
        if self.frame_size[1] * 0.1 < coordinates[0] < self.frame_size[0] - self.frame_size[1] * 0.1 \
                and self.frame_size[1] * 0.1 < coordinates[1] < self.frame_size[1] * 0.9:
            return True
        else:
            return False

    def assign_ants_to_springs(self):
        if self.restart or not os.path.exists(os.path.join(self.output_path, "ants_assigned_to_springs.npz")):
            utils.wait_for_existance(self.output_path, "tracking_data_corrected.mat")
            print(f"Assigning ants to springs (Stage 4/7)... ({self.sub_dirs_names[0]}-{self.sub_dirs_names[-1]})")
            tracked_ants = sio.loadmat(os.path.join(self.output_path, "tracking_data_corrected.mat"))["tracked_blobs_matrix"].astype(np.uint16)
            unique_elements, indices = np.unique(tracked_ants[:, 2, :], return_inverse=True)
            tracked_ants[:,2,:] = indices.reshape(tracked_ants[:, 2, :].shape)
            N_ants_around_springs = np.load(os.path.join(self.output_path, "N_ants_around_springs.npz"))["arr_0"].astype(np.uint8)
            ants_attached_labels = np.load(os.path.join(self.output_path, "ants_attached_labels.npz"))["arr_0"]
            num_of_frames = ants_attached_labels.shape[0]
            ants_assigned_to_springs = np.zeros((num_of_frames, len(unique_elements)-1)).astype(np.uint8)
            for i in range(num_of_frames):
                bool = (~np.isnan(ants_attached_labels[i]))*(tracked_ants[:,2,i]>0)
                labels = tracked_ants[bool, 2, i].astype(int)
                springs = ants_attached_labels[i, bool].astype(int)
                ants_assigned_to_springs[i, labels-1] = springs
            ants_assigned_to_springs = self.interpolate_assigned_ants(ants_assigned_to_springs, num_of_frames)
            ants_assigned_to_springs = self.remove_artificial_cases(N_ants_around_springs, ants_assigned_to_springs)
            # ants_assigned_to_springs = self.connect_occasions(ants_assigned_to_springs, ants_attached_labels)
            np.savez_compressed(os.path.join(self.output_path, "ants_assigned_to_springs.npz"), ants_assigned_to_springs)
            np.savez_compressed(os.path.join(self.output_path, "ants_labels.npz"), unique_elements[1:])

    def interpolate_assigned_ants(self, ants_assigned_to_springs, num_of_frames, max_gap=30):
        arranged_frames = np.arange(num_of_frames)
        for ant in range(ants_assigned_to_springs.shape[1]):
            array = ants_assigned_to_springs[:, ant]
            closing_bool = binary_closing(array.astype(bool).reshape(1, num_of_frames), np.ones((1, max_gap)))
            closing_bool = closing_bool.reshape(closing_bool.shape[1])
            xp = arranged_frames[~closing_bool+(array != 0)]
            fp = array[xp]
            x = arranged_frames[~closing_bool+(array == 0)]
            ants_assigned_to_springs[x, ant] = np.round(np.interp(x, xp, fp))
        return ants_assigned_to_springs

    def remove_artificial_cases(self, N_ants_around_springs, ants_assigned_to_springs):
        print(f"Removing artificial ants pulling cases (Stage 5/7)... ({self.sub_dirs_names[0]}-{self.sub_dirs_names[-1]})")
        unreal_ants_attachments = np.full(N_ants_around_springs.shape, np.nan)
        switch_attachments = np.full(N_ants_around_springs.shape, np.nan)
        unreal_detachments = np.full(N_ants_around_springs.shape, np.nan)
        for ant in range(ants_assigned_to_springs.shape[1]):
            attachment = ants_assigned_to_springs[:,ant]
            if not np.all(attachment == 0):
                first_attachment = np.where(attachment != 0)[0][0]
                for count, spring in enumerate(np.unique(attachment)[1:]):
                    frames = np.where(attachment == spring)[0]
                    sum_assign = np.sum(ants_assigned_to_springs[frames[0]:frames[-1]+3,:] == spring, axis=1)
                    real_sum = N_ants_around_springs[frames[0]:frames[-1]+3,int(spring-1)]
                    if frames[0] == first_attachment and np.all(sum_assign[0:2] != real_sum[0:2]):
                        unreal_ants_attachments[frames,int(spring-1)] = ant
                    elif np.all(sum_assign[0:2] != real_sum[0:2]):
                        switch_attachments[frames,int(spring-1)] = ant
                    else:
                        if np.all(sum_assign[-1] != real_sum[-1]):
                            unreal_detachments[frames[-1], int(spring-1)] = ant
        for frame in range(ants_assigned_to_springs.shape[0]):
            detach_ants = np.unique(unreal_detachments[frame,:])
            for detach_ant in detach_ants[~np.isnan(detach_ants)]:
                spring = np.where(unreal_detachments[frame,:] == detach_ant)[0][0]
                assign_ant = unreal_ants_attachments[frame,spring]
                if not np.isnan(assign_ant):
                    ants_assigned_to_springs[unreal_ants_attachments[:,spring]==assign_ant,int(detach_ant)] = spring+1
                    ants_assigned_to_springs[unreal_ants_attachments[:,spring]==assign_ant,int(assign_ant)] = 0
            switch_ants = np.unique(switch_attachments[frame, :])
            for switch_ant in switch_ants[~np.isnan(switch_ants)]:
                spring = np.where(switch_attachments[frame,:] == switch_ant)[0][0]
                switch_from_spring = np.where(switch_attachments[frame-1,:] == switch_ant)[0]
                switch_from_spring = switch_from_spring[switch_from_spring != spring]
                if len(switch_from_spring) > 0:
                    switch_frames = np.where(switch_attachments[:,spring] == switch_ant)[0]
                    ants_assigned_to_springs[switch_frames, int(switch_ant)] = switch_from_spring[0]
        ants_assigned_to_springs = ants_assigned_to_springs[:,:-1]
        return ants_assigned_to_springs

    def connect_occasions(self):
        if self.restart or not os.path.exists(os.path.join(self.output_path, "ants_assigned_to_springs_fixed.npz")):
            print(f"Connecting occasions of the same ant (Stage 6/7)... ({self.sub_dirs_names[0]}-{self.sub_dirs_names[-1]})")
            missing_info = np.load(os.path.join(self.output_path, "missing_info.npz"))["arr_0"]
            ants_assigned_to_springs = np.load(os.path.join(self.output_path, "ants_assigned_to_springs.npz"))["arr_0"]
            last_value = ants_assigned_to_springs[0,:]
            last_time_value = np.full(last_value.shape, np.nan)
            last_time_value[last_value != 0] = 0
            for frame in range(1,ants_assigned_to_springs.shape[0]):
                print("\r Frame: ", frame, end="")
                current_info = ants_assigned_to_springs[frame,:]
                current_springs = np.unique(current_info[current_info != 0])
                springs_were_nan = current_springs[np.isnan(missing_info[frame-1,current_springs-1])]
                ants_to_connect = np.where(np.isin(last_value, springs_were_nan))[0]
                for ant in ants_to_connect.astype(int):
                    last_frame_with_value = last_time_value[ant].astype(int)
                    ants_assigned_to_springs[last_frame_with_value:frame, ant] = current_info[ant]
                last_value[current_info != 0] = current_info[current_info != 0]
                last_time_value[current_info != 0] = frame
            np.savez_compressed(os.path.join(self.output_path, "ants_assigned_to_springs_fixed.npz"), ants_assigned_to_springs)

    def create_ant_profiles(self):
        import pandas as pd
        print(f"\rCreating ants profiles (Stage 7/7)... ({self.sub_dirs_names[0]}-{self.sub_dirs_names[-1]})")
        ants_assigned_to_springs = np.load(os.path.join(self.output_path, "ants_assigned_to_springs_fixed.npz"))["arr_0"][:, :-1].astype(np.uint8)
        ants_attached_labels = np.load(os.path.join(self.output_path, "ants_attached_labels.npz"))["arr_0"]
        self.profiles = np.full(6, np.nan)  # ant, spring, start, end, precedence, sudden_appearance
        for ant in range(ants_assigned_to_springs.shape[1]):
            print("\r Ant number: ", ant, end="")
            attachment = ants_assigned_to_springs[:, ant]
            events_springs = np.split(attachment, np.arange(len(attachment[1:]))[np.diff(attachment) != 0] + 1)
            events_frames = np.split(np.arange(len(attachment)),
                                     np.arange(len(attachment[1:]))[np.diff(attachment) != 0] + 1)
            precedence = 0
            for event in range(len(events_springs)):
                if events_springs[event][0] != 0 and len(events_springs[event]) > 1:
                    precedence += 1
                    start, end = events_frames[event][0] + self.set_frames[0], events_frames[event][-1] + self.set_frames[0]
                    sudden_appearance = np.any(np.isnan(ants_attached_labels[start-3:start, events_springs[event][0] - 1])).astype(np.uint8)
                    self.profiles = np.vstack((self.profiles, np.array([ant + 1, events_springs[event][0], start, end, precedence, sudden_appearance])))
        self.profiles = self.profiles[1:, :]
        np.savez_compressed(os.path.join(self.output_path, "ant_profiles.npz"), self.profiles)
        # self.profiles = pd.DataFrame(self.profiles[1:, :])
        # self.profiles.columns = ["ant", "spring", "start", "end", "precedence", "sudden_appearance"]
        # self.profiles.to_pickle(os.path.join(self.output_path, "ant_profiles.pkl"))


# if __name__ == "__main__":
#     spring_type = "plus_0.2"
#     output_path = f"Z:\\Dor_Gabay\\ThesisProject\\data\\3-data_analysis\\summer_2023\\{spring_type}\\"
#     sets_video_paths = pickle.load(open(os.path.join(output_path, "sets_video_paths.pkl"), "rb"))
#     sets_frames = [(video_set[0][0], video_set[-1][1]) for video_set in pickle.load(open(os.path.join(output_path, "sets_frames.pkl"), "rb"))]
#     pool = Pool()
#     pool.starmap(AntTracking, zip(sets_video_paths, repeat(output_path), sets_frames, repeat((2160, 3840)), repeat(False)))
#     pool.close()
#     pool.join()





# def profile_ants_behavior(profiles):
#     for count, query_data in enumerate(
#             ["force_magnitude", "force_direction", "N_ants_around_springs", "fixed_end_angle_to_nest",
#              "angular_velocity", "tangential_force"]):
#         if query_data == "angular_velocity":
#             data = np.load(os.path.join(path, "fixed_end_angle_to_nest.npz"))["arr_0"]
#             data = np.nanmedian(utils.calc_angular_velocity(data, diff_spacing=1) / 1, axis=1)
#         elif query_data == "N_ants_around_springs":
#             data = np.load(os.path.join(path, f"{query_data}.npz"))["arr_0"]
#             all_profiles_data2 = np.full((len(profiles), 10000), np.nan)
#         else:
#             data = np.load(os.path.join(path, f"{query_data}.npz"))["arr_0"]
#         all_profiles_data = np.full((len(profiles), 10000), np.nan)
#         for profile in range(len(profiles)):
#             # ant = int(profiles[profile, 0] - 1)
#             spring = int(profiles[profile, 1])
#             start = int(profiles[profile, 2])
#             end = int(profiles[profile, 3])
#             if not end - start + 1 > 10000:
#                 if query_data == "angular_velocity":
#                     all_profiles_data[profile, 0:end - start + 1] = data[start:end + 1]
#                 elif query_data == "N_ants_around_springs":
#                     all_profiles_data[profile, 0:end - start + 1] = data[start:end + 1, int(spring - 1)]
#                     all_profiles_data2[profile, 0:end - start + 1] = np.nansum(data[start:end + 1], axis=1)
#                 else:
#                     all_profiles_data[profile, 0:end - start + 1] = data[start:end + 1, int(spring - 1)]
#         if query_data == "N_ants_around_springs":
#             np.savez_compressed(os.path.join(path, f"all_profiles_{query_data}_sum.npz"),
#                                 all_profiles_data2)
#         np.savez_compressed(os.path.join(path, f"all_profiles_{query_data}.npz"), all_profiles_data)
#     np.savez_compressed(os.path.join(path, "all_profiles_information.npz"), profiles)


