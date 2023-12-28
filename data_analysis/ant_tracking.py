import os
import apytl
import numpy as np
import scipy.io as sio
import subprocess
# local imports:
import utils


class AntTracking:
    def __init__(self, data_paths, output_path, set_frames, frame_resolution, restart=False):
        self.data_paths = data_paths
        self.restart = restart
        self.frame_size = frame_resolution
        self.set_frames = set_frames
        self.sub_dirs_names = [os.path.basename(os.path.split(os.path.normpath(path))[-1]) for path in data_paths]
        self.output_path = os.path.join(output_path, f"{self.sub_dirs_names[0]}-{self.sub_dirs_names[-1]}")
        self.first_tracking_calculation()
        CorrectTrackedAnts(self.output_path, self.sub_dirs_names, self.restart, self.frame_size)
        self.assign_ants_to_springs()
        self.create_ant_profiles()
        print(f"\rFinished tracking ants for {self.sub_dirs_names[0]}-{self.sub_dirs_names[-1]}")

    def first_tracking_calculation(self):
        if not os.path.exists(os.path.join(self.output_path, "ants_centers.mat")):
            ants_centers_x = np.concatenate([np.loadtxt(os.path.join(path, "ants_centers_x.csv"), delimiter=",") for path in self.data_paths], axis=0)
            ants_centers_y = np.concatenate([np.loadtxt(os.path.join(path, "ants_centers_y.csv"), delimiter=",") for path in self.data_paths], axis=0)
            ants_centers = np.stack((ants_centers_x, ants_centers_y), axis=2)
            del ants_centers_x, ants_centers_y
            ants_centers_mat = np.zeros((ants_centers.shape[0], 1), dtype=object)
            for i in range(ants_centers.shape[0]):
                ants_centers_mat[i, 0] = ants_centers[i, :, :]
            os.makedirs(self.output_path, exist_ok=True)
            if self.restart:
                print(f"removing old tracking data from {self.output_path}")
                os.system(f"del {self.output_path}\\ants_centers.mat")
            print(f"\rSaving ants centers data (Stage 1/5)... ({self.sub_dirs_names[0]}-{self.sub_dirs_names[-1]})")
            sio.savemat(os.path.join(self.output_path, "ants_centers.mat"), {"ants_centers": ants_centers_mat})
        if self.restart or not os.path.exists(os.path.join(self.output_path, "tracking_data.mat")):
            matlab_script_path = "Z:\\Dor_Gabay\\ThesisProject\\scripts\\munkres_tracker\\"
            os.chdir(matlab_script_path)
            os.system("PATH=$PATH:'C:\Program Files\MATLAB\R2022a\bin'")
            execution_string = f"matlab -r ""ants_tracking('" + self.output_path + "\\')"""
            print(f"\rRunning matlab script for ants tracking (Stage 2/5)... ({self.sub_dirs_names[0]}-{self.sub_dirs_names[-1]})")
            run_output = subprocess.run(execution_string, shell=True, capture_output=True)
            print(run_output.stdout.decode("utf-8"))

    def assign_ants_to_springs(self):
        if self.restart or not os.path.exists(os.path.join(self.output_path, "ants_assigned_to_springs.npz")):
            utils.wait_for_existance(self.output_path, "tracking_data_corrected.mat")
            print(f"\nAssigning ants to springs (Stage 4/5)... ({self.sub_dirs_names[0]}-{self.sub_dirs_names[-1]})")
            tracked_ants = sio.loadmat(os.path.join(self.output_path, "tracking_data_corrected.mat"))["tracked_blobs_matrix"].astype(np.uint32)
            unique_elements, indices = np.unique(tracked_ants[:, 2, :], return_inverse=True)
            tracked_ants[:, 2, :] = indices.reshape(tracked_ants[:, 2, :].shape)
            ants_attached_labels = np.load(os.path.join(self.output_path, "ants_attached_labels.npz"))["arr_0"]
            ants_attached_labels[np.isnan(ants_attached_labels)] = 0
            ants_attached_labels = ants_attached_labels.astype(np.uint8)
            if tracked_ants.shape[0] > ants_attached_labels.shape[1]:
                ants_attached_labels = np.concatenate((ants_attached_labels,
                                                       np.zeros((ants_attached_labels.shape[0], tracked_ants.shape[0] - ants_attached_labels.shape[1]), dtype=np.uint8)), axis=1)
            num_of_frames = ants_attached_labels.shape[0]
            ants_assigned_to_springs = np.zeros((num_of_frames, len(unique_elements)-1), dtype=np.uint8)
            for i in range(num_of_frames):
                apytl.Bar().drawbar(i, num_of_frames, fill='*')
                boolean = (tracked_ants[:, 2, i] > 0) * (ants_attached_labels[i] != 0)
                labels = tracked_ants[boolean, 2, i].astype(int)
                springs = ants_attached_labels[i, boolean].astype(int)
                ants_assigned_to_springs[i, labels-1] = springs
            del ants_attached_labels, tracked_ants
            ants_assigned_to_springs = utils.interpolate_assigned_ants(ants_assigned_to_springs, num_of_frames)
            np.savez_compressed(os.path.join(self.output_path, "ants_assigned_to_springs.npz"), ants_assigned_to_springs)
            del ants_assigned_to_springs

    def create_ant_profiles(self):
        """Columns order:
            ant | spring | start | end | precedence"""
        if self.restart or not os.path.exists(os.path.join(self.output_path, "ant_profiles.npz")):
            print(f"\nCreating ants profiles (Stage 5/5)... ({self.sub_dirs_names[0]}-{self.sub_dirs_names[-1]})")
            ants_assigned_to_springs = np.load(os.path.join(self.output_path, "ants_assigned_to_springs.npz"))["arr_0"][:, :-1].astype(np.uint8)
            profiles = np.full(5, np.nan)  # ant, spring, start, end, precedence
            for ant in range(ants_assigned_to_springs.shape[1]):
                apytl.Bar().drawbar(ant, ants_assigned_to_springs.shape[1], fill='*')
                attachment = ants_assigned_to_springs[:, ant]
                events_springs = np.split(attachment, np.arange(len(attachment[1:]))[np.diff(attachment) != 0] + 1)
                events_frames = np.split(np.arange(len(attachment)), np.arange(len(attachment[1:]))[np.diff(attachment) != 0] + 1)
                precedence = 0
                for event in range(len(events_springs)):
                    if events_springs[event][0] != 0 and len(events_springs[event]) > 1:
                        precedence += 1
                        start, end = events_frames[event][0] + self.set_frames[0], events_frames[event][-1] + self.set_frames[0]
                        profiles = np.vstack((profiles, np.array([ant + 1, events_springs[event][0], start, end, precedence])))
            del ants_assigned_to_springs
            profiles = profiles[1:, :]
            np.savez_compressed(os.path.join(self.output_path, "ant_profiles.npz"), profiles)
            del profiles


class CorrectTrackedAnts:
    def __init__(self, output_path, sub_dirs_names, restart, frame_size):
        self.output_path = output_path
        self.sub_dirs_names = sub_dirs_names
        self.restart = restart
        self.frame_size = frame_size
        self.main()

    def main(self):
        """
        This function removes all new labels that were create inside the frame, and not on the boundaries (10% of the frame)
        """
        if self.restart or not os.path.exists(os.path.join(self.output_path, "tracking_data_corrected.mat")):
            self.load_tracked_ants()
            self.appeared_labels = np.array([])
            self.appeared_labels_idx = []
            self.appeared_last_coordinates = np.array([]).reshape(0, 2)
            self.disappeared_labels = []
            self.disappeared_labels_partners = []
            self.disappeared_labels_partners_relative_distance = []
            self.disappeared_labels_partners_idx = []
            for frame in range(1, self.num_of_frames - 1):
                apytl.Bar().drawbar(frame, self.num_of_frames - 1, fill='*')
                if frame + 20 >= self.num_of_frames:
                    break
                if frame >= 11:
                    self.min_frame = frame - 10
                else:
                    self.min_frame = 1
                self.max_frame = frame + 15000
                self.sort_labels(frame)
                self.treat_appearing_labels()
                self.treat_disappearing_labels(frame)
                self.remove_disappeared_labels_duplicates()
                self.remove_short_occurrences()
            print(f"\nSaving corrected tracking data (Stage 3/5)... ({self.sub_dirs_names[0]}-{self.sub_dirs_names[-1]})")
            sio.savemat(os.path.join(self.output_path, "tracking_data_corrected.mat"), {"tracked_blobs_matrix": self.tracked_ants})
            del self.tracked_ants

    def load_tracked_ants(self):
        utils.wait_for_existance(self.output_path, "tracking_data.mat")
        print(f"\nCorrecting tracked ants (Stage 3/5)... ({self.sub_dirs_names[0]}-{self.sub_dirs_names[-1]})")
        self.tracked_ants = sio.loadmat(os.path.join(self.output_path, "tracking_data.mat"))["tracked_blobs_matrix"]
        self.num_of_frames = self.tracked_ants.shape[2]
        self.frames_arrange = np.repeat(np.arange(self.tracked_ants.shape[2])[np.newaxis, :], self.tracked_ants.shape[0], axis=0)
        self.ants_arrange = np.repeat(np.arange(self.tracked_ants.shape[0])[np.newaxis, :], self.tracked_ants.shape[2], axis=0).T
        self.tracked_ants[np.isnan(self.tracked_ants)] = 0

    def sort_labels(self, frame):
        labels_in_window = set(self.tracked_ants[:, 2, self.min_frame:frame + 20].flatten())
        labels_scanned_for_appearing = set(self.tracked_ants[:, 2, self.min_frame - 1].flatten())
        labels_now = set(self.tracked_ants[:, 2, frame])
        labels_before = set(self.tracked_ants[:, 2, frame - 1])
        self.appearing_labels = labels_in_window.difference(labels_scanned_for_appearing)
        self.disappearing_labels = labels_before.difference(labels_now)

    def treat_appearing_labels(self):
        appeared_not_collected = self.appearing_labels.difference(set(self.appeared_labels))
        for label in appeared_not_collected:
            label_idx = np.isin(self.tracked_ants[:, 2, self.min_frame:self.max_frame], label)
            label_idx = (self.ants_arrange[:, self.min_frame:self.max_frame][label_idx], self.frames_arrange[:, self.min_frame:self.max_frame][label_idx])
            coordinates = self.tracked_ants[label_idx[0], 0:2, label_idx[1]]
            first_appear = np.argmin(label_idx[1])
            if utils.test_if_on_boundaries(coordinates[first_appear, :], self.frame_size):
                self.appeared_labels = np.append(self.appeared_labels, label)
                median_coordinates = coordinates[first_appear, :].reshape(1, 2)
                self.appeared_last_coordinates = np.concatenate((self.appeared_last_coordinates, median_coordinates), axis=0)
                self.appeared_labels_idx.append(label_idx)

    def treat_disappearing_labels(self, frame):
        currently_appearing = np.array([x for x in range(len(self.appeared_labels)) if self.appeared_labels[x] in self.appearing_labels])
        if len(currently_appearing) > 0:
            currently_appearing_labels = self.appeared_labels[currently_appearing]
            currently_appearing_idx = [self.appeared_labels_idx[x] for x in currently_appearing]
            currently_appearing_last_coords = self.appeared_last_coordinates[currently_appearing]
            for label in self.disappearing_labels:
                coordinates = self.tracked_ants[self.tracked_ants[:, 2, frame - 1] == label, 0:2, frame - 1].flatten()
                if utils.test_if_on_boundaries(coordinates, self.frame_size):
                    if currently_appearing_last_coords.shape[0] > 0:
                        distances = np.linalg.norm(currently_appearing_last_coords - coordinates, axis=1)
                        closest_appearing_label_argmin = np.argsort(distances)[0]
                        if label == currently_appearing_labels[closest_appearing_label_argmin] and len(distances) > 1:
                            closest_appearing_label_argmin = np.argsort(distances)[1]
                        partner_label = currently_appearing_labels[closest_appearing_label_argmin]
                        distance_closest = distances[closest_appearing_label_argmin]
                        idx_closest_appearing_label = currently_appearing_idx[closest_appearing_label_argmin]
                        if distance_closest < 300:
                            self.disappeared_labels.append(label)
                            self.disappeared_labels_partners.append(partner_label)
                            self.disappeared_labels_partners_relative_distance.append(distance_closest)
                            self.disappeared_labels_partners_idx.append(idx_closest_appearing_label)

    def remove_disappeared_labels_duplicates(self):
        for i in range(len(self.disappeared_labels)):
            idx = self.disappeared_labels_partners_idx[i]
            self.tracked_ants[idx[0], 2, idx[1]] = self.disappeared_labels[i]
            if self.disappeared_labels_partners[i] in self.disappeared_labels:
                self.disappeared_labels[self.disappeared_labels.index(self.disappeared_labels_partners[i])] = self.disappeared_labels[i]
            duplicated_idxs = np.arange(len(self.disappeared_labels_partners))[np.array(self.disappeared_labels_partners) == self.disappeared_labels_partners[i]]
            duplicated_labels = [self.disappeared_labels[x] for x in duplicated_idxs]
            self.disappeared_labels = [np.min(duplicated_labels) if x in duplicated_labels else x for x in self.disappeared_labels]

    def remove_short_occurrences(self):
        for unique_label in np.unique(self.tracked_ants[:, 2, :]):
            occurrences = np.count_nonzero(self.tracked_ants[:, 2, :] == unique_label)
            if occurrences < 20:
                idx = np.where(self.tracked_ants[:, 2, :] == unique_label)
                self.tracked_ants[idx[0], 2, idx[1]] = 0

