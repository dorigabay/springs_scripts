import os
import cv2
import glob
import apytl
import pickle
import numpy as np
import scipy.io as sio
# local imports:
# from data_analysis.analysis import Analyser
from data_analysis import Analyser
import utils


class VisualData:
    def __init__(self, video_path, video_analysis_path, data_analysis_path, output_path, spring_type, n_springs, n_frames_to_save,
                       start_frame=0, reduction_factor=1., draw_amoeba=False):
        self.video_path = video_path
        missing_sub_dirs = video_path.split('.MP4')[0].split(spring_type)[1:][0].split(os.sep)[1:]
        self.video_analysis_path = os.path.join(video_analysis_path, *missing_sub_dirs)
        self.data_analysis_path = data_analysis_path
        self.output_path = output_path
        self.start_frame = start_frame
        self.reduction_factor = reduction_factor
        self.n_springs = n_springs
        self.draw_amoeba = draw_amoeba
        self.load_data(Analyser(self.data_analysis_path, os.path.basename(video_analysis_path), spring_type))
        self.n_frames_to_save = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        self.n_frames_to_save = self.n_frames_to_save if (n_frames_to_save == -1) or (n_frames_to_save is None) or (n_frames_to_save > self.n_frames_to_save) else n_frames_to_save
        self.color_data()
        self.create_video()

    def load_data(self, data):
        data_analysis_sub_path, start, end, set_s, set_e = self.load_video_info()
        self.object_fixed_end_angle_to_nest = np.load(os.path.join(data_analysis_sub_path, "object_fixed_end_angle_to_nest.npz"))['arr_0'][start:end]
        self.fixed_end_angle_to_nest = np.load(os.path.join(data_analysis_sub_path, "fixed_end_angle_to_nest.npz"))['arr_0'][start:end]
        # rearrangement = np.append(np.arange(arrangement, self.n_springs), np.arange(0, arrangement))
        self.fixed_ends_coordinates = np.stack((np.loadtxt(os.path.join(self.video_analysis_path, "fixed_ends_coordinates_x.csv"), delimiter=",").reshape(-1, self.n_springs),
        np.loadtxt(os.path.join(self.video_analysis_path, "fixed_ends_coordinates_y.csv"), delimiter=",").reshape(-1, self.n_springs)), axis=2)[self.start_frame:, :]
        self.free_ends_coordinates = np.stack((np.loadtxt(os.path.join(self.video_analysis_path, "free_ends_coordinates_x.csv"), delimiter=",").reshape(-1, self.n_springs),
        np.loadtxt(os.path.join(self.video_analysis_path, "free_ends_coordinates_y.csv"), delimiter=",").reshape(-1, self.n_springs)), axis=2)[self.start_frame:, :]
        self.needle_part_coordinates = np.stack((np.loadtxt(os.path.join(self.video_analysis_path, "needle_part_coordinates_x.csv"), delimiter=","),
        np.loadtxt(os.path.join(self.video_analysis_path, "needle_part_coordinates_y.csv"), delimiter=",")), axis=2)[self.start_frame:]
        self.object_center_coordinates = self.needle_part_coordinates[:, 0]
        self.needle_tip_coordinates = self.needle_part_coordinates[:, -1]
        self.number_of_springs = self.fixed_ends_coordinates.shape[1]
        self.force_magnitude = data.force_magnitude[set_s:set_e][start:end]
        self.force_direction = data.force_direction[set_s:set_e][start:end]
        self.fixed_end_angle_to_nest = data.fixed_end_angle_to_nest[set_s:set_e][start:end]
        self.tangential_force = data.tangential_force[set_s:set_e][start:end]
        self.net_tangential_force = np.sum(self.tangential_force, axis=1)
        self.angular_velocity = data.angular_velocity[set_s:set_e][start:end]
        self.load_ants_data(data_analysis_sub_path, start, end)

    def load_video_info(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        sets_frames = pickle.load(open(os.path.join(self.data_analysis_path, "sets_frames.pkl"), "rb"))
        sets_video_paths = pickle.load(open(os.path.join(self.data_analysis_path, "sets_video_paths.pkl"), "rb"))
        set_count, video_count = [(set_count, video_count) for set_count, video_paths_set in enumerate(sets_video_paths) for video_count, video in enumerate(video_paths_set)
                                  if os.path.normpath(video) == os.path.normpath(self.video_analysis_path)][0]
        set_first_video = os.path.basename(sets_video_paths[set_count][0]).split(".")[0]
        set_last_video = os.path.basename(sets_video_paths[set_count][-1]).split(".")[0]
        data_analysis_sub_path = os.path.join(self.data_analysis_path, f"{set_first_video}-{set_last_video}")
        set_s, set_e = sets_frames[set_count][0][0], sets_frames[set_count][-1][1]
        start, end = sets_frames[set_count][video_count] - sets_frames[set_count - 1][-1][1] if set_count != 0 else sets_frames[set_count][video_count]
        start, end = start + self.start_frame, end + self.start_frame + 1
        return data_analysis_sub_path, start, end, set_s, set_e

    def load_ants_data(self, path, start, end):
        self.ants_assigned_to_springs = np.load(os.path.join(path, "ants_assigned_to_springs.npz"))['arr_0'][start:end]
        tracked_ants = sio.loadmat(os.path.join(path, "tracking_data_corrected.mat"))["tracked_blobs_matrix"].astype(np.uint32)
        unique_elements, indices = np.unique(tracked_ants[:, 2, :], return_inverse=True)
        tracked_ants[:, 2, :] = indices.reshape(tracked_ants[:, 2, :].shape)
        self.tracked_ants = tracked_ants[:, :, start:end]

    def color_data(self):
        self.force_magnitude_color_range, self.force_magnitude_color_range_bins = utils.create_color_space(self.force_magnitude)
        self.tangential_force_color_range, self.tangential_force_color_range_bins = utils.create_color_space(self.tangential_force, around_zero=True)
        self.net_tangential_force_color_range, self.net_tangential_force_color_range_bins = utils.create_color_space(self.net_tangential_force, around_zero=True)
        self.velocity_color_range, self.velocity_color_range_bins = utils.create_color_space(self.angular_velocity, around_zero=True)

    def create_video(self):
        os.makedirs(self.output_path, exist_ok=True)
        video_name = f"results_video_{os.path.split(self.video_path)[-1].split('.')[0]}.MP4"
        video_name += "-amoeba_mask.MP4" if self.draw_amoeba else ""
        save_path = os.path.join(self.output_path, video_name)
        previous_points = None
        _, frame = self.cap.read()
        video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (frame.shape[1], frame.shape[0]))
        for frame_idx in range(1, self.n_frames_to_save):
            apytl.Bar().drawbar(frame_idx, self.n_frames_to_save, fill='*')
            _, frame = self.cap.read()
            if frame is None:
                break
            cv2.resize(frame, (0, 0), fx=self.reduction_factor, fy=self.reduction_factor)
            if self.draw_amoeba:
                previous_points = self.draw_amoeba_results(frame, frame_idx, self.reduction_factor, previous_points=previous_points)
            else:
                # self.draw_ants_tracked(frame, frame_idx, self.reduction_factor)
                self.draw_springs_vectors(frame, frame_idx, self.reduction_factor)
                self.draw_parts_detections(frame, frame_idx, self.reduction_factor)
                self.write_net_values(frame, frame_idx, self.reduction_factor)
            video_writer.write(frame)
        video_writer.release()
        print("\rFinished saving video to: ", save_path)

    def draw_amoeba_results(self, frame, frame_idx, reduction_factor, previous_points=None):
        vector_end_points = np.full((self.number_of_springs, 2), np.nan)
        object_center_coordinates = self.object_center_coordinates[frame_idx]
        for spring_idx in range(self.number_of_springs):
            midpoint = (object_center_coordinates + self.fixed_ends_coordinates[frame_idx, spring_idx, :]) / 2
            universal_angle = self.object_fixed_end_angle_to_nest[frame_idx, spring_idx] + np.pi / 2 + self.force_direction[frame_idx, spring_idx]
            vector_end_point = midpoint + 100 * self.force_magnitude[frame_idx, spring_idx] * np.array([np.cos(universal_angle), np.sin(universal_angle)])
            if not np.isnan(vector_end_point).any():
                vector_end_point = tuple((vector_end_point * reduction_factor).astype(int))
                vector_end_points[spring_idx, :] = vector_end_point
        frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        if previous_points is not None:
            previous_points = np.concatenate((previous_points, vector_end_points[np.newaxis, :, :]), axis=0)
            previous_points[:, :, 0] = utils.interpolate_columns(previous_points[:, :, 0])
            previous_points[:, :, 1] = utils.interpolate_columns(previous_points[:, :, 1])
            points = previous_points[-1, :, :].reshape((-1, 1, 2)).astype(int)
            cv2.fillPoly(frame, [points], color=(255, 255, 255))
        else:
            previous_points = vector_end_points[np.newaxis, :, :]
        if not np.isnan(vector_end_points).any():
            cv2.circle(frame, (int(object_center_coordinates[0]), int(object_center_coordinates[1])), 5, (0, 0, 255), -1)
        return previous_points

    def draw_ants_tracked(self, frame, frame_idx, reduction_factor):
        tracked_ants = self.tracked_ants[:, :, frame_idx].astype(int)
        for count, ant in enumerate(tracked_ants):
            if (ant[:2] != 0).all():
                coordinates = tuple((ant[:2] * reduction_factor).astype(int))
                label = ant[2].astype(int)
                cv2.circle(frame, coordinates, int(3 * reduction_factor), (0, 0, 255), -1)
                # cv2.putText(frame, str(label), coordinates, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * reduction_factor, (0, 0, 255), 2)
                spring = self.ants_assigned_to_springs[frame_idx, label - 1].astype(int)
                if spring != 0:
                    # cv2.putText(frame, str(spring), coordinates, cv2.FONT_HERSHEY_SIMPLEX,  0.7*reduction_factor, (0, 255, 0), 2)
                    cv2.circle(frame, coordinates, int(3 * reduction_factor), (255, 0, 255), -1)

    def draw_springs_vectors(self, frame, frame_idx, reduction_factor):
        for spring in range(self.number_of_springs):
            angle_to_nest = self.fixed_end_angle_to_nest[frame_idx, spring]
            tangential_force_color = [int(x) for x in self.tangential_force_color_range[np.argmin(np.abs(self.tangential_force[frame_idx, spring] - self.tangential_force_color_range_bins))]]
            start_point = self.fixed_ends_coordinates[frame_idx, spring, :]
            end_point = self.free_ends_coordinates[frame_idx, spring, :]
            universal_angle = self.object_fixed_end_angle_to_nest[frame_idx, spring] + np.pi / 2 + self.force_direction[frame_idx, spring]
            vector_end_point = start_point + 100 * self.force_magnitude[frame_idx, spring] * np.array([np.cos(universal_angle), np.sin(universal_angle)])
            if not np.isnan(start_point).any() and not np.isnan(end_point).any() and not np.isnan(vector_end_point).any():
                start_point = tuple((start_point*reduction_factor).astype(int))
                end_point = tuple((end_point*reduction_factor).astype(int))
                vector_end_point = tuple((vector_end_point*reduction_factor).astype(int))
                cv2.arrowedLine(frame, start_point, vector_end_point, color=tangential_force_color, thickness=int(3*reduction_factor))
                cv2.circle(frame, end_point, int(3*reduction_factor), (0, 0, 0), thickness=-1)
                cv2.circle(frame, start_point, int(3*reduction_factor), (0, 0, 0), thickness=-1)
                cv2.putText(frame, str(spring+1), start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.7*reduction_factor, (0, 200, 255), 2)
                # write angle to nest:
                # cv2.putText(frame, str(int(angle_to_nest*180/np.pi)), vector_end_point, cv2.FONT_HERSHEY_SIMPLEX, 0.7*reduction_factor, (0, 200, 255), 2)

    def draw_parts_detections(self, frame, frame_idx, reduction_factor):
        circle_coordinates = (np.concatenate((self.object_center_coordinates[frame_idx, :].reshape(1, 2),
                                              self.needle_tip_coordinates[frame_idx, :].reshape(1, 2)), axis=0) * reduction_factor).astype(int)
        # object_center = tuple((self.object_center_coordinates[frame_idx, :] * reduction_factor).astype(int))
        # needle_tip = tuple((self.needle_tip_coordinates[frame_idx, :] * reduction_factor).astype(int))
        four_colors = [(0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]
        for (x, y), c in zip(circle_coordinates, four_colors[:circle_coordinates.shape[0]]):
            cv2.circle(frame, (x, y), int(3*reduction_factor), (0, 255, 0), thickness=-1)

    def write_net_values(self, frame, frame_idx, reduction_factor=0.4):
        colors = [int(x) for x in self.net_tangential_force_color_range[np.argmin(np.abs(self.net_tangential_force[frame_idx] - self.net_tangential_force_color_range_bins))]]
        coordinates = tuple((np.array((1440, 300))*reduction_factor).astype(int))
        text = "Net tangential force (mN): "+str(np.round(self.net_tangential_force[frame_idx], 3))
        cv2.putText(frame, text, coordinates, cv2.FONT_HERSHEY_SIMPLEX, 1*reduction_factor, colors, int(2*reduction_factor), cv2.LINE_AA)
        colors = [int(x) for x in self.velocity_color_range[np.argmin(np.abs(self.angular_velocity[frame_idx] - self.velocity_color_range_bins))]]
        coordinates = tuple((np.array((1440, 200))*reduction_factor).astype(int))
        text = "Angular velocity (rad/s): "+str(np.round(self.angular_velocity[frame_idx], 3))
        cv2.putText(frame, text, coordinates, cv2.FONT_HERSHEY_SIMPLEX, 1*reduction_factor, colors, int(2*reduction_factor), cv2.LINE_AA)


if __name__ == "__main__":
    spring = "plus_0.1"
    video_idx = 1
    video_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\1-videos\\summer_2023\\experiment\\{spring}\\"
    video_path = os.path.normpath(glob.glob(os.path.join(video_dir, "**", "*.MP4"), recursive=True)[video_idx])
    video_analysis_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\2-video_analysis\\summer_2023\\experiment\\{spring}\\"
    data_analysis_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\3-data_analysis\\summer_2023\\experiment\\{spring}\\"
    results_output_dir = f"Z:\\Dor_Gabay\\ThesisProject\\results\\summer_2023\\{spring}\\"
    self = VisualData(video_path, video_analysis_dir, data_analysis_dir, results_output_dir, spring, n_springs=20,
                      n_frames_to_save=1000, reduction_factor=1, draw_amoeba=False)

