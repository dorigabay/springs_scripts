import pickle
import pandas as pd
import cv2
import numpy as np
import os
import scipy.io as sio
# local imports:
import utils
# from data_analysis.data_preparation import PostProcessing
from data_analysis.analysis import AnalyserPerSpring

def create_video(class_object):
    save_path = os.path.join(class_object.data_analysis_path,
                             f"results_video_{os.path.split(class_object.video_path)[-1].split('.')[0]}.MP4")
    print("saving video to: ", save_path)
    for frame_num in range(class_object.frames_num):
        print("\rframe: ", frame_num, end="")
        ret, frame = class_object.cap.read()
        if frame_num == class_object.n_frames_to_save or frame is None:
            break
        frame = cv2.resize(frame, (0, 0), fx=class_object.reduction_factor, fy=class_object.reduction_factor)
        frame = class_object.draw_results_on_frame(frame, frame_num, reduction_factor=class_object.reduction_factor)
        if frame_num == 0:
            video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), class_object.fps, (frame.shape[1], frame.shape[0]))
        video_writer.write(frame)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
    video_writer.release()
    print("\rFinished in creating the results video")


def create_color_space(hue_data, around_zero=False):
    x = np.linspace(0, 1, 100)
    blue = (0, 0, 1)
    white = (1, 1, 1)
    red = (1, 0, 0)
    colors = np.empty((100, 3))
    if around_zero:
        for i in range(3):
            colors[:, i] = np.interp(x, [0, 0.5, 1], [blue[i], white[i], red[i]])
    else:  # range color from white to red
        for i in range(3):
            colors[:, i] = np.interp(x, [0, 1], [white[i], blue[i]])
    color_range = (colors * 255).astype(int)
    flatten = hue_data.flatten()#[0:5000]
    flatten = flatten[~np.isnan(flatten)]
    median_biggest = np.median(np.sort(flatten)[-100:])
    median_smallest = np.median(np.sort(flatten)[:100])
    color_range_bins = np.linspace(median_smallest, median_biggest, 100)
    if around_zero:
        color_range_bins = np.linspace(-np.median(np.sort(np.abs(flatten))[-100:]), np.median(np.sort(np.abs(flatten))[-100:]), 100)
    return color_range, color_range_bins


# class TrackingResultVideo():
#     def __init__(self, data_path, video_path, n_frames_to_save=100, reduction_factor=1.):
#         self.data_path = data_path
#         self.video_path = video_path
#         self.reduction_factor = reduction_factor
#         self.n_frames_to_save = n_frames_to_save
#         self.cap = cv2.VideoCapture(video_path)
#         self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
#         self.frames_num = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         self.load(data_path)
#         create_video(self)
#         print("Finished creating tracking results video")
#
#     def load(self,data_path):
#         self.ants_assigned_to_springs = np.load(os.path.join(data_path, "ants_assigned_to_springs_fixed.npz"))['arr_0']
#         self.tracked_ants = sio.loadmat(os.path.join(data_path,"tracking_data_corrected.mat"))["tracked_blobs_matrix"]
#         self.N_ants_around_springs = np.load(os.path.join(data_path, "N_ants_around_springs.npz"))['arr_0']
#
#     def draw_results_on_frame(self, frame, frame_num):
#         self.left_upper_corner = np.array([0,0])
#         tracked_ants = self.tracked_ants[~np.isnan(self.tracked_ants[:, 0, frame_num]), :, frame_num].astype(int)
#         # assigned_ants = self.ants_assigned_to_springs[frame_num]
#         for ant in tracked_ants[tracked_ants[:,2]!=0, :]:
#             cv2.circle(frame, tuple(ant[:2]), 5, (0, 0, 255), -1)
#             cv2.putText(frame, str(ant[2]), tuple(ant[:2]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         return frame


class ResultsVideo:
    def __init__(self, video_path, video_analysis_path, data_analysis_path, start_frame=0, n_frames_to_save=100, reduction_factor=1., n_springs=20):
        self.video_path = video_path
        self.n_springs = n_springs
        missing_sub_dirs = video_path.split('.MP4')[0].split(video_analysis_path.split(os.sep)[-2])[1:][0].split(os.sep)[1:]
        self.video_analysis_path = os.path.join(video_analysis_path, *missing_sub_dirs)
        self.data_analysis_path = data_analysis_path
        self.start_frame = start_frame
        self.n_frames_to_save = 10e5 if (n_frames_to_save == -1) or (n_frames_to_save is None) else n_frames_to_save
        self.reduction_factor = reduction_factor
        # self.calculations()
        data = AnalyserPerSpring(self.data_analysis_path, os.path.basename(video_analysis_path))
        self.load_data(data)
        self.force_magnitude_color_range, self.force_magnitude_color_range_bins = create_color_space(self.force_magnitude)
        self.tangential_force_color_range, self.tangential_force_color_range_bins = create_color_space(self.tangential_force, around_zero=True)
        self.net_tangential_force_color_range, self.net_tangential_force_color_range_bins = create_color_space(self.net_tangential_force, around_zero=True)
        self.velocity_color_range, self.velocity_color_range_bins = create_color_space(self.angular_velocity, around_zero=True)
        create_video(self)

    def load_data(self, data):
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frames_num = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        sets_frames = pickle.load(open(os.path.join(self.data_analysis_path, "sets_frames.pkl"), "rb"))
        sets_video_paths = pickle.load(open(os.path.join(self.data_analysis_path, "sets_video_paths.pkl"), "rb"))
        set_count, video_count = [(set_count, video_count) for set_count, video_paths_set in enumerate(sets_video_paths) for video_count, video in enumerate(video_paths_set)
                              if os.path.normpath(video) == os.path.normpath(self.video_analysis_path)][0]
        set_first_video = os.path.basename(sets_video_paths[set_count][0]).split(".")[0]
        set_last_video = os.path.basename(sets_video_paths[set_count][-1]).split(".")[0]
        data_analysis_sub_path = os.path.join(self.data_analysis_path, f"{set_first_video}-{set_last_video}")
        set_s, set_e = sets_frames[set_count][0][0], sets_frames[set_count][-1][1]
        start, end = sets_frames[set_count][video_count] - sets_frames[set_count-1][-1][1] if set_count != 0 else sets_frames[set_count][video_count]
        start, end = start+self.start_frame, end+self.start_frame
        self.needle_tip_coordinates = np.load(os.path.join(data_analysis_sub_path, "needle_tip_coordinates.npz"))['arr_0'][start:end]
        self.object_center_coordinates = np.load(os.path.join(data_analysis_sub_path, "object_center_coordinates.npz"))['arr_0'][start:end]
        self.fixed_ends_coordinates = np.load(os.path.join(data_analysis_sub_path, "fixed_ends_coordinates.npz"))['arr_0'][start:end]
        self.free_ends_coordinates = np.load(os.path.join(data_analysis_sub_path, "free_ends_coordinates.npz"))['arr_0'][start:end]

        rearrangement = np.append(np.arange(0, self.n_springs), np.arange(0, 0))
        self.fixed_ends_coordinates = np.stack((np.loadtxt(os.path.join(self.video_analysis_path, "fixed_ends_coordinates_x.csv"), delimiter=",").reshape(-1, self.n_springs),
                                                np.loadtxt(os.path.join(self.video_analysis_path, "fixed_ends_coordinates_y.csv"), delimiter=",").reshape(-1, self.n_springs)), axis=2)[self.start_frame:, rearrangement]
        self.free_ends_coordinates = np.stack((np.loadtxt(os.path.join(self.video_analysis_path, "free_ends_coordinates_x.csv"), delimiter=",").reshape(-1, self.n_springs),
                                               np.loadtxt(os.path.join(self.video_analysis_path, "free_ends_coordinates_y.csv"), delimiter=",").reshape(-1, self.n_springs)), axis=2)[self.start_frame:, rearrangement]
        self.needle_part_coordinates = np.stack((np.loadtxt(os.path.join(self.video_analysis_path, "needle_part_coordinates_x.csv"), delimiter=","),
                                                   np.loadtxt(os.path.join(self.video_analysis_path, "needle_part_coordinates_y.csv"), delimiter=",")), axis=2)[self.start_frame:]
        self.object_center_coordinates = self.needle_part_coordinates[:, 0]
        self.needle_tip_coordinates = self.needle_part_coordinates[:, -1]

        self.perspective_squares_coordinates = np.load(os.path.join(data_analysis_sub_path, "perspective_squares_coordinates.npz"))['arr_0'][start:end]
        self.object_fixed_end_angle_to_nest = np.load(os.path.join(data_analysis_sub_path, "object_fixed_end_angle_to_nest.npz"))['arr_0'][start:end]
        self.number_of_springs = self.fixed_ends_coordinates.shape[1]
        self.N_ants_around_springs = data.N_ants_around_springs[set_s:set_e][start:end]
        self.force_magnitude = data.force_magnitude[set_s:set_e][start:end]
        self.force_direction = data.force_direction[set_s:set_e][start:end]
        self.fixed_end_angle_to_nest = data.fixed_end_angle_to_nest[set_s:set_e][start:end]
        self.tangential_force = data.tangential_force[set_s:set_e][start:end]
        self.net_tangential_force = data.net_tangential_force[set_s:set_e][start:end]
        self.angular_velocity = data.angular_velocity[set_s:set_e][start:end]

        # self.object_center_repeated = np.copy(np.repeat(self.object_center_coordinates[:, np.newaxis, :], self.free_ends_coordinates.shape[1], axis=1))
        # self.needle_tip_repeated = np.copy(np.repeat(self.needle_tip_coordinates[:, np.newaxis, :], self.free_ends_coordinates.shape[1], axis=1))
        # self.pulling_angle = utils.calc_pulling_angle_matrix(self.fixed_ends_coordinates, self.object_center_repeated, self.free_ends_coordinates)
        # self.spring_length = np.linalg.norm(self.free_ends_coordinates - self.fixed_ends_coordinates, axis=2)
        # self.fixed_to_center_length = np.linalg.norm(self.object_center_repeated - self.fixed_ends_coordinates, axis=2)
        # self.fixed_to_tip_angle = utils.calc_pulling_angle_matrix(self.object_center_repeated, self.needle_tip_repeated, self.fixed_ends_coordinates)
        # self.N_ants_around_springs = np.load(os.path.join(data_analysis_sub_path, "N_ants_around_springs.npz"))['arr_0'][start:end]
        # self.force_magnitude = np.load(os.path.join(data_analysis_sub_path, "force_magnitude.npz"))['arr_0'][start:end]
        # self.force_direction = np.load(os.path.join(data_analysis_sub_path, "force_direction.npz"))['arr_0'][start:end]
        # self.fixed_end_angle_to_nest = np.load(os.path.join(data_analysis_sub_path, "fixed_end_angle_to_nest.npz"))['arr_0'][start:end]
        # self.ants_assigned_to_springs = np.load(os.path.join(data_analysis_sub_path, "ants_assigned_to_springs_fixed.npz"))['arr_0'][frames_idx[0]:frames_idx[1]]
        # self.tracked_ants = sio.loadmat(os.path.join(data_analysis_sub_path, "tracking_data_corrected.mat"))["tracked_blobs_matrix"][frames_idx[0]:frames_idx[1]]


    def draw_results_on_frame(self, frame, frame_num, reduction_factor=0.4):
        circle_coordinates = (np.concatenate((self.object_center_coordinates[frame_num, :].reshape(1, 2),
                                              self.needle_tip_coordinates[frame_num, :].reshape(1, 2),
                                              self.perspective_squares_coordinates[frame_num, :],), axis=0)*reduction_factor).astype(int)
        object_center = tuple((self.object_center_coordinates[frame_num, :]*reduction_factor).astype(int))
        needle_tip = tuple((self.needle_tip_coordinates[frame_num, :]*reduction_factor).astype(int))
        four_colors = [(0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]
        for (x, y), c in zip(circle_coordinates, four_colors):
            cv2.circle(frame, (x, y), int(5*reduction_factor), (0, 0, 0), thickness=-1)
        # cv2.circle(frame, circles_coordinates, int(5*reduction_factor), color=[255, 0, 0], thickness=-1)
        # cv2.circle(frame, tuple((self.object_center_coordinates[frame_num, :]*reduction_factor).astype(int)), int(5*reduction_factor), color=[255, 0, 0], thickness=-1)
        # cv2.circle(frame, tuple((self.needle_tip_coordinates[frame_num, :]*reduction_factor).astype(int)), int(5*reduction_factor), color=[255, 0, 0], thickness=-1)

        # tracked_ants = self.tracked_ants[~np.isnan(self.tracked_ants[:, 0, frame_num]), :, frame_num].astype(int)
        # for ant in tracked_ants[tracked_ants[:,2]!=0, :]:
        #     cv2.circle(frame, tuple(ant[:2]), 2, (0, 0, 255), -1)
        #     cv2.putText(frame, str(ant[2]), tuple(ant[:2]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        # cv2.circle(frame, tuple((self.perspective_squares_coordinates[frame_num, :]*reduction_factor).astype(int)), 2, (0, 255, 0), -1)

        for spring in range(self.number_of_springs):
            tangential_force_color = [int(x) for x in self.tangential_force_color_range[np.argmin(np.abs(self.tangential_force[frame_num, spring] - self.tangential_force_color_range_bins))]]
            force_magnitude_color = [int(x) for x in self.force_magnitude_color_range[np.argmin(np.abs(self.force_magnitude[frame_num, spring] - self.force_magnitude_color_range_bins))]]
            start_point = self.fixed_ends_coordinates[frame_num, spring, :]
            end_point = self.free_ends_coordinates[frame_num, spring, :]
            universal_angle = self.object_fixed_end_angle_to_nest[frame_num, spring] + np.pi/2 + self.force_direction[frame_num, spring]*-1
            vector_end_point = start_point + 100 * self.force_magnitude[frame_num, spring] * np.array([np.cos(universal_angle), np.sin(universal_angle)])
            if not np.isnan(start_point).any() and not np.isnan(end_point).any() and not  np.isnan(vector_end_point).any():
                start_point = tuple((start_point*reduction_factor).astype(int))
                end_point = tuple((end_point*reduction_factor).astype(int))
                vector_end_point = tuple((vector_end_point*reduction_factor).astype(int))
                # cv2.line(frame, start_point, end_point, color=(0, 0, 0), thickness=2)
                # cv2.line(frame, start_point, object_center, color=(0, 0, 0), thickness=2)
                # cv2.line(frame, object_center, needle_tip, color=(0, 0, 0), thickness=2)
                cv2.arrowedLine(frame, start_point, vector_end_point, color=tangential_force_color, thickness=1)
                # cv2.arrowedLine(frame, start_point, vector_end_point, color=tangential_force_color, thickness=int(3*reduction_factor))
                # cv2.putText(frame, str(self.N_ants_around_springs[frame_num, spring]), vector_end_point,
                #             cv2.FONT_HERSHEY_SIMPLEX, 1*reduction_factor, (0,0,0), 2)
                # cv2.putText(frame, str(np.round(self.pulling_angle[frame_num, spring], 4)), start_point, cv2.FONT_HERSHEY_SIMPLEX, 1*reduction_factor, (0, 0, 0), 2)
                # cv2.putText(frame, str(np.round(self.spring_length[frame_num, spring], 4)), end_point, cv2.FONT_HERSHEY_SIMPLEX, 1*reduction_factor, (0, 0, 0), 2)
                # cv2.putText(frame, str(np.round(self.fixed_to_center_length[frame_num, spring], 4)), object_center, cv2.FONT_HERSHEY_SIMPLEX, 1*reduction_factor, (0, 0, 0), 2)
                # cv2.putText(frame, str(np.round(self.fixed_to_tip_angle[frame_num, spring], 4)), object_center, cv2.FONT_HERSHEY_SIMPLEX, 1*reduction_factor, (0, 0, 0), 2)
                cv2.circle(frame, end_point, int(5*reduction_factor), (0, 0, 0), thickness=-1)
                cv2.circle(frame, start_point, int(5*reduction_factor), (0, 0, 0), thickness=-1)


        # write the net tangential force on the frame
        net_tangential_force_color = [int(x) for x in self.net_tangential_force_color_range[np.argmin(np.abs(self.net_tangential_force[frame_num] - self.net_tangential_force_color_range_bins))]]
        coordinates = tuple((np.array((1440, 300))*reduction_factor).astype(int))
        cv2.putText(frame, "Net tangential force (mN): "+str(np.round(self.net_tangential_force[frame_num],3)), coordinates,
                    cv2.FONT_HERSHEY_SIMPLEX, 1*reduction_factor, net_tangential_force_color, int(2*reduction_factor), cv2.LINE_AA)
        # write the velocity on the frame
        velocity_color = [int(x) for x in self.velocity_color_range[np.argmin(np.abs(self.angular_velocity[frame_num] - self.velocity_color_range_bins))]]
        coordinates = tuple((np.array((1440, 200))*reduction_factor).astype(int))
        cv2.putText(frame, "Angular velocity: " + str(np.round(self.angular_velocity[frame_num], 5)), coordinates,
                    cv2.FONT_HERSHEY_SIMPLEX, 1*reduction_factor, velocity_color, int(2*reduction_factor), cv2.LINE_AA)
        return frame


if __name__ == "__main__":
    import glob
    spring_type = "plus_0.1"
    # calib_or_experiment = "calibration"
    calib_or_experiment = "experiment"
    video_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\1-videos\\summer_2023\\{calib_or_experiment}\\{spring_type}\\"
    video_analysis_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\2-video_analysis\\summer_2023\\{calib_or_experiment}\\{spring_type}\\"
    data_analysis_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\3-data_analysis\\summer_2023\\{calib_or_experiment}\\{spring_type}\\"
    video_path = os.path.normpath(glob.glob(os.path.join(video_dir, "**", "*.MP4"), recursive=True)[1])
    self = ResultsVideo(video_path, video_analysis_dir, data_analysis_dir, n_frames_to_save=300, reduction_factor=0.4, n_springs=20)

