import cv2
import numpy as np
import os
from data_analysis.data_preparation import PostProcessing
from data_analysis.analysis import DataAnalyser
import pickle


# class TrackingResultVideo(PostProcessing):
class TrackingResultVideo():
    def __init__(self, data_path, video_path, calibration_model,n_frames_to_save=None):
        print("_"*50)
        # super().__init__(data_path, calibration_model,correct_tracking=False)
        # super().__init__(data_path)
        self.load(data_path)
        self.save_path = os.path.join(data_path, "ants_tracking_corrected.MP4")
        print("Creating tracking results video, for video: ", video_path)
        print("Save the video in: ", self.save_path)
        self.collect_cropping_coordinates(video_path)
        if n_frames_to_save is None:
            n_frames_to_save = self.N_ants_around_springs.shape[0]
        self.create_video(video_path,n_frames_to_save=n_frames_to_save)
        print("Finished creating tracking results video")

    def load(self,data_path):
        import scipy.io as sio
        data_path = os.path.join(data_path, "two_vars_post_processing")
        self.tracked_ants = sio.loadmat(os.path.join(data_path,"tracking_data_corrected.mat"))["tracked_blobs_matrix"]
        self.N_ants_around_springs = np.load(os.path.join(data_path, "N_ants_around_springs.npz"))['arr_0']

    def collect_cropping_coordinates(self,video_path):
        import pickle
        video_preferences_path = os.path.normpath("\\".join(video_path.split("\\")[:-2])+"\\video_preferences.pickle")
        video_preferences = pickle.load(open(video_preferences_path, "rb"))
        normed_paths_preferences = {os.path.normpath(path): preferences for path, preferences in video_preferences.items()}
        crop_coordinates = normed_paths_preferences[video_path]["crop_coordinates"]
        self.left_upper_corner = np.array(( crop_coordinates[0], crop_coordinates[2]))
        self.left_upper_corner = self.left_upper_corner[[1, 0]]

    def present_results_on_frame(self, frame, frame_num):#,blobs_labels):
        # self.left_upper_corner = np.array([0,0])
        # put circle on the object_center
        self.left_upper_corner = np.array([0,0])
        tracked_ants = self.tracked_ants[~np.isnan(self.tracked_ants[:, 0, frame_num]), :, frame_num]
        for i in range(tracked_ants.shape[0]):
            ant_label = tracked_ants[i, 2]
            ant_coordinates = tracked_ants[i, :2]
            object_center = ant_coordinates.astype(int)
            cv2.circle(frame, tuple(object_center), 5, (0, 0, 255), -1)
            # put text on the object_center
            cv2.putText(frame, str(ant_label), tuple(object_center), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def create_video(self, video_path,n_frames_to_save):
        cap = cv2.VideoCapture(video_path)
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_num in range(self.frames_num):
            if frame_num ==n_frames_to_save:
                break
            ret, frame = cap.read()
            #resize fra
            self.present_results_on_frame(frame, frame_num)
            self.save_video(frame, frame_num)
            # cv2.imshow('frame', frame)
            # cv2.waitKey(1)
        self.video_writer.release()

    def save_video(self,frame,frame_num):
        if frame_num == 0:
            self.video_writer = cv2.VideoWriter(self.save_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps,
                                                (frame.shape[1], frame.shape[0]))
        self.video_writer.write(frame)

class ResultVideo(DataAnalyser):
# class ResultVideo(PostProcessing):
    def __init__(self, data_path, video_path, calibration_model,n_frames_to_save=None):
        # self.blobs_mat = blobs_mat
        self.load(data_path)
        super().__init__(data_path)
        self.angular_velocity = self.angular_velocity*-1
        # super().__init__(data_path,calibration_model)
        self.save_path = os.path.join(data_path, "results_video.S5280008.MP4")
        print("save_path: ", self.save_path)
        self.collect_cropping_coordinates(video_path)
        # self.net_force = np.nansum(self.total_force,axis=1)
        if n_frames_to_save is None:
            n_frames_to_save = self.N_ants_around_springs.shape[0]
        self.force_color_range, self.force_color_range_bins = self.create_color_space(self.total_force,around_zero=True)
        self.net_force_color_range, self.net_force_color_range_bins = self.create_color_space(self.net_force,around_zero=True)
        self.velocity_color_range, self.velocity_color_range_bins = self.create_color_space(self.angular_velocity,around_zero=True)
        # self.pulling_angle_color_range, self.pulling_angle_color_range_bins = self.create_color_space(self.pulling_angle,around_zero=True)
        self.create_video(video_path,n_frames_to_save=n_frames_to_save)

    def load(self,data_path):
        # rearrange = np.array([19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
        rearrange = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        directory = os.path.join(data_path, "raw_analysis")
        self.N_ants_around_springs = np.loadtxt(os.path.join(directory,"N_ants_around_springs.csv"), delimiter=",")[:,rearrange]
        self.num_of_springs = self.N_ants_around_springs.shape[1]
        self.size_ants_around_springs = np.loadtxt(os.path.join(directory, "size_ants_around_springs.csv"), delimiter=",")[:,rearrange]
        fixed_ends_coordinates_x = np.loadtxt(os.path.join(directory, "fixed_ends_coordinates_x.csv"), delimiter=",")
        fixed_ends_coordinates_y = np.loadtxt(os.path.join(directory, "fixed_ends_coordinates_y.csv"), delimiter=",")
        self.fixed_ends_coordinates = np.stack((fixed_ends_coordinates_x, fixed_ends_coordinates_y), axis=2)[:,rearrange]
        import pandas as pd
        for col in range(self.fixed_ends_coordinates.shape[1]):
            x_smoothed = pd.DataFrame(self.fixed_ends_coordinates[:, col,0]).rolling(10, center=True, min_periods=1).mean().values[:, 0]
            y_smoothed = pd.DataFrame(self.fixed_ends_coordinates[:, col,1]).rolling(10, center=True, min_periods=1).mean().values[:, 0]
            self.fixed_ends_coordinates[:, col] = np.vstack((x_smoothed, y_smoothed)).T
        free_ends_coordinates_x = np.loadtxt(os.path.join(directory, "free_ends_coordinates_x.csv"), delimiter=",")
        free_ends_coordinates_y = np.loadtxt(os.path.join(directory, "free_ends_coordinates_y.csv"), delimiter=",")
        self.free_ends_coordinates = np.stack((free_ends_coordinates_x, free_ends_coordinates_y), axis=2)[:,rearrange]
        blue_part_coordinates_x = np.loadtxt(os.path.join(directory, "blue_part_coordinates_x.csv"), delimiter=",")
        blue_part_coordinates_y = np.loadtxt(os.path.join(directory, "blue_part_coordinates_y.csv"), delimiter=",")
        blue_part_coordinates = np.stack((blue_part_coordinates_x, blue_part_coordinates_y), axis=2)
        self.object_center = blue_part_coordinates[:, 0, :]
        self.blue_tip_coordinates = blue_part_coordinates[:, -1, :]
        self.num_of_frames = self.N_ants_around_springs.shape[0]
        # self.num_of_springs = self.N_ants_around_springs.shape[1]

    def create_color_space(self,hue_data,around_zero=False):
        # # Create a 100-element array ranging from 0 to 1
        x = np.linspace(0, 1, 100)
        # Define the blue, white, and red colors as RGB tuples
        blue = (0, 0, 1)
        white = (1, 1, 1)
        red = (1, 0, 0)
        # Create the gradient by interpolating between the blue and white colors
        # and then between the white and red colors
        colors = np.empty((100, 3))
        for i in range(3):
            colors[:, i] = np.interp(x, [0, 0.5, 1], [blue[i], white[i], red[i]])
        # colors = np.zeros((100, 3))
        # start_color = np.array([0, 0, 255])
        # end_color = np.array([255, 0, 0])
        # for i in range(100):
        #     colors[i] = start_color * (1 - i / 99) + end_color * (i / 99)
        # Convert the color array to integer values
        color_range = (colors*255).astype(int)
        # values range for the hue data
        flatten = hue_data.flatten()#[0:5000]
        flatten = flatten[~np.isnan(flatten)]
        median_biggest = np.median(np.sort(flatten)[-100:])
        median_smallest = np.median(np.sort(flatten)[:100])
        color_range_bins = np.linspace(median_smallest, median_biggest, 100)
        if around_zero:
            color_range_bins = np.linspace(-np.median(np.sort(np.abs(flatten))[-100:]), np.median(np.sort(np.abs(flatten))[-100:]), 100)
        return color_range, color_range_bins

    def present_results_on_frame(self, frame, frame_num):#,blobs_labels):
        # self.left_upper_corner = np.array([0,0])
        # put circle on the object_center
        center = tuple(self.object_center[frame_num, :].astype(int)+self.left_upper_corner.astype(int))
        cv2.circle(frame, center, 5, color=[255, 0, 0], thickness=-1)
        # put circle on the blue_tip_coordinates
        center = tuple(self.blue_tip_coordinates[frame_num, :].astype(int)+self.left_upper_corner.astype(int))
        cv2.circle(frame, center, 5, color=[255, 0, 0], thickness=-1)

        self.left_upper_corner = np.array([0,0])
        for i in range(self.num_of_springs):
            spring_color = [int(x) for x in self.force_color_range[np.argmin(np.abs(self.total_force[frame_num, i] - self.force_color_range_bins))]]
            # pulling_angle_color = [int(x) for x in self.pulling_angle_color_range[np.argmin(np.abs(self.pulling_angle[frame_num, i] - self.pulling_angle_color_range_bins))]]
            # spring_color = [0, 0, 255]
            start_point = self.free_ends_coordinates[frame_num, i, :]
            # end_point = self.fixed_end_fixed_coordinates[frame_num, i, :]
            end_point = self.fixed_ends_coordinates[frame_num, i, :]
            if np.isnan(start_point).any() or np.isnan(end_point).any():
                continue
            else:
                # print((start_point+self.left_upper_corner).astype(int))
                start_point = tuple(start_point.astype(int)+self.left_upper_corner.astype(int))
                end_point = tuple(end_point.astype(int)+self.left_upper_corner.astype(int))
                # draw line between the two points and make contour around the line
                # cv2.line(frame, start_point, end_point, color=[255,255,255], thickness=10)
                cv2.line(frame, start_point, end_point, color=spring_color, thickness=10)
                # put circle on the fixed end
                cv2.circle(frame, end_point, 5, color=spring_color, thickness=-1)
                # put circle on the free end
                cv2.circle(frame, start_point, 5, color=spring_color, thickness=-1)
                cv2.putText(frame, str(i), end_point,  cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,0], 2, cv2.LINE_AA)
                cv2.putText(frame, str(i), start_point,  cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,0], 2, cv2.LINE_AA)
                #write the pulling_angle for each spring on the frame
                # cv2.putText(frame, str(np.round(self.pulling_angle[frame_num, i],5)), start_point, cv2.FONT_HERSHEY_SIMPLEX, 1, pulling_angle_color, 2, cv2.LINE_AA)
                # draw a line from start_point, to and end_point which is calculated by the pulling_angle and the length of the spring
                # length_spring = np.linalg.norm(self.free_ends_coordinates[frame_num, i, :]-self.fixed_ends_coordinates[frame_num, i, :])
                # pulling_fixed_point = self.fixed_end_fixed_coordinates[frame_num,i, :].astype(int)+self.left_upper_corner.astype(int)
                # pulling_free_point = pulling_fixed_point + np.array([np.sin(self.pulling_angle[frame_num, i]+self.fixed_end_angle_to_nest[frame_num,i]), np.cos(self.pulling_angle[frame_num, i]+self.fixed_end_fixed_coordinates_angle_to_nest[frame_num,i])]) * length_spring
                # pulling_free_point_fixed = pulling_fixed_point + np.array([np.sin(self.fixed_end_angle_to_nest[frame_num,i]), np.cos(self.fixed_end_angle_to_nest[frame_num,i])]) * length_spring
                # print(pulling_start_point, end_point)
                # cv2.line(frame, pulling_fixed_point, tuple(pulling_free_point.astype(int)), color=[0, 0, 0], thickness=2)
                #draw triangle on the pulling_free_point
                # cv2.line(frame, pulling_fixed_point, tuple(pulling_free_point.astype(int)), color=[0, 0, 0], thickness=2)
                # cv2.line(frame, pulling_fixed_point, tuple(pulling_free_point_fixed.astype(int)), color=[0, 0, 0], thickness=2)
                # cv2.line(frame, tuple(pulling_free_point.astype(int)), tuple(pulling_free_point_fixed.astype(int)), color=[0, 0, 0], thickness=2)
                # # fill the triangle with the pulling_angle_color
                # triangle_cnt = np.array( [pulling_fixed_point, tuple(pulling_free_point.astype(int)), tuple(pulling_free_point_fixed.astype(int))])
                # cv2.drawContours(frame, [triangle_cnt], 0, pulling_angle_color, -1)

                # cv2.putText(frame, str(np.round(self.total_force[frame_num, i],7)), start_point, cv2.FONT_HERSHEY_SIMPLEX, 1, spring_color, 2, cv2.LINE_AA)

        #draw arrow between the two points
        #write a clock wise arrow
        # cv2.arrowedLine(frame, (1500,500), (1500,600), color=[0, 0, 255], thickness=5)
        # cv2.arrowedLine(frame, (1440,600), (1440,500), color=[255, 0, 0], thickness=5)
            #write the total force on the frame
        net_force_color = [int(x) for x in self.net_force_color_range[np.argmin(np.abs(self.net_force[frame_num] - self.net_force_color_range_bins))]]
        cv2.putText(frame, "Net_force (mN): "+str(np.round(self.net_force[frame_num]*1000,3)), (1440, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, net_force_color, 2, cv2.LINE_AA)
            # # write the velocity on the frame
        velcity_color = [int(x) for x in self.velocity_color_range[np.argmin(np.abs(self.angular_velocity[frame_num] - self.velocity_color_range_bins))]]
        cv2.putText(frame, "Velocity: " + str(np.round(self.angular_velocity[frame_num], 5)), (1440, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, velcity_color, 2, cv2.LINE_AA)

        # cv2.putText(frame, "0:", (1530, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 0], 1, cv2.LINE_AA)
        # total_force_above = np.nansum(self.total_force[frame_num, :][self.total_force[frame_num, :] > 0])
        # total_force_below = np.nansum(self.total_force[frame_num, :][self.total_force[frame_num, :] < 0])
        # cv2.rectangle(frame, (1530, 550), (1580, 550 - int(total_force_above/np.nanmax(self.total_force) * 100)), color=[255, 0, 0], thickness=-1)
        # cv2.rectangle(frame, (1530, 550 - int(total_force_below/np.nanmax(self.total_force) * 100)), (1580, 550), color=[0, 0, 255], thickness=-1)
        #
        # cv2.putText(frame, "2e-5:", (1600, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 0], 1, cv2.LINE_AA)
        # total_force_above = np.nansum(self.total_force[frame_num, :][self.total_force[frame_num, :] > 2e-5])
        # total_force_below = np.nansum(self.total_force[frame_num, :][self.total_force[frame_num, :] < -2e-5])
        # cv2.rectangle(frame, (1610, 550), (1660, 550 - int(total_force_above/np.nanmax(self.total_force) * 100)), color=[255, 0, 0], thickness=-1)
        # cv2.rectangle(frame, (1610, 550 - int(total_force_below/np.nanmax(self.total_force) * 100)), (1660, 550), color=[0, 0, 255], thickness=-1)

        # cv2.putText(frame, "pulling:", (1680, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 0], 1, cv2.LINE_AA)
        # pulling_angle_above = np.nansum(self.pulling_angle[frame_num, :][self.pulling_angle[frame_num, :] > 0])
        # # pulling_angle_above = np.nansum(self.pulling_angle[frame_num, :][self.total_force[frame_num, :] > 2e-5])
        # pulling_angle_below = np.nansum(self.pulling_angle[frame_num, :][self.pulling_angle[frame_num, :] < 0])
        # # pulling_angle_below = np.nansum(self.pulling_angle[frame_num, :][self.total_force[frame_num, :] < -2e-5])
        # cv2.rectangle(frame, (1690, 550), (1740, 550 - int(pulling_angle_above/np.nanmax(self.pulling_angle) * 100)), color=[255, 0, 0], thickness=-1)
        # cv2.rectangle(frame, (1690, 550 - int(pulling_angle_below/np.nanmax(self.pulling_angle) * 100)), (1740, 550), color=[0, 0, 255], thickness=-1)

        # blobs_labels = blobs_labels[~np.isnan(blobs_labels).any(axis=1)]
        # print("blobs_labels.shape", blobs_labels.shape)
        # for ant in range(blobs_labels.shape[0]):
        #     print("ant: ",blobs_labels[ant])
        #     point = (int(blobs_labels[ant][0]), int(blobs_labels[ant][1]))
        #     cv2.circle(frame, point, 5, (0, 0, 255), -1)
        #     label = str(blobs_labels[ant][2])
        #     cv2.putText(frame, label, point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def create_video(self, video_path,n_frames_to_save):
        cap = cv2.VideoCapture(video_path)
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_num in range(self.frames_num):
            if frame_num ==n_frames_to_save:
                break
            ret, frame = cap.read()
            #resize fra
            self.present_results_on_frame(frame, frame_num)
            self.save_video(frame, frame_num)
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
        self.video_writer.release()

        # start_frame = 21000
        # cap = cv2.VideoCapture(video_path)
        #
        # self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        # self.frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # for frame_num in range(self.frames_num):
        #     if frame_num ==20000:
        #         break
        #         # print("frame_num: ", frame_num)
        #     ret, frame = cap.read()
        #     # blobs_labels = self.blobs_mat[:, :, frame_num]
        #     self.present_results_on_frame(frame, frame_num)#, blobs_labels)
        #     self.save_video(frame, frame_num)
        #     # cv2.imshow('frame', frame)
        #     # cv2.waitKey(1)
        # self.video_writer.release()

    def save_video(self,frame,frame_num):
        if frame_num == 0:
            self.video_writer = cv2.VideoWriter(self.save_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps,
                                                (frame.shape[1], frame.shape[0]))
        self.video_writer.write(frame)

    def collect_cropping_coordinates(self,video_path):
        import pickle
        video_preferences_path = os.path.normpath("\\".join(video_path.split("\\")[:-2])+"\\video_preferences.pickle")
        video_preferences = pickle.load(open(video_preferences_path, "rb"))
        normed_paths_preferences = {os.path.normpath(path): preferences for path, preferences in video_preferences.items()}
        crop_coordinates = normed_paths_preferences[video_path]["crop_coordinates"]
        self.left_upper_corner = np.array(( crop_coordinates[0], crop_coordinates[2]))
        self.left_upper_corner = self.left_upper_corner[[1, 0]]


if __name__ == "__main__":
    # for i in [1,3,4,5,6,7,8,9]:
    for i in [1]:
        vidname = f"S528000{i}"
        calibration_dir = "Z:\\Dor_Gabay\\ThesisProject\\data\\calibration\\post_slicing\\calibration_perfect2\\sliced_videos\\"
        data_path = f"Z:\\Dor_Gabay\\ThesisProject\\data\\analysed_with_tracking\\15.9.22\\plus0.3mm_force\\{vidname}\\"
        video_path = f"Z:\\Dor_Gabay\\ThesisProject\\data\\videos\\15.9.22\\plus0.3mm_force\\{vidname}.MP4"

        calibration_model = pickle.load(open(calibration_dir + "calibration_model.pkl", "rb"))
        # ResultVideo(data_path, video_path, calibration_model)
        ResultVideo(data_path, video_path, calibration_model,n_frames_to_save=1000)
        # data_path = "Z:\\Dor_Gabay\\ThesisProject\\data\\analysed_with_tracking\\15.9.22\\plus0.3mm_force\\S5280008\\two_vars_post_processing"
        # TrackingResultVideo(data_path, video_path, calibration_model,n_frames_to_save=5000)


