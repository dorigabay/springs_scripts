import cv2
import numpy as np
import os
from data_analysis.data_preparation import PostProcessing

class ResultVideo(PostProcessing):
    def __init__(self, data_path, video_path):
        super().__init__(data_path)
        self.save_path = os.path.join(data_path, "results_video.MP4")
        self.collect_cropping_coordinates(video_path)
        # self.load_data(data_path)
        self.create_color_space()
        self.create_video(video_path)

    # def load_data(self,directory):
    #     print("loading data from:", directory)
    #     directory_coordinates = os.path.join(directory, "coordinates")+"\\"
    #     directory_normalized_data = os.path.join(directory, "normalized_data")+"\\"
    #     directory_post_calculation = os.path.join(directory, "post_calculation")+"\\"
    #     # self.N_ants_around_springs = np.loadtxt(f"{directory}N_ants_around_springs.csv", delimiter=",")
    #     # self.size_ants_around_springs = np.loadtxt(f"{directory}size_ants_around_springs.csv", delimiter=",")
    #     fixed_ends_coordinates_x = np.loadtxt(f"{directory_coordinates}fixed_ends_coordinates_x.csv", delimiter=",")
    #     fixed_ends_coordinates_y = np.loadtxt(f"{directory_coordinates}fixed_ends_coordinates_y.csv", delimiter=",")
    #     self.fixed_ends_coordinates = np.stack((fixed_ends_coordinates_x, fixed_ends_coordinates_y), axis=2)
    #     free_ends_coordinates_x = np.loadtxt(f"{directory_coordinates}free_ends_coordinates_x.csv", delimiter=",")
    #     free_ends_coordinates_y = np.loadtxt(f"{directory_coordinates}free_ends_coordinates_y.csv", delimiter=",")
    #     self.free_ends_coordinates = np.stack((free_ends_coordinates_x, free_ends_coordinates_y), axis=2)
    #     blue_part_coordinates_x = np.loadtxt(f"{directory_coordinates}blue_part_coordinates_x.csv", delimiter=",")
    #     blue_part_coordinates_y = np.loadtxt(f"{directory_coordinates}blue_part_coordinates_y.csv", delimiter=",")
    #     self.blue_part_coordinates = np.stack((blue_part_coordinates_x, blue_part_coordinates_y), axis=2)
    #     self.object_center = self.blue_part_coordinates[:, 0, :]
    #     self.blue_tip_coordinates = self.blue_part_coordinates[:, -1, :]
    #     # self.angle_to_blue_part = np.loadtxt(f"{directory_normalized_data}angle_to_blue_part.csv", delimiter=",")
    #     self.pulling_angle = np.loadtxt(f"{directory_normalized_data}pulling_angle.csv", delimiter=",")
    #     fixed_end_fixed_coordinates_x = np.loadtxt(f"{directory_post_calculation}fixed_end_fixed_coordinates_x.csv", delimiter=",")
    #     fixed_end_fixed_coordinates_y = np.loadtxt(f"{directory_post_calculation}fixed_end_fixed_coordinates_y.csv", delimiter=",")
    #     self.fixed_end_fixed_coordinates = np.stack((fixed_end_fixed_coordinates_x, fixed_end_fixed_coordinates_y), axis=2)


    def create_color_space(self):
        # take the median of 100 biggest angles, and the median of 100 smallest angles
        # create a color space between them, and color the lines according to the angle
        # the color space will be from red to green
        #flatten the array of angle to blue part, and remove the nans
        angles_flatten = self.pulling_angle.flatten()
        #remove nans
        angles_flatten = angles_flatten[~np.isnan(angles_flatten)]
        median_biggest_angles = np.median(np.sort(angles_flatten)[-100:])
        median_smallest_angles = np.median(np.sort(angles_flatten)[:100])
        # # create color array with 100 colors, from red to blue
        # Define the start and end colors
        color_range = np.zeros((100, 3))
        start_color = np.array([0, 0, 255])
        end_color = np.array([255, 0, 0])
        for i in range(100):
            color_range[i] = start_color * (1 - i / 99) + end_color * (i / 99)
        # Convert the color array to integer values
        self.color_range = color_range.astype(int)
        self.color_range_bins = np.linspace(median_smallest_angles, median_biggest_angles, 100)

    def present_results_on_frame(self, frame,frame_num):
        # put circle on the object_center
        center = tuple(self.object_center[frame_num, :].astype(int)+self.left_upper_corner.astype(int))
        cv2.circle(frame, center, 5, color=[255, 0, 0], thickness=-1)
        # put circle on the blue_tip_coordinates
        center = tuple(self.blue_tip_coordinates[frame_num, :].astype(int)+self.left_upper_corner.astype(int))
        cv2.circle(frame, center, 5, color=[255, 0, 0], thickness=-1)
        #create line between free ends and center of blue part, with using skimage.measure.line
        for i in range(self.free_end_fixed_coordinates.shape[1]):
            color = [int(x) for x in self.color_range[np.argmin(np.abs(self.pulling_angle[frame_num, i] - self.color_range_bins))]]
            # color = [0, 0, 255]
            start_point = self.free_ends_coordinates[frame_num, i, :]
            # end_point = self.fixed_ends_coordinates[frame_num, i, :]
            end_point = self.fixed_ends_coordinates[frame_num, i, :]
            if np.isnan(start_point).any() or np.isnan(end_point).any():
                continue
            else:
                # print((start_point+self.left_upper_corner).astype(int))
                start_point = tuple(start_point.astype(int)+self.left_upper_corner.astype(int))
                end_point = tuple(end_point.astype(int)+self.left_upper_corner.astype(int))
                # draw line between the two points and make contour around the line
                # cv2.line(frame, start_point, end_point, color=[255,255,255], thickness=10)
                cv2.line(frame, start_point, end_point, color=color, thickness=5)
                # put circle on the fixed end
                cv2.circle(frame, end_point, 5, color=color, thickness=-1)
                # put circle on the free end
                cv2.circle(frame, start_point, 5, color=color, thickness=-1)
                #draw arrow between the two points
                cv2.arrowedLine(frame, (1500,500), (1500,600), color=[0, 0, 255], thickness=5)
                cv2.arrowedLine(frame, (1440,600), (1440,500), color=[255, 0, 0], thickness=5)

    def create_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_num in range(self.frames_num):
            ret, frame = cap.read()
            self.present_results_on_frame(frame, frame_num)
            self.save_video(frame, frame_num)
            # cv2.imshow('frame', frame)
            # cv2.waitKey(1)

    def save_video(self,frame,frame_num):
        # in the first frame, create the video writer
        # in the rest of the frames, write the frame to the video
        if frame_num == 0:
            self.video_writer = cv2.VideoWriter(self.save_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps,
                                                (frame.shape[1], frame.shape[0]))
        self.video_writer.write(frame)

    def collect_cropping_coordinates(self,video_path):
        import pickle
        video_preferences_path = os.path.normpath("\\".join(video_path.split("\\")[:6])+"\\video_preferences.pickle")
        video_preferences = pickle.load(open(video_preferences_path, "rb"))
        crop_coordinates = video_preferences[video_path]["crop_coordinates"]
        self.left_upper_corner = np.array(( crop_coordinates[0], crop_coordinates[2]))
        self.left_upper_corner = self.left_upper_corner[[1, 0]]

if __name__ == "__main__":
    data_path = "Z:\\Dor_Gabay\\ThesisProject\\data\\test3\\15.9.22\\plus0.3mm_force\\S5280006\\"
    video_path = "Z:\\Dor_Gabay\\ThesisProject\\data\\videos\\15.9.22\\plus0.3mm_force\\S5280006.MP4"
    from data_analysis  import utils
    spring_types_directories = utils.find_dirs("Z:/Dor_Gabay/ThesisProject/data/test3/")
    ResultVideo("Z:\\Dor_Gabay\\ThesisProject\\data\\test3\\15.9.22\\plus0.3mm_force\\S5280006\\", video_path)

