import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label,sum_labels
from scipy.signal import savgol_filter
import os
from data_analysis import utils

def iter_dirs(spring_type_directories,function):
    springs_analysed = {}
    for spring_type in spring_type_directories:
        springs_analysed[spring_type] = {}
        for video_data in spring_type_directories[spring_type]:
            video_name = video_data.split("\\")[-2]
            object = function(video_data)
            object.save_data(video_data)
            springs_analysed[spring_type][video_name] = object
    return springs_analysed


class PostProcessing:
    def __init__(self, directory):
        self.load_data(directory)
        self.N_ants_proccessing()
        self.calc_distances()
        self.calc_angle()
        self.calc_spring_length()

        # self.save_data(directory)

    def load_data(self,directory):
        print("loading data from:", directory)
        directory_coordinates = os.path.join(directory, "coordinates")+"\\"
        self.N_ants_around_springs = np.loadtxt(f"{directory}N_ants_around_springs.csv", delimiter=",")
        self.size_ants_around_springs = np.loadtxt(f"{directory}size_ants_around_springs.csv", delimiter=",")
        fixed_ends_coordinates_x = np.loadtxt(f"{directory_coordinates}fixed_ends_coordinates_x.csv", delimiter=",")
        fixed_ends_coordinates_y = np.loadtxt(f"{directory_coordinates}fixed_ends_coordinates_y.csv", delimiter=",")
        self.fixed_ends_coordinates = np.stack((fixed_ends_coordinates_x, fixed_ends_coordinates_y), axis=2)
        free_ends_coordinates_x = np.loadtxt(f"{directory_coordinates}free_ends_coordinates_x.csv", delimiter=",")
        free_ends_coordinates_y = np.loadtxt(f"{directory_coordinates}free_ends_coordinates_y.csv", delimiter=",")
        self.free_ends_coordinates = np.stack((free_ends_coordinates_x, free_ends_coordinates_y), axis=2)
        blue_part_coordinates_x = np.loadtxt(f"{directory_coordinates}blue_part_coordinates_x.csv", delimiter=",")
        blue_part_coordinates_y = np.loadtxt(f"{directory_coordinates}blue_part_coordinates_y.csv", delimiter=",")
        blue_part_coordinates = np.stack((blue_part_coordinates_x, blue_part_coordinates_y), axis=2)
        self.object_center = blue_part_coordinates[:, 0, :]
        self.blue_tip_coordinates = blue_part_coordinates[:, -1, :]
        self.num_of_frames = self.N_ants_around_springs.shape[0]
        self.num_of_springs = self.N_ants_around_springs.shape[1]

    def N_ants_proccessing(self):
        def smoothing_n_ants(array):
            for col in range(array.shape[1]):
                array[:,col] = np.abs(np.round(savgol_filter(array[:,col], 31, 2)))
            return array
        undetected_springs_for_long_time = utils.filter_continuity(utils.convert_bool_to_binary(
            np.isnan(self.fixed_ends_coordinates[:, :, 0])),min_size=8)
        self.N_ants_around_springs = np.round(utils.interpolate_data(self.N_ants_around_springs,
                                                    utils.find_cells_to_interpolate(self.N_ants_around_springs)))
        self.N_ants_around_springs[np.isnan(self.N_ants_around_springs)] = 0
        self.N_ants_around_springs = smoothing_n_ants(self.N_ants_around_springs)
        self.N_ants_around_springs[undetected_springs_for_long_time] = np.nan
        all_small_attaches = np.zeros(self.N_ants_around_springs.shape,int)
        for n in np.unique(self.N_ants_around_springs)[1:]:
            if not np.isnan(n):
                short_attaches = utils.filter_continuity(self.N_ants_around_springs==n,max_size=30)
                all_small_attaches[short_attaches] = 1
        self.N_ants_around_springs = np.round(utils.interpolate_data(self.N_ants_around_springs,
                                                                     all_small_attaches.astype(bool)))
        self.rest_bool = self.N_ants_around_springs == 0

    def bound_angle(self,angle):
        angle[angle > 2 * np.pi] -= 2 * np.pi
        return angle

    def calc_distances(self):
        #repeat object center to have the same shape as free ends coordinates
        object_center = np.repeat(self.object_center[:, np.newaxis, :], self.num_of_springs, axis=1)
        self.object_center_to_free_end_distance = np.linalg.norm(self.free_ends_coordinates - object_center, axis=2)
        self.object_center_to_fixed_end_distance = np.linalg.norm(self.fixed_ends_coordinates - object_center, axis=2)
        self.blue_length = np.linalg.norm(self.blue_tip_coordinates - self.object_center, axis=1)

    def norm_values(self, X, Y, bias_bool=None,column_on_boundry=None):
        # takes array of x and y values and returns the normalized values
        # X and Y should have the same shape
        # X is the independent variable, that creates the bias over Y
        # iterates over the second axis
        if bias_bool is not None:
            X_bias = copy.copy(X)
            Y_bias = copy.copy(Y)
            X_bias[np.invert(bias_bool)] = np.nan
            Y_bias[np.invert(bias_bool)] = np.nan
        else:
            X_bias = X
            Y_bias = Y
        # subtract the median of the y values from the y values
        if column_on_boundry is not None:
            # add pi to Y_bias[:,0] if it is bigger than pi
            Y_bias[Y_bias[:, column_on_boundry] > np.pi, column_on_boundry] -= 2 * np.pi
        Y_bias -= np.nanmedian(Y_bias, axis=0)
        normed_Y = np.zeros(Y.shape)
        import pandas as pd
        for i in range(X.shape[1]):
            df = pd.DataFrame({"x": X_bias[:, i], "y": Y_bias[:, i]}).dropna()
            bias_equation = utils.deduce_bias_equation(df["x"], df["y"])
            normed_Y[:, i] = utils.normalize(Y[:, i], X[:, i], bias_equation)
        return normed_Y

    def find_fixed_coordinates(self,angle_to_blue,distance_to_object_center,remove_non_rest=False):
        object_center = copy.copy(np.repeat(self.object_center[:, np.newaxis, :], self.num_of_springs, axis=1))
        if remove_non_rest:
            # for free_ends norm the angle to the blue part, that is biased by the angle to the nest
            angle_to_blue = self.norm_values(self.free_end_angle_to_nest, angle_to_blue, self.rest_bool, 0)
            distance_to_object_center = copy.copy(distance_to_object_center)
            distance_to_object_center[np.invert(self.rest_bool)] = np.nan
        else:
            #median angle between the blue part and the fixed end
            angle_to_blue = copy.copy(angle_to_blue)
            angle_to_blue = np.nanmedian(angle_to_blue, axis=0)
            angle_to_blue = np.repeat(angle_to_blue[np.newaxis, :], self.num_of_frames, axis=0)
        # median distance between the blue part and the end
        median_distance = np.nanmedian(distance_to_object_center, axis=0)
        median_distance = np.repeat(median_distance[np.newaxis, :], self.num_of_frames, axis=0)
        # subtract the angle of the blue part end to the nest from the angle of the fixed end to the blue part
        angle_to_blue_part_normed = self.bound_angle(self.blue_part_angle_to_nest+angle_to_blue-np.pi/2)
        # find the fixed end coordinates with fixed_end_angle_to_blue_part_normed and median_distance_to_fixed_end
        fixed_coordinates = object_center + np.stack((np.cos(angle_to_blue_part_normed) * median_distance,
                                                                    np.sin(angle_to_blue_part_normed) * median_distance), axis=2)
        return fixed_coordinates

    def calc_angle(self):
        #repeat object center to have the same shape as free ends coordinates
        nest_direction = np.stack((self.object_center[:, 0], self.object_center[:, 1]-100), axis=1)
        nest_direction = np.repeat(nest_direction[:, np.newaxis, :], self.free_ends_coordinates.shape[1], axis=1)
        object_center = copy.copy(np.repeat(self.object_center[:, np.newaxis, :], self.free_ends_coordinates.shape[1], axis=1))
        blue_tip_coordinates = np.repeat(self.blue_tip_coordinates[:, np.newaxis, :], self.free_ends_coordinates.shape[1], axis=1)
        #angles to the nest
        self.free_end_angle_to_nest = utils.calc_angle_matrix(self.free_ends_coordinates, object_center, nest_direction)+np.pi
        self.fixed_end_angle_to_nest = utils.calc_angle_matrix(self.fixed_ends_coordinates, object_center, nest_direction)+np.pi
        self.blue_part_angle_to_nest = utils.calc_angle_matrix(nest_direction, object_center, blue_tip_coordinates)+np.pi
        # angles to the blue part
        self.free_end_angle_to_blue_part = utils.calc_angle_matrix(blue_tip_coordinates, object_center, self.free_ends_coordinates)+np.pi
        self.fixed_end_angle_to_blue_part = utils.calc_angle_matrix(blue_tip_coordinates,object_center,self.fixed_ends_coordinates)+np.pi
        # find fixed coordinates
        self.fixed_end_fixed_coordinates = self.find_fixed_coordinates(self.fixed_end_angle_to_blue_part,self.object_center_to_fixed_end_distance)
        self.free_end_fixed_coordinates = self.find_fixed_coordinates(self.free_end_angle_to_blue_part,self.object_center_to_free_end_distance,remove_non_rest=True)
        # pulling angle between fixed coordinates of free end and fixed end
        self.pulling_angle = utils.calc_angle_matrix(self.free_end_fixed_coordinates,self.fixed_end_fixed_coordinates,object_center)+2*np.pi
        self.pulling_angle = self.bound_angle(self.pulling_angle)
        median_pulling_angle = copy.copy(self.pulling_angle)
        median_pulling_angle[np.invert(self.rest_bool)] = np.nan
        median_pulling_angle = np.nanmedian(median_pulling_angle,axis=0)
        median_pulling_angle = np.repeat(median_pulling_angle[np.newaxis, :], self.num_of_frames, axis=0)
        self.pulling_angle-=median_pulling_angle
        # splitting the coordinates to x and y
        # self.free_end_fixed_coordinates_x,self.free_end_fixed_coordinates_y = self.free_end_fixed_coordinates[:,: ,0],self.free_end_fixed_coordinates[:,: ,1]
        # self.fixed_end_fixed_coordinates_x,self.fixed_end_fixed_coordinates_y = self.fixed_end_fixed_coordinates[:,: ,0],self.fixed_end_fixed_coordinates[:,: ,1]

    def calc_spring_length(self):
        self.spring_length = np.linalg.norm(self.free_ends_coordinates - self.fixed_end_fixed_coordinates , axis=2)
        self.spring_length = utils.interpolate_data(self.spring_length,
                                                       utils.find_cells_to_interpolate(self.spring_length))

    # def collect_parameters(self,video_path):
    #     import pickle
    #     video_preferences_path = os.path.normpath("\\".join(video_path.split("\\")[:6])+"\\video_preferences.pickle")
    #     video_preferences = pickle.load(open(video_preferences_path, "rb"))
    #     self.starting_frame = video_preferences["starting_frame"]


    def save_data(self, directory):
        print("saving data to:", directory)
        directory = os.path.join(directory, "post_processed_data/")
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.savetxt(f"{directory}N_ants_around_springs.csv", self.N_ants_around_springs, delimiter=",")
        np.savetxt(f"{directory}spring_length.csv", self.spring_length, delimiter=",")
        np.savetxt(f"{directory}fixed_end_angle_to_nest.csv", self.fixed_end_angle_to_nest, delimiter=",")
        np.savetxt(f"{directory}pulling_angle.csv", self.pulling_angle, delimiter=",")
        # np.savetxt(f"{directory}size_ants_around_springs.csv", self.size_ants_around_springs, delimiter=",")
        # np.savetxt(f"{directory}center_to_free_end_distance.csv", self.center_to_free_end_distance, delimiter=",")
        # np.savetxt(f"{directory}free_end_angle_to_nest.csv", self.free_end_angle_to_nest, delimiter=",")
        # np.savetxt(f"{directory}free_end_angle_to_blue_part.csv", self.free_end_angle_to_blue_part, delimiter=",")
        # np.savetxt(f"{directory}fixed_end_angle_to_nest.csv", self.fixed_end_angle_to_nest, delimiter=",")
        # np.savetxt(f"{directory}fixed_end_angle_to_blue_part.csv", self.fixed_end_angle_to_blue_part, delimiter=",")
        # np.savetxt(f"{directory}blue_part_angle_to_nest.csv", self.blue_part_angle_to_nest, delimiter=",")
        # np.savetxt(f"{directory}blue_length.csv", self.blue_length, delimiter=",")
        # np.savetxt(f"{directory}fixed_end_fixed_coordinates_x.csv", self.fixed_end_fixed_coordinates_x, delimiter=",")
        # np.savetxt(f"{directory}fixed_end_fixed_coordinates_y.csv", self.fixed_end_fixed_coordinates_y, delimiter=",")

# class DataPreparation:
#     def __init__(self,directory):
#         self.directory = os.path.join(directory, "post_calculation")+"\\"
#         self.load_data(self.directory)
#         self.load_starting_frame()
#         self.N_ants_proccessing()
#         self.springs_length_processing()
#         self.springs_angles_processing()
#         # self.springs_rest_lengths()
#         # self.attaches_events(self.N_ants_around_springs)
#         # self.labeled_zero_to_one_filtered = self.fiter_attaches_events(self.labeled_zero_to_one)
#         # self.data_names = ["N_ants_around_springs","size_ants_around_springs","springs_length","springs_angles_to_nest","springs_angles_to_object"]
#
#     # def load_data(self,directory):
#     #     print("loading data from:",directory)
#     #     self.N_ants_around_springs = np.loadtxt(f"{directory}N_ants_around_springs.csv",delimiter=",")
#     #     self.size_ants_around_springs = np.loadtxt(f"{directory}size_ants_around_springs.csv",delimiter=",")
#     #     self.spring_length = np.loadtxt(f"{directory}spring_length.csv",delimiter=",")
#     #     self.angle_to_nest = np.loadtxt(f"{directory}angle_to_nest.csv",delimiter=",")
#     #     self.free_end_angle_to_blue_part = np.loadtxt(f"{directory}free_end_angle_to_blue_part.csv",delimiter=",")
#     #     self.fixed_end_angle_to_blue_part = np.loadtxt(f"{directory}fixed_end_angle_to_blue_part.csv",delimiter=",")
#     #     self.blue_length = np.loadtxt(f"{directory}blue_length.csv",delimiter=",")
#     #     # self.angles_to_object_free = np.loadtxt(f"{directory}angles_to_object_free.csv",delimiter=",")
#     #     # self.angles_to_object_fixed = np.loadtxt(f"{directory}angles_to_object_fixed.csv",delimiter=",")
#     #     # self.angle_to_blue_part = np.loadtxt(f"{directory}angle_to_blue_part.csv",delimiter=",")
#     #     self.pulling_angle = np.loadtxt(f"{directory}pulling_angle.csv",delimiter=",")
    # def springs_angles_processing(self):
    #     cells_to_interp = utils.find_cells_to_interpolate(self.free_end_angle_to_blue_part)
    #     # self.angle_to_blue_part = utils.interpolate_data(self.angle_to_blue_part + np.pi, cells_to_interp)
    #     #free end angle
    #     self.free_end_angle_to_blue_part = utils.interpolate_data(self.free_end_angle_to_blue_part + np.pi, cells_to_interp)
    #     free_end_rest_angles = copy.copy(self.free_end_angle_to_blue_part)
    #     free_end_rest_angles[self.N_ants_around_springs == 0] = np.nan
    #     free_end_median_rest_angle = np.nanmedian(free_end_rest_angles,axis=0)
    #     self.free_end_angle_to_blue_part = ((self.free_end_angle_to_blue_part - free_end_median_rest_angle)+np.pi)%(2*np.pi)-np.pi
#
#     def load_starting_frame(self):
#         # from video_analysis.collect_color_parameters import get_parameters
#         # path_parts =  os.path.normpath(self.directory).split('\\')
#         # video_dir =os.path.normpath(os.path.join("Z:/Dor_Gabay/ThesisProject/data/videos",path_parts[-3]))
#         # video_path = os.path.normpath(os.path.join("Z:/Dor_Gabay/ThesisProject/data/videos",path_parts[-3],path_parts[-2],path_parts[-1]+".MP4"))
#         #
#         # parameters = get_parameters(video_dir, video_path)
#         # self.starting_frame = parameters["starting_frame"]
#         # with open(os.path.join("Z:/Dor_Gabay/ThesisProject/data/videos",path_parts[-3],"video_preferences.pickle"), 'rb') as handle:
#         #     path = f"Z:\\Dor_Gabay\\videos\\{path_parts[-3]}\\{path_parts[-2]}\\{path_parts[-1]}.MP4"
#         #     self.starting_frame = pickle.load(handle)[path]['starting_frame']
#         # return int(input("starting frame:"))
#         return 0
#
#
#
#
#         # fixed end angle
#
#         # fixed_end_first_frame_angle = copy.copy(self.fixed_end_angle_to_blue_part[0])
#         # fixed_end_median_angle = np.nanmedian(self.fixed_end_angle_to_blue_part , axis=0)
#
#         # self.pulling_angle = (self.free_end_angle_to_blue_part - fixed_end_median_rest_angle)%(2*np.pi)
#         # self.fixed_end_angle_to_blue_part = utils.interpolate_data(self.fixed_end_angle_to_blue_part + np.pi, cells_to_interp)
#         # self.fixed_end_angle_to_blue_part = ((self.fixed_end_angle_to_blue_part - fixed_end_median_rest_angle)+np.pi)%(2*np.pi)-np.pi
#
#     # def springs_angles_processing(self):
#     #     cells_to_interp = utils.find_cells_to_interpolate(self.angle_to_nest)
#     #     self.angles_to_nest = utils.interpolate_data(self.angle_to_nest+np.pi,cells_to_interp)
#     #     self.angles_to_object_free = utils.interpolate_data(self.angles_to_object_free+np.pi,cells_to_interp)
#     #     self.angles_to_object_fixed = utils.interpolate_data(self.angles_to_object_fixed+np.pi,cells_to_interp)
#     #     self.pulling_angle = self.angles_to_object_free-self.angles_to_object_fixed
#     #     self.pulling_angle = (self.pulling_angle+np.pi)%(2*np.pi)-np.pi
#     #     # pulling_angle_rest_median = np.median(self.pulling_angle[self.N_ants_around_springs == 0],axis=0)
#     #     # self.pulling_angle = self.pulling_angle - pulling_angle_rest_median
#
#
#
#
#     def attaches_events(self,N_ants_array):
#         N_ants_array = copy.copy(N_ants_array)
#         N_ants_array = utils.column_dilation(N_ants_array)
#         diff = np.vstack((np.diff(N_ants_array,axis=0),np.zeros(N_ants_array.shape[1]).reshape(1,N_ants_array.shape[1])))
#         labeled_single_attaches,num_labels_single = label(utils.convert_bool_to_binary(N_ants_array==1))
#         labeled_single_attaches_adapted = np.vstack((labeled_single_attaches,np.zeros(labeled_single_attaches.shape[1]).reshape(1,labeled_single_attaches.shape[1])))[1:,:]
#         labels_0_to_1 = np.unique(labeled_single_attaches_adapted[(diff==1)&(N_ants_array==0)])
#         labels_1_to_0 = np.unique(labeled_single_attaches_adapted[(diff==-1)&(N_ants_array==1)])
#         self.labeled_zero_to_one,_ = label(np.isin(labeled_single_attaches,labels_0_to_1))
#         self.labeled_zero_to_one = self.labeled_zero_to_one[:,list(range(0,self.labeled_zero_to_one.shape[1],2))]
#         self.labeled_one_to_zero,_ = label(np.isin(labeled_single_attaches,labels_1_to_0))
#         self.labeled_one_to_zero = self.labeled_one_to_zero[:,list(range(0,self.labeled_one_to_zero.shape[1],2))]
#
#     def fiter_attaches_events(self, labeled):
#         EVENT_LENGTH = 150
#         PRE_EVENT_LENGTH = 15
#         ar = self.labeled_zero_to_one
#         events_to_keep = []
#         self.events_sts = []
#         for event in np.unique(ar):
#             frames = np.where(ar[:,:]==event)[0]
#             # print(event,":",len(frames))
#             pre_frames = np.arange(frames[0]-PRE_EVENT_LENGTH,frames[0]-1,1)
#             spring = np.where(ar[:,:]==event)[1][0]
#             # if len(frames)>EVENT_LENGTH:
#             if True:
#                 if np.sum(np.isnan(self.N_ants_around_springs[frames[0]-10:frames[0],spring]))==0:
#                     pre_frames_lengths = np.take(self.springs_length_processed[:,spring],pre_frames)
#                     pre_frames_median = np.nanmedian(pre_frames_lengths)
#                     # if (not pre_frames_median<self.rest_length*0.8) & (not pre_frames_median>self.rest_length*1.2):
#                     if True:
#                         self.events_sts.append(np.nanstd(pre_frames_lengths)/pre_frames_median)
#                         # if not np.std()
#                         events_to_keep.append(event)
#         labeled = copy.copy(labeled)
#         labeled[np.invert(np.isin(labeled,events_to_keep))] = 0
#         labeled = utils.column_dilation(labeled)
#         labeled, _ =  label(labeled!=0)
#         labeled = labeled[:,list(range(0,labeled.shape[1],2))]
#         # print(.shape)
#         return labeled
#
#     def clean_ant_switching(self,N_ants_array):
#         N_ants_array = copy.copy(N_ants_array)
#         N_ants_array = utils.column_dilation(N_ants_array)
#         labeled_all_attaches,num_labels_all = label(utils.convert_bool_to_binary(N_ants_array>=1))
#
#         labels_to_remove = np.unique(labeled_all_attaches[N_ants_array>1&np.isnan(N_ants_array)])
#         N_ants_array[np.isin(labeled_all_attaches,labels_to_remove)] = 0
#         N_ants_array = N_ants_array[:,list(range(0,N_ants_array.shape[1],2))]
#         return N_ants_array
#
#     def save_data(self,directory):
#         pass

if __name__ == "__main__":
    spring_types_directories = utils.find_dirs("Z:/Dor_Gabay/ThesisProject/data/test3/")
    iter_dirs(spring_types_directories, PostProcessing)


