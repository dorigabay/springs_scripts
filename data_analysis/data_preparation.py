import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label,sum_labels
from scipy.signal import savgol_filter
import os, pickle
import pandas as pd
import seaborn as sns
from data_analysis import utils

def prepare_multiple(spring_type_directories):
    springs_analysed = {}
    for spring_type in spring_type_directories:
        springs_analysed[spring_type] = {}
        for video_data in spring_type_directories[spring_type]:
            video_name = video_data.split("\\")[-2]
            springs_analysed[spring_type][video_name] = PrepareData(video_data)
    return springs_analysed

class PrepareData:
    def __init__(self,directory):
        self.directory = directory
        self.load_data(directory)
        self.original_N_ants = copy.copy(self.N_ants_around_springs)
        self.original_angles = copy.copy(self.springs_angles_to_nest)
        self.resolve_bad_rows()
        self.load_starting_frame()
        self.N_ants_proccessing()
        self.springs_length_processing()
        self.springs_angles_processing()
        # self.angular_velocity_nest = self.calc_angular_velocity(self.springs_angles_to_nest,diff_spacing=1)
        # self.angular_velocity_nest_spaced = self.calc_angular_velocity(self.springs_angles_to_nest,diff_spacing=10)
        self.springs_rest_lengths()
        self.attaches_events(self.N_ants_around_springs)
        self.labeled_zero_to_one_filtered = self.fiter_attaches_events(self.labeled_zero_to_one)
        # if save_events:
        #     self.save_events(self.labeled_zero_to_one_filtered,self.directory)

    def load_data(self,directory):
        print("loading data from:",directory)
        self.N_ants_around_springs = np.loadtxt(f"{directory}N_ants_around_springs.csv",delimiter=",")
        self.size_ants_around_springs = np.loadtxt(f"{directory}size_ants_around_springs.csv",delimiter=",")
        self.springs_length = np.loadtxt(f"{directory}springs_length.csv",delimiter=",")
        self.springs_angles_to_nest = np.loadtxt(f"{directory}springs_angles_to_nest.csv",delimiter=",")
        self.springs_angles_to_object = np.loadtxt(f"{directory}springs_angles_to_object.csv",delimiter=",")

    def load_starting_frame(self):
        from video_analysis.collect_color_parameters import get_parameters
        path_parts =  os.path.normpath(self.directory).split('\\')
        video_dir =os.path.normpath(os.path.join("Z:/Dor_Gabay/ThesisProject/data/videos",path_parts[-3]))
        video_path = os.path.normpath(os.path.join("Z:/Dor_Gabay/ThesisProject/data/videos",path_parts[-3],path_parts[-2],path_parts[-1]+".MP4"))
        parameters = get_parameters(video_dir, video_path)
        self.starting_frame = parameters["starting_frame"]
        # with open(os.path.join("Z:/Dor_Gabay/ThesisProject/data/videos",path_parts[-3],"video_preferences.pickle"), 'rb') as handle:
        #     path = f"Z:\\Dor_Gabay\\videos\\{path_parts[-3]}\\{path_parts[-2]}\\{path_parts[-1]}.MP4"
        #     self.starting_frame = pickle.load(handle)[path]['starting_frame']

    def resolve_bad_rows(self):
        # for rows with more than 2 Nan values, put nan in all the row
        bad_rows = np.sum(np.isnan(self.springs_length),axis=1)>2
        self.springs_length[bad_rows,:] = np.nan
        self.springs_angles_to_nest[bad_rows,:] = np.nan
        self.springs_angles_to_object[bad_rows,:] = np.nan
        self.N_ants_around_springs[bad_rows,:] = np.nan
        self.size_ants_around_springs[bad_rows,:] = np.nan

    def N_ants_proccessing(self):
        undetected_springs_for_long_time = utils.filter_continuity(utils.convert_bool_to_binary(np.isnan(self.springs_length)),min_size=8)
        self.N_ants_around_springs = np.round(utils.interpolate_data(self.N_ants_around_springs,
                                                    utils.find_cells_to_interpolate(self.N_ants_around_springs)))
        self.N_ants_around_springs[np.isnan(self.N_ants_around_springs)] = 0
        self.N_ants_around_springs = self.smoothing_n_ants(self.N_ants_around_springs)
        self.N_ants_around_springs[undetected_springs_for_long_time] = np.nan
        all_small_attaches = np.zeros(self.N_ants_around_springs.shape,int)
        for n in np.unique(self.N_ants_around_springs)[1:]:
            if not np.isnan(n):
                short_attaches = utils.filter_continuity(self.N_ants_around_springs==n,max_size=30)
                all_small_attaches[short_attaches] = 1
        self.N_ants_around_springs = np.round(utils.interpolate_data(self.N_ants_around_springs,
                                                                     all_small_attaches.astype(bool)))

    def springs_length_processing(self):
        self.springs_length_processed = copy.copy(self.springs_length)
        # self.springs_length_processed[np.invert(np.isnan(self.springs_length))&np.isnan(self.springs_length)]=0
        self.springs_length_processed = utils.interpolate_data(self.springs_length_processed,
                                                               utils.find_cells_to_interpolate(self.springs_length))

    def springs_angles_processing(self):
        cells_to_interp = utils.find_cells_to_interpolate(self.springs_angles_to_nest)
        self.springs_angles_to_nest = utils.interpolate_data(self.springs_angles_to_nest+np.pi,cells_to_interp)
        self.springs_angles_to_object = utils.interpolate_data(self.springs_angles_to_nest+np.pi,cells_to_interp)

    # def calc_angular_velocity(self,angles,diff_spacing=1):
    #     thershold = 5.5
    #     diff = utils.difference(angles,spacing=diff_spacing,axis=0)
    #     diff[(diff>thershold)] = diff[diff>thershold]-2*np.pi
    #     diff[(diff<-thershold)] = diff[diff<-thershold]+2*np.pi
    #     return diff

    def smoothing_n_ants(self, array):
        for col in range(array.shape[1]):
            array[:,col] = np.abs(np.round(savgol_filter(array[:,col], 31, 2)))
        return array

    def attaches_events(self,N_ants_array):
        N_ants_array = copy.copy(N_ants_array)
        N_ants_array = utils.column_dilation(N_ants_array)
        diff = np.vstack((np.diff(N_ants_array,axis=0),np.zeros(N_ants_array.shape[1]).reshape(1,N_ants_array.shape[1])))
        labeled_single_attaches,num_labels_single = label(utils.convert_bool_to_binary(N_ants_array==1))
        labeled_single_attaches_adapted = np.vstack((labeled_single_attaches,np.zeros(labeled_single_attaches.shape[1]).reshape(1,labeled_single_attaches.shape[1])))[1:,:]
        labels_0_to_1 = np.unique(labeled_single_attaches_adapted[(diff==1)&(N_ants_array==0)])
        labels_1_to_0 = np.unique(labeled_single_attaches_adapted[(diff==-1)&(N_ants_array==1)])
        self.labeled_zero_to_one,_ = label(np.isin(labeled_single_attaches,labels_0_to_1))
        self.labeled_zero_to_one = self.labeled_zero_to_one[:,list(range(0,self.labeled_zero_to_one.shape[1],2))]
        self.labeled_one_to_zero,_ = label(np.isin(labeled_single_attaches,labels_1_to_0))
        self.labeled_one_to_zero = self.labeled_one_to_zero[:,list(range(0,self.labeled_one_to_zero.shape[1],2))]

    def fiter_attaches_events(self, labeled):
        EVENT_LENGTH = 150
        PRE_EVENT_LENGTH = 15
        ar = self.labeled_zero_to_one
        events_to_keep = []
        self.events_sts = []
        for event in np.unique(ar):
            frames = np.where(ar[:,:]==event)[0]
            # print(event,":",len(frames))
            pre_frames = np.arange(frames[0]-PRE_EVENT_LENGTH,frames[0]-1,1)
            spring = np.where(ar[:,:]==event)[1][0]
            # if len(frames)>EVENT_LENGTH:
            if True:
                if np.sum(np.isnan(self.N_ants_around_springs[frames[0]-10:frames[0],spring]))==0:
                    pre_frames_lengths = np.take(self.springs_length_processed[:,spring],pre_frames)
                    pre_frames_median = np.nanmedian(pre_frames_lengths)
                    # if (not pre_frames_median<self.rest_length*0.8) & (not pre_frames_median>self.rest_length*1.2):
                    if True:
                        self.events_sts.append(np.nanstd(pre_frames_lengths)/pre_frames_median)
                        # if not np.std()
                        events_to_keep.append(event)
        labeled = copy.copy(labeled)
        labeled[np.invert(np.isin(labeled,events_to_keep))] = 0
        labeled = utils.column_dilation(labeled)
        labeled, _ =  label(labeled!=0)
        labeled = labeled[:,list(range(0,labeled.shape[1],2))]
        # print(.shape)
        return labeled

    def clean_ant_switching(self,N_ants_array):
        N_ants_array = copy.copy(N_ants_array)
        N_ants_array = utils.column_dilation(N_ants_array)
        labeled_all_attaches,num_labels_all = label(utils.convert_bool_to_binary(N_ants_array>=1))

        labels_to_remove = np.unique(labeled_all_attaches[N_ants_array>1&np.isnan(N_ants_array)])
        N_ants_array[np.isin(labeled_all_attaches,labels_to_remove)] = 0
        N_ants_array = N_ants_array[:,list(range(0,N_ants_array.shape[1],2))]
        return N_ants_array

    def filter_by_ant_size(self):
        from scipy.ndimage import median
        self.size_ants_around_springs[np.isnan(self.size_ants_around_springs)] = 0
        N_ants_for_mean = copy.copy(self.N_ants_around_springs)
        N_ants_for_mean[self.size_ants_around_springs==0] = 0
        median_size_over_N_ants = median(self.size_ants_around_springs,N_ants_for_mean.astype(int),
                                         index=np.unique(N_ants_for_mean))
        # size_ants_around_springs[(smoothed_N_ants_without_short_attaches==1)&size_ants_around_springs*1.3<median_size_over_N_ants[1]] = 0
        # false_single_ants = (self.N_ants_around_springs==1)&(self.size_ants_around_springs>median_size_over_N_ants[1]*1.5)

    def springs_rest_lengths(self):
        ar = copy.copy(self.springs_length_processed)
        ar[self.N_ants_around_springs!=0] = np.nan
        ar = np.sort(ar,axis=0)
        self.rest_lenghts = np.nanmedian(ar[:100],axis=0)
        self.rest_length = np.median(self.rest_lenghts)
