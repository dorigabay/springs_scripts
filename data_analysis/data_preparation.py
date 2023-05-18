import copy
import numpy as np
import pandas as pd
import os
import pickle
import scipy.io as sio
from scipy.signal import savgol_filter
from data_analysis import utils
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from data_analysis import plots

def save_as_mathlab_matrix(output_dir):
    ants_centers_x = np.loadtxt(os.path.join(output_dir, "ants_centers_x.csv"), delimiter=",")
    ants_centers_y = np.loadtxt(os.path.join(output_dir, "ants_centers_y.csv"), delimiter=",")
    ants_centers = np.stack((ants_centers_x, ants_centers_y), axis=2)
    ants_centers_mat = np.zeros((ants_centers.shape[0], 1), dtype=np.object)
    for i in range(ants_centers.shape[0]):
        ants_centers_mat[i, 0] = ants_centers[i, :, :]
    sio.savemat(os.path.join(output_dir, "ants_centers.mat"), {"ants_centers": ants_centers_mat})
    matlab_script_path = "Z:\\Dor_Gabay\\ThesisProject\\scripts\\munkres_tracker\\"
    os.chdir(matlab_script_path)
    os.system("PATH=$PATH:'C:\Program Files\MATLAB\R2022a\bin'")
    execution_string = f"matlab -r ""ants_tracking('" + output_dir + "\\')"""
    os.system(execution_string)


def save_blue_areas_median(output_dir):
    blue_area_sizes = np.loadtxt(os.path.join(output_dir, "blue_area_sizes.csv"), delimiter=",")
    median_blue_area_size = np.median(blue_area_sizes)
    with open(os.path.join(output_dir, "blue_median_area.pickle"), 'wb') as f:
        pickle.dump(median_blue_area_size, f)


class PostProcessing:
    def __init__(self, directory, calibration_model,length_models=None, angle_models=None, slice=None, two_vars=True):
        # self.frame_center = np.array([1920, 540])/2
        self.two_vars = two_vars
        self.directory = directory
        self.load_data(slice=slice)
        self.assign_ants_to_springs()
        self.n_ants_processing()
        self.calc_distances()
        self.repeat_values()
        self.calc_angle()
        self.calibration_model = calibration_model
        if length_models is not None:
            self.calc_spring_length(models=length_models)
            self.calc_pulling_angle(models=angle_models)
            self.calc_force(calibration_model)
            # self.calculations()

    def slice_data(self, start_frame, end_frame):
        self.N_ants_around_springs = self.N_ants_around_springs[start_frame:end_frame]
        self.size_ants_around_springs = self.size_ants_around_springs[start_frame:end_frame]
        self.fixed_ends_coordinates = self.fixed_ends_coordinates[start_frame:end_frame]
        self.free_ends_coordinates = self.free_ends_coordinates[start_frame:end_frame]
        self.blue_tip_coordinates = self.blue_tip_coordinates[start_frame:end_frame]
        self.object_center = self.object_center[start_frame:end_frame]
        self.num_of_frames = self.N_ants_around_springs.shape[0]

    def load_data(self, slice=None):
        print("loading data from:", self.directory)
        directory = os.path.join(self.directory, "raw_analysis")+"\\"
        self.tracked_ants = sio.loadmat(os.path.join(directory, "tracking_data.mat"))["tracked_blobs_matrix"]
        self.ants_attached_labels  = np.loadtxt(os.path.join(directory, "ants_attached_labels.csv"), delimiter=",")

        # self.ants_attached_labels_path  = os.path.join(directory, "ants_attached_labels_complete.pickle")
        if os.path.exists(os.path.join(self.directory,"blue_median_area.pickle")):
            self.norm_size = pickle.load(open(os.path.join(self.directory,"blue_median_area.pickle"), "rb"))
        else:
            self.norm_size = pickle.load(open(os.path.join(directory,"blue_median_area.pickle"), "rb"))
        self.N_ants_around_springs = np.loadtxt(os.path.join(directory,"N_ants_around_springs.csv"), delimiter=",")
        self.num_of_springs = self.N_ants_around_springs.shape[1]
        self.size_ants_around_springs = np.loadtxt(os.path.join(directory,"size_ants_around_springs.csv"), delimiter=",")
        fixed_ends_coordinates_x = np.loadtxt(os.path.join(directory,"fixed_ends_coordinates_x.csv"), delimiter=",")
        fixed_ends_coordinates_y = np.loadtxt(os.path.join(directory,"fixed_ends_coordinates_y.csv"), delimiter=",")
        self.fixed_ends_coordinates = np.stack((fixed_ends_coordinates_x, fixed_ends_coordinates_y), axis=2)
        free_ends_coordinates_x = np.loadtxt(os.path.join(directory,"free_ends_coordinates_x.csv"), delimiter=",")
        free_ends_coordinates_y = np.loadtxt(os.path.join(directory,"free_ends_coordinates_y.csv"), delimiter=",")
        self.free_ends_coordinates = np.stack((free_ends_coordinates_x, free_ends_coordinates_y), axis=2)
        blue_part_coordinates_x = np.loadtxt(os.path.join(directory,"blue_part_coordinates_x.csv"), delimiter=",")
        blue_part_coordinates_y = np.loadtxt(os.path.join(directory,"blue_part_coordinates_y.csv"), delimiter=",")
        blue_part_coordinates = np.stack((blue_part_coordinates_x, blue_part_coordinates_y), axis=2)
        self.object_center = blue_part_coordinates[:, 0, :]
        self.blue_tip_coordinates = blue_part_coordinates[:, -1, :]
        self.num_of_frames = self.N_ants_around_springs.shape[0]
        if slice is not None:
            self.slice_data(*slice)

    def assign_ants_to_springs(self):
        self.assigned_to_springs = np.zeros((self.num_of_frames, np.nanmax(self.tracked_ants[:,2,:]).astype(np.int32)))
        print(self.assigned_to_springs.shape)
        for i in range(self.num_of_frames):
            labels = self.tracked_ants[~np.isnan(self.ants_attached_labels[i]),2,i]
            labels = labels[~np.isnan(labels)].astype(int)
            springs = self.ants_attached_labels[i,~np.isnan(self.ants_attached_labels[i])].astype(int)
            print(labels, springs)
            self.assigned_to_springs[i,labels-1] = springs
        print(self.assigned_to_springs)

    def smoothing_n_ants(self, array):
        for col in range(array.shape[1]):
            array[:,col] = np.abs(np.round(savgol_filter(array[:,col], 31, 2)))
        return array

    def n_ants_processing(self):
        undetected_springs_for_long_time = utils.filter_continuity(utils.convert_bool_to_binary(
            np.isnan(self.fixed_ends_coordinates[:, :, 0])),min_size=8)
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
        self.rest_bool = self.N_ants_around_springs == 0

    def calc_distances(self):
        object_center = np.repeat(self.object_center[:, np.newaxis, :], self.num_of_springs, axis=1)
        blue_tip_coordinates = np.repeat(self.blue_tip_coordinates[:, np.newaxis, :], self.num_of_springs, axis=1)
        self.blue_length = np.nanmedian(np.linalg.norm(self.blue_tip_coordinates - self.object_center, axis=1))
        self.object_center_to_free_end_distance = np.linalg.norm(self.free_ends_coordinates - object_center, axis=2)
        self.object_center_to_fixed_end_distance = np.linalg.norm(self.fixed_ends_coordinates - object_center, axis=2)
        # print("distance: ",np.nanmedian(self.object_center_to_fixed_end_distance)-np.nanmedian(self.object_center_to_fixed_end_distance)/10)
        self.object_blue_tip_to_fixed_end_distance = np.linalg.norm(self.fixed_ends_coordinates - blue_tip_coordinates, axis=2)

    def repeat_values(self):
        nest_direction = np.stack((self.object_center[:, 0], self.object_center[:, 1]-100), axis=1)
        self.nest_direction_repeated = np.repeat(nest_direction[:, np.newaxis, :], self.free_ends_coordinates.shape[1], axis=1)
        self.object_center_repeated = copy.copy(np.repeat(self.object_center[:, np.newaxis, :], self.free_ends_coordinates.shape[1], axis=1))
        self.blue_tip_coordinates_repeated = np.repeat(self.blue_tip_coordinates[:, np.newaxis, :], self.free_ends_coordinates.shape[1], axis=1)

    def calc_angle(self):
        self.free_end_angle_to_nest = utils.calc_angle_matrix(self.free_ends_coordinates, self.object_center_repeated, self.nest_direction_repeated)+np.pi
        self.fixed_end_angle_to_nest = utils.calc_angle_matrix(self.fixed_ends_coordinates, self.object_center_repeated, self.nest_direction_repeated)+np.pi
        self.blue_part_angle_to_nest = utils.calc_angle_matrix(self.nest_direction_repeated, self.object_center_repeated, self.blue_tip_coordinates_repeated)+np.pi
        self.free_end_angle_to_blue_part = utils.calc_angle_matrix(self.blue_tip_coordinates_repeated, self.object_center_repeated, self.free_ends_coordinates)+np.pi
        self.fixed_end_angle_to_blue_part = utils.calc_angle_matrix(self.blue_tip_coordinates_repeated, self.object_center_repeated,self.fixed_ends_coordinates)+np.pi
        self.fixed_to_blue_angle_change = utils.calc_pulling_angle_matrix(self.blue_tip_coordinates_repeated, self.object_center_repeated, self.fixed_ends_coordinates)

    # def norm_to_sections(self,array,section_min_size):
    #     import scipy.ndimage as ndi
    #     if_rest = copy.copy(array)
    #     if_rest[~self.rest_bool] = np.nan
    #     for col in range(if_rest.shape[1]):
    #         label_min_size = section_min_size
    #         values = if_rest[:, col]
    #         if label_min_size > np.sum(~np.isnan(values)):
    #             label_min_size = np.sum(~np.isnan(values))
    #         labeled, n_labels = ndi.label(~np.isnan(values))
    #         labeled_nan, n_labels_nan = ndi.label(np.isnan(values))
    #         if labeled_nan[0] == 1:
    #             labeled_nan[(labeled_nan != 1) + (labeled_nan != 0)] -= 1
    #         labeled_nan = labeled_nan + labeled
    #         for i in range(1, n_labels + 1):
    #             if np.sum(labeled == i) < label_min_size:
    #                 labeled[labeled == i] = i + 1
    #                 labeled_nan[labeled_nan == i] = i + 1
    #             else:
    #                 values[labeled_nan == i] = np.nanmedian(values[labeled == i])
    #         if_rest[:, col] = values
    #     return if_rest

    # def find_best_sectioning(self):
    #     self.calc_spring_length()
    #     self.calc_pulling_angle(zero_angles=True)
    #     saved_pulling_angle = copy.copy(self.pulling_angle)
    #     saved_spring_length = copy.copy(self.spring_length)
    #     corr_results = []
    #     section_sizes = range(1,70000,100)
    #     for section_size in section_sizes:
    #         self.pulling_angle = copy.copy(saved_pulling_angle)
    #         pulling_angle_if_rest = self.norm_to_sections(self.pulling_angle, section_size)
    #         self.pulling_angle -= pulling_angle_if_rest
    #
    #         self.spring_length = copy.copy(saved_spring_length)
    #         median_spring_length_at_rest = self.norm_to_sections(self.spring_length, section_size)
    #         self.spring_length /= median_spring_length_at_rest
    #         self.spring_extension = self.spring_length - 1
    #
    #         self.calc_force(calibration_model)
    #         self.calculations()
    #         corr = self.test_correlation()
    #         print("corr: ",corr)
    #         corr_results.append(corr)
    #
    #     self.best_section_size = section_sizes[np.argmax(corr_results)]
    #     self.pulling_angle = copy.copy(saved_pulling_angle)
    #     pulling_angle_if_rest = self.norm_to_sections(self.pulling_angle, self.best_section_size)
    #     self.pulling_angle -= pulling_angle_if_rest
    #
    #     self.spring_length = copy.copy(saved_spring_length)
    #     median_spring_length_at_rest = self.norm_to_sections(self.spring_length, 1000)
    #     self.spring_length /= median_spring_length_at_rest
    #     self.spring_extension = self.spring_length - 1
    #
    #     self.calc_force(calibration_model)
    #     self.calculations()

    def norm_values(self,matrix,models=None):
        angles_to_nest = np.expand_dims(self.fixed_end_angle_to_nest, axis=2)
        fixed_end_distance = np.expand_dims(self.object_center_to_fixed_end_distance, axis=2)
        object_center = self.object_center_repeated
        fixed_to_tip_distance = np.expand_dims(self.object_blue_tip_to_fixed_end_distance, axis=2)
        fixed_to_blue_angle_change = np.expand_dims(np.repeat(np.expand_dims(np.nanmedian(self.fixed_to_blue_angle_change,axis=1),axis=1),self.num_of_springs,axis=1),axis=2)
        blue_length = np.expand_dims(np.repeat(np.expand_dims(np.linalg.norm(
            self.blue_tip_coordinates - self.object_center, axis=1), axis=1), self.num_of_springs, axis=1), axis=2)
        X = np.concatenate((np.sin(angles_to_nest), np.cos(angles_to_nest),object_center,fixed_end_distance,
                            fixed_to_tip_distance,fixed_to_blue_angle_change,blue_length,), axis=2)
        not_nan_idx = ~(np.isnan(matrix) + np.isnan(X).any(axis=2))
        prediction_matrix = np.zeros(matrix.shape)
        for col in range(matrix.shape[1]):
            model = models[col]
            prediction_matrix[not_nan_idx[:, col], col] = model.predict(X[not_nan_idx[:, col], col, :])
        return prediction_matrix

    def calc_pulling_angle(self,models):
        self.pulling_angle = utils.calc_pulling_angle_matrix(self.fixed_ends_coordinates,self.object_center_repeated,
                                                             self.free_ends_coordinates)
        pred_pulling_angle = self.norm_values(self.pulling_angle,models)
        self.pulling_angle -= pred_pulling_angle
        pulling_angle_if_rest = copy.copy(self.pulling_angle)
        pulling_angle_copy = copy.copy(self.pulling_angle)
        pulling_angle_if_rest[~self.rest_bool] = np.nan
        pulling_angle_copy[self.rest_bool] = np.nan
        self.pulling_angle -= np.nanmedian(pulling_angle_if_rest, axis=0)

    def calc_spring_length(self,models):
        self.spring_length = np.linalg.norm(self.free_ends_coordinates - self.fixed_ends_coordinates , axis=2)
        pred_spring_length = self.norm_values(self.spring_length,models)
        self.spring_length /= pred_spring_length
        self.spring_length = utils.interpolate_data(self.spring_length,
                                                       utils.find_cells_to_interpolate(self.spring_length))
        self.spring_length /= self.norm_size
        median_spring_length_at_rest = copy.copy(self.spring_length)
        median_spring_length_at_rest[np.invert(self.rest_bool)] = np.nan
        median_spring_length_at_rest = np.nanmedian(median_spring_length_at_rest, axis=0)
        self.spring_length /= median_spring_length_at_rest
        self.spring_extension = self.spring_length -1

    def calc_force(self,model):
        self.force_direction = np.zeros((self.num_of_frames,self.num_of_springs))
        self.force_magnitude = np.zeros((self.num_of_frames,self.num_of_springs))
        self.total_force = np.zeros((self.num_of_frames,self.num_of_springs))
        for s in range(self.num_of_springs):
            X = np.stack((self.pulling_angle[:,s],self.spring_extension[:,s]),axis=1)
            un_nan_bool = ~np.isnan(X).any(axis=1)
            X = X[un_nan_bool]
            forces_predicted = model.predict(X)
            if self.two_vars:
                self.force_direction[un_nan_bool,s] = forces_predicted[:,0]
                self.force_magnitude[un_nan_bool,s] = forces_predicted[:,1]
            else:
                self.total_force[un_nan_bool,s] = forces_predicted

        if self.two_vars:
            self.total_force = np.sin(self.force_direction) * self.force_magnitude
            # self.total_force[self.negative_pulling_angle] *= -1
            self.total_force *= -1
        else:
            self.total_force *= -1

    def save_data(self, save_path):
        print("-" * 60)
        print("saving data to:", save_path)
        # directory = os.path.join(directory, "post_processed_data/")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.savetxt(os.path.join(save_path,"N_ants_around_springs.csv"), self.N_ants_around_springs, delimiter=",")
        np.savetxt(os.path.join(save_path,"fixed_end_angle_to_nest.csv"), self.fixed_end_angle_to_nest, delimiter=",")
        np.savetxt(os.path.join(save_path,"force_direction.csv"), self.force_direction, delimiter=",")
        np.savetxt(os.path.join(save_path,"force_magnitude.csv"), self.force_magnitude, delimiter=",")
        # np.savetxt(os.path.join(save_path,"pulling_angle.csv"), self.pulling_angle, delimiter=",")
        # np.savetxt(os.path.join(save_path,"total_force.csv"), self.total_force, delimiter=",")
        # np.savetxt(os.path.join(save_path,"spring_length.csv"), self.spring_length, delimiter=",")

    def calculations(self):
        self.net_force = np.nansum(self.total_force, axis=1)
        self.net_force = np.array(pd.Series(self.net_force).rolling(window=5,center=True).median())
        self.net_magnitude = np.nansum(self.force_magnitude, axis=1)
        self.net_magnitude = np.array(pd.Series(self.net_magnitude).rolling(window=5,center=True).median())
        self.angular_velocity = np.nanmedian(utils.calc_angular_velocity(self.fixed_end_angle_to_nest, diff_spacing=20)/20, axis=1)
        self.total_n_ants = np.nansum(self.N_ants_around_springs, axis=1)
        self.sum_pulling_angle = np.nansum(self.pulling_angle, axis=1)
        self.angle_to_nest = np.nansum(self.fixed_end_angle_to_nest, axis=1)
        self.spring_extension = np.nanmean(self.spring_length, axis=1)
        self.total_force[np.isnan(self.pulling_angle)] = np.nan

    def test_correlation(self):
        corr_df = pd.DataFrame({"net_force": self.net_force, "angular_velocity": self.angular_velocity})
        corr_df = corr_df.dropna()
        correlation_score = corr_df.corr()["net_force"]["angular_velocity"]
        print(f"correlation score between net force and angular velocity: {correlation_score}")
        return correlation_score


def create_model(objects):
    X = None
    y_length = None
    y_angle = None
    idx = None
    for count,object in enumerate(objects):
        if count == 0:
            y_length = np.linalg.norm(object.free_ends_coordinates - object.fixed_ends_coordinates , axis=2)
            angles_to_nest = np.expand_dims(object.fixed_end_angle_to_nest, axis=2)
            fixed_end_distance = np.expand_dims(object.object_center_to_fixed_end_distance, axis=2)
            fixed_to_tip_distance = np.expand_dims(object.object_blue_tip_to_fixed_end_distance, axis=2)
            fixed_to_blue_angle_change = np.expand_dims(np.repeat(np.expand_dims(np.nanmedian(object.fixed_to_blue_angle_change,axis=1),axis=1),object.num_of_springs,axis=1),axis=2)
            object_center = object.object_center_repeated
            blue_length = np.expand_dims(np.repeat(np.expand_dims(np.linalg.norm(
                object.blue_tip_coordinates - object.object_center, axis=1), axis=1), object.num_of_springs, axis=1), axis=2)
            X = np.concatenate((np.sin(angles_to_nest), np.cos(angles_to_nest),object_center,fixed_end_distance,
                                fixed_to_tip_distance,fixed_to_blue_angle_change,blue_length), axis=2)
            y_angle = utils.calc_pulling_angle_matrix(object.fixed_ends_coordinates,
                                                                 object.object_center_repeated,
                                                                 object.free_ends_coordinates)
            idx = object.rest_bool
        else:
            y_length = np.concatenate((y_length,np.linalg.norm(object.free_ends_coordinates - object.fixed_ends_coordinates , axis=2)),axis=0)
            angles_to_nest = np.expand_dims(object.fixed_end_angle_to_nest,axis=2)
            fixed_end_distance = np.expand_dims(object.object_center_to_fixed_end_distance, axis=2)
            fixed_to_tip_distance = np.expand_dims(object.object_blue_tip_to_fixed_end_distance, axis=2)
            fixed_to_blue_angle_change = np.expand_dims(np.repeat(np.expand_dims(np.nanmedian(object.fixed_to_blue_angle_change,axis=1),axis=1),object.num_of_springs,axis=1),axis=2)
            object_center = object.object_center_repeated
            blue_length = np.expand_dims(np.repeat(np.expand_dims(np.linalg.norm(
                object.blue_tip_coordinates - object.object_center, axis=1), axis=1), object.num_of_springs, axis=1), axis=2)
            X = np.concatenate((X,np.concatenate((np.sin(angles_to_nest),np.cos(angles_to_nest),object_center,
                                                  fixed_end_distance,fixed_to_tip_distance,fixed_to_blue_angle_change,
                                                  blue_length),axis=2)),axis=0)
            y_angle = np.concatenate((y_angle,utils.calc_pulling_angle_matrix(object.fixed_ends_coordinates,
                                                                    object.object_center_repeated,
                                                                    object.free_ends_coordinates)),axis=0)
            idx = np.concatenate((idx,object.rest_bool),axis=0)
    models_lengths = []
    models_angles = []
    not_nan_idx = ~(np.isnan(y_angle) + np.isnan(X).any(axis=2))*idx
    for col in range(y_length.shape[1]):
        X_fit = X[not_nan_idx[:,col],col]
        y_length_fit = y_length[not_nan_idx[:,col],col]
        y_angle_fit = y_angle[not_nan_idx[:,col],col]
        model_length = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
        model_angle = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
        models_lengths.append(model_length.fit(X_fit, y_length_fit))
        models_angles.append(model_angle.fit(X_fit, y_angle_fit))
    return models_lengths,models_angles


if __name__=="__main__":
    calibration_dir = "Z:\\Dor_Gabay\\ThesisProject\\data\\calibration\\post_slicing\\calibration_perfect2\\sliced_videos\\"
    # data_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\exp_collected_data\\15.9.22\\plus0.3mm_force"
    data_dir = "Z:\\Dor_Gabay\\ThesisProject\\data\\analysed_with_tracking3\\15.9.22\\plus0.3mm_force"
    objects = []
    two_vars = True
    # for i in [1,3,4,5,6,7,8,9]:
    for i in [1]:
        path = os.path.join(data_dir, f"S528000{i}")
        calibration_model = pickle.load(open(calibration_dir + "calibration_model.pkl", "rb"))
        # save_blue_areas_median(os.path.join(path,"raw_analysis"))
        # save_as_mathlab_matrix(os.path.join(path,"raw_analysis"))
        # if i == 7:
        #     object = PostProcessing(data_dir, calibration_model=calibration_model,slice=[30000,68025])
        # elif i == 6:
        #     object = PostProcessing(data_dir, calibration_model=calibration_model,slice=[0,30000])
        # else:
        object = PostProcessing(path, calibration_model=calibration_model,two_vars=two_vars)
        object.video_name = path.split("\\")[-1]
        if two_vars:
            object.output_path = os.path.join(path, "two_vars_post_processing")
            object.figures_output_path = os.path.join(data_dir,"figures_two_vars_magnitude")
        else:
            object.output_path = os.path.join(path, "one_var_post_processing")
            object.figures_output_path = os.path.join(data_dir, "figures_one_var")
        objects.append(object)
    models_lengths, models_angles = create_model(objects)
    for object in objects:
        print("-" * 60)
        object.calc_spring_length(models=models_lengths)
        object.calc_pulling_angle(models=models_angles)
        object.calc_force(calibration_model)
        # object.calculations()
        # correlation_score = object.test_correlation()
        # # object.save_data(object.output_path)
        #
        # # correlation plots:
        # os.makedirs(object.figures_output_path, exist_ok=True)
        # plots.plot_overall_behavior(object, start=0, end=None, window_size=200,
        #                             title=object.video_name+"_corr"+str(np.round(correlation_score,2)),
        #                             output_dir=object.figures_output_path)

