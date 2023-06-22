import copy
import numpy as np
import pandas as pd
import os
from data_analysis import utils
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pickle


class CalibrationModeling:
    def __init__(self, directories, weights, output_path, video_paths=None):
        self.directories = directories
        self.weights = weights
        self.output_path = output_path
        self.video_paths = video_paths
        self.zero_weight_dir = self.directories[np.where(np.array(self.weights) == 0)[0][0]]
        self.get_bias_equations()
        self.combine_all_data()
        self.create_calibration_model()

    def get_bias_equations(self):
        self.post_processing(self.zero_weight_dir)
        self.video_path = self.video_paths[np.where(np.array(self.weights) == 0)[0][0]]
        y_length = np.linalg.norm(self.free_ends_coordinates - self.fixed_ends_coordinates, axis=2)
        angles_to_nest = np.expand_dims(self.fixed_end_angle_to_nest, axis=2)
        fixed_end_distance = np.expand_dims(self.object_center_to_fixed_end_distance, axis=2)
        fixed_to_tip_distance = np.expand_dims(self.object_blue_tip_to_fixed_end_distance, axis=2)
        fixed_to_blue_angle_change = np.expand_dims(
            np.repeat(np.expand_dims(np.nanmedian(self.fixed_to_blue_angle_change, axis=1), axis=1),
                      self.num_of_springs, axis=1), axis=2)
        object_center = self.object_center_repeated
        blue_length = np.expand_dims(np.repeat(np.expand_dims(np.linalg.norm(
            self.blue_tip_coordinates - self.object_center, axis=1), axis=1), self.num_of_springs,
            axis=1), axis=2)
        X = np.concatenate((np.sin(angles_to_nest), np.cos(angles_to_nest), object_center, fixed_end_distance,
                            fixed_to_tip_distance, fixed_to_blue_angle_change, blue_length), axis=2)
        y_angle = utils.calc_pulling_angle_matrix(self.fixed_ends_coordinates,
                                                  self.object_center_repeated,
                                                  self.free_ends_coordinates)
        idx = self.rest_bool
        not_nan_idx = ~(np.isnan(y_angle) + np.isnan(X).any(axis=2)) * idx
        X_fit = X[not_nan_idx[:, 0], 0]
        y_length_fit = y_length[not_nan_idx[:, 0], 0]
        y_angle_fit = y_angle[not_nan_idx[:, 0], 0]
        self.model_length = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
        self.model_angle = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
        self.model_length.fit(X_fit, y_length_fit)
        self.model_angle.fit(X_fit, y_angle_fit)

    def get_nest_direction(self,video_path):
        import cv2
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        #collect two points from the user
        from general_video_scripts.collect_analysis_parameters import collect_points
        points = collect_points(frame, 2)
        upper_point = points[0]
        lower_point = points[1]
        lower_artificial_point = (lower_point[0], upper_point[1])
        ba = lower_artificial_point - upper_point
        bc = lower_point - upper_point
        ba_y = ba[0]
        ba_x = ba[1]
        dot = ba_y * bc[0] + ba_x * bc[1]
        det = ba_y * bc[1] - ba_x * bc[0]
        angle = np.arctan2(det, dot)
        self.calib_nest_angle = angle

    def combine_all_data(self):
        calibration_force_direction = np.array(())
        calibration_force_magnitude = np.array(())
        pulling_angle = np.array(())
        extension = np.array(())
        weights = np.array(())
        nest_direction = np.array(())
        self.angular_force = False
        for dir, weight, video_path in zip(self.directories[:], self.weights[:], self.video_paths[:]):
            self.post_processing(dir)
            self.video_path = video_path
            self.calc_pulling_angle()
            self.calc_spring_length()
            self.calc_calibration_force(weight)
            self.weight = weight
            calibration_force_magnitude = np.append(calibration_force_magnitude, self.calibration_force_magnitude)
            pulling_angle = np.append(pulling_angle, (self.pulling_angle).flatten())
            extension = np.append(extension, self.spring_extension.flatten())
            if weight == 0:
                calibration_force_direction = np.append(calibration_force_direction, np.zeros(self.calibration_force_direction.shape))
            else:
                calibration_force_direction = np.append(calibration_force_direction, self.calibration_force_direction)
            weights = np.append(weights, np.repeat(weight, len(self.calibration_force_magnitude)))
            nest_direction = np.append(nest_direction, self.fixed_end_angle_to_nest.flatten())#+self.calib_nest_angle

        data = np.vstack((pulling_angle,
                          extension,
                          calibration_force_direction,
                          calibration_force_magnitude,
                          weights,
                          nest_direction
                          )).transpose()
        data = data[~np.isnan(data).any(axis=1)]

        if self.angular_force:
            self.y = np.sin(data[:, 2]) * data[:, 3]
            negative = self.y > 0
            self.y[negative] *= -1
            self.y = np.array(pd.concat([pd.DataFrame(self.y), pd.DataFrame(self.y)*-1], axis=0)[0])
        else:
            self.y = data[:, 2:4]
            negative = self.y[:, 0] > 0
            self.y[negative, 0] *= -1
            add_this = np.array((copy.copy(self.y[:, 0]) * -1, self.y[:, 1])).transpose()
            self.y = np.array(pd.concat([pd.DataFrame(self.y), pd.DataFrame(add_this)], axis=0))

        self.X = data[:, 0:2]
        # negative = self.X[:, 0] > 0
        self.X[negative, 0] *= -1
        add_this = np.array((copy.copy(self.X[:, 0]) * -1, self.X[:, 1])).transpose()
        self.X = np.array(pd.concat([pd.DataFrame(self.X), pd.DataFrame(add_this)], axis=0))

        self.weights_labels = data[:, 4]
        self.nest_direction = data[:, 5]
        self.weights = self.weights[:]
        self.weights_labels = np.array(pd.concat([pd.DataFrame(self.weights_labels), pd.DataFrame(self.weights_labels)], axis=0)[0])
        # self.nest_direction = np.array(pd.concat([pd.DataFrame(self.nest_direction), pd.DataFrame(self.nest_direction)], axis=0)[0])

    def create_calibration_model(self):
        number_degrees = list(range(1, 2))
        plt_mean_squared_error = []
        plt_r_squared = []
        best_degrees = []
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(self.X, self.y, self.weights_labels, test_size=0.2, random_state=42)
        for degree in number_degrees:
            model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            plt_mean_squared_error.append(mean_squared_error(y_test, y_pred, squared=False))
            plt_r_squared.append(model.score(X_test, y_test))
        best_degrees.append(number_degrees[np.argmin(plt_mean_squared_error)])
        self.degree = max(set(best_degrees), key=best_degrees.count)
        print("Best polynomial degree is for force calibration equation: ", self.degree)
        self.model = make_pipeline(PolynomialFeatures(degree=self.degree), LinearRegression())
        self.model.fit(X_train, y_train)
        with open(os.path.join(self.output_path,"calibration_model.pkl"), "wb") as f:
            pickle.dump(self.model, f)
        y_pred = self.model.predict(X_test)
        if self.angular_force:
            y_pred = np.repeat(y_pred[:, np.newaxis], 3, axis=1)
        print("r squared (before removing out layers): ", self.model.score(X_test, y_test))
        if self.angular_force:
            y_test = np.repeat(y_test[:, np.newaxis], 3, axis=1)
        print("mean squared error (before removing out layers): ", mean_squared_error(y_test, y_pred, squared=False))
        self.ploting_fitting_results_data = (number_degrees, plt_mean_squared_error, plt_r_squared,y_test,y_pred, weights_test)
        print("-"*20)
        print(f"min pulling_angel: {np.min(self.X[:,0])}, max pulling_angel: {np.max(self.X[:,0])}, mean pulling_angel: {np.mean(self.X[:,0])}")
        print(f"min extension: {np.min(self.X[:,1])}, max extension: {np.max(self.X[:,1])}, mean extension: {np.mean(self.X[:,1])}")
        if not self.angular_force:
            print(f"min force_direction: {np.min(self.y[:,0])}, max force_direction: {np.max(self.y[:,0])}, mean force_direction: {np.mean(self.y[:,0])}")
            print(f"min force_magnitude: {np.min(self.y[:,1])}, max force_magnitude: {np.max(self.y[:,1])}, mean force_magnitude: {np.mean(self.y[:,1])}")
        else:
            print(f"min force_magnitude: {np.min(self.y)}, max force_magnitude: {np.max(self.y)}, mean force_magnitude: {np.mean(self.y)}")

    def post_processing(self,directory):
        self.current_dir = directory
        self.load_data(self.current_dir)
        self.calc_distances()
        self.repeat_values()
        self.calc_angle()

    def load_data(self,directory):
        self.num_of_springs = 1
        # self.norm_size = pickle.load(open(os.path.join(directory,"blue_median_area.pickle"), "rb"))
        directory = os.path.join(directory, "raw_analysis")+"\\"
        self.norm_size = np.median(np.loadtxt(os.path.join(directory, "blue_area_sizes.csv"), delimiter=","))
        fixed_ends_coordinates_x = np.loadtxt(os.path.join(directory,"fixed_ends_coordinates_x.csv"), delimiter=",")
        fixed_ends_coordinates_y = np.loadtxt(os.path.join(directory,"fixed_ends_coordinates_y.csv"), delimiter=",")
        self.fixed_ends_coordinates = np.stack((np.expand_dims(fixed_ends_coordinates_x,1), np.expand_dims(fixed_ends_coordinates_y,1)), axis=2)
        free_ends_coordinates_x = np.loadtxt(os.path.join(directory,"free_ends_coordinates_x.csv"), delimiter=",")
        free_ends_coordinates_y = np.loadtxt(os.path.join(directory,"free_ends_coordinates_y.csv"), delimiter=",")
        self.free_ends_coordinates = np.stack((np.expand_dims(free_ends_coordinates_x,1), np.expand_dims(free_ends_coordinates_y,1)), axis=2)
        blue_part_coordinates_x = np.loadtxt(os.path.join(directory,"blue_part_coordinates_x.csv"), delimiter=",")
        blue_part_coordinates_y = np.loadtxt(os.path.join(directory,"blue_part_coordinates_y.csv"), delimiter=",")
        blue_part_coordinates = np.stack((blue_part_coordinates_x, blue_part_coordinates_y), axis=2)
        self.object_center = blue_part_coordinates[:, 0, :]
        self.blue_tip_coordinates = blue_part_coordinates[:, -1, :]
        self.num_of_frames = self.free_ends_coordinates.shape[0]
        self.rest_bool = np.ones((self.num_of_frames, self.num_of_springs), dtype=bool)

    def calc_distances(self):
        object_center = np.repeat(self.object_center[:, np.newaxis, :], self.num_of_springs, axis=1)
        blue_tip_coordinates = np.repeat(self.blue_tip_coordinates[:, np.newaxis, :], self.num_of_springs, axis=1)
        self.blue_length = np.nanmedian(np.linalg.norm(self.blue_tip_coordinates - self.object_center, axis=1))
        self.object_center_to_free_end_distance = np.linalg.norm(self.free_ends_coordinates - object_center, axis=2)
        self.object_center_to_fixed_end_distance = np.linalg.norm(self.fixed_ends_coordinates - object_center, axis=2)
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

    def norm_values(self,matrix,model):
        angles_to_nest = np.expand_dims(self.fixed_end_angle_to_nest, axis=2)
        fixed_end_distance = np.expand_dims(self.object_center_to_fixed_end_distance, axis=2)
        object_center = self.object_center_repeated
        fixed_to_tip_distance = np.expand_dims(self.object_blue_tip_to_fixed_end_distance, axis=2)
        fixed_to_blue_angle_change = np.expand_dims(np.repeat(np.expand_dims(np.nanmedian(self.fixed_to_blue_angle_change,axis=1),axis=1),self.num_of_springs,axis=1),axis=2)
        blue_length = np.expand_dims(np.repeat(np.expand_dims(np.linalg.norm(
            self.blue_tip_coordinates - self.object_center, axis=1), axis=1), self.num_of_springs, axis=1), axis=2)
        X = np.concatenate((np.sin(angles_to_nest), np.cos(angles_to_nest),object_center,fixed_end_distance,fixed_to_tip_distance,fixed_to_blue_angle_change,blue_length), axis=2)
        not_nan_idx = ~(np.isnan(matrix) + np.isnan(X).any(axis=2))
        prediction_matrix = np.zeros(matrix.shape)
        for col in range(matrix.shape[1]):
            prediction_matrix[not_nan_idx[:, col], col] = model.predict(X[not_nan_idx[:, col], col, :])
        return prediction_matrix

    def calc_pulling_angle(self):
        self.pulling_angle = utils.calc_pulling_angle_matrix(self.fixed_ends_coordinates,self.object_center_repeated,
                                                             self.free_ends_coordinates)
        pred_pulling_angle = self.norm_values(self.pulling_angle,self.model_angle)
        self.pulling_angle -= pred_pulling_angle
        pulling_angle_if_rest = copy.copy(self.pulling_angle)
        pulling_angle_copy = copy.copy(self.pulling_angle)
        pulling_angle_if_rest[~self.rest_bool] = np.nan
        pulling_angle_copy[self.rest_bool] = np.nan
        self.pulling_angle -= np.nanmedian(pulling_angle_if_rest, axis=0)

    def calc_spring_length(self):
        self.spring_length = np.linalg.norm(self.free_ends_coordinates - self.fixed_ends_coordinates , axis=2)
        pred_spring_length = self.norm_values(self.spring_length,self.model_length)
        self.spring_length /= pred_spring_length
        self.spring_length = utils.interpolate_data(self.spring_length,
                                                       utils.find_cells_to_interpolate(self.spring_length))
        self.spring_length /= self.norm_size
        median_spring_length_at_rest = copy.copy(self.spring_length)
        median_spring_length_at_rest[np.invert(self.rest_bool)] = np.nan
        median_spring_length_at_rest = np.nanmedian(median_spring_length_at_rest, axis=0)
        self.spring_length /= median_spring_length_at_rest
        self.spring_extension = self.spring_length -1

    def calc_calibration_force(self, calibration_weight):
        self.calibration_force_direction = (self.fixed_end_angle_to_nest).flatten()
        above = self.calibration_force_direction > np.pi
        self.calibration_force_direction[above] = self.calibration_force_direction[above] - 2 * np.pi
        G = 9.81
        weight_in_Kg = calibration_weight*1e-3
        self.calibration_force_magnitude = np.repeat(weight_in_Kg * G, self.num_of_frames)


if __name__ == "__main__":
    # data_dir = "Z:\\Dor_Gabay\\ThesisProject\\data\\exp_collected_data\\15.9.22\\plus0.3mm_force\\S5280006\\"

    # calibration_dir1 = "Z:\\Dor_Gabay\\ThesisProject\\data\\calibration\\post_slicing\\calibration1\\"
    # directories_1 = [os.path.join(calibration_dir1, o) for o in os.listdir(calibration_dir1)
    #                  if os.path.isdir(os.path.join(calibration_dir1, o)) and "_sliced" in os.path.basename(o)]
    # weights1 = list(np.array([0.10606, 0.14144, 0.16995, 0.19042, 0.16056, 0.15082]) - 0.10506)
    #
    # calibration_dir2 = "Z:\\Dor_Gabay\\ThesisProject\\data\\calibration\\post_slicing\\calibration2\\"
    # calibration_video_dir2 = "Z:\\Dor_Gabay\\ThesisProject\\data\\videos\\calibrations\\calibration2\\"
    # directories_2 = [os.path.join(calibration_dir2, o) for o in os.listdir(calibration_dir2)
    #                  if os.path.isdir(os.path.join(calibration_dir2, o)) and "_sliced" in os.path.basename(o) and "6" not in os.path.basename(o)]# and "9" not in os.path.basename(o)]
    # # weights2 = list(np.array([0.10582,0.13206,0.15650,0.18405,0.21030,0.46612])-0.10582)
    # weights2 = list(np.array([0.10582,0.13206,0.15650,0.18405,0.46612])-0.10582)
    # video_paths2 = [os.path.join(calibration_video_dir2, o) for o in os.listdir(calibration_video_dir2)
    #                  if "MP4" in os.path.basename(o) and "_sliced" in os.path.basename(o) and "6" not in os.path.basename(o)]
    # #
    # calibration_dir3 = "Z:\\Dor_Gabay\\ThesisProject\\data\\calibration\\post_slicing\\calibration3\\"
    # calibration_video_dir3 = "Z:\\Dor_Gabay\\ThesisProject\\data\\videos\\calibrations\\calibration3\\"
    # directories_3 = [os.path.join(calibration_dir3, o) for o in os.listdir(calibration_dir3)
    #                  if os.path.isdir(os.path.join(calibration_dir3, o)) and "_sliced" in os.path.basename(o) and "S5430006_sliced" not in os.path.basename(o)]# not in os.path.basename(o) and "S5430003_sliced" not in os.path.basename(o)]
    # # weights3 = list(np.array([0.41785, 0.36143, 0.3008, 0.2389, 0.18053, 0.15615, 0.11511, 0.12561, 0.10610]) - 0.10610)
    # weights3 = list(np.array([0.41785, 0.36143, 0.3008, 0.2389, 0.18053, 0.11511, 0.12561, 0.10610]) - 0.10610)
    # # weights3 = list(np.array([0.41785, 0.2389, 0.18053, 0.15615, 0.11511, 0.12561, 0.10610]) - 0.10610)
    # video_paths3 = [os.path.join(calibration_video_dir3, o) for o in os.listdir(calibration_video_dir3)
    #                if "MP4" in os.path.basename(o) and "_sliced" in os.path.basename(o) and "S5430006_sliced" not in os.path.basename(o)]## and "S5430002_sliced" not in os.path.basename(o) and "S5430003_sliced" not in os.path.basename(o)]

    calibration_dir4 = "Z:\\Dor_Gabay\\ThesisProject\\data\\calibration\\post_slicing\\calibration_perfect2\\sliced_videos\\"
    calibration_video_dir4 = "Z:\\Dor_Gabay\\ThesisProject\\data\\videos\\calibration_perfect2\\sliced_videos\\"
    directories_4 = [os.path.join(calibration_dir4, o) for o in os.listdir(calibration_dir4)
                     if os.path.isdir(os.path.join(calibration_dir4, o)) and "plots" not in os.path.basename(o)]# and "S5430006_sliced" not in os.path.basename(o)]  # not in os.path.basename(o) and "S5430003_sliced" not in os.path.basename(o)]
    weights4 = list(np.array([0.11656, 0.16275, 0.21764, 0.28268, 0.10657]) - 0.10657)
    # weights3 = list(np.array([0.41785, 0.2389, 0.18053, 0.15615, 0.11511, 0.12561, 0.10610]) - 0.10610)
    video_paths4 = [os.path.join(calibration_video_dir4, o) for o in os.listdir(calibration_video_dir4)
                    if "MP4" in os.path.basename(o) and "plots" not in os.path.basename(o)]#]# and "_sliced" in os.path.basename(o)]# and "S5430006_sliced" not in os.path.basename(o)]

    CalibrationModeling(directories=directories_4, weights=weights4, output_path=calibration_dir4, video_paths=video_paths4)
    # Calibration(directories=directories_3, weights=weights3, output_path=calibration_dir3, video_paths=video_paths3)
    # Calibration(directories=directories_2, weights=weights2, output_path=calibration_dir2, video_paths=video_paths2)
    print("-"*60)
    print("Calibration model has been created.")
    print("-"*60)

