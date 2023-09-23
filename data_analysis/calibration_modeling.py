import os
import cv2
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# local imports:
from data_analysis import utils


class CalibrationModeling:
    def __init__(self, output_path, videos_dir, weights=None):
        self.calibration_name = os.path.basename(os.path.normpath(videos_dir))
        self.output_path = os.path.join(output_path, self.calibration_name)
        self.load_weights(weights)
        self.num_of_springs = 1
        video_names = [os.path.splitext(file)[0] for file in os.listdir(videos_dir) if file.endswith(".MP4")]
        self.directories = np.array([os.path.join(self.output_path, video_name) for video_name in video_names])
        self.zero_weight_dir = self.directories[np.where(np.array(self.weights) == 0)[0][0]]
        self.create_bias_correction_models()
        self.combine_all_data()
        self.create_calibration_model()

    def load_weights(self, weights=None):
        if weights is None:
            self.weights = pickle.load(open(os.path.join(self.output_path, self.calibration_name, "weights.pickle"), 'rb'))
            raise ValueError("No weights were given and no weights were found in the output path.")
        else:
            self.weights = np.array(weights)
            os.makedirs(self.output_path, exist_ok=True)
            pickle.dump(self.weights, open(os.path.join(self.output_path, "weights.pickle"), 'wb'))

    def create_bias_correction_models(self):
        self.post_processing(self.zero_weight_dir)
        y_length = np.linalg.norm(self.free_ends_coordinates - self.fixed_ends_coordinates, axis=2)
        y_angle = utils.calc_pulling_angle_matrix(self.fixed_ends_coordinates, self.object_center_repeated, self.free_ends_coordinates)
        angles_to_nest = np.expand_dims(self.fixed_end_angle_to_nest, axis=2)
        X = np.concatenate((np.sin(angles_to_nest), np.cos(angles_to_nest)), axis=2)
        idx = np.ones((self.free_ends_coordinates.shape[0], 1), dtype=bool)
        self.models_lengths = []
        self.models_angles = []
        not_nan_idx = ~(np.isnan(y_angle) + np.isnan(X).any(axis=2)) * idx
        for col in range(y_length.shape[1]):
            X_fit = X[not_nan_idx[:, col], col]
            y_length_fit = y_length[not_nan_idx[:, col], col]
            y_angle_fit = y_angle[not_nan_idx[:, col], col]
            model_length = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
            model_angle = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
            self.models_lengths.append(model_length.fit(X_fit, y_length_fit))
            self.models_angles.append(model_angle.fit(X_fit, y_angle_fit))

    def combine_all_data(self):
        calibration_force_direction = np.array(())
        calibration_force_magnitude = np.array(())
        pulling_angle = np.array(())
        length = np.array(())
        a = [0, 3, 5, 7, 9]
        for dir, weight in zip(self.directories[a], self.weights[a]):
        # for dir, weight in zip(self.directories, self.weights):
            self.post_processing(dir)
            self.calc_pulling_angle(weight)
            self.calc_spring_length(weight)
            self.calc_calibration_force(weight)
            calibration_force_direction = np.append(calibration_force_direction, self.calibration_force_direction)
            calibration_force_magnitude = np.append(calibration_force_magnitude, self.calibration_force_magnitude)
            pulling_angle = np.append(pulling_angle, self.pulling_angle.flatten())
            length = np.append(length, self.spring_length.flatten())
        data = np.vstack((pulling_angle, length, calibration_force_direction, calibration_force_magnitude)).transpose()
        data[data[:, 2] < 0, 0] *= -1
        data[data[:, 2] < 0, 2] *= -1
        data[data[:, 0] < 0, :] = np.nan
        max_value = np.nanmax(data[~(data[:, 3] == 0), 2])
        data[data[:, 2] > max_value, :] = np.nan
        data = data[~np.isnan(data).any(axis=1)]
        # data = np.concatenate((data, data), axis=0)
        # data[0:data.shape[0]//2, [0, 2]] *= -1
        self.data = data
        self.plot()

    def plot(self):
        # plt.subplots(1, 2, figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.scatter(self.data[self.data[:, 3] == 0, 0],self.data[self.data[:, 3] == 0, 2], alpha=0.5)
        # plt.subplot(1, 2, 2)
        # plt.scatter(self.data[self.data[:, 3] == 0, 1],self.data[self.data[:, 3] == 0, 2], alpha=0.5)
        # plt.show()
        for force_magnitude in np.unique(self.data[:, 3]):
            data = self.data[self.data[:, 3] == force_magnitude]
            coordinates = np.array([np.sin(data[:, 0]), np.cos(data[:, 0])]).transpose() * (data[:, 1:2])#-self.median_spring_length_at_rest)
            plt.scatter(coordinates[:, 0], coordinates[:, 1], label=np.round(force_magnitude,3), alpha=0.5)
        plt.legend()
        plt.show()

    def create_calibration_model(self):
        number_degrees = list(range(1, 2))
        plt_mean_squared_error = []
        plt_r_squared = []
        best_degrees = []
        X_train, X_test, y_train, y_test = train_test_split(self.data[:, 0:2], self.data[:, 2:4], test_size=0.2)
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
        pickle.dump(self.model, open(os.path.join(self.output_path, "calibration_model.pkl"), 'wb'))
        y_prediction = self.model.predict(X_test)
        print("r squared (before removing out layers): ", self.model.score(X_test, y_test))
        print("mean squared error (before removing out layers): ", mean_squared_error(y_test, y_prediction, squared=False))
        print("Model saved to: ", os.path.join(self.output_path,"calibration_model.pkl"))
        print("-"*20)

    def post_processing(self,directory):
        self.current_dir = directory
        self.load_data(self.current_dir)
        self.create_missing_perspective_squares()
        self.rearrange_perspective_squares_order()
        self.correct_perspectives()
        self.calc_distances()
        self.repeat_values()
        self.calc_angle()

    def load_data(self,directory):
        print(directory)
        perspective_squares_sizes_height = np.loadtxt(os.path.join(directory, "perspective_squares_sizes_height.csv"), delimiter=",")
        perspective_squares_sizes_width = np.loadtxt(os.path.join(directory, "perspective_squares_sizes_width.csv"), delimiter=",")
        self.perspective_squares_sizes = np.stack((np.expand_dims(perspective_squares_sizes_height,1), np.expand_dims(perspective_squares_sizes_width,1)), axis=2)
        perspective_squares_coordinates_x = np.loadtxt(os.path.join(directory, "perspective_squares_coordinates_y.csv"), delimiter=",") # TODO: fix this
        perspective_squares_coordinates_y = np.loadtxt(os.path.join(directory, "perspective_squares_coordinates_x.csv"), delimiter=",") # TODO: fix this
        self.perspective_squares_coordinates = np.stack((perspective_squares_coordinates_x, perspective_squares_coordinates_y), axis=2)
        self.perspective_squares_squareness = np.loadtxt(os.path.join(directory, "perspective_squares_squareness.csv"), delimiter=",")
        # self.norm_size = np.median((perspective_squares_sizes_height*perspective_squares_sizes_width)[self.perspective_squares_squareness < 0.001])
        free_ends_coordinates_x = np.loadtxt(os.path.join(directory, "free_ends_coordinates_x.csv"), delimiter=",")
        free_ends_coordinates_y = np.loadtxt(os.path.join(directory, "free_ends_coordinates_y.csv"), delimiter=",")
        self.free_ends_coordinates = np.stack((np.expand_dims(free_ends_coordinates_x,1), np.expand_dims(free_ends_coordinates_y,1)), axis=2)
        fixed_ends_coordinates_x = np.loadtxt(os.path.join(directory, "fixed_ends_coordinates_x.csv"), delimiter=",")
        fixed_ends_coordinates_y = np.loadtxt(os.path.join(directory, "fixed_ends_coordinates_y.csv"), delimiter=",")
        self.fixed_ends_coordinates = np.stack((np.expand_dims(fixed_ends_coordinates_x,1), np.expand_dims(fixed_ends_coordinates_y,1)), axis=2)
        needle_part_coordinates_x = np.loadtxt(os.path.join(directory, "needle_part_coordinates_x.csv"), delimiter=",")
        needle_part_coordinates_y = np.loadtxt(os.path.join(directory, "needle_part_coordinates_y.csv"), delimiter=",")
        needle_part_coordinates = np.stack((needle_part_coordinates_x, needle_part_coordinates_y), axis=2)
        self.object_center_coordinates = needle_part_coordinates[:, 0, :]
        self.needle_tip_coordinates = needle_part_coordinates[:, -1, :]

    def create_missing_perspective_squares(self, threshold=0.01):
        real_squares = (self.perspective_squares_squareness < threshold)
        self.perspective_squares_coordinates[~real_squares] = np.nan
        self.reference_coordinates = self.perspective_squares_coordinates[real_squares.all(axis=1)][0]
        print(self.reference_coordinates)
        # try:
        # except:
        #     self.reference_coordinates = self.reference_coordinates
        diff = self.perspective_squares_coordinates - self.reference_coordinates[np.newaxis, :]
        median_real_squares_diff = np.nanmedian(diff, axis=1)
        predicted_square_coordinates = np.repeat(self.reference_coordinates[np.newaxis, :], self.perspective_squares_coordinates.shape[0], axis=0)
        predicted_square_coordinates += median_real_squares_diff[:, np.newaxis, :]
        self.perspective_squares_coordinates[~np.isnan(predicted_square_coordinates)] = predicted_square_coordinates[~np.isnan(predicted_square_coordinates)]

    def rearrange_perspective_squares_order(self):
        print(np.nanmedian(self.perspective_squares_coordinates, axis=0))
        x_assort = np.argsort(self.perspective_squares_coordinates[:, :, 0], axis=1)
        y_assort = np.argsort(self.perspective_squares_coordinates[:, :, 1], axis=1)
        for count, (frame_x_assort, frame_y_assort) in enumerate(zip(x_assort, y_assort)):
            if not np.any(np.isnan(self.perspective_squares_coordinates[count])):
                top_left_column = set(frame_x_assort[:2]).intersection(set(frame_y_assort[:2])).pop()
                top_right_column = set(frame_x_assort[2:]).intersection(set(frame_y_assort[:2])).pop()
                bottom_right_column = set(frame_x_assort[2:]).intersection(set(frame_y_assort[2:])).pop()
                bottom_left_column = set(frame_x_assort[:2]).intersection(set(frame_y_assort[2:])).pop()
                rearrangement = np.array([top_left_column, bottom_left_column, top_right_column, bottom_right_column])
                if np.any(rearrangement != np.array([0, 1, 3, 2])):
                    print("rearrangement", rearrangement)
                self.perspective_squares_coordinates[count] = self.perspective_squares_coordinates[count, rearrangement, :]

    def correct_perspectives(self):
        PTMs = utils.create_projective_transform_matrix(self.perspective_squares_coordinates)
        self.fixed_ends_coordinates = utils.apply_projective_transform(self.fixed_ends_coordinates, PTMs)
        self.free_ends_coordinates = utils.apply_projective_transform(self.free_ends_coordinates, PTMs)
        self.object_center_coordinates = utils.apply_projective_transform(self.object_center_coordinates, PTMs)
        self.needle_tip_coordinates = utils.apply_projective_transform(self.needle_tip_coordinates, PTMs)
        self.perspective_squares_coordinates = utils.apply_projective_transform(self.perspective_squares_coordinates, PTMs)
        # self.perspective_squares_corner_coordinates = utils.apply_projective_transform(self.perspective_squares_corner_coordinates, PTMs)

    def calc_distances(self):
        object_center = np.repeat(self.object_center_coordinates[:, np.newaxis, :], self.num_of_springs, axis=1)
        needle_tip_coordinates = np.repeat(self.needle_tip_coordinates[:, np.newaxis, :], self.num_of_springs, axis=1)
        self.needle_length = np.nanmedian(np.linalg.norm(self.needle_tip_coordinates - self.object_center_coordinates, axis=1))
        self.object_center_to_free_end_distance = np.linalg.norm(self.free_ends_coordinates - object_center, axis=2)
        self.object_center_to_fixed_end_distance = np.linalg.norm(self.fixed_ends_coordinates - object_center, axis=2)
        self.object_needle_tip_to_fixed_end_distance = np.linalg.norm(self.fixed_ends_coordinates - needle_tip_coordinates, axis=2)

    def repeat_values(self):
        nest_direction = np.stack((self.fixed_ends_coordinates[:, 0, 0], self.fixed_ends_coordinates[:, 0, 1]-500), axis=1)
        self.nest_direction_repeated = np.repeat(nest_direction[:, np.newaxis, :], self.fixed_ends_coordinates.shape[1], axis=1)
        self.object_center_repeated = copy.copy(np.repeat(self.object_center_coordinates[:, np.newaxis, :], self.fixed_ends_coordinates.shape[1], axis=1))

    def calc_angle(self):
        self.fixed_end_angle_to_nest = utils.calc_angle_matrix(self.object_center_repeated, self.fixed_ends_coordinates, self.nest_direction_repeated)+np.pi

    def calc_pulling_angle(self, weight):
        self.pulling_angle = utils.calc_pulling_angle_matrix( self.free_ends_coordinates, self.object_center_repeated, self.fixed_ends_coordinates)
        prediction_pulling_angle = utils.norm_values(self.pulling_angle, self.fixed_end_angle_to_nest, self.models_angles)
        self.pulling_angle -= prediction_pulling_angle
        if weight == 0:
            self.median_angle_at_rest = np.nanmedian(self.pulling_angle)
        self.pulling_angle -= self.median_angle_at_rest

    def calc_spring_length(self, weight):
        self.spring_length = np.linalg.norm(self.free_ends_coordinates - self.fixed_ends_coordinates, axis=2)
        prediction_spring_length = utils.norm_values(self.spring_length, self.fixed_end_angle_to_nest, self.models_lengths)
        self.spring_length /= prediction_spring_length
        # if weight== 0:
        #     self.spring_length *= 0.70

    def calc_calibration_force(self, mass1):
        self.calibration_force_direction = self.fixed_end_angle_to_nest.flatten() - np.pi
        mass1 *= 1e-3  # convert to kg
        G = 6.67430e-11
        mass2 = 5.972e24  # Mass of the Earth (in kilograms)
        distance = 6371000  # Approximate radius of the Earth (in meters)
        force = (G * mass1 * mass2) / (distance ** 2)
        force_mN = force * 1000
        self.calibration_force_magnitude = np.repeat(force_mN, self.free_ends_coordinates.shape[0])


if __name__ == '__main__':
    # videos_paths = "Z:/Dor_Gabay/ThesisProject/data/1-videos/summer_2023/calibration/plus_0.1/"
    videos_paths = "Z:/Dor_Gabay/ThesisProject/data/1-videos/summer_2023/calibration/plus_0.2/"
    calibration_output_path = "Z:\\Dor_Gabay\\ThesisProject\\data\\2-video_analysis\\summer_2023\\calibration\\"
    # calibration_weights = [0, 0.00364, 0.00967, 0.02355, 0.03424, 0.05675, 0.07668, 0.09281, 0.14015]
    # calibration_weights = [0., 0.00356, 0.01449, 0.01933, 0.02986, 0.04515, 0.05, 0.06307, 0.08473, 0.10512, 0.13058]
    calibration_weights = [0., 0.00356, 0.01449, 0.01933, 0.02986, 0.04515, 0.06307, 0.08473, 0.10512, 0.13058]
    self = CalibrationModeling(calibration_output_path, videos_paths, calibration_weights)

