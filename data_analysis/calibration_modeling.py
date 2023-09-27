import os
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
from post_processing import PostProcessing


class CalibrationModeling:
    def __init__(self, output_path, videos_dir, weights=None):
        self.videos_dir = videos_dir
        self.calibration_name = os.path.basename(os.path.normpath(self.videos_dir))
        self.output_path = os.path.join(output_path, self.calibration_name)
        self.load_weights(weights)
        self.num_of_springs = 1
        video_names = [os.path.splitext(file)[0] for file in os.listdir(self.videos_dir) if file.endswith(".MP4")]
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
        data = PostProcessing(self.zero_weight_dir, self.videos_dir, n_springs=1)
        y_length = np.linalg.norm(data.free_ends_coordinates - data.fixed_ends_coordinates, axis=2)
        y_angle = utils.calc_pulling_angle_matrix(data.fixed_ends_coordinates, data.object_center_repeated, data.free_ends_coordinates)
        angles_to_nest = np.expand_dims(data.fixed_end_angle_to_nest, axis=2)
        X = np.concatenate((np.sin(angles_to_nest), np.cos(angles_to_nest)), axis=2)
        idx = np.ones((data.free_ends_coordinates.shape[0], 1), dtype=bool)
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
        a = [0, 1, 3, 5, 7, 8, 9]
        for dir, weight in zip(self.directories[a], self.weights[a]):
        # for dir, weight in zip(self.directories, self.weights):
            data = PostProcessing(dir, self.videos_dir, n_springs=1)
            self.calc_pulling_angle(data, weight)
            self.calc_spring_length(data)
            self.calc_calibration_force(data, weight)
            calibration_force_direction = np.append(calibration_force_direction, self.calibration_force_direction)
            calibration_force_magnitude = np.append(calibration_force_magnitude, self.calibration_force_magnitude)
            pulling_angle = np.append(pulling_angle, self.pulling_angle.flatten())
            length = np.append(length, self.spring_length.flatten())
        modeling_data = np.vstack((pulling_angle, length, calibration_force_direction, calibration_force_magnitude)).transpose()
        modeling_data[modeling_data[:, 2] < 0, 0] *= -1
        modeling_data[modeling_data[:, 2] < 0, 2] *= -1
        modeling_data[modeling_data[:, 0] < 0, :] = np.nan
        # max_value = np.nanmax(modeling_data[~(modeling_data[:, 3] == 0), 2])
        max_value = np.pi/2
        modeling_data[modeling_data[:, 2] > max_value, :] = np.nan
        modeling_data = modeling_data[~np.isnan(modeling_data).any(axis=1)]
        modeling_data = np.concatenate((modeling_data, modeling_data), axis=0)
        modeling_data[0:modeling_data.shape[0]//2, [0, 2]] *= -1
        self.data = modeling_data
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
            coordinates = np.array([np.sin(data[:, 0]), np.cos(data[:, 0])]).transpose() * (data[:, 1:2])
            plt.scatter(coordinates[:, 0], coordinates[:, 1], label=np.round(force_magnitude,3), alpha=0.5)
        plt.legend()
        print("Presenting data. Close the plot to continue.")
        plt.show()

    def create_calibration_model(self):
        print("-"*20)
        degree = 1
        print("Polynomial degree of the force calibration model: ", degree)
        X_train, X_test, y_train, y_test = train_test_split(self.data[:, 0:2], self.data[:, 2:4], test_size=0.2)
        self.model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
        self.model.fit(X_train, y_train)
        pickle.dump(self.model, open(os.path.join(self.output_path, "calibration_model.pkl"), 'wb'))
        print("Model saved to: ", os.path.join(self.output_path,"calibration_model.pkl"))
        y_prediction = self.model.predict(X_test)
        print("r squared: ", self.model.score(X_test, y_test))
        print("mean squared error: ", mean_squared_error(y_test, y_prediction, squared=False))
        print("-"*20)

    def calc_pulling_angle(self, data, weight):
        self.pulling_angle = utils.calc_pulling_angle_matrix( data.free_ends_coordinates, data.object_center_repeated, data.fixed_ends_coordinates)
        prediction_pulling_angle = utils.norm_values(self.pulling_angle, data.fixed_end_angle_to_nest, self.models_angles)
        self.pulling_angle -= prediction_pulling_angle
        if weight == 0:
            self.median_angle_at_rest = np.nanmedian(self.pulling_angle)
        self.pulling_angle -= self.median_angle_at_rest

    def calc_spring_length(self, data):
        self.spring_length = np.linalg.norm(data.free_ends_coordinates - data.fixed_ends_coordinates, axis=2)
        prediction_spring_length = utils.norm_values(self.spring_length, data.fixed_end_angle_to_nest, self.models_lengths)
        self.spring_length /= prediction_spring_length
        # if weight== 0:
        #     self.spring_length *= 0.70

    def calc_calibration_force(self, data, mass1):
        self.calibration_force_direction = data.fixed_end_angle_to_nest.flatten() - np.pi
        mass1 *= 1e-3  # convert to kg
        G = 6.67430e-11
        mass2 = 5.972e24  # Mass of the Earth (in kilograms)
        distance = 6371000  # Approximate radius of the Earth (in meters)
        force = (G * mass1 * mass2) / (distance ** 2)
        force_mN = force * 1000
        self.calibration_force_magnitude = np.repeat(force_mN, data.free_ends_coordinates.shape[0])


if __name__ == '__main__':
    spring_type = "plus_0.3"
    videos_paths = f"Z:/Dor_Gabay/ThesisProject/data/1-videos/summer_2023/calibration/{spring_type}/"
    calibration_output_path = "Z:\\Dor_Gabay\\ThesisProject\\data\\2-video_analysis\\summer_2023\\calibration\\"
    # calibration_weights = [0, 0.00364, 0.00967, 0.02355, 0.03424, 0.05675, 0.07668, 0.09281, 0.14015]
    calibration_weights = [0., 0.00356, 0.01449, 0.01933, 0.02986, 0.04515, 0.06307, 0.08473, 0.10512, 0.13058]
    # calibration_weights = [0., 0.00356, 0.01449, 0.01933, 0.02986, 0.04515, 0.08473, 0.10512, 0.13058]
    self = CalibrationModeling(calibration_output_path, videos_paths, calibration_weights)

