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
from data_analysis.data_preparation import DataPreparation


class CalibrationModeling:
    def __init__(self, video_path, output_path, weights=None, videos_idx=None):
        self.num_of_springs = 1
        self.video_path = video_path
        self.calibration_name = os.path.basename(os.path.normpath(self.video_path))
        self.output_path = os.path.join(output_path, self.calibration_name)
        self.load_weights(weights)
        self.data_paths = np.array([root for root, dirs, files in os.walk(self.output_path) if not dirs])
        self.weights = self.weights[videos_idx] if videos_idx is not None else self.weights
        self.data_paths = self.data_paths[videos_idx] if videos_idx is not None else self.data_paths
        self.zero_weight_dir = self.data_paths[np.where(np.array(self.weights) == 0)[0][0]]
        self.load_data()
        self.plot()
        self.create_calibration_model()

    def load_weights(self, weights=None):
        if weights is None:
            self.weights = pickle.load(open(os.path.join(self.output_path, self.calibration_name, "weights.pickle"), 'rb'))
            raise ValueError("No weights were given and no weights were found in the output path.")
        else:
            self.weights = np.array(weights)
            os.makedirs(self.output_path, exist_ok=True)
            pickle.dump(self.weights, open(os.path.join(self.output_path, "weights.pickle"), 'wb'))

    def load_data(self):
        analysis_data = DataPreparation(self.data_paths, self.video_path, n_springs=1)
        weights_repeated = np.concatenate([np.repeat(weight, analysis_data.video_n_frames[count]) for count, weight in enumerate(self.weights)], axis=0)
        magnitude, direction = self.calc_calib_force(analysis_data.fixed_end_angle_to_nest, weights_repeated)
        self.calib_data = np.concatenate((analysis_data.pulling_angle.flatten(), analysis_data.spring_length.flatten(), direction, magnitude), axis=1)
        self.calib_data[self.calib_data[:, 2] < 0, 0] *= -1
        self.calib_data[self.calib_data[:, 2] < 0, 2] *= -1
        self.calib_data[self.calib_data[:, 0] < 0, :] = np.nan
        self.calib_data = self.remove_bin_outliers(self.calib_data, analysis_data.video_n_frames, bins=300, percentile=5, max_angle_value=np.pi*(13/24))
        self.calib_data = self.calib_data[~np.isnan(self.calib_data).any(axis=1)]
        self.calib_data = np.concatenate((self.calib_data, self.calib_data), axis=0)
        self.calib_data[0:self.calib_data.shape[0]//2, [0, 2]] *= -1

    def remove_bin_outliers(self, data, frames_per_weight, bins=200, percentile=5, max_angle_value=np.pi*(13/24)):
        # max_angle_value = np.nanpercentile(data[~(data[:, 3] == 0), 2], 90)
        angle_bins = np.linspace(0, max_angle_value, bins)
        data[data[:, 2] > max_angle_value, :] = np.nan
        for count, n_frames in enumerate(frames_per_weight):
            s, e = frames_per_weight[:count].sum(), frames_per_weight[:count + 1].sum()
            for i in range(len(angle_bins) - 1):
                idx = (data[s:e, 2] > angle_bins[i]) * (data[s:e, 2] < angle_bins[i + 1])
                pa_upper, pa_lower = np.nanpercentile(data[s:e, 0][idx], 100-percentile), np.nanpercentile(data[s:e, 0][idx], percentile)
                l_upper, l_lower = np.nanpercentile(data[s:e, 1][idx], 100-percentile), np.nanpercentile(data[s:e, 1][idx], percentile)
                angle_bin_outliers = (data[s:e, 0] > pa_upper) * idx + (data[s:e, 0] < pa_lower) * idx + (data[s:e, 1] > l_upper) * idx + (data[s:e, 1] < l_lower) * idx
                data[s:e][angle_bin_outliers, :] = np.nan
        return data

    def plot(self):
        for force_magnitude in np.unique(self.calib_data[:, 3]):
            data = self.calib_data[self.calib_data[:, 3] == force_magnitude]
            coordinates = np.array([np.sin(data[:, 0]), np.cos(data[:, 0])]).transpose() * (data[:, 1:2])
            plt.scatter(coordinates[:, 0], coordinates[:, 1], label=np.round(force_magnitude,3), alpha=0.5)
        plt.legend()
        print("Presenting data. Close the plot to continue.")
        plt.show()

    def create_calibration_model(self):
        degree = 1
        print("Polynomial degree of the force calibration model: ", degree)
        X_train, X_test, y_train, y_test = train_test_split(self.calib_data[:, 0:2], self.calib_data[:, 2:4], test_size=0.2)
        self.model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
        self.model.fit(X_train, y_train)
        pickle.dump(self.model, open(os.path.join(self.output_path, "calibration_model.pkl"), 'wb'))
        print("Model saved to: ", os.path.join(self.output_path,"calibration_model.pkl"))
        y_prediction = self.model.predict(X_test)
        print("r squared: ", self.model.score(X_test, y_test))
        print("mean squared error: ", mean_squared_error(y_test, y_prediction, squared=False))

    def calc_calib_force(self, angle_to_nest, mass1):
        mass1 *= 1e-3  # convert to kg
        gravitational_constant = 6.67430e-11  # Gravitational constant (in m^3 kg^-1 s^-2)
        mass2 = 5.972e24  # Mass of the Earth (in kilograms)
        distance = 6371000  # Approximate radius of the Earth (in meters)
        force = (gravitational_constant * mass1 * mass2) / (distance ** 2)
        force_millinewton = force * 1e-6  # convert to mN
        calibration_force_magnitude = force_millinewton
        calibration_force_direction = angle_to_nest.flatten() - np.pi
        return calibration_force_magnitude, calibration_force_direction


if __name__ == '__main__':
    spring_type = "plus_0.1"
    video_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\1-videos\\summer_2023\\experiment\\{spring_type}\\"
    calibration_output_path = "Z:\\Dor_Gabay\\ThesisProject\\data\\2-video_analysis\\summer_2023\\calibration\\"
    print("-" * 60 + "\nCreating calibration model...\n" + "-" * 20)
    calibration_weights = [0., 0.00356, 0.01449, 0.01933, 0.02986, 0.04515, 0.06307, 0.08473, 0.10512, 0.13058]
    # calibration_weights = [0., 0.00356, 0.01449, 0.01933, 0.02986, 0.04515, 0.08473, 0.10512, 0.13058]
    self = CalibrationModeling(video_dir, calibration_output_path, calibration_weights)



    # def create_bias_correction_models(self):
    #     analysis_data = DataPreparation(self.zero_weight_dir, self.video_dir, n_springs=1)
    #     y_length = np.linalg.norm(analysis_data.free_ends_coordinates - analysis_data.fixed_ends_coordinates, axis=2)
    #     y_angle = utils.calc_pulling_angle_matrix(analysis_data.fixed_ends_coordinates, analysis_data.object_center_repeated, analysis_data.free_ends_coordinates)
    #     angles_to_nest = np.expand_dims(analysis_data.fixed_end_angle_to_nest, axis=2)
    #     X = np.concatenate((np.sin(angles_to_nest), np.cos(angles_to_nest)), axis=2)
    #     # rest_bool = analysis_data.N_ants_around_springs == 0
    #     self.model_length = []
    #     self.model_angle = []
    #     for col in range(y_length.shape[1]):
    #         not_nan_idx = ~(np.isnan(y_angle[:, col]) + np.isnan(X[:, col]).any(axis=1) + np.isnan(y_length[:, col]))
    #         X_fit = X[not_nan_idx, col, :]
    #         y_length_fit = y_length[not_nan_idx, col:col+1]
    #         y_angle_fit = y_angle[not_nan_idx, col:col+1]
    #         model_length = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
    #         model_angle = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
    #         self.model_angle.append(model_angle.fit(X_fit, y_angle_fit))
    #         self.model_length.append(model_length.fit(X_fit, y_length_fit))

    # def create_bias_correction_models(self):
    #     print(self.zero_weight_dir)
    #     analysis_data = DataPreparation(self.zero_weight_dir, self.video_dir, n_springs=1)
    #     analysis_data.fixed_ends_coordinates = self.plane_perpective_projection(analysis_data.fixed_ends_coordinates, 0.0025)
    #     analysis_data.object_center_repeated = self.plane_perpective_projection(analysis_data.object_center_repeated, 0.0025)
    #     # spring_length = np.linalg.norm(analysis_data.free_ends_coordinates - analysis_data.object_center_repeated, axis=2)
    #     spring_length = np.linalg.norm(analysis_data.free_ends_coordinates - analysis_data.fixed_ends_coordinates, axis=2)
    #     needle_tip_length = np.linalg.norm(analysis_data.needle_tip_repeated - analysis_data.object_center_repeated, axis=2)
    #     # print("needle_tip_length at row 98", needle_tip_length[98], "needle_tip_length at row 230", needle_tip_length[230], "needle_tip_length at row 537", needle_tip_length[538])
    #     object_center_to_fixed_end_length = np.linalg.norm(analysis_data.fixed_ends_coordinates - analysis_data.object_center_repeated, axis=2)
    #
    #     object_center_to_free_end_length = np.linalg.norm(analysis_data.free_ends_coordinates - analysis_data.object_center_repeated, axis=2)
    #     spring_angle = utils.calc_pulling_angle_matrix(analysis_data.fixed_ends_coordinates, analysis_data.object_center_repeated, analysis_data.free_ends_coordinates)
    #     fixed_to_tip_angle = utils.calc_pulling_angle_matrix(analysis_data.fixed_ends_coordinates, analysis_data.needle_tip_repeated, analysis_data.object_center_repeated)
    #     angles_to_nest = analysis_data.fixed_end_angle_to_nest
    #     nan_idx = np.isnan(spring_length) + np.isnan(analysis_data.fixed_end_angle_to_nest)+ np.isnan(spring_angle)
    #     spring_length = spring_length[~nan_idx]
    #     spring_angle = spring_angle[~nan_idx]
    #     spring_angle -= np.median(spring_angle)
    #     angles_to_nest = angles_to_nest[~nan_idx]
    #     fixed_to_tip_angle = fixed_to_tip_angle[~nan_idx]
    #     needle_tip_length = needle_tip_length[~nan_idx]
    #     object_center_to_fixed_end_length = object_center_to_fixed_end_length[~nan_idx]
    #     object_center_to_free_end_length = object_center_to_free_end_length[~nan_idx]
    #     p0 = [np.max(spring_length) - np.min(spring_length), 1, 0, np.mean(spring_length)]
    #     self.length_model_parameters = curve_fit(utils.sine_function, angles_to_nest, spring_length, p0=p0)[0]
    #     p0 = [np.max(spring_angle) - np.min(spring_angle), 1, 0, np.mean(spring_angle)]
    #     self.angle_model_parameters = curve_fit(utils.sine_function, angles_to_nest, spring_angle, p0=p0)[0]
    #     # make two plot in one figure
    #     plt.scatter(angles_to_nest, object_center_to_fixed_end_length)
    #     plt.show()

        # spring_length_sine_fit = utils.sine_function(angles_to_nest, *self.length_model_parameters) - self.length_model_parameters[3]
        # spring_angle_sine_fit = utils.sine_function(angles_to_nest, *self.angle_model_parameters)
        # print(spring_length.shape)
        # angles_to_nest = np.expand_dims(analysis_data.fixed_end_angle_to_nest, axis=2)
        # X = np.concatenate((np.sin(angles_to_nest), np.cos(angles_to_nest)), axis=2)
        # # idx = np.ones((analysis_data.free_ends_coordinates.shape[0], 1), dtype=bool)
        # not_nan_idx = ~(np.isnan(y_angle) + np.isnan(X).any(axis=2)) #* idx
        # X_fit = X[not_nan_idx[:, 0], 0]
        # y_length_fit = y_length[not_nan_idx[:, 0], 0]
        # y_angle_fit = y_angle[not_nan_idx[:, 0], 0]
        # self.length_model = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
        # self.angle_model = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
        # self.length_model.fit(X_fit, y_length_fit)
        # self.angle_model.fit(X_fit, y_angle_fit)
        # self.calc_pulling_angle(analysis_data)
        # self.calc_spring_length(analysis_data)
        # pulling_angle = utils.calc_pulling_angle_matrix(analysis_data.free_ends_coordinates, analysis_data.object_center_repeated, analysis_data.fixed_ends_coordinates)
        # spring_length = np.linalg.norm(analysis_data.free_ends_coordinates - analysis_data.fixed_ends_coordinates, axis=2)
        # prediction = utils.norm_values(spring_length, analysis_data.fixed_end_angle_to_nest, [self.length_model])
        # half_min_max = (np.nanmin(spring_length) + np.nanmax(spring_length)) / 2
        # self.spring_length_gravity_effect = prediction - half_min_max
        # self.pulling_angle_gravity_effect = utils.norm_values(pulling_angle, analysis_data.fixed_end_angle_to_nest, [self.angle_model])

    # def search_spring_ends_height(self, data, n_iterations=10000):
    #     """
    #     Iteratively search for the spring ends' height (h), by minimizing the fit of a sine wave to the relationship between the spring length and the angle to the nest.
    #     Calculate the projected distances of the spring ends from the center of the frame (spdx, spdy),
    #       using the coordinates of the spring ends (sx, sy) and center of the frame (x0=3840/2, y0=2160/2).
    #     Calculate the displacement distances (sddx, sddy) of the spring ends from center of the frame, using the equations: sddx = spdx*h/(1-h), and sddy = spdy*h/(1-h).
    #     Correct the springs ends coordinates (csx, csy) by adding the displacement distances (sddx, sddy) to the spring ends coordinates (sx,sy).
    #     Calculate the score of the fit of a sine wave to the relationship between the spring length and the angle to the nest,
    #       using the corrected spring ends coordinates (csx, csy).
    #     Repeat the process for different values of h, and choose the value of h that gives the lowest score.
    #     The score is calculated as the sum of the squared differences between the sine wave and the spring length.
    #     Do the minimization using the package scipy.optimize.minimize.
    #     """
    #     def sine_function(x, amplitude, frequency, phase, offset):
    #         return amplitude * np.sin(frequency * x + phase) + offset

            # p0 = [np.max(correct_distance) - np.min(correct_distance), 1, 0, np.mean(correct_distance)]
    #         popt = curve_fit(sine_function, sine_x, correct_distance, p0=p0)[0]
    #
    #         correct_distance = np.linalg.norm(plane1_correct_coordinates - plane2_correct_coordinates, axis=1)
    #         # coefficient_of_variation = (np.std(correct_distance) / np.mean(correct_distance))
    #         # fit to sine wave
    #         sine_fit = sine_function(sine_x, *popt)
    #         r_squared = np.corrcoef(correct_distance, sine_fit)[0, 1] ** 2
    #

    #
    #     def custom_loss(heights, plane1_coordinates, plane2_coordinates, sine_x):
    #         plane1_correct_coordinates = correct_coordinates(heights[0], plane1_coordinates)
    #         plane2_correct_coordinates = correct_coordinates(heights[1], plane2_coordinates)
    #         plane1_correct_distance_to_origin = np.linalg.norm(plane1_correct_coordinates, axis=1)
    #         plane2_correct_distance_to_origin = np.linalg.norm(plane2_correct_coordinates, axis=1)
    #         p0 = [np.max(correct_distance)-np.min(correct_distance), 1, 0, np.mean(correct_distance)]
    #         popt = curve_fit(sine_function, sine_x, correct_distance, p0=p0)[0]
    #
    #         correct_distance = np.linalg.norm(plane1_correct_coordinates - plane2_correct_coordinates, axis=1)
    #         # coefficient_of_variation = (np.std(correct_distance) / np.mean(correct_distance))
    #         # fit to sine wave
    #         sine_fit = sine_function(sine_x, *popt)
    #         r_squared = np.corrcoef(correct_distance, sine_fit)[0, 1] ** 2
    #         print(heights, r_squared)
    #         return r_squared
    #
    #     nan_idx = (np.isnan(self.data_prep.fixed_ends_coordinates) + np.isnan(self.data_prep.free_ends_coordinates)).any(axis=2).flatten()
    #     fixed_end_coordinates = self.data_prep.fixed_ends_coordinates[~nan_idx, 0, :]
    #     free_end_coordinates = self.data_prep.free_ends_coordinates[~nan_idx, 0, :]
    #     angle_to_nest = self.data_prep.fixed_end_angle_to_nest[~nan_idx, 0]
    #     object_center_coordinates = self.data_prep.object_center_coordinates[~nan_idx, :]
    #
    #     initial_guess = np.array([0.01, 0.01])
    #     # heights = np.linspace(0., -0.05, n_iterations)
    #     # fixed_end_scores = np.zeros(n_iterations)
    #     # object_center_scores = np.zeros(n_iterations)
    #     # for count, height in enumerate(heights):
    #     #     fixed_end_scores[count] = custom_loss(height, free_end_coordinates, fixed_end_coordinates)
    #     #     object_center_scores[count] = custom_loss(height, free_end_coordinates, object_center_coordinates)
    #     heights_estimated = minimize(custom_loss, initial_guess, args=(free_end_coordinates, fixed_end_coordinates, angle_to_nest)).x
    #     self.fixed_end_height, self.free_end_height = heights_estimated
    #     # self.fixed_end_height = minimize(custom_loss, initial_guess, args=(free_end_coordinates, fixed_end_coordinates, angle_to_nest)).x[0]
    #     # self.object_center_height = minimize(custom_loss, initial_guess, args=(free_end_coordinates, object_center_coordinates, angle_to_nest)).x[0]
    #     self.fixed_end_height = 0.0025
    #     self.object_center_height = 0.0025
    #     # self.fixed_end_height = heights[np.argmin(fixed_end_scores)]
    #     # self.object_center_height = heights[np.argmin(object_center_scores)]
    #     score = custom_loss(heights_estimated, free_end_coordinates, fixed_end_coordinates, angle_to_nest)
    #     # score = custom_loss(self.fixed_end_height, free_end_coordinates, fixed_end_coordinates, angle_to_nest)
    #     print("fixed_end_height:", self.fixed_end_height, "free_end_height:", self.free_end_height, "score:", score, )#"object_center_height:", self.object_center_height)
    #     fixed_end_correct_coordinates = correct_coordinates(self.fixed_end_height, fixed_end_coordinates)
    #     free_end_correct_coordinates = correct_coordinates(self.fixed_end_height, free_end_coordinates)
    #     # object_center_correct_coordinates = correct_coordinates(self.object_center_height, object_center_coordinates)
    #     # correct_spring_length = np.linalg.norm(fixed_end_correct_coordinates - free_end_coordinates, axis=1)
    #     spring_length = np.linalg.norm(fixed_end_coordinates - free_end_coordinates, axis=1)
    #     correct_spring_length = np.linalg.norm(fixed_end_correct_coordinates - free_end_correct_coordinates, axis=1)
    #     fixed_to_center_length = np.linalg.norm(fixed_end_coordinates - object_center_coordinates, axis=1)
    #     free_to_center_length = np.linalg.norm(free_end_coordinates - object_center_coordinates, axis=1)
    #     correct_fixed_to_center_length = np.linalg.norm(fixed_end_correct_coordinates - object_center_coordinates, axis=1)
    #     correct_free_to_center_length = np.linalg.norm(free_end_correct_coordinates - object_center_coordinates, axis=1)
    #     fixed_end_to_origin_distance = np.linalg.norm(fixed_end_coordinates, axis=1)
    #     correct_fixed_end_to_origin_distance = np.linalg.norm(fixed_end_correct_coordinates, axis=1)
    #
    #     plt.figure(figsize=(20, 10))
    #     plt.subplot(2, 4, 1)
    #     plt.scatter(angle_to_nest, spring_length)
    #     # plt.ylim(35, 50)
    #     plt.title("original spring length")
    #     plt.subplot(2, 4, 2)
    #     plt.scatter(angle_to_nest, correct_spring_length)
    #     # plt.ylim(35, 50)
    #     plt.title("corrected spring length")
    #     plt.show()
    #     plt.subplot(2, 4, 3)
    #     plt.scatter(angle_to_nest, fixed_to_center_length)
    #     # plt.ylim(92, 105)
    #     plt.title("fixed to center length")
    #     plt.show()
    #     plt.subplot(2, 4, 4)
    #     plt.scatter(angle_to_nest, correct_fixed_to_center_length)
    #     # plt.ylim(92, 105)
    #     plt.title("correct fixed to center length")
    #     plt.show()
    #     plt.subplot(2, 4, 5)
    #     plt.scatter(angle_to_nest, free_to_center_length)
    #     # plt.ylim(92, 105)
    #     plt.title("free to center length")
    #     plt.show()
    #     plt.subplot(2, 4, 6)
    #     plt.scatter(angle_to_nest, correct_free_to_center_length)
    #     # plt.ylim(92, 105)
    #     plt.title("correct free to center length")
    #     plt.show()
    #     plt.subplot(2, 4, 7)
    #     plt.scatter(angle_to_nest, fixed_end_to_origin_distance)
    #     # plt.ylim(92, 105)
    #     plt.title("fixed end to origin distance")