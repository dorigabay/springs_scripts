import os
import pickle
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# local imports:
from data_analysis import utils
from data_analysis.data_preparation import DataPreparation


class ForceCalculator:
    def __init__(self, video_dir, data_paths, output_path, calibration_model_path, n_springs=20):
        self.video_dir = video_dir
        self.output_path = output_path
        self.data_paths = data_paths
        analysis_data = DataPreparation(data_paths, video_dir, n_springs=n_springs)
        self.calibration_model = pickle.load(open(calibration_model_path, "rb"))
        self.calc_spring_length(analysis_data)
        self.calc_pulling_angle(analysis_data)
        self.calc_force()
        self.save_data(analysis_data)

    def calc_pulling_angle(self, data):
        # self.pulling_angle = utils.calc_pulling_angle_matrix(data.fixed_ends_coordinates, data.object_center_repeated, data.free_ends_coordinates)
        self.pulling_angle = utils.calc_pulling_angle_matrix(data.fixed_ends_coordinates, data.needle_tip_repeated, data.free_ends_coordinates)
        for count, set_idx in enumerate(data.sets_frames):
            s, e = set_idx[0][0], set_idx[-1][1]
            self.pulling_angle[s:e] -= np.nanmedian(np.where(~data.rest_bool[s:e], np.nan, self.pulling_angle[s:e]))

    def calc_spring_length(self, data):
        self.spring_length = np.linalg.norm(data.free_ends_coordinates - data.fixed_ends_coordinates, axis=2)

    def calc_force(self):
        self.force_direction = np.full(self.pulling_angle.shape, np.nan, dtype=np.float64)
        self.force_magnitude = np.full(self.pulling_angle.shape, np.nan, dtype=np.float64)
        for s in range(self.pulling_angle.shape[1]):
            X = np.stack((self.pulling_angle[:, s], self.spring_length[:, s]), axis=1)
            exclude_idx = np.isnan(X).any(axis=1) + np.isinf(X).any(axis=1)
            X = X[~exclude_idx]
            forces_predicted = self.calibration_model.predict(X)
            self.force_direction[~exclude_idx, s] = forces_predicted[:, 0]
            self.force_magnitude[~exclude_idx, s] = forces_predicted[:, 1]
        self.tangential_force = np.sin(self.force_direction) * self.force_magnitude

    def save_data(self, data):
        for count, set_paths in enumerate(data.sets_video_paths):
            sub_dirs_names = [os.path.basename(os.path.normpath(path)) for path in set_paths]
            set_save_path = os.path.join(self.output_path, f"{sub_dirs_names[0]}-{sub_dirs_names[-1]}")
            os.makedirs(set_save_path, exist_ok=True)
            start, end = data.sets_frames[count][0][0], data.sets_frames[count][-1][1]
            np.savez_compressed(os.path.join(set_save_path, "needle_tip_coordinates.npz"), data.needle_tip_coordinates[start:end])
            np.savez_compressed(os.path.join(set_save_path, "object_center_coordinates.npz"), data.object_center_coordinates[start:end])
            np.savez_compressed(os.path.join(set_save_path, "fixed_ends_coordinates.npz"), data.fixed_ends_coordinates[start:end])
            np.savez_compressed(os.path.join(set_save_path, "free_ends_coordinates.npz"), data.free_ends_coordinates[start:end])
            np.savez_compressed(os.path.join(set_save_path, "perspective_squares_coordinates.npz"), data.perspective_squares_coordinates[start:end])
            np.savez_compressed(os.path.join(set_save_path, "N_ants_around_springs.npz"), data.N_ants_around_springs[start:end])
            np.savez_compressed(os.path.join(set_save_path, "size_ants_around_springs.npz"), data.size_ants_around_springs[start:end])
            np.savez_compressed(os.path.join(set_save_path, "ants_attached_labels.npz"), data.ants_attached_labels[start:end])
            np.savez_compressed(os.path.join(set_save_path, "fixed_end_angle_to_nest.npz"), data.fixed_end_angle_to_nest[start:end])
            np.savez_compressed(os.path.join(set_save_path, "object_fixed_end_angle_to_nest.npz"), data.object_fixed_end_angle_to_nest[start:end])
            np.savez_compressed(os.path.join(set_save_path, "force_direction.npz"), self.force_direction[start:end])
            np.savez_compressed(os.path.join(set_save_path, "force_magnitude.npz"), self.force_magnitude[start:end])
            np.savez_compressed(os.path.join(set_save_path, "tangential_force.npz"), self.tangential_force[start:end])
            np.savez_compressed(os.path.join(set_save_path, "missing_info.npz"), np.isnan(data.free_ends_coordinates[start:end, :, 0]))
        pickle.dump(data.sets_frames, open(os.path.join(self.output_path, "sets_frames.pkl"), "wb"))
        pickle.dump(data.sets_video_paths, open(os.path.join(self.output_path, "sets_video_paths.pkl"), "wb"))
        print("Saved data to: ", self.output_path)


if __name__ == "__main__":
    spring_type = "plus_0.1"
    video_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\1-videos\\summer_2023\\calibration\\{spring_type}\\"
    data_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\2-video_analysis\\summer_2023\\calibration\\{spring_type}\\"
    data_paths = [root for root, dirs, files in os.walk(data_dir) if not dirs]
    data_paths = data_paths[:1]
    output_path = f"Z:\\Dor_Gabay\\ThesisProject\\data\\3-data_analysis\\summer_2023\\calibration\\{spring_type}\\"
    calibration_model_path = os.path.join("Z:\\Dor_Gabay\\ThesisProject\\data\\2-video_analysis\\summer_2023\\calibration\\", spring_type, "calibration_model.pkl")
    self = ForceCalculator(video_dir, data_paths, output_path, calibration_model_path, n_springs=1)



    # def create_bias_correction_models(self, data):
    #     y_length = np.linalg.norm(data.free_ends_coordinates - data.fixed_ends_coordinates, axis=2).flatten()
    #     y_angle = utils.calc_pulling_angle_matrix(data.fixed_ends_coordinates, data.object_center_repeated, data.free_ends_coordinates).flatten()
    #     angles_to_nest = np.expand_dims(data.fixed_end_angle_to_nest, axis=2)
    #     position_from_center = data.object_center_coordinates-data.video_resolution/2
    #     position_from_center = np.expand_dims(position_from_center, axis=1).repeat(20, axis=1)
    #     position_from_center_x = position_from_center[:, :, 0]
    #     position_from_center_y = position_from_center[:, :, 1]
    #     X = np.vstack((np.sin(angles_to_nest).flatten(), np.cos(angles_to_nest).flatten(), position_from_center_x.flatten(),  position_from_center_y.flatten())).T
    #     # X = np.concatenate((np.sin(angles_to_nest), np.cos(angles_to_nest), position_from_center), axis=2).flatten()
    #     idx = data.rest_bool.flatten()
    #     not_nan_idx = ~(np.isnan(y_angle) + np.isnan(X).any(axis=1)) * idx
    #     X_fit = X[not_nan_idx]
    #     y_length_fit = y_length[not_nan_idx].reshape(-1, 1)
    #     y_angle_fit = y_angle[not_nan_idx].reshape(-1, 1)
    #     self.model_length = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
    #     self.model_angle = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
    #     self.model_length.fit(X_fit, y_length_fit)
    #     self.model_angle.fit(X_fit, y_angle_fit)

    # def create_bias_correction_models(self, data):
    #     y_length = np.linalg.norm(data.free_ends_coordinates - data.fixed_ends_coordinates, axis=2)
    #     y_angle = utils.calc_pulling_angle_matrix(data.fixed_ends_coordinates, data.object_center_repeated, data.free_ends_coordinates)
    #     angles_to_nest = np.expand_dims(data.fixed_end_angle_to_nest, axis=2)
    #     X = np.concatenate((np.sin(angles_to_nest), np.cos(angles_to_nest)), axis=2)
    #     # not_nan_idx = ~(np.isnan(y_angle).flatten() + np.isnan(X).any(axis=1).flatten() + np.isnan(y_length).flatten())
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

    # def norm_values(self, explained, data, model):
    #     angles_to_nest = np.expand_dims(data.fixed_end_angle_to_nest, axis=2)
    #     position_from_center = data.object_center_coordinates-data.video_resolution/2
    #     position_from_center = np.expand_dims(position_from_center, axis=1).repeat(20, axis=1)
    #     X = np.concatenate((np.sin(angles_to_nest), np.cos(angles_to_nest), position_from_center), axis=2)
    #     # X = np.concatenate((np.sin(angles_to_nest), np.cos(angles_to_nest)), axis=2)
    #     idx = data.rest_bool
    #     not_nan_idx = ~(np.isnan(explained) + np.isnan(X).any(axis=2)) * idx
    #     prediction_matrix = np.zeros(explained.shape)
    #     for col in range(explained.shape[1]):
    #         X_fit = X[not_nan_idx[:, col], col, :]
    #         prediction_matrix[not_nan_idx[:, col], col] = model.predict(X_fit).flatten()
    #     return prediction_matrix