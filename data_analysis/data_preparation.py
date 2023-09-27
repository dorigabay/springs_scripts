import os
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# local imports:
from data_analysis import utils
from post_processing import PostProcessing
from data_analysis.ant_tracking import process_ant_tracking_data


class CalculateForce:
    def __init__(self, video_dir, data_paths, output_path):
        self.video_dir = video_dir
        self.output_path = output_path
        self.data_paths = data_paths
        data = PostProcessing(data_paths, video_dir, n_springs=20)
        self.calibration_model = pickle.load(open(calibration_model_path, "rb"))
        self.create_bias_correction_models(data)
        self.calc_spring_length(data)
        self.calc_pulling_angle(data)
        self.calc_force()
        self.save_data(data)
        self.test_correlation(data)
        # process_ant_tracking_data(data, self.output_path, restart=False)

    def create_bias_correction_models(self, data):
        y_length = np.linalg.norm(data.free_ends_coordinates - data.fixed_ends_coordinates, axis=2)
        y_angle = utils.calc_pulling_angle_matrix(data.fixed_ends_coordinates, data.object_center_repeated, data.free_ends_coordinates)
        angles_to_nest = np.expand_dims(data.fixed_end_angle_to_nest, axis=2)
        X = np.concatenate((np.sin(angles_to_nest), np.cos(angles_to_nest)), axis=2)
        idx = data.rest_bool
        self.models_lengths, self.models_angles = [], []
        not_nan_idx = ~(np.isnan(y_angle) + np.isnan(X).any(axis=2)) * idx
        for col in range(y_length.shape[1]):
            X_fit = X[not_nan_idx[:, col], col]
            y_length_fit = y_length[not_nan_idx[:, col], col]
            y_angle_fit = y_angle[not_nan_idx[:, col], col]
            model_length = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
            model_angle = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
            self.models_lengths.append(model_length.fit(X_fit, y_length_fit))
            self.models_angles.append(model_angle.fit(X_fit, y_angle_fit))

    def calc_pulling_angle(self, data):
        self.pulling_angle = utils.calc_pulling_angle_matrix(data.fixed_ends_coordinates, data.object_center_repeated, data.free_ends_coordinates)
        pulling_angle_prediction = utils.norm_values(self.pulling_angle, data.fixed_end_angle_to_nest, self.models_angles)
        self.pulling_angle -= pulling_angle_prediction
        for count, set_idx in enumerate(data.sets_frames):
            start, end = set_idx[0][0], set_idx[-1][1]
            rest_pull_angle = np.copy(self.pulling_angle[start:end])
            rest_pull_angle[~data.rest_bool[start:end]] = np.nan
            self.pulling_angle[start:end] -= np.nanmedian(rest_pull_angle, axis=0)

    def calc_spring_length(self, data):
        self.spring_length = np.linalg.norm(data.free_ends_coordinates - data.fixed_ends_coordinates, axis=2)
        spring_length_prediction = utils.norm_values(self.spring_length, data.fixed_end_angle_to_nest, self.models_lengths)
        smallest_zero_for_np_float64 = np.finfo(np.float64).tiny
        self.spring_length[spring_length_prediction != 0] /= spring_length_prediction[spring_length_prediction != 0]
        self.spring_length[spring_length_prediction == 0] /= smallest_zero_for_np_float64

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

    def save_data(self, data):
        for count, set_paths in enumerate(data.sets_video_paths):
            sub_dirs_names = [os.path.basename(os.path.normpath(path)) for path in set_paths]
            set_save_path = os.path.join(self.output_path, f"{sub_dirs_names[0]}-{sub_dirs_names[-1]}")
            os.makedirs(set_save_path, exist_ok=True)
            print("-" * 60)
            print("saving data to:", set_save_path)
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
            np.savez_compressed(os.path.join(set_save_path, "missing_info.npz"), np.isnan(data.free_ends_coordinates[start:end, :, 0]))
        pickle.dump(data.sets_frames, open(os.path.join(self.output_path, "sets_frames.pkl"), "wb"))
        pickle.dump(data.sets_video_paths, open(os.path.join(self.output_path, "sets_video_paths.pkl"), "wb"))
        pickle.dump(data.num_of_frames_per_video, open(os.path.join(self.output_path, "num_of_frames_per_video.pkl"), "wb"))

    def test_correlation(self, data, sets_idx=(0, -1)):
        first_set_idx, last_set_idx = (sets_idx, sets_idx) if isinstance(sets_idx, int) else sets_idx
        start, end = data.sets_frames[first_set_idx][0][0], data.sets_frames[last_set_idx][-1][1]
        self.calculations(data)
        corr_df = pd.DataFrame({"net_tangential_force": self.net_tangential_force[start:end], "angular_velocity": self.angular_velocity[start:end],
                                "movement_magnitude": self.movement_magnitude[start:end], "movement_direction": self.movement_direction[start:end],
                                "net_force_magnitude": self.net_force_magnitude[start:end], "net_force_direction": self.net_force_direction[start:end]
                                })
        self.corr_df = corr_df.dropna()
        angular_velocity_correlation_score = corr_df.corr()["net_tangential_force"]["angular_velocity"]
        translation_direction_correlation_score = corr_df.corr()["net_force_direction"]["movement_direction"]
        translation_magnitude_correlation_score = corr_df.corr()["net_force_magnitude"]["movement_magnitude"]
        print(f"correlation score between net tangential force and angular velocity: {angular_velocity_correlation_score}")
        print(f"correlation score between net force direction and translation direction: {translation_direction_correlation_score}")
        print(f"correlation score between net force magnitude and translation magnitude: {translation_magnitude_correlation_score}")

    def calculations(self, data):
        # self.force_magnitude[~np.isnan(self.force_magnitude)*self.rest_bool] -= np.nanmean(self.force_magnitude[~np.isnan(self.force_magnitude)*self.rest_bool])
        # self.force_direction[~np.isnan(self.force_direction)*self.rest_bool] -= np.nanmean(self.force_direction[~np.isnan(self.force_direction)*self.rest_bool])
        self.calc_net_force(data)
        self.angular_velocity = utils.calc_angular_velocity(data.fixed_end_angle_to_nest, diff_spacing=20) / 20
        self.angular_velocity = np.where(np.isnan(self.angular_velocity).all(axis=1), np.nan, np.nanmedian(self.angular_velocity, axis=1))
        self.movement_direction, self.movement_magnitude = utils.calc_translation_velocity(data.object_center_coordinates, spacing=40)
        self.net_force_direction = np.array(pd.Series(self.net_force_direction).rolling(window=40, center=True).median())
        self.net_force_magnitude = np.array(pd.Series(self.net_force_magnitude).rolling(window=40, center=True).median())
        self.net_tangential_force = np.array(pd.Series(self.net_tangential_force).rolling(window=5, center=True).median())

    def calc_net_force(self, data):
        horizontal_component = self.force_magnitude * np.cos(self.force_direction + data.fixed_end_angle_to_nest)
        vertical_component = self.force_magnitude * np.sin(self.force_direction + data.fixed_end_angle_to_nest)
        self.net_force_direction = np.arctan2(np.nansum(vertical_component, axis=1), np.nansum(horizontal_component, axis=1))
        self.net_force_magnitude = np.sqrt(np.nansum(horizontal_component, axis=1) ** 2 + np.nansum(vertical_component, axis=1) ** 2)
        self.tangential_force = np.sin(self.force_direction) * self.force_magnitude
        # self.net_tangential_force = np.nansum(self.tangential_force, axis=1)
        self.net_tangential_force = np.where(np.isnan(self.tangential_force).all(axis=1), np.nan, np.nansum(self.tangential_force, axis=1))


if __name__ == "__main__":
    spring_type = "plus_0.2"
    video_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\1-videos\\summer_2023\\{spring_type}\\"
    data_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\2-video_analysis\\summer_2023\\{spring_type}\\"
    data_paths = [root for root, dirs, files in os.walk(data_dir) if not dirs]
    data_paths = data_paths[:5]
    output_path = f"Z:\\Dor_Gabay\\ThesisProject\\data\\3-data_analysis\\summer_2023\\{spring_type}\\"
    calibration_model_path = os.path.join("Z:\\Dor_Gabay\\ThesisProject\\data\\2-video_analysis\\summer_2023\\calibration\\", spring_type, "calibration_model.pkl")
    self = CalculateForce(video_dir, data_paths, output_path)

