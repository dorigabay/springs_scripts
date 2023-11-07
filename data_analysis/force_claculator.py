import os
import pickle
import numpy as np
# local imports:
from data_analysis.data_preparation import DataPreparation


class ForceCalculator(DataPreparation):
    def __init__(self, video_paths, data_paths, output_path, calibration_model_path, n_springs=20):
        super().__init__(data_paths, video_paths, n_springs=n_springs)
        self.video_dir = video_paths
        self.output_path = output_path
        self.data_paths = data_paths
        self.calibration_model = pickle.load(open(calibration_model_path, "rb"))
        self.calc_force()
        self.save_data()

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

    def save_data(self):
        for count, set_paths in enumerate(self.sets_video_paths):
            sub_dirs_names = [os.path.basename(os.path.normpath(path)) for path in set_paths]
            set_save_path = os.path.join(self.output_path, f"{sub_dirs_names[0]}-{sub_dirs_names[-1]}")
            os.makedirs(set_save_path, exist_ok=True)
            s, e = self.sets_frames[count][0][0], self.sets_frames[count][-1][1] + 1
            np.savez_compressed(os.path.join(set_save_path, "needle_tip_coordinates.npz"), self.needle_tip_coordinates[s:e])
            np.savez_compressed(os.path.join(set_save_path, "object_center_coordinates.npz"), self.object_center_coordinates[s:e])
            np.savez_compressed(os.path.join(set_save_path, "fixed_ends_coordinates.npz"), self.fixed_ends_coordinates[s:e])
            np.savez_compressed(os.path.join(set_save_path, "free_ends_coordinates.npz"), self.free_ends_coordinates[s:e])
            np.savez_compressed(os.path.join(set_save_path, "perspective_squares_coordinates.npz"), self.perspective_squares_coordinates[s:e])
            np.savez_compressed(os.path.join(set_save_path, "N_ants_around_springs.npz"), self.N_ants_around_springs[s:e])
            np.savez_compressed(os.path.join(set_save_path, "size_ants_around_springs.npz"), self.size_ants_around_springs[s:e])
            np.savez_compressed(os.path.join(set_save_path, "ants_attached_labels.npz"), self.ants_attached_labels[s:e])
            np.savez_compressed(os.path.join(set_save_path, "fixed_end_angle_to_nest.npz"), self.fixed_end_angle_to_nest[s:e])
            np.savez_compressed(os.path.join(set_save_path, "object_fixed_end_angle_to_nest.npz"), self.object_fixed_end_angle_to_nest[s:e])
            np.savez_compressed(os.path.join(set_save_path, "force_direction.npz"), self.force_direction[s:e])
            np.savez_compressed(os.path.join(set_save_path, "force_magnitude.npz"), self.force_magnitude[s:e])
            np.savez_compressed(os.path.join(set_save_path, "tangential_force.npz"), self.tangential_force[s:e])
            np.savez_compressed(os.path.join(set_save_path, "missing_info.npz"), self.missing_info[s:e])
        pickle.dump(self.sets_frames, open(os.path.join(self.output_path, "sets_frames.pkl"), "wb"))
        pickle.dump(self.sets_video_paths, open(os.path.join(self.output_path, "sets_video_paths.pkl"), "wb"))
        print("Saved data to: ", self.output_path)


if __name__ == "__main__":
    spring_type = "plus_0.1"
    video_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\1-videos\\summer_2023\\experiment\\{spring_type}\\"
    video_analysis_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\2-video_analysis\\summer_2023\\experiment\\{spring_type}_final\\"
    data_analysis_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\3-data_analysis\\summer_2023\\experiment\\{spring_type}_final\\"
    calibration_output_path = "Z:\\Dor_Gabay\\ThesisProject\\data\\2-video_analysis\\summer_2023\\calibration\\"
    calibration_model_path = os.path.join(calibration_output_path, spring_type, "calibration_model.pkl")
    video_analysis_paths = [root for root, dirs, files in os.walk(video_analysis_dir) if not dirs]
    video_analysis_paths = video_analysis_paths[:1]
    self = ForceCalculator(video_dir, video_analysis_paths, data_analysis_dir, calibration_model_path)
