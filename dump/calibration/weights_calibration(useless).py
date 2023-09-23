import os
import glob
import numpy as np
import pickle

# local imports:
from data_analysis.calibration_modeling import CalibrationModeling


class WeightsCalibration:
    def __init__(self, videos_path, output_path, weights=None, nCPU=5):
        self.videos_path = videos_path
        self.output_path = output_path
        self.nCPU = nCPU
        self.calibration_name = os.path.basename(os.path.normpath(self.videos_path))
        self.videos_names = [os.path.basename(x).split(".")[0] for x in
                             glob.glob(os.path.normpath(os.path.join(self.videos_path, "*.MP4")))]
        # print("videos_names", self.videos_names)
        # self.analyze_calibration_videos()
        self.save_sum_weights(weights)
        self.model_calibration()

    def analyze_calibration_videos(self):
        print("Analyzing calibration videos...")
        command_line = f"python Z:\Dor_Gabay\ThesisProject\scripts\springs_scripts\command_line_operator.py" \
                       f" --dir_path {os.path.normpath(self.videos_path)}" \
                       f" --output_dir {self.output_path}" \
                       f" --nCPU {self.nCPU}" \
                       f" --iter_dir" \
                       f" --collect_parameters" \
                       #f" --complete_unanalyzed"
        os.system(command_line)

    def save_sum_weights(self, weights):
        """
        First weight is the weight of the object without any weights on it.
        The other weights are the weights that were added to the object.
        """
        if weights is None:
            try:
                self.weights = pickle.load(
                    open(os.path.join(self.output_path, self.calibration_name, "weights.pickle"), 'rb'))
            except:
                raise ValueError("No weights were given and no weights were found in the output path.")
        else:
            self.weights = np.array(weights)
            weights_output_path = os.path.join(self.output_path, self.calibration_name)
            os.makedirs(weights_output_path, exist_ok=True)
            pickle.dump(self.weights, open(os.path.join(weights_output_path, "weights.pickle"), 'wb'))

    def model_calibration(self):
        print("-" * 50)
        print("Creating weights calibration model...")
        data_paths = [os.path.join(self.output_path, self.calibration_name, video_name) for video_name in
                      self.videos_names]
        model_output_path = os.path.join(self.output_path, self.calibration_name)
        weights_only = self.weights - self.weights[0]
        CalibrationModeling(data_paths, weights_only, model_output_path, self.videos_path)


if __name__ == "__main__":
    calibration_videos_dir = "Z:/Dor_Gabay/ThesisProject/data/1-videos/summer_2023/calibration/plus_0.1/"
    calibration_output_path = "Z:\\Dor_Gabay\\ThesisProject\\data\\2-video_analysis\\summer_2023\\calibration\\"
    calibration_weights = [0, 0.00364, 0.00967, 0.02355, 0.03424, 0.05675, 0.07668, 0.09281, 0.14015]
    calib = WeightsCalibration(calibration_videos_dir, calibration_output_path, calibration_weights, nCPU=12)
