import os
import glob
import numpy as np
import pickle
from weights_calibration_modeling import CalibrationModeling
from data_analysis.utils import difference
import cv2


class WeightsCalibration:
    def __init__(self, videos_path, output_path, weights=None, nCPU=5):
        self.videos_path = videos_path
        self.output_path = output_path
        self.nCPU = nCPU
        self.calibration_name = os.path.basename(os.path.normpath(self.videos_path))
        self.videos_names = [os.path.basename(x).split(".")[0] for x in
                             glob.glob(os.path.normpath(os.path.join(self.videos_path, "*.MP4")))]
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
                       f" --collect_crop"\
                       #f" --complete_unanalyzed"
        os.system(command_line)
        print("Removing frames where the object is not moving...")
        for video_name in self.videos_names:
            data_path = os.path.join(self.output_path, self.calibration_name, video_name)
            frames_to_keep = self.find_frames_to_keep(data_path)
            # print(len(frames_to_keep))
            # cropping_coordinates = [0, 1920, 0, 1080]
            # cropping_coordinates = [0,1080,0,1920]
            # video_path = os.path.join(self.videos_path, video_name + ".MP4")yes, frames_to_keep=frames_to_keep)
            csv_files = glob.glob(os.path.join(data_path, "raw_analysis", "*.csv"))
            for csv_file in csv_files:
                csv_data = np.loadtxt(csv_file, delimiter=",")
                csv_data = csv_data[frames_to_keep]
                sliced_output_path = os.path.join(data_path, "raw_analysis", "sliced_data")
                os.makedirs(sliced_output_path, exist_ok=True)
                np.savetxt(os.path.join(sliced_output_path, os.path.basename(csv_file)), csv_data, delimiter=",")

    def find_frames_to_keep(self, data_path):
        free_ends_coordinates_x = np.loadtxt(os.path.join(data_path, "raw_analysis", "free_ends_coordinates_x.csv"),
                                             delimiter=",")#[:, 0]

        free_ends_coordinates_y = np.loadtxt(os.path.join(data_path, "raw_analysis", "free_ends_coordinates_y.csv"),
                                             delimiter=",")#[:, 0]
        free_end_coordinates = np.array([free_ends_coordinates_x, free_ends_coordinates_y]).T
        free_end_coordinates_diff = np.sum(np.abs(difference(free_end_coordinates, spacing=10)), axis=1)
        still_frames = np.where(np.abs(free_end_coordinates_diff) < 10)[0]
        return still_frames

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
            weights_sum = 0
            for count, weight in enumerate(weights):
                weights_sum += weight
                weights[count] = weights_sum
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

    def slice_video(self, video_path, output_path, cropping_coordinates=None, frames_to_keep=None):
        print(f"Slicing video: {video_path}")
        # read the video file
        cap = cv2.VideoCapture(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(output_path, os.path.basename(video_path))
        resoloution = (
            cropping_coordinates[3] - cropping_coordinates[2], cropping_coordinates[1] - cropping_coordinates[0])
        print(resoloution)
        out = cv2.VideoWriter(output_path, fourcc, fps, resoloution)
        for i in range(n_frames):
            ret, frame = cap.read()
            if i in frames_to_keep:
                if cropping_coordinates is not None:
                    frame = frame[cropping_coordinates[0]:cropping_coordinates[1],
                            cropping_coordinates[2]:cropping_coordinates[3]]
                    cv2.imshow('frame', frame)
                    cv2.waitKey(0)
                out.write(frame)
        out.release()
        cap.release()


if __name__ == "__main__":
    calibration_videos_path = "Z:/Dor_Gabay/ThesisProject/data/videos/calibration_plus0/"
    calibration_output_path = "Z:/Dor_Gabay/ThesisProject/data/calibration/"
    calibration_weights = [0.10482, 0.00665, 0.0068, 0.0077, 0.0219, 0.0234, 0.0388, 0.0240, 0.0275]
    # calibration_weights = [0.10530, 0.00966, 0.01204, 0.01775, 0.02438, 0.02568, 0.02742, 0.03380, 0.03060]
    WeightsCalibration(calibration_videos_path, calibration_output_path, calibration_weights, nCPU=9)
