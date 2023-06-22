import os
import glob
import numpy as np
import pickle
from weights_calibration_modeling import CalibrationModeling
from data_analysis.utils import difference


class WeightsCalibration:
    def __init__(self, videos_path, output_path, weights, nCPU=5):
        self.videos_path = videos_path
        self.output_path = output_path
        self.nCPU = nCPU
        self.calibration_name = os.path.basename(os.path.normpath(self.videos_path))
        self.videos_names = [os.path.basename(x).split(".")[0] for x in
                             glob.glob(os.path.normpath(os.path.join(self.videos_path, "*.MP4")))]
        self.analyze_calibration_videos()
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
                       f" --collect_crop"
                       # f" --complete_unanalyzed"
        os.system(command_line)
        print("Removing frames that the object is not moving...")
        for video_name in self.videos_names:
            data_path = os.path.join(self.output_path, self.calibration_name, video_name)
            frames_to_keep = self.find_frames_to_keep(data_path)
            csv_files = glob.glob(os.path.join(data_path,"raw_analysis","*.csv"))
            for csv_file in csv_files:
                csv_data = np.loadtxt(csv_file,delimiter=",")
                csv_data = csv_data[frames_to_keep,:]
                np.savetxt(csv_file,csv_data,delimiter=",")

    def find_frames_to_keep(self, data_path):
        free_ends_coordinates_x = np.loadtxt(os.path.join(data_path, "raw_analysis", "free_ends_coordinates_x.csv"), delimiter=",")
        free_ends_coordinates_y = np.loadtxt(os.path.join(data_path, "raw_analysis", "free_ends_coordinates_y.csv"), delimiter=",")
        free_end_coordinates = np.array([free_ends_coordinates_x, free_ends_coordinates_y]).T
        free_end_coordinates_diff = np.sum(np.abs(difference(free_end_coordinates, spacing=10)),axis=1)
        still_frames = np.where(np.abs(free_end_coordinates_diff) < 10)[0]
        return still_frames

    def save_sum_weights(self, weights):
        """
        First weight is the weight of the object without any weights on it.
        The other weights are the weights that were added to the object.
        """
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
        data_paths = [os.path.join(self.output_path, self.calibration_name, video_name) for video_name in self.videos_names]
        model_output_path = os.path.join(self.output_path, self.calibration_name)
        weights_only = self.weights-self.weights[0]
        CalibrationModeling(data_paths, model_output_path, weights_only, self.videos_path)


if __name__ == "__main__":
    calibration_videos_path = "Z:/Dor_Gabay/ThesisProject/data/videos/calibration_plus0.5/"
    calibration_output_path = "Z:/Dor_Gabay/ThesisProject/data/calibration/"
    calibration_weights = [0.10804, 0.00731, 0.01239, 0.01832, 0.03023, 0.03182, 0.03351, 0.03501, 0.03894]
    WeightsCalibration(calibration_videos_path, calibration_output_path, calibration_weights, nCPU=5)










        # parameters_path = os.path.join(os.path.join(dir_path,'unsliced_videos'), "parameters")
        # parameters = get_parameters(parameters_path, video_path)
        # output_path = os.path.join(dir_path,"sliced_videos")
        # os.makedirs(output_path, exist_ok=True)
        # slice_video(video_path, output_path, frames_to_keep=frames_to_keep,
        #             cropping_coordinates=parameters["crop_coordinates"])

    # # copy parameters folder to sliced_videos
    # parameters_path = os.path.normpath(os.path.join(os.path.join(dir_path,'unsliced_videos'), "parameters"))
    # output_path = os.path.normpath(os.path.join(dir_path,"sliced_videos"))
    # os.makedirs(os.path.join(parameters_path,'parameters'), exist_ok=True)
    # os.system(f"xcopy {parameters_path} {os.path.join(output_path,'parameters')} /E /I /Y")
    # unanalyzed_videos_path = os.path.normpath(os.path.join(os.path.join(dir_path,'unsliced_videos'), "Unanalyzed_videos.txt"))
    # os.system(f"xcopy {unanalyzed_videos_path} {output_path} /E /I /Y")
    #
    # # open the "Unanalyzed_videos.txt" file and in each line change 'unsliced_videos' to 'sliced_videos'
    # with open(os.path.join(output_path,"Unanalyzed_videos.txt"),'r') as f:
    #     lines = f.readlines()
    #     line_parts = [line.split("\\") for line in lines]
    #     new_lines = ["Z:\\"+os.path.join(*line_parts[i][1:-2],"sliced_videos",line_parts[i][-1]) for i in range(len(line_parts))]
    # with open(os.path.join(output_path,"Unanalyzed_videos.txt"),'w') as f:
    #     for line in new_lines:
    #         f.write(line)
    #
    # # open each parameters pickle file and change the path to the video
    # parameters_path = os.path.normpath(os.path.join(os.path.join(dir_path,'sliced_videos'), "parameters"))
    # parameters_files = glob.glob(os.path.join(parameters_path,"*.pickle"))
    # import pickle
    # for file in parameters_files:
    #     with open(file,'rb') as f:
    #         parameters_dict = pickle.load(f)
    #         old_video_path = list(parameters_dict.keys())[0]
    #         vid_name = os.path.basename(old_video_path).split(".")[0].split("_")[0]
    #         new_video_path = os.path.normpath(os.path.join(os.path.join(dir_path,'sliced_videos'),vid_name+".MP4"))
    #         crop_coordinates = parameters_dict[old_video_path]["crop_coordinates"]
    #         zero_crop_coordinates = [0,crop_coordinates[1]-crop_coordinates[0],0,crop_coordinates[3]-crop_coordinates[2]]
    #         parameters_dict[old_video_path]["crop_coordinates"] = zero_crop_coordinates
    #         new_paramerters_dict = {new_video_path:parameters_dict[old_video_path]}
    #     with open(file,'wb') as f:
    #         pickle.dump(new_paramerters_dict,f)

# # second analysis
# output_dir = "Z:/Dor_Gabay/ThesisProject/data/calibration/post_slicing/"
# command_line2 = f"python Z:\Dor_Gabay\ThesisProject\scripts\springs_scripts\command_line_operator.py --dir_path {os.path.normpath(os.path.join(dir_path,'sliced_videos'))} --output_dir {output_dir} --nCPU {nCPU} --complete_unanalyzed --iter_dir"
# os.system(command_line2)
#
# import utils
# for video in glob.glob(os.path.join(os.path.normpath(os.path.join(dir_path,'sliced_videos')),"*.MP4")):
#     utils.collect_normalization_length(video,os.path.join(output_dir,calibration_videos_dirname,"sliced_videos"))