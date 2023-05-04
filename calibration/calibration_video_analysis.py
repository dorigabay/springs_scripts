import os
from calibration import video_slicer
import glob
from general_video_scripts.collect_color_parameters import get_parameters

# first analysis
calibration_videos_dirname = "calibration_perfect2"
dir_path = os.path.join("Z:/Dor_Gabay/ThesisProject/data/videos/", calibration_videos_dirname)
output_dir = "Z:/Dor_Gabay/ThesisProject/data/calibration/pre_slicing/"
nCPU = 5
command_line = f"python Z:\Dor_Gabay\ThesisProject\scripts\springs_scripts\command_line_operator.py" \
               f" --dir_path {os.path.normpath(os.path.join(dir_path,'unsliced_videos'))}" \
               f" --output_dir {output_dir}" \
               f" --nCPU {nCPU}" \
               f" --iter_dir" \
               f" --collect_parameters" \
               f" --collect_crop"
               # f" --complete_unanalyzed"
# os.system(command_line)
#
# #video slicing
# videos = glob.glob(os.path.normpath(os.path.join(dir_path,"unsliced_videos","*.MP4")))
# for video_path in videos:
#     video_name = os.path.basename(video_path).split(".")[0]
#     data_path = os.path.join("Z:\\Dor_Gabay\\ThesisProject\\data\\calibration\\pre_slicing\\",calibration_videos_dirname,"unsliced_videos", video_name)
#     frames_to_keep = video_slicer.find_frames_to_keep(data_path)
#     parameters_path = os.path.join(os.path.join(dir_path,'unsliced_videos'), "parameters")
#     parameters = get_parameters(parameters_path, video_path)
#     output_path = os.path.join(dir_path,"sliced_videos")
#     os.makedirs(output_path, exist_ok=True)
#     video_slicer.slice_video(video_path, output_path, frames_to_keep=frames_to_keep,
#                 cropping_coordinates=parameters["crop_coordinates"])
#
# #copy parameters folder to sliced_videos
# parameters_path = os.path.normpath(os.path.join(os.path.join(dir_path,'unsliced_videos'), "parameters"))
# output_path = os.path.normpath(os.path.join(dir_path,"sliced_videos"))
# os.makedirs(os.path.join(parameters_path,'parameters'), exist_ok=True)
# os.system(f"xcopy {parameters_path} {os.path.join(output_path,'parameters')} /E /I /Y")
# unanalyzed_videos_path = os.path.normpath(os.path.join(os.path.join(dir_path,'unsliced_videos'), "Unanalyzed_videos.txt"))
# os.system(f"xcopy {unanalyzed_videos_path} {output_path} /E /I /Y")
# #open the "Unanalyzed_videos.txt" file and in each line change 'unsliced_videos' to 'sliced_videos'
# with open(os.path.join(output_path,"Unanalyzed_videos.txt"),'r') as f:
#     lines = f.readlines()
#     line_parts = [line.split("\\") for line in lines]
#     new_lines = ["Z:\\"+os.path.join(*line_parts[i][1:-2],"sliced_videos",line_parts[i][-1]) for i in range(len(line_parts))]
# with open(os.path.join(output_path,"Unanalyzed_videos.txt"),'w') as f:
#     for line in new_lines:
#         f.write(line)
#
# #open each parameters pickle file and change the path to the video
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
#
# # second analysis
output_dir = "Z:/Dor_Gabay/ThesisProject/data/calibration/post_slicing/"
# command_line2 = f"python Z:\Dor_Gabay\ThesisProject\scripts\springs_scripts\command_line_operator.py --dir_path {os.path.normpath(os.path.join(dir_path,'sliced_videos'))} --output_dir {output_dir} --nCPU {nCPU} --complete_unanalyzed --iter_dir"
# os.system(command_line2)

import utils
for video in glob.glob(os.path.join(os.path.normpath(os.path.join(dir_path,'sliced_videos')),"*.MP4")):
    utils.collect_normalization_length(video,os.path.join(output_dir,calibration_videos_dirname,"sliced_videos"))