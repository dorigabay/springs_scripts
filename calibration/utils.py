import os
import pickle
import numpy as np
import cv2
from general_video_scripts.collect_color_parameters import neutrlize_colour
from video_analysis.springs_detector import Springs
from scipy.ndimage import label, sum


def collect_normalization_length(video_path, output_path, staring_frame=0, n_frames=100):
    # TODO: make sure that this function is not needed anymore, and delete it, and delete "if __name__ == "__main__":"
    print("-" * 50)
    print(f"collecting for: {video_path}")
    video_name = video_path.split("\\")[-1].split(".")[0]
    parameters_path = os.path.join("\\".join(video_path.split("\\")[:-2]), "parameters",
                                   f"{video_name}_video_parameters.pickle")
    with open(os.path.normpath(parameters_path), "rb") as f:
        video_parameters = pickle.load(f)[video_path]
    video_parameters["n_springs"] = 20
    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, staring_frame)
    output_path = os.path.join(output_path, video_name)
    print(f"output path: {output_path}")
    areas_medians = []
    blue_areas_medians = []
    for i in range(n_frames):
        ret, frame = video.read()
        frame = neutrlize_colour(frame)
        springs = Springs(video_parameters, image=frame, previous_detections=None)
        fixed_ends_labeled = springs.bundles_labeled
        fixed_ends_labeled[np.isin(fixed_ends_labeled, springs.fixed_ends_bundles_labels, invert=True)] = 0
        fixed_ends_labeled, num_labels = label(fixed_ends_labeled)
        areas = sum(fixed_ends_labeled, fixed_ends_labeled, index=np.arange(1, num_labels + 1)) / np.arange(1,
                                                                                                            num_labels + 1)
        blue_labeled, num_labels = label(springs.mask_blue_full)
        blue_areas = sum(blue_labeled, blue_labeled, index=np.arange(1, num_labels + 1)) / np.arange(1, num_labels + 1)
        if len(areas) > 0 and len(blue_areas) > 0:
            median_area = np.median(areas)
            blue_median_area = np.median(blue_areas)
            areas_medians.append(median_area)
            blue_areas_medians.append(blue_median_area)
    # save the median area in data folder
    # median_medians = np.median(areas_medians)
    blue_median_medians = np.median(blue_areas_medians)
    with open(os.path.join(output_path, "blue_median_area.pickle"), "wb") as f:
        pickle.dump(blue_median_medians, f)
    print(f"Finished collecting for: {video_path}")
    return blue_median_medians


def bound_angle(angle):
    above_2pi = angle > 2 * np.pi
    below_0 = angle < 0
    angle[above_2pi] = angle[above_2pi] - 2 * np.pi
    angle[below_0] = angle[below_0] + 2 * np.pi
    return angle


def load_first_frame(video_path):
    video = cv2.VideoCapture(video_path)
    ret, frame = video.read()
    return frame


if __name__ == "__main__":
    videos_path = "Z:\\Dor_Gabay\\ThesisProject\\data\\videos\\15.9.22\\plus0.3mm_force\\"
    output_path = "Z:\\Dor_Gabay\\ThesisProject\\data\\exp_collected_data\\15.9.22\\plus0.3mm_force\\"
    videos_paths = []
    # output_paths = []
    for vid in os.listdir(videos_path):
        if vid.endswith(".MP4"):
            vidname = vid.split(".")[0]
            video_path = os.path.join(videos_path, vid)
            # output_path = os.path.join(output_path, vidname)
            videos_paths.append(video_path)
            # output_paths.append(output_path)
    # parrallelize
    # import multiprocessing as mp
    # pool = mp.Pool(8)
    # pool.starmap(collect_normalization_length, zip(videos_paths, output_paths))
    for video_path in videos_paths[7:]:
        collect_normalization_length(video_path, output_path, staring_frame=0, n_frames=100)
