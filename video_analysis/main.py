import os
import cv2
import glob
import pickle
import datetime
import argparse
from multiprocessing import Pool, cpu_count
# local imports:
from parameters_collector import CollectAnalysisParameters
import calculator
from ants_detector import Ants
from springs_detector import Springs
import perspective_squares
import utils


def parse_args():
    parser = argparse.ArgumentParser(description='Detecting soft pendulum object')
    parser.add_argument('--path', type=str, help='Directory path of the videos. Branched directories are allowed.', required=True)
    parser.add_argument('--output_path', '-o', type=str, help='Path to output directory. Default is the same as the input directory.')
    parser.add_argument("--nCPU", type=int, help='Number of CPU processors to use for multiprocessing', default=1)
    parser.add_argument('--collect_parameters', '-cp', help='', action='store_true')
    parser.add_argument('--continue_from_last', '-con', help='If True, then the program will continue from the last frame analysed', action='store_true')
    parser.add_argument('--starting_frame', '-sf', help='Frame to start with. (for debugging)', type=int, default=None)
    # parser.add_argument('--parameters_path', '-p', help='')
    arguments = vars(parser.parse_args())
    return arguments


def load_parameters(video_path):
    try:
        video_analysis_parameters = pickle.load(open(os.path.join(args["path"], "video_analysis_parameters.pickle"), 'rb'))[os.path.normpath(video_path)]
    except:
        raise ValueError("Video parameters for video: ", video_path, " not found. Please run the script with the flag --collect_parameters (-cp)")
    video_analysis_parameters["starting_frame"] = args["starting_frame"] if args["starting_frame"] is not None else video_analysis_parameters["starting_frame"]
    video_analysis_parameters["continue_from_last"] = args["continue_from_last"]
    sub_dirs = os.path.normpath(video_path).split(args["path"])[1].split(".MP4")[0].split("\\")
    video_analysis_parameters["output_path"] = os.path.join(args["path"], "analysis_output", *sub_dirs)\
        if args["output_path"] is None else os.path.join(args["output_path"], *sub_dirs)
    return video_analysis_parameters


def create_snapshot_data(parameters=None, snapshot_data=None, calculations=None, squares=None, springs=None):
    if snapshot_data is None:
        if parameters["continue_from_last"] and len([f for f in os.listdir(parameters["output_path"]) if f.startswith("snap_data")]) != 0:
            snaps = [f for f in os.listdir(parameters["output_path"]) if f.startswith("snap_data")]
            snapshot_data = pickle.load(open(os.path.join(parameters["output_path"], snaps[-1]), "rb"))
            parameters["starting_frame"] = snapshot_data["frame_count"]
            snapshot_data["current_time"] = datetime.datetime.now().strftime("%d.%m.%Y-%H%M")
        else:
            snapshot_data = {"object_center_coordinates": parameters["object_center_coordinates"][0],
                             "tip_point": None, "springs_angles_reference_order": None,
                             "sum_needle_radius": 0, "analysed_frame_count": 0, "frame_count": 0,"skipped_frames": 0,
                             "current_time": datetime.datetime.now().strftime("%d.%m.%Y-%H%M"),
                             "perspective_squares_coordinates": parameters["perspective_squares_coordinates"]}
    else:
        snapshot_data["object_center_coordinates"] = springs.object_center_coordinates[[1, 0]]
        snapshot_data["tip_point"] = springs.tip_point
        snapshot_data["sum_needle_radius"] += int(springs.object_needle_radius)
        snapshot_data["springs_angles_reference_order"] = calculations.springs_angles_reference_order
        snapshot_data["perspective_squares_coordinates"] = utils.swap_columns(squares.perspective_squares_properties[:, 0:2])
    return snapshot_data


# def cut_access_data(video_n_frames, parameters):
#     print("Cutting access data", parameters["output_path"])
#     import numpy as np
#     data_names = ["perspective_squares_coordinates_x", "perspective_squares_coordinates_y", "perspective_squares_rectangle_similarity", "perspective_squares_squareness",
#                   "N_ants_around_springs", "size_ants_around_springs", "fixed_ends_coordinates_x", "fixed_ends_coordinates_y", "free_ends_coordinates_x",
#                   "free_ends_coordinates_y", "needle_part_coordinates_x", "needle_part_coordinates_y",
#                   "ants_centers_x", "ants_centers_y", "ants_attached_labels", "ants_attached_forgotten_labels"]
#     for data_name in data_names:
#         file = np.loadtxt(os.path.join(parameters["output_path"], data_name+".csv"), delimiter=",")
#         np.savetxt(os.path.join(parameters["output_path"], data_name+".csv"), file[:video_n_frames], delimiter=",")


def main(video_path, parameters):
    snapshot_data = create_snapshot_data(parameters=parameters)
    print("Started processing video: ", video_path)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, parameters["starting_frame"])
    # if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) < snapshot_data["frame_count"]:
    #     cut_access_data(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), parameters)
    while int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) >= snapshot_data["frame_count"]:
        _, frame = cap.read()
        if snapshot_data["skipped_frames"] % 25 == 0 and snapshot_data["skipped_frames"] != 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, parameters["starting_frame"] + snapshot_data["frame_count"] + 24)
            for i in range(24):
                perspective_squares.save_data(snapshot_data, parameters)
                snapshot_data = calculator.save_data(snapshot_data, parameters)
        else:
            try:
                squares = perspective_squares.PerspectiveSquares(parameters, frame, snapshot_data)
                perspective_squares.save_data(snapshot_data, parameters, squares)
                try:
                    springs = Springs(parameters, frame, snapshot_data)
                    ants = Ants(frame, springs, squares)
                    calculations = calculator.Calculation(parameters, snapshot_data, springs, ants)
                    snapshot_data = create_snapshot_data(snapshot_data=snapshot_data, calculations=calculations, squares=squares, springs=springs)
                    snapshot_data = calculator.save_data(snapshot_data, parameters, calculations)
                    utils.present_analysis_result(frame, calculations, springs, os.path.basename(video_path).split(".")[0])
                except:
                    snapshot_data = calculator.save_data(snapshot_data, parameters)
            except:
                perspective_squares.save_data(snapshot_data, parameters)
                snapshot_data = calculator.save_data(snapshot_data, parameters)
        parameters["continue_from_last"] = False
    cap.release()
    utils.convert_ants_centers_to_mathlab(parameters["output_path"])
    print("Finished processing video: ", video_path)


if __name__ == '__main__':
    args = parse_args()
    videos = [args["path"]] if args["path"].endswith(".MP4") else glob.glob(os.path.join(args["path"], "**", "*.MP4"), recursive=True)
    print("Number of video found to be analyzed: ", len(videos))
    if args['collect_parameters']:
        print("Collecting parameters for all videos in directory: ", args["path"])
        CollectAnalysisParameters(videos, args["path"])
    video_parameters = [load_parameters(video_path) for video_path in videos]
    print("Number of processors exist:", cpu_count())
    print("Number of processors asked for this task:", str(args["nCPU"]))
    pool = Pool(args["nCPU"])
    pool.starmap(main, zip(videos, video_parameters))
    pool.close()
    print("-"*80)
    print("Finished processing all videos in directory: ", args["path"])

# python video_analysis\main.py --path Z:\Dor_Gabay\ThesisProject\data\1-videos\summer_2023\plus_0.2\ --output_path Z:\Dor_Gabay\ThesisProject\data\2-video_analysis\summer_2023\plus_0.2\ --nCPU 5
# python video_analysis\main.py --path Z:\Dor_Gabay\ThesisProject\data\1-videos\summer_2023\calibration\plus_0.5\ --output_path Z:\Dor_Gabay\ThesisProject\data\2-video_analysis\summer_2023\calibration\plus_0.5\ --nCPU 5
# python video_analysis\main.py --path Z:\Dor_Gabay\ThesisProject\data\1-videos\summer_2023\calibration\plus_0.3\ --output_path Z:\Dor_Gabay\ThesisProject\data\2-video_analysis\summer_2023\calibration\plus_0.3\ --nCPU 5
# python video_analysis\main.py --path Z:\Dor_Gabay\ThesisProject\data\1-videos\summer_2023\calibration\plus_0.1\ --output_path Z:\Dor_Gabay\ThesisProject\data\2-video_analysis\summer_2023\calibration\plus_0.1\ --nCPU 5 -cp
# python video_analysis\main.py --path Z:\Dor_Gabay\ThesisProject\data\1-videos\summer_2023\calibration\plus_0\ --output_path Z:\Dor_Gabay\ThesisProject\data\2-video_analysis\summer_2023\calibration\plus_0\ --nCPU 5 -cp

# python video_analysis\main.py --path Z:\Dor_Gabay\ThesisProject\data\1-videos\summer_2023\plus_0.2\ --output_path Z:\Dor_Gabay\ThesisProject\data\2-video_analysis\summer_2023\plus_0.2\ --nCPU 8 -con -cp