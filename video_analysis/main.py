import os
import cv2
import glob
import pickle
import argparse
from multiprocessing import Pool, cpu_count
# local imports:
from parameters_collector import CollectParameters
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
    parser.add_argument('--continue_from_last_snapshot', '-con', help='If True, then the program will continue from the last frame analysed', action='store_true')
    parser.add_argument('--starting_frame', '-sf', help='Frame to start with. (for debugging)', type=int, default=None)
    # parser.add_argument('--parameters_path', '-p', help='')
    arguments = vars(parser.parse_args())
    return arguments


def main(video_path, parameters):
    snapshot_data = utils.create_snapshot_data(parameters=parameters)
    print("Started processing video: ", video_path)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, parameters["STARTING_FRAME"])
    total_n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while total_n_frames > snapshot_data["frame_count"]:
        _, frame = cap.read()
        if snapshot_data["skipped_frames"] % 25 == 0 and snapshot_data["skipped_frames"] != 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, parameters["STARTING_FRAME"] + snapshot_data["frame_count"] + 24)
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
                    snapshot_data = utils.create_snapshot_data(snapshot_data=snapshot_data, calculations=calculations, squares=squares, springs=springs, ants=ants)
                    snapshot_data = calculator.save_data(snapshot_data, parameters, calculations)
                    utils.present_analysis_result(frame, calculations, springs, ants, os.path.basename(video_path).split(".")[0])
                except:
                    snapshot_data = calculator.save_data(snapshot_data, parameters)
            except:
                perspective_squares.save_data(snapshot_data, parameters)
                snapshot_data = calculator.save_data(snapshot_data, parameters)
        parameters["CONTINUE_FROM_LAST_SNAPSHOT"] = False
    cap.release()
    print("Finished processing video: ", video_path)


if __name__ == '__main__':
    args = parse_args()
    videos = [args["path"]] if args["path"].endswith(".MP4") else glob.glob(os.path.join(args["path"], "**", "*.MP4"), recursive=True)
    print("Number of video found to be analyzed: ", len(videos))
    if args['collect_parameters']:
        print("Collecting parameters for all videos in directory: ", args["path"])
        CollectParameters(videos, args["path"])
    video_parameters = [utils.load_parameters(video_path, args) for video_path in videos]
    print("Number of processors exist:", cpu_count())
    print("Number of processors asked for this task:", str(args["nCPU"]))
    pool = Pool(args["nCPU"])
    pool.starmap(main, zip(videos, video_parameters))
    pool.close()
    pool.join()
    print("-"*80)
    print("Finished processing all videos in directory: ", args["path"])


# python video_analysis\main.py --path Z:\Dor_Gabay\ThesisProject\data\1-videos\summer_2023\calibration\plus_0.1\ --output_path Z:\Dor_Gabay\ThesisProject\data\2-video_analysis\summer_2023\calibration\plus_0.1\ -cp
# python video_analysis\main.py --path \\phys-guru-cs\ants\Dor_Gabay\ThesisProject\data\1-videos\summer_2023\experiment\plus_0.5\ --output_path \\phys-guru-cs\ants\Dor_Gabay\ThesisProject\data\2-video_analysis\summer_2023\experiment\plus_0.5\ -cp

# python video_analysis\main.py --path \\phys-guru-cs\ants\Dor_Gabay\ThesisProject\data\1-videos\summer_2023\experiment\plus_0.5\ --output_path \\phys-guru-cs\ants\Dor_Gabay\ThesisProject\data\2-video_analysis\summer_2023\experiment\plus_0.5\ -nCPU 18

# python video_analysis\main.py --path \\phys-guru-cs\ants\Dor_Gabay\ThesisProject\data\1-videos\summer_2023\experiment\plus_0.2\ --output_path \\phys-guru-cs\ants\Dor_Gabay\ThesisProject\data\2-video_analysis\summer_2023\experiment\plus_0.2\ -cp
# python video_analysis\main.py --path \\phys-guru-cs\ants\Dor_Gabay\ThesisProject\data\1-videos\summer_2023\experiment\plus_0.1\ --output_path \\phys-guru-cs\ants\Dor_Gabay\ThesisProject\data\2-video_analysis\summer_2023\experiment\plus_0.1\ -cp
# python video_analysis\main.py --path Z:\Dor_Gabay\ThesisProject\data\1-videos\summer_2023\experiment\plus_0.1\ --output_path Z:\Dor_Gabay\ThesisProject\data\2-video_analysis\summer_2023\experiment\plus_0.1_final_final\ -con

# python video_analysis\main.py --path Z:\Dor_Gabay\ThesisProject\data\1-videos\summer_2023\experiment\plus_0.2\ --output_path Z:\Dor_Gabay\ThesisProject\data\2-video_analysis\summer_2023\experiment\plus_0.2\ -cp
# python video_analysis\main.py --path Z:\Dor_Gabay\ThesisProject\data\1-videos\summer_2023\experiment\plus_0.3\ --output_path Z:\Dor_Gabay\ThesisProject\data\2-video_analysis\summer_2023\experiment\plus_0.3\ -cp
# python video_analysis\main.py --path Z:\Dor_Gabay\ThesisProject\data\1-videos\summer_2023\experiment\plus_0.5\ --output_path Z:\Dor_Gabay\ThesisProject\data\2-video_analysis\summer_2023\experiment\plus_0.5\ -cp
# python video_analysis\main.py --path Y:\Dor_Gabay\ThesisProject\data\1-videos\summer_2023\experiment\plus_0.1\ --output_path Y:\Dor_Gabay\Trash\_lior_test\ -cp

# python video_analysis\main.py --path Z:\Dor_Gabay\ThesisProject\data\1-videos\summer_2023\experiment\plus_0.2\ --output_path Z:\Dor_Gabay\ThesisProject\data\2-video_analysis\summer_2023\experiment\plus_0.2\ -cp
# python video_analysis\main.py --path Z:\Dor_Gabay\ThesisProject\data\1-videos\summer_2023\calibration\plus_0.2\ --output_path Z:\Dor_Gabay\ThesisProject\data\2-video_analysis\summer_2023\calibration\plus_0.2\ -cp
# python video_analysis\main.py --path Z:\Dor_Gabay\ThesisProject\data\1-videos\summer_2023\calibration\plus_0.2\ --output_path Z:\Dor_Gabay\ThesisProject\data\Trash\

# python video_analysis\main.py --path Z:\Dor_Gabay\ThesisProject\data\1-videos\summer_2023\calibration\plus_0.5\ --output_path Z:\Dor_Gabay\ThesisProject\data\2-video_analysis\summer_2023\calibration\plus_0.5\ --nCPU 10


