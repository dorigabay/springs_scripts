import os
import cv2
import glob
import argparse
from multiprocessing import Pool, cpu_count
import warnings

# local imports:
from parameters_collector import CollectParameters
import integrator
from ants_detector import Ants
from springs_detector import Springs
import perspective_squares_detector
import utils


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser(description='Detecting soft pendulum object')
    parser.add_argument('--path', type=str, help='Directory path of the videos. Branched directories are allowed.', required=True)
    parser.add_argument('--output_path', '-o', type=str, help='Path to output directory. Default is the same as the input directory.')
    parser.add_argument("--nCPU", type=int, help='Number of CPU processors to use for multiprocessing.', default=1)
    parser.add_argument('--collect_parameters', '-cp', help='', action='store_true')
    parser.add_argument('--continue', '-con', help='If True, then the program will continue from the last frame analysed', action='store_true')
    parser.add_argument('--starting_frame', '-sf', help='Frame to start with. (for debugging)', type=int, default=None)
    arguments = vars(parser.parse_args())
    return arguments


def main(video_path, parameters, run_id, progress_bar):
    checkpoint = utils.CheckpointFile(parameters)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, checkpoint.starting_frame)
    checkpoint.total_n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while checkpoint.total_n_frames > checkpoint.frame_count:
        _, frame = cap.read()
        if checkpoint.skipped_frames != 0 and checkpoint.skipped_frames % 25 == 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, checkpoint.starting_frame + checkpoint.frame_count + 24)
            for i in range(24):
                perspective_squares_detector.save(checkpoint)
                integrator.save(checkpoint)
        else:
            try:
                squares = perspective_squares_detector.PerspectiveSquares(parameters, frame, checkpoint)
                perspective_squares_detector.save(checkpoint, squares)
                try:
                    springs = Springs(parameters, frame, checkpoint)
                    ants = Ants(frame, springs, squares)
                    integration = integrator.Integration(parameters, checkpoint, springs, ants)
                    checkpoint.update(integration, squares, springs, ants)
                    integrator.save(checkpoint, integration)
                    utils.present_analysis_result(frame, integration, springs, ants, os.path.basename(video_path).split(".")[0])
                except Exception as e:
                    # print("\r Error in ants or springs detection: ", e)
                    integrator.save(checkpoint)
            except Exception as e:
                # print("\r Error in perspective squares detection: ", e)
                perspective_squares_detector.save(checkpoint)
                integrator.save(checkpoint)
            checkpoint.continuation = False
        progress_bar.update(run_id, checkpoint.frame_count)
    cap.release()


if __name__ == '__main__':
    args = parse_args()
    videos = [args["path"]] if args["path"].endswith(".MP4") else glob.glob(os.path.join(args["path"], "**", "*.MP4"), recursive=True)
    print("Number of video found to be analyzed: ", len(videos))
    if args['collect_parameters']:
        print("Collecting parameters for all videos in directory: ", args["path"])
        CollectParameters(videos, args["path"])
    print("Number of processors exist:", cpu_count())
    print("Number of processors asked for this task:", str(args["nCPU"]))
    video_parameters = [utils.load_parameters(video_path, args) for video_path in videos]
    pb = utils.ProgressBar(videos, args["path"])
    pool = Pool(args["nCPU"])
    pool.starmap(main, zip(videos, video_parameters, range(len(videos)), [pb]*len(videos)))
    pool.close()
    pool.join()
    pb.end()


# python video_analysis\main.py --path Z:\Dor_Gabay\ThesisProject\data\1-videos\summer_2023\experiment\plus_0.3\ --output_path Z:\Dor_Gabay\ThesisProject\data\2-video_analysis\summer_2023\experiment\plus_0.3\ --nCPU 17 -con
# python video_analysis\main.py --path Z:\Dor_Gabay\ThesisProject\data\1-videos\summer_2023\experiment\plus_0.0\ --output_path Z:\Dor_Gabay\ThesisProject\data\2-video_analysis\summer_2023\experiment\plus_0.0\ --nCPU 15
# python video_analysis\main.py --path Z:\Dor_Gabay\ThesisProject\data\1-videos\summer_2023\calibration\plus_0.3\ --output_path Z:\Dor_Gabay\ThesisProject\data\2-video_analysis\summer_2023\calibration\plus_0.3\ --nCPU 10
# python video_analysis\main.py --path Z:\Dor_Gabay\ThesisProject\data\1-videos\summer_2023\calibration\stiff\ --output_path Z:\Dor_Gabay\ThesisProject\data\2-video_analysis\summer_2023\stiff\ --nCPU 17 -cp

# python video_analysis\main.py --path Z:\Dor_Gabay\ThesisProject\data\1-videos\summer_2023\calibration\plus_0.5\ --output_path Z:\Dor_Gabay\ThesisProject\data\2-video_analysis\summer_2023\calibration\plus_0.5\ --nCPU 10
# python video_analysis\main.py --path Z:\Dor_Gabay\ThesisProject\data\1-videos\summer_2023\calibration\plus_0.0\ --output_path Z:\Dor_Gabay\ThesisProject\data\2-video_analysis\summer_2023\calibration\plus_0.0\ --nCPU 10


