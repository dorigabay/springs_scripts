import os
import glob
import pickle
from multiprocessing import Pool
from itertools import repeat
import argparse
import configparser
# local packages:
from calibration_modeling import CalibrationModeling
from force_claculator import ForceCalculator
from ant_tracking import AntTracking
from results_video import ResultsVideo


def parse_args():
    parser = argparse.ArgumentParser(description='Analyse the data collected from the videos')
    parser.add_argument('--config', type=str, help='Path to config file', required=True)
    arguments = vars(parser.parse_args())
    return arguments


def main(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    spring = config.get('General', 'spring_type')
    n_springs = eval(config.get('General', 'n_springs'))
    resolution = eval(config.get('General', 'resolution'))  # (height, width)
    videos_path = os.path.join(os.path.normpath(config.get('Paths', 'video_path')), spring)
    video_analysis_path = os.path.join(os.path.normpath(config.get('Paths', 'video_analysis_path')), spring)
    data_analysis_path = os.path.join(os.path.normpath(config.get('Paths', 'data_analysis_path')), spring)
    results_output_path = os.path.join(os.path.normpath(config.get('Paths', 'results_output_path')), spring)
    calibration_output_path = config.get('Paths', 'calibration_output_path')
    calibration_model_path = os.path.join(calibration_output_path, spring, "calibration_model.pkl")

    if eval(config.get('General', 'run_calibration')):
        print("-" * 60 + "\nCreating calibration model...\n" + "-" * 20)
        # weights = [float(weight) for weight in config.get('Calibration', 'weights').split(',')]
        weights = eval(config.get('Calibration', 'weights'))
        # videos_idx = [int(video_idx) for video_idx in config.get('Calibration', 'videos_idx').split(',')]
        videos_idx = eval(config.get('Calibration', 'videos_idx'))
        CalibrationModeling(videos_path, calibration_output_path, weights, videos_idx)

    if eval(config.get('General', 'run_force_calculation')):
        print("-" * 60 + "\nCalculating force...\n" + "-" * 20)
        video_analysis_paths = [root for root, dirs, files in os.walk(video_analysis_path) if not dirs]
        ForceCalculator(videos_path, video_analysis_paths, data_analysis_path, calibration_model_path, n_springs)

    if eval(config.get('General', 'run_ant_tracking')):
        print("-" * 60 + "\nTracking ants...\n"+"-"*20)
        restart_ant_tracking = eval(config.get('General', "restart_ant_tracking"))  # For debugging
        sets_video_paths = pickle.load(open(os.path.join(data_analysis_path, "sets_video_paths.pkl"), "rb"))
        sets_frames = [(video_set[0][0], video_set[-1][1]) for video_set in pickle.load(open(os.path.join(data_analysis_path, "sets_frames.pkl"), "rb"))]
        pool = Pool()
        pool.starmap(AntTracking, zip(sets_video_paths, repeat(data_analysis_path), sets_frames, repeat(resolution), repeat(restart_ant_tracking)))

    if eval(config.get('General', 'create_results_video')):
        print("-" * 60 + "\nCreating a sample video with the results...\n" + "-" * 20)
        if eval(config.get('Results_video', 'use_video_idx')):
            video_idx = eval(config.get('Results_video', 'video_idx'))
            all_videos = glob.glob(os.path.join(videos_path, "**", "*.MP4"), recursive=True)
            video_to_analyse = os.path.normpath(all_videos[video_idx])
        else:
            video_to_analyse = config.get('Results_video', 'video_path')
        start_frame = eval(config.get('Results_video', 'start_frame'))
        n_frames_to_save = eval(config.get('Results_video', 'n_frames_to_save'))
        reduction_factor = eval(config.get('Results_video', 'reduction_factor'))
        draw_amoeba = eval(config.get('Results_video', 'draw_amoeba'))
        ResultsVideo(video_to_analyse, video_analysis_path, data_analysis_path, results_output_path, spring, n_springs, n_frames_to_save,
                     start_frame=start_frame, reduction_factor=reduction_factor, draw_amoeba=draw_amoeba)

    print("-" * 60 + "\nDone!\n" + "-" * 60)


if __name__ == "__main__":
    params = parse_args()
    main(params['config'])

