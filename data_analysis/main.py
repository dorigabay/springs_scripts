import os
import pickle
from multiprocessing import Pool
from itertools import repeat
import argparse
import configparser
# local packages:
from data_analysis.force_claculator import ForceCalculator
from data_analysis.ant_tracking import AntTracking
from data_analysis.calibration_modeling import CalibrationModeling
from data_analysis.analysis import Analyser
from data_analysis.results_video import ResultsVideo


def parse_args():
    parser = argparse.ArgumentParser(description='Analyse the data collected from the videos')
    parser.add_argument('--config', type=str, help='Path to config file', required=True)
    arguments = vars(parser.parse_args())
    return arguments


def main(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    spring = config.get('General', 'spring_type')
    resolution = tuple([int(res) for res in config.get('General', 'resolution').split(',')])  # (height, width)
    restart_ant_tracking = False  # For debugging
    video_path = os.path.join(config.get('Paths', 'video_dir'), spring)
    video_analysis_path = os.path.join(config.get('Paths', 'video_analysis_dir'), spring)
    data_analysis_path = os.path.join(config.get('Paths', 'data_analysis_dir'), spring)
    results_output_path = os.path.join(config.get('Paths', 'results_output_dir'), spring)
    calibration_output_path = config.get('Paths', 'calibration_output_path')
    calibration_model_path = os.path.join(calibration_output_path, spring, "calibration_model.pkl")

    if bool(config.get('General', 'run_calibration')):
        print("-" * 60 + "\nCreating calibration model...\n" + "-" * 20)
        weights = [float(weight) for weight in config.get('Calibration', 'weights').split(',')]
        videos_idx = [int(video_idx) for video_idx in config.get('Calibration', 'videos_idx').split(',')]
        CalibrationModeling(video_path, calibration_output_path, weights, videos_idx)

    if bool(config.get('General', 'run_force_calculation')):
        print("-" * 60 + "\nCalculating force...\n" + "-" * 20)
        video_analysis_paths = [root for root, dirs, files in os.walk(video_analysis_path) if not dirs]
        ForceCalculator(video_path, video_analysis_paths, data_analysis_path, calibration_model_path)

    if bool(config.get('General', 'run_ant_tracking')):
        print("-" * 60 + "\nTracking ants...\n"+"-"*20)
        sets_video_paths = pickle.load(open(os.path.join(data_analysis_path, "sets_video_paths.pkl"), "rb"))
        sets_frames = [(video_set[0][0], video_set[-1][1]) for video_set in pickle.load(open(os.path.join(data_analysis_path, "sets_frames.pkl"), "rb"))]
        pool = Pool()
        pool.starmap(AntTracking, zip(sets_video_paths, repeat(data_analysis_path), sets_frames, repeat(resolution), repeat(restart_ant_tracking)))

    if bool(config.get('Results_video', 'create_video')):
        print("-" * 60 + "\nCreating a sample video with the results...\n" + "-" * 20)
        results_video_path = config.get('Results_video', 'video_path')
        start_frame = int(config.get('Results_video', 'start_frame'))
        n_frames_to_save = int(config.get('Results_video', 'n_frames_to_save'))
        reduction_factor = float(config.get('Results_video', 'reduction_factor'))
        n_springs = int(config.get('Results_video', 'n_springs'))
        draw_amoeba = bool(config.get('Results_video', 'draw_amoeba'))
        ResultsVideo(results_video_path, video_analysis_path, data_analysis_path, results_output_path, spring,
                     start_frame=start_frame, n_frames_to_save=n_frames_to_save, reduction_factor=reduction_factor, n_springs=n_springs, draw_amoeba=draw_amoeba)

    print("Finished analysing all videos in directory: ", data_analysis_path)


if __name__ == "__main__":
    params = parse_args()
    main(params['config'])




    # spring_type = "plus_0.5"
    # video_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\1-videos\\summer_2023\\experiment\\{spring_type}\\"
    # video_analysis_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\2-video_analysis\\summer_2023\\experiment\\{spring_type}\\"
    # data_analysis_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\3-data_analysis\\summer_2023\\experiment\\{spring_type}\\"
    # calibration_output_path = "Z:\\Dor_Gabay\\ThesisProject\\data\\2-video_analysis\\summer_2023\\calibration\\"
    # calibration_model_path = os.path.join(calibration_output_path, spring_type, "calibration_model.pkl")
    # results_output_dir = f"Z:\\Dor_Gabay\\ThesisProject\\results\\summer_2023\\{spring_type}\\"
    # print("-" * 60 + "\nCreating calibration model...\n" + "-" * 20)
    # calibration_weights = [0., 0.00356, 0.01449, 0.01933, 0.02986, 0.04515, 0.06307, 0.08473, 0.10512, 0.13058]
    # videos_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # CalibrationModeling(video_dir, calibration_output_path, calibration_weights, videos_idx)
    # print("-" * 60 + "\nCalculating force...\n" + "-" * 20)
    # video_analysis_paths = [root for root, dirs, files in os.walk(video_analysis_dir) if not dirs]
    # ForceCalculator(video_dir, video_analysis_paths, data_analysis_dir, calibration_model_path)
    # print("-" * 60 + "\nTracking ants...\n"+"-"*20)
    # sets_video_paths = pickle.load(open(os.path.join(data_analysis_dir, "sets_video_paths.pkl"), "rb"))
    # sets_frames = [(video_set[0][0], video_set[-1][1]) for video_set in pickle.load(open(os.path.join(data_analysis_dir, "sets_frames.pkl"), "rb"))]
    # pool = Pool()
    # pool.starmap(AntTracking, zip(sets_video_paths, repeat(data_analysis_dir), sets_frames, repeat((2160, 3840)), repeat(False)))
    # print("-" * 60 + "\nAnalysing...\n" + "-" * 20)
    # Analyser(data_analysis_dir, results_output_dir, spring_type)
    # print("-" * 60 + "\nCreating a sample video with the results...\n" + "-" * 20)
    # results_video_path = os.path.normpath(glob.glob(os.path.join(video_dir, "**", "*.MP4"), recursive=True)[0])
    # ResultsVideo(results_video_path, video_analysis_dir, data_analysis_dir, start_frame=4700, n_frames_to_save=1000, reduction_factor=0.4)
    # ResultsVideo(results_video_path, video_analysis_dir, data_analysis_dir, spring_type, n_frames_to_save=500, reduction_factor=1, n_springs=20, arrangement=0)

    # print("Finished analysing all videos in directory: ", data_analysis_dir)

