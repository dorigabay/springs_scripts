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
from data_visualization import VisualData
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class Config:
    def __init__(self, config_path):
        self.config_file = configparser.ConfigParser()
        self.config_file.read(config_path)
        self.spring = self._get('General', 'spring_type')
        self.stiff_load = self._get('General', 'stiff_load')
        self.n_springs = self._get('General', 'n_springs')
        self.resolution = self._get('General', 'resolution')  # (height, width)
        self.run_calibration = self._get('General', 'run_calibration')
        self.run_force_calculation = self._get('General', 'run_force_calculation')
        self.run_ant_tracking = self._get('General', 'run_ant_tracking')
        self.restart_ant_tracking = self._get('General', "restart_ant_tracking")  # For debugging purposes
        self.create_results_video = self._get('General', 'create_results_video')
        self.videos_path = os.path.join(os.path.normpath(self._get('Paths', 'video_path')), self.spring)
        self.video_analysis_path = os.path.join(os.path.normpath(self._get('Paths', 'video_analysis_path')), self.spring)
        self.data_analysis_path = os.path.join(os.path.normpath(self._get('Paths', 'data_analysis_path')), self.spring)
        self.results_output_path = os.path.join(os.path.normpath(self._get('Paths', 'results_output_path')), self.spring)
        self.calib_output_path = self._get('Paths', 'calibration_output_path')
        self.calib_model_path = os.path.join(self.calib_output_path, self.spring, "calibration_model.pkl")
        self.calib_weights = self._get('Calibration', 'weights')
        # self.calib_videos_idx = self._get('Calibration', 'videos_idx')
        self.viz_use_video_idx = self._get('Data_Visualization', 'use_video_idx')
        self.viz_video_idx = self._get('Data_Visualization', 'video_idx') if self.viz_use_video_idx else None
        self.viz_video_path = self._get('Data_Visualization', 'video_path') if not self.viz_use_video_idx else None
        self.viz_start_frame = self._get('Data_Visualization', 'start_frame')
        self.viz_n_frames_to_save = self._get('Data_Visualization', 'n_frames_to_save')
        self.viz_reduction_factor = self._get('Data_Visualization', 'reduction_factor')
        self.viz_draw_amoeba = self._get('Data_Visualization', 'draw_amoeba')

    def _get(self, section, parameter):
        try:
            parameter = eval(self.config_file.get(section, parameter))
        except NameError:
            parameter = str(self.config_file.get(section, parameter))
        except SyntaxError:
            parameter = self.config_file.get(section, parameter)
        return parameter


def parse_args():
    parser = argparse.ArgumentParser(description='Analyse the data collected from the videos')
    parser.add_argument('--config', type=str, help='Path to config file', required=True)
    arguments = vars(parser.parse_args())
    return arguments


def main(config_path):
    config = Config(config_path)
    if not config.stiff_load and config.run_calibration:
        print("-" * 60 + "\nCreating calibration model\n" + "-" * 20)
        CalibrationModeling(config.videos_path, config.calib_output_path, config.calib_weights) #, config.calib_videos_idx)

    if config.run_force_calculation:
        print("-" * 60 + "\nCalculating force\n" + "-" * 20)
        video_analysis_paths = [root for root, dirs, files in os.walk(config.video_analysis_path) if not dirs]
        calib_model_path = config.calib_model_path if not config.stiff_load else None
        ForceCalculator(config.videos_path, video_analysis_paths, config.data_analysis_path, config.n_springs, calib_model_path)

    if config.run_ant_tracking:
        print("-" * 60 + "\nTracking ants\n"+"-"*20)
        sets_video_paths = pickle.load(open(os.path.join(config.data_analysis_path, "sets_video_paths.pkl"), "rb"))
        sets_frames = [(video_set[0][0], video_set[-1][1]) for video_set in pickle.load(open(os.path.join(config.data_analysis_path, "sets_frames.pkl"), "rb"))]
        pool = Pool()
        pool.starmap(AntTracking, zip(sets_video_paths, repeat(config.data_analysis_path), sets_frames,
                                      repeat(config.resolution), repeat(config.restart_ant_tracking)))

    if config.create_results_video:
        print("-" * 60 + "\nCreating a sample video with the results\n" + "-" * 20)
        if config.viz_use_video_idx:
            all_videos = glob.glob(os.path.join(config.videos_path, "**", "*.MP4"), recursive=True)
            config.video_to_analyse_path = os.path.normpath(all_videos[config.viz_video_idx])
        VisualData(config.video_to_analyse_path, config.video_analysis_path, config.data_analysis_path,
                   config.results_output_path, config.spring, config.n_springs, config.viz_n_frames_to_save,
                   start_frame=config.viz_start_frame, reduction_factor=config.viz_reduction_factor, draw_amoeba=config.viz_draw_amoeba)

    print("-" * 60 + "\nDone!\n" + "-" * 60)


if __name__ == "__main__":
    command_parameters = parse_args()
    main(command_parameters['config'])
    # python data_analysis\main.py --config data_analysis\configure.ini
