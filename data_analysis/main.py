import os
import pickle
import glob
from multiprocessing import Pool
from itertools import repeat
# local packages:
from data_analysis.force_claculator import ForceCalculator
from data_analysis.ant_tracking import AntTracking
from data_analysis.calibration_modeling import CalibrationModeling
from data_analysis.analysis import Analyser
from data_analysis.results_video import ResultsVideo

if __name__ == "__main__":
    spring_type = "plus_0.1"
    # spring_type = "plus_0.2"
    video_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\1-videos\\summer_2023\\experiment\\{spring_type}\\"
    video_analysis_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\2-video_analysis\\summer_2023\\experiment\\{spring_type}\\"
    data_analysis_dir = f"Z:\\Dor_Gabay\\ThesisProject\\data\\3-data_analysis\\summer_2023\\experiment\\{spring_type}\\"
    calibration_output_path = "Z:\\Dor_Gabay\\ThesisProject\\data\\2-video_analysis\\summer_2023\\calibration\\"
    calibration_model_path = os.path.join(calibration_output_path, spring_type, "calibration_model.pkl")
    results_output_dir = f"Z:\\Dor_Gabay\\ThesisProject\\results\\summer_2023\\{spring_type}\\"
    print("-" * 60 + "\nCreating calibration model...\n" + "-" * 20)
    calibration_weights = [0., 0.00356, 0.01449, 0.01933, 0.02986, 0.04515, 0.06307, 0.08473, 0.10512, 0.13058]
    # calibration_weights = [0., 0.00356, 0.01449, 0.01933, 0.02986, 0.04515, 0.08473, 0.10512, 0.13058]
    videos_idx = [0, 1, 2, 3, 4, 7, 8, 9]
    # videos_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # CalibrationModeling(video_dir, calibration_output_path, calibration_weights, videos_idx)
    print("-" * 60 + "\nCalculating force...\n" + "-" * 20)
    video_analysis_paths = [root for root, dirs, files in os.walk(video_analysis_dir) if not dirs]
    video_analysis_paths = video_analysis_paths[:1]
    ForceCalculator(video_dir, video_analysis_paths, data_analysis_dir, calibration_model_path)
    print("-" * 60 + "\nTracking ants...\n"+"-"*20)
    # sets_video_paths = pickle.load(open(os.path.join(data_analysis_dir, "sets_video_paths.pkl"), "rb"))
    # sets_frames = [(video_set[0][0], video_set[-1][1]) for video_set in pickle.load(open(os.path.join(data_analysis_dir, "sets_frames.pkl"), "rb"))]
    # pool = Pool()
    # pool.starmap(AntTracking, zip(sets_video_paths, repeat(data_analysis_dir), sets_frames, repeat((2160, 3840)), repeat(False)))
    # pool.close()
    # pool.join()
    print("-" * 60 + "\nAnalysing...\n" + "-" * 20)
    # self = Analyser(data_analysis_dir, results_output_dir, spring_type)
    print("-" * 60 + "\nCreating a sample video with the results...\n" + "-" * 20)
    # video_path = os.path.normpath(glob.glob(os.path.join(video_dir, "**", "*.MP4"), recursive=True)[13])
    # ResultsVideo(video_path, video_analysis_dir, data_analysis_dir, start_frame=4700, n_frames_to_save=1000, reduction_factor=0.4)

    print("Finished analysing all videos in directory: ", data_analysis_dir)

