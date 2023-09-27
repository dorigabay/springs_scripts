import os
import glob
import argparse
from multiprocessing import Pool, cpu_count
import pickle
#local packages
from video_analysis.parameters_collector import CollectAnalysisParameters
import video_analysis.main as main





if __name__ == '__main__':
    args = parse_args()
    videos = [args["path"]] if args["path"].endswith(".MP4") else glob.glob(os.path.join(args["path"], "**", "*.MP4"), recursive=True)
    print("Number of video found to be analyzed: ", len(videos))
    if args['collect_parameters']:
        print("Collecting parameters for all videos in directory: ", args["path"])
        CollectAnalysisParameters(videos, args["path"])
    parameters = [load_parameters(video_path) for video_path in videos]
    print("Number of processors exist:", cpu_count())
    print("Mumber of processors used for this task:", str(args["nCPU"]))
    pool = Pool(args["nCPU"])
    pool.starmap(main.main, zip(videos, parameters))
    pool.close()
    print("-"*80)
    print("Finished processing all videos in directory: ", args["path"])


# python video_analysis\main.py --dir_path Z:\Dor_Gabay\ThesisProject\data\1-videos\summer_2023\calibration\plus_0.2\ --output_dir Z:\Dor_Gabay\ThesisProject\data\2-video_analysis\summer_2023\calibration\plus_0.2\test\ --nCPU 1

