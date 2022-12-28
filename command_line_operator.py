import os
import re
import glob
import argparse
import time
import multiprocessing as mp
import numpy as np
#local packages
import collect_color_parameters
import main

parser = argparse.ArgumentParser(description='Detecting soft pendulum object')
parser.add_argument('--dirpath', type=str, help='Path to video for analysis', required=True)
parser.add_argument('--vidpath', type=str, help='Path to video for analysis')
parser.add_argument("--iter_dir", help='Recursavly search for videos in vidpath. Vidpath will be given as a path to a directory rather than a file.', action='store_true')
parser.add_argument("--nCPU", type=int, help='Number of CPU processors to use for multiprocessin (when iter_dir is added)',default=1)
parser.add_argument('--Xmargin', type=int, help='Margins from contour to cut frames (X_value)"')
parser.add_argument('--Ymargin', type=int, help='Margins from contour to cut frames (X_value)"')
parser.add_argument('--output_dir', type=str, help='Path to output directory. If not given the program will create a directory within the input folder')
parser.add_argument('--startFrame', type=int, help='Frame to start analysing',default=0)
parser.add_argument('--skip', type=int, help='Number of skipping frames',default=1)
parser.add_argument('--crop',help='Number of pixels to slice out from each direction (top,bottom,left,right)',action='store_true')
parser.add_argument('--collect_parameters',help='', action='store_true')
parser.add_argument('--complete_unanalyzed',help='',action='store_true')
args = vars(parser.parse_args())


def create_output_dir(video_path):
    """
    creates a output directory for all the created data
    :param video_path: The video path,
    :return: the output directory path, the video name.
    """
    main_dir = args["dirpath"].split("/")[-2]
    vidname = re.search(".*[/\\\\]([^/\\\\]+).[mM][pP]4", video_path).group(1)
    subdirs = re.search(".*" + "(" + main_dir + ".*)" + vidname + ".*", video_path)
    if subdirs != None:
        subdirs= subdirs.group(1)
    else: subdirs = ""

    if "output_dir" in args:
        output_dir = args["output_dir"]+"/"+subdirs+vidname
    else:
        output_dir = os.path.join(args["dirpath"], subdirs,vidname)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir,vidname


def write_or_remove_files_paths_in_txt_file(videos_to_analyze=None,video_path=None):
    """
    Creates a txt file with all the paths of the videos to be analysed.
    after being analysed this function will remove the path from the file.
    Therefore if the process what inturupted in the middle, the user can access the file to continue from the last stop
    """
    # write:
    if videos_to_analyze!=None:
        with open(os.path.join(args["dirpath"], "Unanalyzed_videos.txt"), 'w') as f:
            for vid in videos_to_analyze:
                f.write(os.path.normpath(vid) + '\n')
    # remove successfully analysed video:
    elif video_path != None:
        with open(os.path.join(args["dirpath"], "Unanalyzed_videos.txt"), 'r') as f:
            lines = f.readlines()
        with open(os.path.join(args["dirpath"], "Unanalyzed_videos.txt"), 'w') as f:
            for line in lines:
                if line.strip("\n") != video_path:
                    f.write(line)
    else: exit("You need to provide either list of files to be written, of a single file path to be removed.")


def find_videos_to_analyze():
    """ Iterates over the directory given in the parsing (that is way this function recives no input)"""
    if args["complete_unanalyzed"]:
        videos_to_analyze = [os.path.normpath(x) for x in open(os.path.join(args["dirpath"], "Unanalyzed_videos.txt"), 'r').read().split('\n') if len(x)>1]
    else:
        videos_to_analyze = []
        # find video to analyse:
        directories_to_search = [args["dirpath"]]
        while directories_to_search:
            dir = directories_to_search.pop()
            found_dirs = [folder_name for folder_name in [x for x in os.walk(dir)][0][1] if "_force" in folder_name]
            if len(found_dirs) > 0:
                for found_dir in found_dirs:
                    videos_to_analyze += [os.path.normpath(x) for x in glob.glob(os.path.join(dir, found_dir, "*.MP4"))]
            else:
                for subdir in [x for x in os.walk(dir)][0][1]:
                    directories_to_search.append(os.path.join(dir, subdir))
    print("Number of video found to be analyzed: ", len(videos_to_analyze))
    return videos_to_analyze

def run_analysis(video_path):
    output_dir,vidname = create_output_dir(video_path)
    print("Start processing video: ",video_path)
    video_parameters = collect_color_parameters.get_parameters(args["dirpath"],video_path)
    main.main(video_path, output_dir, video_parameters)
    # remove vidoe path from 'Unanalyzed_videos.txt' file:
    write_or_remove_files_paths_in_txt_file(video_path=video_path)



if __name__ == '__main__':
    if not args["iter_dir"]:
        run_analysis(os.path.normpath(args["vidpath"]))
    elif args["iter_dir"]:
        videos_to_analyze = find_videos_to_analyze()
        # print(videos_to_analyze)
        # write videos  to be analysed in a txt file:
        write_or_remove_files_paths_in_txt_file(videos_to_analyze=videos_to_analyze)
        if args['collect_parameters']:
            # make parameters file:
            collect_color_parameters.main(videos_to_analyze,args["dirpath"], starting_frame=args["startFrame"], collect_crop=args["crop"])
        # analyze videos:
        print("Number of processors exist:",mp.cpu_count())
        print("Mumber of processors used for this task:",str(args["nCPU"]))
        pool = mp.Pool(args["nCPU"])
        pool.map(run_analysis, [v for v in videos_to_analyze])
        pool.close()

# python command_line_operator.py --dirpath Z:/Dor_Gabay/videos/11.9.22/ --iter_dir --output_dir Z:/Dor_Gabay/data/summer_2022/ --nCPU 15 --complete_unanalyzed


