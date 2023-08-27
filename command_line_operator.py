import os
import glob
import argparse
import multiprocessing as mp
#local packages
import general_video_scripts.collect_analysis_parameters as collect_color_parameters
import video_analysis.main as main


def parse_args():
    parser = argparse.ArgumentParser(description='Detecting soft pendulum object')
    parser.add_argument('--dir_path', type=str, help='Path to video for analysis', required=True)
    parser.add_argument('--vid_path', type=str, help='Path to video for analysis')
    parser.add_argument("--iter_dir", help='Recursavly search for videos in vidpath. Vidpath will be given as a path to a directory rather than a file.', action='store_true')
    parser.add_argument("--nCPU", type=int, help='Number of CPU processors to use for multiprocessin (when iter_dir is added)',default=1)
    parser.add_argument('--object_crop_margin', '-ocm', type=int, default=200, help='Number of pixels to slice out from each direction (top,bottom,left,right)')
    parser.add_argument('--perspective_squares_crop_margin', '-pcm', type=int, default=100, help='Number of pixels to slice out from each direction (top,bottom,left,right)')
    parser.add_argument('--output_dir','-o', type=str, help='Path to output directory. If not given the program will create a directory within the input folder')
    parser.add_argument('--collect_start_frame', help='If True, then the program will request frame to start analysing for each vidoe',action='store_true')
    parser.add_argument('--continue_from_last', '-con', help='If True, then the program will continue from the last frame analysed',action='store_true')
    parser.add_argument('--starting_frame', help='Frame to start with',type=int,default=None)
    parser.add_argument('--skip', type=int, help='Number of skipping frames',default=1)
    parser.add_argument('--resolution', '-r', help='Resolution of the video', default=[2160, 3840])
    # parser.add_argument('--collect_crop',help='Number of pixels to slice out from each direction (top,bottom,left,right)',action='store_true')
    parser.add_argument('--collect_parameters', '-cp', help='', action='store_true')
    parser.add_argument('--complete_unanalyzed', '-cu', help='',action='store_true')
    args = vars(parser.parse_args())
    return args

def create_output_dir(video_path,args):
    """
    creates a output directory for all the created data
    :param video_path: The video path,
    :return: the output directory path, the video name.
    """
    video_path = os.path.normpath(video_path)
    subdirs = video_path.split('\\')[5:-1]
    vid_name = video_path.split('\\')[-1].split('.')[0]
    if "output_dir" in args:
        output_dir = os.path.normpath(os.path.join(args["output_dir"],*subdirs,vid_name))
    else:
        output_dir = os.path.join(args["dir_path"], subdirs,vid_name)
    os.makedirs(output_dir, exist_ok=True)
    print("Output_dir: ", output_dir)
    return output_dir, vid_name


def write_or_remove_files_paths_in_txt_file(videos_to_analyze=None,video_path=None):
    """
    Creates a txt file with all the paths of the videos to be analysed.
    after being analysed this function will remove the path from the file.
    Therefore if the process what inturupted in the middle, the user can access the file to continue from the last stop
    """
    # write:
    if videos_to_analyze!=None:
        with open(os.path.join(args["dir_path"], "Unanalyzed_videos.txt"), 'w') as f:
            for vid in videos_to_analyze:
                f.write(os.path.normpath(vid) + '\n')
    # remove successfully analysed video:
    elif video_path != None:
        with open(os.path.join(args["dir_path"], "Unanalyzed_videos.txt"), 'r') as f:
            lines = f.readlines()
        with open(os.path.join(args["dir_path"], "Unanalyzed_videos.txt"), 'w') as f:
            for line in lines:
                if line.strip("\n") != video_path:
                    f.write(line)
    else: exit("You need to provide either list of files to be written, of a single file path to be removed.")


def find_videos_to_analyze(args):
    """ Iterates over the directory given in the parsing (that is way this function recives no input)"""
    if args["complete_unanalyzed"]:
        videos_to_analyze = [os.path.normpath(x) for x in open(os.path.join(args["dir_path"], "Unanalyzed_videos.txt"), 'r').read().split('\n') if len(x)>1]
    else:
        videos_to_analyze = []
        dirs_to_iter = [args["dir_path"]]
        while dirs_to_iter:
            dir = dirs_to_iter.pop()
            # dirs_to_iter += [os.path.join(dir, subdir) for subdir in os.listdir(dir) if os.path.isdir(os.path.join(dir, subdir)) and (("_force" in subdir) or ("_sliced" in subdir))]
            dirs_to_iter += [os.path.join(dir, subdir) for subdir in os.listdir(dir) if os.path.isdir(os.path.join(dir, subdir))]
            videos_to_analyze += [os.path.normpath(x) for x in glob.glob(os.path.join(dir, "*.MP4"))]\
                                 +[os.path.normpath(x) for x in glob.glob(os.path.join(dir, "*.MTS"))]\
                                 +[os.path.normpath(x) for x in glob.glob(os.path.join(dir, "*.MOV"))]
    print("Number of video found to be analyzed: ", len(videos_to_analyze))
    return videos_to_analyze

def run_analysis(video_path_args):
    video_path = video_path_args[0]
    args = video_path_args[1]
    output_dir,vidname = create_output_dir(video_path,args)
    print("Start processing video: ",video_path)
    video_parameters = collect_color_parameters.get_parameters(os.path.join(args["dir_path"], "parameters"), video_path)
    # video_parameters["crop_coordinates"] = None #only for calibration videos
    main.main(video_path, output_dir, video_parameters,starting_frame=args["starting_frame"],continue_from_last=args["continue_from_last"])
    print("Finished processing video: ", video_path)
    # remove vidoe path from 'Unanalyzed_videos.txt' file:
    # write_or_remove_files_paths_in_txt_file(video_path=video_path)

if __name__ == '__main__':
    args = parse_args()
    if not args["iter_dir"]:
        if args['collect_parameters']:
            print("Collecting parameters for all videos in directory: ",args["dir_path"])
            collect_color_parameters.main([args["vid_path"]], args["dir_path"], starting_frame=args["collect_start_frame"])
        run_analysis(([args["vid_path"],args]))
    elif args["iter_dir"]:
        videos_to_analyze = find_videos_to_analyze(args)
        write_or_remove_files_paths_in_txt_file(videos_to_analyze=videos_to_analyze)
        if args['collect_parameters']:
            print("Collecting parameters for all videos in directory: ",args["dir_path"])
            collect_color_parameters.main(videos_to_analyze, args["dir_path"], starting_frame=args["collect_start_frame"])
        print("Number of processors exist:",mp.cpu_count())
        print("Mumber of processors used for this task:",str(args["nCPU"]))
        # pool = mp.Pool(args["nCPU"])
        videos_to_analyze = [(x,args) for x in videos_to_analyze]
        for video_path_args in videos_to_analyze:
            run_analysis(video_path_args)
        # pool.map(run_analysis, videos_to_analyze)
        # pool.close()
    print("-"*80)
    print("Finished processing all videos in directory: ", args["dir_path"])

# python command_line_operator.py --dir_path Z:\Dor_Gabay\ThesisProject\data\1-videos\summer_2023\13.8\ --output_dir Z:\Dor_Gabay\ThesisProject\data\2-videos_analysis\ --collect_parameters --iter_dir --nCPU 1