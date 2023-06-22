# this scripts takes a csv file and a video file and slices the video into frames according to the csv file.
# the csv file should be a 2 column file with the first column being the start frame number and the second column being the end frame number for each slice.
# finally it joins the slices into a new video file.

import cv2
import numpy as np
import os

def slice_video(video_path, output_path, cropping_coordinates=None, frames_to_keep=None):
    print(f"Slicing video: {video_path}")
    # read the video file
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(output_path,video_path.split("\\")[-1].split(".")[0] + ".MP4")
    resoloution = (cropping_coordinates[3]-cropping_coordinates[2],cropping_coordinates[1]-cropping_coordinates[0])
    out = cv2.VideoWriter(output_path, fourcc, fps, resoloution)
    for i in range(n_frames):
        ret, frame = cap.read()
        if i in frames_to_keep:
            if cropping_coordinates is not None:
                frame = frame[cropping_coordinates[0]:cropping_coordinates[1],cropping_coordinates[2]:cropping_coordinates[3]]
            out.write(frame)
    out.release()
    cap.release()

# def find_still_frames(video_path, parameters):
#     cap = cv2.VideoCapture(video_path)
#     from video_analysis.springs_detector import Springs
#     from video_analysis.ants_detector import Ants
#     from video_analysis.calculator import Calculation
#     from video_analysis.collect_color_parameters import neutrlize_colour
#     from video_analysis.utils import crop_frame_by_coordinates
#     from video_analysis.main import present_analysis_result
#     previous_detections = None
#     count = 0
#     frames_analysed = 0
#     frames_until_saving = 0
#     print("video_path: ")
#     while True:
#         ret, frame = cap.read()
#         frames_until_saving += 1
#         if frame is None:
#             print("End of video")
#             break # break the loop if there are no additional frame in the video
#         try:
#             frame_neutrlized = neutrlize_colour(frame)
#             if parameters["crop_coordinates"] != None:
#                 frame = crop_frame_by_coordinates(frame, parameters["crop_coordinates"])
#                 frame_neutrlized = crop_frame_by_coordinates(frame_neutrlized, parameters["crop_coordinates"])
#             springs = Springs(parameters, frame_neutrlized, previous_detections = previous_detections)
#             ants = Ants(frame, springs)
#             if count == 0:
#                 calculations = Calculation(springs, ants)
#                 sum_blue_radius = 0
#             else:
#                 calculations.make_calculations(springs, ants)  # ,previous_calculations=previous_detections)
#             sum_blue_radius += springs.blue_radius
#             frames_analysed += 1
#             previous_detections = [springs.object_center, springs.tip_point, sum_blue_radius, frames_analysed]
#             print("frame number:", count)#, end="\r")
#             # present_analysis_result(frame_neutrlized, springs, calculations, ants)
#             count += 1
#             # if count == 100:
#             #     break
#         except:
#             print("skipped frame ", end="\r")
#             calculations.add_blank_row(number_of_rows=frames_until_saving)
#             count += 1
#             continue
#     # find still frames based of minimal movement of springs free end
#     free_end_coordinates_x = calculations.fixed_ends_coordinates_x
#     free_end_coordinates_y = calculations.fixed_ends_coordinates_y
#     free_end_coordinates = np.array([free_end_coordinates_x, free_end_coordinates_y]).T[0,:,:]
#     # find the difference between every 5 frames in each column
#     # free_end_coordinates_diff = np.sum(np.diff(free_end_coordinates, n=10, axis=0),axis=1)
#     from data_analysis.utils import difference
#     free_end_coordinates_diff = np.sum(np.abs(difference(free_end_coordinates, spacing=10)),axis=1)
#     # find the frames where the difference is less than 5
#     still_frames = np.where(np.abs(free_end_coordinates_diff) < 10)[0]
#     cap.release()
#     return still_frames

def collect_points(image,n_points):
    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([y, x])

    points = []
    print("Please pick 2 coordinates for the cropping area.")
    cv2.imshow("Pick two points", image)
    cv2.setMouseCallback("Pick two points", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if len(points) != n_points: collect_points(image, n_points)
    return np.array(points)

def find_frames_to_keep(data_path):
    free_ends_coordinates_x = np.loadtxt(os.path.join(data_path, "raw_analysis", "free_ends_coordinates_x.csv"), delimiter=",")
    free_ends_coordinates_y = np.loadtxt(os.path.join(data_path, "raw_analysis", "free_ends_coordinates_y.csv"), delimiter=",")
    free_end_coordinates = np.array([free_ends_coordinates_x, free_ends_coordinates_y]).T
    from data_analysis.utils import difference
    free_end_coordinates_diff = np.sum(np.abs(difference(free_end_coordinates, spacing=10)),axis=1)
    still_frames = np.where(np.abs(free_end_coordinates_diff) < 10)[0]
    return still_frames

if __name__ == "__main__":
    # videos_numbers = [1,2, 3, 4, 5, 6, 7, 8, 9, 10]
    videos_numbers = [10]
    for v in videos_numbers:
        video_name = f"S54300{v}"
        output_path = "Z:\\Dor_Gabay\\ThesisProject\\data\\videos\\calibration3\\"
        video_path = os.path.join(output_path,f"{video_name}.MP4")
        frames_to_keep = find_frames_to_keep(video_name)
        parameters_path = os.path.join(output_path,"parameters")
        from general_video_scripts.collect_analysis_parameters import get_parameters
        parameters = get_parameters(parameters_path, video_path)
        slice_video(video_path, output_path, frames_to_keep=frames_to_keep, cropping_coordinates=parameters["crop_coordinates"])
    print("done")

