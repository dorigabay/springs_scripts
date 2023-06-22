import cv2
import os
import numpy as np
import scipy.io as sio
import pickle
from general_video_scripts.collect_analysis_parameters import neutrlize_colour
import datetime
# local imports:
from video_analysis.calculator import Calculation
from video_analysis import utils


def save_as_mathlab_matrix(output_dir):
    ants_centers_x = np.loadtxt(os.path.join(output_dir, "ants_centers_x.csv"), delimiter=",")
    ants_centers_y = np.loadtxt(os.path.join(output_dir, "ants_centers_y.csv"), delimiter=",")
    ants_centers = np.stack((ants_centers_x, ants_centers_y), axis=2)
    ants_centers_mat = np.zeros((ants_centers.shape[0], 1), dtype=np.object)
    for i in range(ants_centers.shape[0]):
        ants_centers_mat[i, 0] = ants_centers[i, :, :]
    sio.savemat(os.path.join(output_dir, "ants_centers.mat"), {"ants_centers": ants_centers_mat})
    matlab_script_path = "Z:\\Dor_Gabay\\ThesisProject\\scripts\\munkres_tracker\\"
    os.chdir(matlab_script_path)
    os.system("PATH=$PATH:'C:\Program Files\MATLAB\R2022a\bin'")
    execution_string = f"matlab -r ""ants_tracking('" + output_dir + "\\')"""
    os.system(execution_string)


# def save_blue_areas_median(output_dir):
#     blue_area_sizes = np.loadtxt(os.path.join(output_dir, "blue_area_sizes.csv"), delimiter=",")
#     median_blue_area_size = np.median(blue_area_sizes)
#     with open(os.path.join(output_dir, "blue_median_area.pickle"), 'wb') as f:
#         pickle.dump(median_blue_area_size, f)


def get_csv_line_count(csv_file):
    with open(csv_file, 'r') as file:
        line_count = sum(1 for _ in file)
    return line_count


def save_data(output_dir, snap_data, calculations, n_springs=20,continue_from_last=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if calculations is None:
        empty_springs = np.repeat(np.nan, n_springs).reshape(1, n_springs)
        empty_blue_part = np.repeat(np.nan, 2).reshape(1, 2)
        empty_ant_centers = np.repeat(np.nan, 100).reshape(1, 100)
        empty_blue_area_size = np.repeat(np.nan, 1).reshape(1, 1)
        empty_ants_attached_labels = np.array([np.nan for _ in range(100)]).reshape(1, 100)
        arrays = [empty_springs, empty_springs, empty_springs, empty_springs, empty_springs, empty_springs,
                  empty_blue_part, empty_blue_part, empty_ant_centers, empty_ant_centers, empty_blue_area_size,
                  empty_ants_attached_labels, empty_ants_attached_labels]
    else:
        arrays = [calculations.N_ants_around_springs, calculations.size_ants_around_springs,
                  calculations.fixed_ends_coordinates_x, calculations.fixed_ends_coordinates_y,
                  calculations.free_ends_coordinates_x, calculations.free_ends_coordinates_y,
                  calculations.blue_part_coordinates_x, calculations.blue_part_coordinates_y,
                  calculations.ants_centers_x, calculations.ants_centers_y,
                  np.array(calculations.blue_area_size).reshape(-1, 1),
                  calculations.ants_attached_labels.reshape(1, 100),
                  calculations.ants_attached_forgotten_labels.reshape(1, 100)]
    names = ["N_ants_around_springs", "size_ants_around_springs",
             "fixed_ends_coordinates_x", "fixed_ends_coordinates_y", "free_ends_coordinates_x",
             "free_ends_coordinates_y", "blue_part_coordinates_x", "blue_part_coordinates_y",
             "ants_centers_x", "ants_centers_y", "blue_area_sizes",
             "ants_attached_labels", "ants_attached_forgotten_labels"]
    pickle.dump(snap_data, open(os.path.join(output_dir, f"snap_data_{snap_data[6]}.pickle"), "wb"))
    if snap_data[5]==0:
        for d, n in zip(arrays[:], names[:]):
            with open(os.path.join(output_dir, str(n) + '.csv'), 'wb') as f:
                np.savetxt(f, d, delimiter=',')
                f.close()
    else:
        for d, n in zip(arrays[:], names[:]):
            with open(os.path.join(output_dir, str(n) + '.csv'), 'a') as f:
                if continue_from_last:
                    line_count = get_csv_line_count(os.path.join(output_dir, str(n) + '.csv'))
                    if line_count == snap_data[5]+1:
                        continue
                    else:
                        np.savetxt(f, d, delimiter=',')
                else:
                        np.savetxt(f, d, delimiter=',')
                f.close()


def present_analysis_result(frame, calculations):
    image_to_illustrate = frame
    for point_green in calculations.green_centers:
        image_to_illustrate = cv2.circle(image_to_illustrate, point_green.astype(int), 1, (0, 255, 0), 2)
    for point_red in calculations.red_centers:
        image_to_illustrate = cv2.circle(image_to_illustrate, point_red.astype(int), 1, (0, 0, 255), 2)
    for count_, angle in enumerate(calculations.springs_angles_ordered):
        if angle != 0:
            if angle in calculations.fixed_ends_edges_bundles_labels:
                point = calculations.fixed_ends_edges_centers[calculations.fixed_ends_edges_bundles_labels.index(angle)]
                image_to_illustrate = cv2.circle(image_to_illustrate, point, 1, (0, 0, 0), 2)
                image_to_illustrate = cv2.putText(image_to_illustrate, str(count_), point, cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                  (255, 0, 0), 2)
                image_to_illustrate = cv2.circle(image_to_illustrate, point, 1, (0, 0, 0), 2)
            if angle in calculations.free_ends_edges_bundles_labels:
                point = calculations.free_ends_edges_centers[calculations.free_ends_edges_bundles_labels.index(angle)]
                image_to_illustrate = cv2.circle(image_to_illustrate, point, 1, (0, 0, 0), 2)

    image_to_illustrate = cv2.circle(image_to_illustrate, calculations.object_center, 1, (0, 0, 0), 2)
    image_to_illustrate = cv2.circle(image_to_illustrate, calculations.tip_point, 1, (255, 0, 0), 2)
    cv2.imshow("frame", image_to_illustrate)
    cv2.waitKey(1)
    return image_to_illustrate, calculations.joints


def main(video_path, output_dir, parameters, starting_frame=None, continue_from_last=False):
    output_dir = os.path.join(output_dir, "raw_analysis")
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if continue_from_last and len([f for f in os.listdir(output_dir) if f.startswith("snap_data")]) != 0:
        snaps = [f for f in os.listdir(output_dir) if f.startswith("snap_data")]
        snap_data = pickle.load(open(os.path.join(output_dir, snaps[-1]), "rb"))
        parameters["starting_frame"] = snap_data[5]
        snap_data[6] = datetime.datetime.now().strftime("%d.%m.%Y-%H%M")
    else:
        snap_data = [None, None ,None, 0, 0, 0,datetime.datetime.now().strftime("%d.%m.%Y-%H%M")]
        # in snap_data: object_center, tip_point, springs_angles_reference_order, sum_blue_radius, frames_analysed, count, current_time
    if starting_frame is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, parameters["starting_frame"])
    # parameters["n_springs"] = 20
    while True:
        ret, frame = cap.read()
        if frame is None:
            print("End of video")
            break
        frame = neutrlize_colour(frame)
        if parameters["crop_coordinates"] != None:
            frame = utils.crop_frame_by_coordinates(frame, parameters["crop_coordinates"])
        try:
            calculations = Calculation(parameters, frame, snap_data)
            snap_data = [calculations.object_center, calculations.tip_point, calculations.springs_angles_reference_order,
                                   snap_data[3]+int(calculations.blue_radius), snap_data[4]+1, snap_data[5], snap_data[6]]
            save_data(output_dir, calculations=calculations, snap_data=snap_data, continue_from_last=continue_from_last)
            present_analysis_result(frame, calculations)
            del calculations, ret, frame
            print("Analyzed frame number:", snap_data[5], end="\r")
        except:
            print("Skipped frame: ", snap_data[5], end="\r")
            save_data(output_dir, snap_data=snap_data, calculations=None, n_springs=parameters["n_springs"],continue_from_last=continue_from_last)
        snap_data[5] += 1
        continue_from_last = False
    cap.release()
    if not os.path.exists(os.path.join(output_dir, f"analysis_ended_{snap_data[6]}.pickle")):
        save_as_mathlab_matrix(output_dir)
        # save_blue_areas_median(output_dir)
    pickle.dump("video_ended", open(os.path.join(output_dir, f"analysis_ended_{snap_data[6]}.pickle"), "wb"))


if __name__ == "__main__":
    video_path = "Z:\\Dor_Gabay\\ThesisProject\\data\\videos\\15.9.22\\plus0.3mm_force\\S5280007.MP4"
    output_dir = "Z:\\Dor_Gabay\\ThesisProject\\data\\analysed_with_tracking_tesss\\"
    parameters_dir = "Z:\\Dor_Gabay\\ThesisProject\\data\\videos\\15.9.22\\parameters\\"
    from general_video_scripts.collect_analysis_parameters import get_parameters
    parameters = get_parameters(parameters_dir,video_path)
    main(video_path, output_dir, parameters,starting_frame=0, continue_from_last=True)