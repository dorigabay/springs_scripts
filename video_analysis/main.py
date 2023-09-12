import os, pickle, datetime, cv2
import numpy as np
import scipy.io as sio

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


def get_csv_line_count(csv_file):
    with open(csv_file, 'r') as file:
        line_count = sum(1 for _ in file)
    return line_count


def save_data(output_dir, snapshot_data, calculations=None, n_springs=20, max_ants=100, continue_from_last=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if calculations is None:
        empty_springs = np.repeat(np.nan, n_springs).reshape(1, n_springs)
        empty_2_values = np.repeat(np.nan, 2).reshape(1, 2)
        empty_4_values = np.full((1, 4), np.nan)
        empty_ant_centers = np.repeat(np.nan, max_ants).reshape(1, max_ants)
        empty_needle_area_size = np.repeat(np.nan, 1).reshape(1, 1)
        empty_ants_attached_labels = np.array([np.nan for _ in range(max_ants)]).reshape(1, max_ants)
        arrays = [empty_springs, empty_springs, empty_springs, empty_springs, empty_springs, empty_springs,
                  empty_2_values, empty_2_values, empty_ant_centers, empty_ant_centers, empty_needle_area_size,
                  empty_ants_attached_labels, empty_ants_attached_labels, empty_4_values, empty_4_values,
                  empty_4_values, empty_4_values]
    else:
        arrays = [calculations.N_ants_around_springs, calculations.size_ants_around_springs,
                  calculations.fixed_ends_coordinates_x, calculations.fixed_ends_coordinates_y,
                  calculations.free_ends_coordinates_x, calculations.free_ends_coordinates_y,
                  calculations.needle_part_coordinates_x, calculations.needle_part_coordinates_y,
                  calculations.ants_centers_x, calculations.ants_centers_y,
                  np.array(calculations.object_needle_area_size).reshape(-1, 1),
                  calculations.ants_attached_labels.reshape(1, max_ants),
                  calculations.ants_attached_forgotten_labels.reshape(1, max_ants),
                  calculations.perspective_squares_properties[:, 0, 0].reshape(1, 4),
                  calculations.perspective_squares_properties[:, 0, 1].reshape(1, 4),
                  calculations.perspective_squares_properties[:, 1, 0].reshape(1, 4),
                  calculations.perspective_squares_properties[:, 1, 1].reshape(1, 4)]
    names = ["N_ants_around_springs", "size_ants_around_springs",
             "fixed_ends_coordinates_x", "fixed_ends_coordinates_y", "free_ends_coordinates_x",
             "free_ends_coordinates_y", "needle_part_coordinates_x", "needle_part_coordinates_y",
             "ants_centers_x", "ants_centers_y", "needle_area_sizes",
             "ants_attached_labels", "ants_attached_forgotten_labels",
             "perspective_squares_coordinates_y", "perspective_squares_coordinates_x",
             "perspective_squares_sizes_width", "perspective_squares_sizes_height"]
    pickle.dump(snapshot_data, open(os.path.join(output_dir, f'snap_data_{snapshot_data["current_time"]}.pickle'), "wb"))
    if snapshot_data["frame_count"]==0:
        for d, n in zip(arrays[:], names[:]):
            with open(os.path.join(output_dir, str(n) + '.csv'), 'wb') as f:
                np.savetxt(f, d, delimiter=',')
                f.close()
    else:
        for d, n in zip(arrays[:], names[:]):
            with open(os.path.join(output_dir, str(n) + '.csv'), 'a') as f:
                if continue_from_last:
                    line_count = get_csv_line_count(os.path.join(output_dir, str(n) + '.csv'))
                    if line_count == snapshot_data["frame_count"]+1:
                        continue
                    else:
                        np.savetxt(f, d, delimiter=',')
                else:
                        np.savetxt(f, d, delimiter=',')
                f.close()


def present_analysis_result(frame, calculations, video_name=" "):
    image_to_illustrate = frame
    for point_red in calculations.spring_ends_centers:
        image_to_illustrate = cv2.circle(image_to_illustrate, point_red.astype(int), 1, (0, 0, 255), 2)
    for point_blue in calculations.spring_middle_part_centers:
        image_to_illustrate = cv2.circle(image_to_illustrate, point_blue.astype(int), 1, (255, 0, 0), 2)
    for count_, angle in enumerate(calculations.springs_angles_ordered):
        if angle != 0:
            if angle in calculations.fixed_ends_edges_bundles_labels:
                point = calculations.fixed_ends_edges_centers[calculations.fixed_ends_edges_bundles_labels.index(angle)]
                image_to_illustrate = cv2.circle(image_to_illustrate, point, 1, (0, 0, 0), 2)
                image_to_illustrate = cv2.putText(image_to_illustrate, str(count_), point, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                  (255, 0, 0), 1)
                image_to_illustrate = cv2.circle(image_to_illustrate, point, 1, (0, 0, 0), 2)
            if angle in calculations.free_ends_edges_bundles_labels:
                point = calculations.free_ends_edges_centers[calculations.free_ends_edges_bundles_labels.index(angle)]
                image_to_illustrate = cv2.circle(image_to_illustrate, point, 1, (0, 0, 0), 2)
    for label, ant_center_x, ant_center_y in zip(calculations.ants_attached_labels,calculations.ants_centers_x[0], calculations.ants_centers_y[0]):
        if label != 0:
            image_to_illustrate = cv2.putText(image_to_illustrate, str(int(label)-1), (int(ant_center_x), int(ant_center_y)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    image_to_illustrate = cv2.circle(image_to_illustrate, calculations.object_center, 1, (0, 0, 0), 2)
    image_to_illustrate = cv2.circle(image_to_illustrate, calculations.tip_point, 1, (0, 255, 0), 2)
    image_to_illustrate = utils.crop_frame_by_coordinates(image_to_illustrate, calculations.object_crop_coordinates)
    cv2.imshow(video_name, utils.white_balance_bgr(image_to_illustrate))
    cv2.waitKey(1)


def main(video_path, output_dir, parameters, continue_from_last=False):
    cap = cv2.VideoCapture(video_path)
    if continue_from_last and len([f for f in os.listdir(output_dir) if f.startswith("snap_data")]) != 0:
        snaps = [f for f in os.listdir(output_dir) if f.startswith("snap_data")]
        snapshot_data = pickle.load(open(os.path.join(output_dir, snaps[-1]), "rb"))
        parameters["starting_frame"] = snapshot_data["frame_count"]
        snapshot_data["current_time"] = datetime.datetime.now().strftime("%d.%m.%Y-%H%M")
    else:
        snapshot_data = {"object_center_coordinates": parameters["object_center_coordinates"][0],
                     "tip_point": None, "springs_angles_reference_order": None,
                     "sum_needle_radius": 0, "analysed_frame_count": 0, "frame_count": 0,
                     "current_time": datetime.datetime.now().strftime("%d.%m.%Y-%H%M"),
                     "perspective_squares_coordinates": parameters["perspective_squares_coordinates"],
                     "skipped_frames": 0}
    cap.set(cv2.CAP_PROP_POS_FRAMES, parameters["starting_frame"])
    while True:
        ret, frame = cap.read()
        if (snapshot_data["skipped_frames"] % 25 == 0 and snapshot_data["skipped_frames"] != 0):
            print("\r Jumping 25 frames ahead, and analysing the entire frame", end="")
            cap.set(cv2.CAP_PROP_POS_FRAMES, parameters["starting_frame"] + snapshot_data["frame_count"] + 24)
            for i in range(24):
                save_data(output_dir, snapshot_data=snapshot_data, n_springs=parameters["n_springs"], continue_from_last=continue_from_last)
                snapshot_data["frame_count"] += 1
            snapshot_data["skipped_frames"] += 24
        else:
            if frame is None: break
            # cv2.imshow("frame", frame)
            # cv2.waitKey(0)
            try:
                calculations = Calculation(parameters, frame, snapshot_data)
                snapshot_data["object_center_coordinates"] = calculations.object_center[[1, 0]]
                snapshot_data["tip_point"] = calculations.tip_point
                snapshot_data["springs_angles_reference_order"] = calculations.springs_angles_reference_order
                snapshot_data["sum_needle_radius"] += int(calculations.object_needle_radius)
                snapshot_data["analysed_frame_count"] += 1
                snapshot_data["perspective_squares_coordinates"] = utils.swap_columns(calculations.perspective_squares_properties[:, 0])
                snapshot_data["skipped_frames"] = 0
                snapshot_data["perspective_square_dimensions"] = calculations.perspective_squares_properties[:, 1]
                save_data(output_dir, calculations=calculations, snapshot_data=snapshot_data, continue_from_last=continue_from_last,
                          n_springs=parameters["n_springs"], max_ants=parameters["max_ants_number"])
                present_analysis_result(frame, calculations, os.path.basename(video_path).split(".")[0])
                print("\r Analyzed frame number: ", snapshot_data["frame_count"], end=" "*150)
            except:
                print("\r Skipped frame: ", snapshot_data["frame_count"], end=" "*150)
                snapshot_data["skipped_frames"] += 1
                save_data(output_dir, snapshot_data=snapshot_data, n_springs=parameters["n_springs"], continue_from_last=continue_from_last)
            snapshot_data["frame_count"] += 1
        continue_from_last = False
    cap.release()
    if not os.path.exists(os.path.join(output_dir, f'analysis_ended_{snapshot_data["current_time"]}.pickle')):
        save_as_mathlab_matrix(output_dir)
    pickle.dump("video_ended", open(os.path.join(output_dir, f'analysis_ended_{snapshot_data["current_time"]}.pickle'), "wb"))

