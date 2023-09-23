import os
import pickle
import datetime
import cv2
# local imports:
from video_analysis import calculator
from video_analysis.ants_detector import Ants
from video_analysis.springs_detector import Springs
from video_analysis import perspective_squares
from video_analysis import utils


def present_analysis_result(frame, calculations, springs, video_name=" "):
    image_to_illustrate = frame
    for point_red in springs.spring_ends_centers:
        image_to_illustrate = cv2.circle(image_to_illustrate, point_red.astype(int), 1, (0, 0, 255), 2)
    for point_blue in springs.spring_middle_part_centers:
        image_to_illustrate = cv2.circle(image_to_illustrate, point_blue.astype(int), 1, (255, 0, 0), 2)
    for count_, angle in enumerate(calculations.springs_angles_ordered):
        if angle != 0:
            if angle in springs.fixed_ends_edges_bundles_labels:
                point = springs.fixed_ends_edges_centers[springs.fixed_ends_edges_bundles_labels.index(angle)]
                image_to_illustrate = cv2.circle(image_to_illustrate, point, 1, (0, 0, 0), 2)
                image_to_illustrate = cv2.putText(image_to_illustrate, str(count_), point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                image_to_illustrate = cv2.circle(image_to_illustrate, point, 1, (0, 0, 0), 2)
            if angle in springs.free_ends_edges_bundles_labels:
                point = springs.free_ends_edges_centers[springs.free_ends_edges_bundles_labels.index(angle)]
                image_to_illustrate = cv2.circle(image_to_illustrate, point, 1, (0, 0, 0), 2)
    for label, ant_center_x, ant_center_y in zip(calculations.ants_attached_labels, calculations.ants_centers_x[0], calculations.ants_centers_y[0]):
        if label != 0:
            point = (int(ant_center_x), int(ant_center_y))
            image_to_illustrate = cv2.putText(image_to_illustrate, str(int(label)-1), point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    image_to_illustrate = cv2.circle(image_to_illustrate, springs.object_center_coordinates, 1, (0, 0, 0), 2)
    image_to_illustrate = cv2.circle(image_to_illustrate, springs.tip_point, 1, (0, 255, 0), 2)
    image_to_illustrate = utils.crop_frame_by_coordinates(image_to_illustrate, springs.object_crop_coordinates)
    cv2.imshow(video_name, utils.white_balance_bgr(image_to_illustrate))
    cv2.waitKey(1)


def create_snapshot_data(init=True, parameters=None, snapshot_data=None, calculations=None, perspective_squares=None, springs=None):
    if init:
        snapshot_data = {"object_center_coordinates": parameters["object_center_coordinates"][0],
                         "tip_point": None, "springs_angles_reference_order": None,
                         "sum_needle_radius": 0, "analysed_frame_count": 0, "frame_count": 0,
                         "current_time": datetime.datetime.now().strftime("%d.%m.%Y-%H%M"),
                         "perspective_squares_coordinates": parameters["perspective_squares_coordinates"],
                         "skipped_frames": 0}
    else:
        snapshot_data["object_center_coordinates"] = springs.object_center_coordinates[[1, 0]]
        snapshot_data["tip_point"] = springs.tip_point
        snapshot_data["sum_needle_radius"] += int(springs.object_needle_radius)
        snapshot_data["springs_angles_reference_order"] = calculations.springs_angles_reference_order
        snapshot_data["analysed_frame_count"] += 1
        snapshot_data["skipped_frames"] = 0
        snapshot_data["perspective_squares_coordinates"] = utils.swap_columns(perspective_squares.perspective_squares_properties[:, 0:2])
    return snapshot_data


def main(video_path, output_dir, parameters, continue_from_last=False):
    cap = cv2.VideoCapture(video_path)
    if continue_from_last and len([f for f in os.listdir(output_dir) if f.startswith("snap_data")]) != 0:
        snaps = [f for f in os.listdir(output_dir) if f.startswith("snap_data")]
        snapshot_data = pickle.load(open(os.path.join(output_dir, snaps[-1]), "rb"))
        parameters["starting_frame"] = snapshot_data["frame_count"]
        snapshot_data["current_time"] = datetime.datetime.now().strftime("%d.%m.%Y-%H%M")
    else:
        snapshot_data = create_snapshot_data(init=True, parameters=parameters)
    cap.set(cv2.CAP_PROP_POS_FRAMES, parameters["starting_frame"])
    while True:
        ret, frame = cap.read()
        if snapshot_data["skipped_frames"] % 25 == 0 and snapshot_data["skipped_frames"] != 0:
            print("\r Jumping 25 frames ahead, and analysing the entire frame", end=" "*150)
            cap.set(cv2.CAP_PROP_POS_FRAMES, parameters["starting_frame"] + snapshot_data["frame_count"] + 24)
            for i in range(24):
                perspective_squares.save_data(output_dir, snapshot_data, continue_from_last=continue_from_last)
                calculator.save_data(output_dir, snapshot_data, parameters, continue_from_last=False)
                snapshot_data["frame_count"] += 1
            snapshot_data["skipped_frames"] += 24
        else:
            if frame is None:
                break
            try:
                squares = perspective_squares.PerspectiveSquares(parameters, frame, snapshot_data)
                perspective_squares.save_data(output_dir, snapshot_data, squares, continue_from_last)
                try:
                    springs = Springs(parameters, frame, snapshot_data)
                    ants = Ants(frame, springs, squares)
                    calculations = calculator.Calculation(parameters, snapshot_data, springs, ants)
                    snapshot_data = create_snapshot_data(init=False, snapshot_data=snapshot_data, calculations=calculations, perspective_squares=squares, springs=springs)
                    calculator.save_data(output_dir, snapshot_data, parameters, calculations, continue_from_last=False)
                    print("\r Analyzed frame number: ", snapshot_data["frame_count"], end=" "*150)
                    present_analysis_result(frame, calculations, springs, os.path.basename(video_path).split(".")[0])
                except:
                    print("\r Skipped frame: ", snapshot_data["frame_count"], " (a problem in springs detection)", end=" "*150)
                    calculator.save_data(output_dir, snapshot_data, parameters, continue_from_last=False)
                    snapshot_data["skipped_frames"] += 1
            except:
                print("\r Skipped frame: ", snapshot_data["frame_count"], end=" "*150)
                perspective_squares.save_data(output_dir, snapshot_data, continue_from_last=continue_from_last)
                calculator.save_data(output_dir, snapshot_data, parameters, continue_from_last=False)
                snapshot_data["skipped_frames"] += 1
            snapshot_data["frame_count"] += 1
        continue_from_last = False
    cap.release()
    if not os.path.exists(os.path.join(output_dir, f'analysis_ended_{snapshot_data["current_time"]}.pickle')):
        utils.save_as_mathlab_matrix(output_dir)
    pickle.dump("video_ended", open(os.path.join(output_dir, f'analysis_ended_{snapshot_data["current_time"]}.pickle'), "wb"))

