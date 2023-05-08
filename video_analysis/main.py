import cv2
import os
import numpy as np
import scipy.io as sio
import pickle
from general_video_scripts.collect_color_parameters import neutrlize_colour
# local imports:
from video_analysis.springs_detector import Springs
from video_analysis.calculator import Calculation
from video_analysis.ants_detector import Ants
import general_video_scripts


def save_as_mathlab_matrix(output_dir):
    # output_dir = "Z:\\Dor_Gabay\\ThesisProject\\data\\videos\\15"
    output_dir = os.path.join(output_dir, "raw_analysis")
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
    execution_string = f"matlab -r ""ants_tracking('"+output_dir+"\\')"""
    os.system(execution_string)


def save_blue_areas_median(output_dir):
    blue_area_sizes = np.loadtxt(os.path.join(output_dir, "blue_area_sizes.csv"), delimiter=",")
    median_blue_area_size = np.median(blue_area_sizes)
    with open(os.path.join(output_dir, "blue_median_area.pickle"), 'wb') as f:
        pickle.dump(median_blue_area_size, f)


def save_data(output_dir, first_save=False, calculations=None, save_empty=False, n_springs=20):
    output_dir = os.path.join(output_dir, "raw_analysis")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if save_empty:
        empty = np.zeros((1,n_springs))
        empty[empty == 0] = np.nan
        empty_list = [np.nan for _ in range(n_springs)]
        arrays = [empty for _ in range(11)]+[empty_list]
    else:
        arrays = [calculations.N_ants_around_springs,calculations.size_ants_around_springs,
                  calculations.fixed_ends_coordinates_x, calculations.fixed_ends_coordinates_y,
                  calculations.free_ends_coordinates_x, calculations.free_ends_coordinates_y,
                  calculations.blue_part_coordinates_x, calculations.blue_part_coordinates_y,
                  calculations.ants_centers_x, calculations.ants_centers_y, np.array(calculations.blue_area_sizes).reshape(-1,1),
                  calculations.ants_attached_labels]
    names = ["N_ants_around_springs","size_ants_around_springs",
             "fixed_ends_coordinates_x", "fixed_ends_coordinates_y", "free_ends_coordinates_x",
             "free_ends_coordinates_y", "blue_part_coordinates_x", "blue_part_coordinates_y",
             "ants_centers_x", "ants_centers_y","blue_area_sizes",
             "ants_attached_labels"]
    if first_save:
        for d,n in zip(arrays[:11],names[:11]):
            with open(os.path.join(output_dir, str(n)+'.csv'), 'wb') as f:
                np.savetxt(f, d, delimiter=',')
        pickle.dump(arrays[-1], open(os.path.join(output_dir,"ants_attached_labels.pickle"), "wb"))
    else:
        for d,n in zip(arrays[:11],names[:11]):
            with open(os.path.join(output_dir, str(n)+'.csv'), 'a') as f:
                np.savetxt(f, d, delimiter=',')
        pickle.dump(arrays[-1], open(os.path.join(output_dir,"ants_attached_labels.pickle"), "ab"))


def present_analysis_result(frame, springs, calculations, ants):
    # angles_to_object_free = calculations.angles_to_object_free[-1,:]+np.pi
    # angles_to_object_fixed = calculations.angles_to_object_fixed[-1,:]+np.pi
    # pulling_angle = angles_to_object_free- angles_to_object_fixed
    # pulling_angle = (pulling_angle+np.pi)%(2*np.pi)-np.pi
    # pulling_angle = np.round(pulling_angle,4)
    # number_of_ants = calculations.N_ants_around_springs[-1,:]
    image_to_illustrate = frame
    for point_green in springs.green_centers:
        image_to_illustrate = cv2.circle(image_to_illustrate, point_green.astype(int), 1, (0, 255, 0), 2)
    for point_red in springs.red_centers:
        image_to_illustrate = cv2.circle(image_to_illustrate, point_red.astype(int), 1, (0, 0, 255), 2)
    for count_, angle in enumerate(calculations.springs_angles_ordered):
        if angle != 0:
            if angle in springs.fixed_ends_edges_bundles_labels:
                point = springs.fixed_ends_edges_centers[springs.fixed_ends_edges_bundles_labels.index(angle)]
                image_to_illustrate = cv2.circle(image_to_illustrate, point, 1, (0, 0, 0), 2)
                image_to_illustrate = cv2.putText(image_to_illustrate, str(count_), point, cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0), 2)
                image_to_illustrate = cv2.circle(image_to_illustrate, point, 1, (0, 0, 0), 2)
            if angle in springs.free_ends_edges_bundles_labels:
                point = springs.free_ends_edges_centers[springs.free_ends_edges_bundles_labels.index(angle)]
                image_to_illustrate = cv2.circle(image_to_illustrate, point, 1, (0, 0, 0), 2)
                # image_to_illustrate = cv2.putText(image_to_illustrate, str(pulling_angle[count_]), point, cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0), 2)
                # image_to_illustrate = cv2.putText(image_to_illustrate, str(number_of_ants[count_]), point, cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0), 2)
                # image_to_illustrate = cv2.putText(image_to_illustrate, str(count_), point, cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0), 2)

    image_to_illustrate = cv2.circle(image_to_illustrate, springs.object_center, 1, (0, 0, 0), 2)
    # print("object center: ", springs.object_center)
    image_to_illustrate = cv2.circle(image_to_illustrate, springs.tip_point, 1, (255, 0, 0), 2)
    # try:
    #     ants.labeled_ants[~np.isin(ants.labeled_ants, calculations.ants_attached)] = 0
    #     image_to_illustrate = label2rgb(ants.labeled_ants, image=image_to_illustrate, bg_label=0)
    # except: pass
    cv2.imshow("frame", image_to_illustrate)
    cv2.waitKey(1)
    return image_to_illustrate, calculations.joints


def main(video_path, output_dir, parameters,starting_frame=None):
    cap = cv2.VideoCapture(video_path)
    if starting_frame is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, parameters["starting_frame"])
    parameters["n_springs"] = 20
    previous_detections = None
    count = 0
    frames_analysed = 0
    sum_blue_radius = 0
    first_save = True
    import time
    t0 = time.time()
    # for i in range(10):
    while True:
        ret, frame = cap.read()
        if frame is None:
            print("End of video")
            break
        frame = neutrlize_colour(frame)
        if parameters["crop_coordinates"] != None:
            frame = general_video_scripts.utils.crop_frame_by_coordinates(frame, parameters["crop_coordinates"])
        # try:
        springs = Springs(parameters, frame, previous_detections)
        ants = Ants(frame, springs)
        calculations = Calculation(springs, ants, previous_detections)
        frames_analysed += 1
        sum_blue_radius += springs.blue_radius
        previous_detections = [springs.object_center,springs.tip_point, sum_blue_radius, frames_analysed, calculations.springs_angles_ordered]
        save_data(output_dir,calculations=calculations, first_save=first_save)
        # present_analysis_result(frame, springs, calculations, ants)
        # print("Analyzed frame number:",count, end="\r")
        # except:
        #     print("Skipped frame: ",count, end="\r")
        #     save_data(output_dir, save_empty=True, first_save=first_save, n_springs=parameters["n_springs"])
        count += 1
        first_save = False
    cap.release()
    print("Time: ", time.time()-t0)
    save_as_mathlab_matrix(output_dir)
    save_blue_areas_median(output_dir)

