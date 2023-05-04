import os
import cv2
import numpy as np
import pickle
from general_video_scripts.collect_color_parameters import neutrlize_colour
# local imports:
from video_analysis.springs_detector import Springs
from video_analysis.calculator import Calculation
from video_analysis.ants_detector import Ants
import general_video_scripts


def save_data(output_dir, first_save=False, calculations=None, save_empty=False, n_springs=20):
    output_dir = os.path.join(output_dir, "raw_analysis")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if save_empty:
        empty = np.zeros((1,n_springs))
        empty[empty == 0] = np.nan
        arrays = [empty for _ in range(8)]
    else:
        arrays = [calculations.N_ants_around_springs,calculations.size_ants_around_springs,
                            calculations.fixed_ends_coordinates_x, calculations.fixed_ends_coordinates_y,
                            calculations.free_ends_coordinates_x, calculations.free_ends_coordinates_y,
                            calculations.blue_part_coordinates_x, calculations.blue_part_coordinates_y]
    names = ["N_ants_around_springs","size_ants_around_springs",
                              "fixed_ends_coordinates_x", "fixed_ends_coordinates_y", "free_ends_coordinates_x",
                              "free_ends_coordinates_y", "blue_part_coordinates_x", "blue_part_coordinates_y"]
    for d,n in zip(arrays,names):
        if first_save:
            with open(os.path.join(output_dir, str(n)+'.csv'), 'wb') as f:
                np.savetxt(f, d, delimiter=',')
        else:
            with open(os.path.join(output_dir, str(n)+'.csv'), 'a') as f:
                np.savetxt(f, d, delimiter=',')
    # Calculation.clear_data(calculations)

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

# def create_video(output_dir, images, vid_name):
#     print("Creating video...")
#     height, width = images[0].shape[:2]
#     video = cv2.VideoWriter(os.path.join(output_dir, vid_name+'.MP4'), cv2.VideoWriter_fourcc(*'mp4v'), 50, (width, height))
#     for image in images:
#         # convert mask to 3 channel:
#         if len(image.shape) == 2:
#             image = image.astype(np.uint8)*255
#             image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#         video.write(image)
#     video.release()

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
    # frames_until_saving = 1
    # iterated_frame = -1
    sum_blue_radius = 0
    first_save = True

    while True:
        ret, frame = cap.read()
        # if count == 0:
        #     iterated_frame += 1
        # elif count == 1:
        #     with open(os.path.join(output_dir, f"starting_frame_{iterated_frame}.txt"), "w") as f:
        #         print(f"For video {video_path} the starting frame is {iterated_frame}")
        #         f.write(str(iterated_frame))
        # elif count > 0:
        #     frames_until_saving += 1

        # print("Frame number: ", count)
        if frame is None:
            print("End of video")
            # save_data(output_dir,first_save=False)
            break

        frame = neutrlize_colour(frame)
        if parameters["crop_coordinates"] != None:
            frame = general_video_scripts.utils.crop_frame_by_coordinates(frame, parameters["crop_coordinates"])
        # try:
        springs = Springs(parameters, frame, previous_detections)
        ants = Ants(frame, springs)
        calculations = Calculation(springs, ants, previous_detections)
        # if count == 0:
            # calculations = Calculation(springs, ants)
            # sum_blue_radius = 0
            # start_analyzing_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        # else:
        #     calculations.make_calculations(springs, ants)
        sum_blue_radius += springs.blue_radius
        # frames_analysed += 1
        previous_detections = [springs.object_center,springs.tip_point, sum_blue_radius, frames_analysed, calculations.springs_angles_ordered]
        save_data(output_dir,calculations=calculations, first_save=first_save)
        print("frame number:",count, end="\r")
        present_analysis_result(frame, springs, calculations, ants)
        count += 1
            # SAVE_GAP = 1
            # if (count % SAVE_GAP == 0 and count != 0):
            #     if count == SAVE_GAP:
            #         first_save = True
            #     else:
            #         first_save = False
            #     save_data(calculations, output_dir, first_save)
            #     frames_until_saving = 1

            # Presnting analysis:
        # except:
        #     print("Skipped frame: ",count, end="\r")
        #     # if count != 0:
        #     print(first_save)
        #     save_data(output_dir, save_empty=True, first_save=first_save, n_springs=parameters["n_springs"])
        #         # calculations.add_blank_row(number_of_rows=frames_until_saving)
        #     #     calculations.add_blank_row(number_of_rows=frames_until_saving)
        #     count += 1
        first_save = False
        # if first_save:

    cv2.destroyAllWindows()
    cap.release()
    #  save the analysis starting frame:



if __name__  == '__main__':
    # VIDEO_PATH = "Z:/Dor_Gabay/videos/18.9.22/plus0.5mm_force/S5290002.MP4"
    VIDEO_PATH = "Z:/Dor_Gabay/videos/10.9/plus0_force/S5200007.MP4"
    # PARAMETERS_PATH = "Z:/Dor_Gabay/videos/18.9.22/video_preferences.pickle"
    PARAMETERS_PATH = "Z:/Dor_Gabay/videos/10.9/video_preferences.pickle"
    # OUTPUT_DATA = "Z:/Dor_Gabay/data/test/18.9.22/plus0.5mm_force/S5290002/"
    OUTPUT_DATA = "Z:/Dor_Gabay/data/test/10.9/plus0_force/S5200007/"

    # while True:
    #     parameters = set_parameters(VIDEO_PATH, 0, 0, True, image=False)
    #     if show_parameters_result(VIDEO_PATH, parameters,image=False):
    #         break
    #
    # with open(os.path.join(CODE_PATH,'parameters_S523000
    # 7.pickle'), 'wb') as handle:
    #     pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open(os.path.join(CODE_PATH, 'parameters_S5230005.pickle'), 'rb') as handle:
    #     parameters = pickle.load(handle)

    with open(PARAMETERS_PATH, 'rb') as handle:
        # parameters = pickle.load(handle)["Z:\\Dor_Gabay\\videos\\18.9.22\\plus0.5mm_force\\S5290002.MP4"]
        parameters = pickle.load(handle)["..\\..\\videos\\10.9\\plus0_force\\S5200007.MP4"]
        # parameters["starting_frame"] = 75
        # print(parameters)
        main(VIDEO_PATH,OUTPUT_DATA,parameters)

