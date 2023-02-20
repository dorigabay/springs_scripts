import os
import cv2
import numpy as np
from utils import crop_frame_by_coordinates,create_circular_mask
import pickle
from collect_color_parameters import neutrlize_colour
# local imports:
from springs_detector import Springs
from calculator import Calculation
from ants_detector import Ants
from skimage.color import label2rgb

def save_data(calculations,output_dir,first_save):
    print("Saving data...")
    # create coordinates folder, within the output directory, in case it doesn't exist:
    coordinates_output_dir = os.path.join(output_dir, "coordinates")
    if not os.path.exists(coordinates_output_dir):
        os.makedirs(coordinates_output_dir)
    # data_arrays = [calculations.springs_length,calculations.N_ants_around_springs,calculations.size_ants_around_springs,
    #                calculations.angles_to_nest,calculations.angles_to_object_free,calculations.angles_to_object_fixed]
    # data_arrays_names = ["springs_length","N_ants_around_springs","size_ants_around_springs","angles_to_nest",
    #                      "angles_to_object_free","angles_to_object_fixed"]
    data_arrays = [calculations.N_ants_around_springs,calculations.size_ants_around_springs]
    data_arrays_names = ["N_ants_around_springs","size_ants_around_springs"]
    data_coordinates = [calculations.fixed_ends_coordinates_x, calculations.fixed_ends_coordinates_y,
                        calculations.free_ends_coordinates_x, calculations.free_ends_coordinates_y,
                        calculations.blue_part_coordinates_x, calculations.blue_part_coordinates_y]
    data_coordinates_names = ["fixed_ends_coordinates_x", "fixed_ends_coordinates_y", "free_ends_coordinates_x",
                              "free_ends_coordinates_y", "blue_part_coordinates_x", "blue_part_coordinates_y"]

    for d,n in zip(data_arrays,data_arrays_names):
        if first_save:
            with open(os.path.join(output_dir, str(n)+'.csv'), 'wb') as f:
                np.savetxt(f, d[:-1], delimiter=',')
        else:
            with open(os.path.join(output_dir, str(n)+'.csv'), 'a') as f:
                np.savetxt(f, d[:-1], delimiter=',')

    for d,n in zip(data_coordinates,data_coordinates_names):
        if first_save:
            with open(os.path.join(coordinates_output_dir, str(n)+'.csv'), 'wb') as f:
                np.savetxt(f, d[:-1], delimiter=',')
        else:
            with open(os.path.join(coordinates_output_dir, str(n)+'.csv'), 'a') as f:
                np.savetxt(f, d[:-1], delimiter=',')
    Calculation.clear_data(calculations)

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
    for count_, angle in enumerate(calculations.springs_angles_matrix[-1, :]):
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
    image_to_illustrate = cv2.circle(image_to_illustrate, springs.tip_point, 1, (255, 0, 0), 2)
    # try:
    #     ants.labeled_ants[~np.isin(ants.labeled_ants, calculations.ants_attached)] = 0
    #     image_to_illustrate = label2rgb(ants.labeled_ants, image=image_to_illustrate, bg_label=0)
    # except: pass
    cv2.imshow("frame", image_to_illustrate)
    cv2.waitKey(1)
    return image_to_illustrate, calculations.joints

def create_video(output_dir, images, vid_name):
    print("Creating video...")
    height, width = images[0].shape[:2]
    video = cv2.VideoWriter(os.path.join(output_dir, vid_name+'.MP4'), cv2.VideoWriter_fourcc(*'mp4v'), 50, (width, height))
    for image in images:
        # convert mask to 3 channel:
        if len(image.shape) == 2:
            image = image.astype(np.uint8)*255
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        video.write(image)
    video.release()

def main(video_path, output_dir, parameters,start_frame=None):
    print("video_path: ", video_path)

    cap = cv2.VideoCapture(video_path)
    if start_frame is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, parameters["starting_frame"])
    previous_detections = None
    count = 0
    frames_analysed = 0
    frames_until_saving = 0
    while True:
        ret, frame = cap.read()
        frames_until_saving += 1
        if frame is None:
            print("End of video")
            save_data(calculations,output_dir,first_save=False)
            break # break the loop if there are no additional frame in the video
            # When setting the parameters, there's an option to set fixed coordinates for cropping the frame
        try:
            frame_neutrlized = neutrlize_colour(frame)
            if parameters["crop_coordinates"] != None:
                frame = crop_frame_by_coordinates(frame, parameters["crop_coordinates"])
                frame_neutrlized = crop_frame_by_coordinates(frame_neutrlized, parameters["crop_coordinates"])
            springs = Springs(parameters, frame_neutrlized, previous_detections)
            ants = Ants(frame, springs)
            if count == 0:
                calculations = Calculation(springs, ants)
                sum_blue_radius = 0
                # previous_detections = [springs.object_center,springs.mask_blue_full, ants.labaled_ants, calculations.springs_angles_to_nest]
            else:
                calculations.make_calculations(springs, ants)#,previous_calculations=previous_detections)
            sum_blue_radius += springs.blue_radius
            frames_analysed += 1
            previous_detections = [springs.object_center,springs.tip_point, sum_blue_radius, frames_analysed]#, calculations.springs_angles_to_nest[-1,:].reshape(1,-1)]

            print("frame number:",count, end="\r")
            SAVE_GAP = 100
            if (count%SAVE_GAP == 0 and count != 0):
                if count==SAVE_GAP: first_save = True
                else: first_save = False
                save_data(calculations, output_dir, first_save)
                frames_until_saving = 1

            # Presnting analysis:
            present_analysis_result(frame_neutrlized, springs, calculations, ants)
            count += 1
        except:
            print("skipped frame ", end="\r")
            calculations.add_blank_row(number_of_rows=frames_until_saving)
            count += 1
            continue

    cv2.destroyAllWindows()
    cap.release()



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

