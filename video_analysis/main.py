import os
import cv2
import numpy as np
from utils import crop_frame_by_coordinates,create_circular_mask
import pickle
from collect_color_parameters import neutrlize_colour
from skimage.color import label2rgb
# local imports:
from springs_detector import Springs
from calculations import Calculation
from ants_detector import Ants

def save_data(calculations,output_dir,first_save):
    print("Saving data...")
    data_arrays = [calculations.springs_length,calculations.N_ants_around_springs,calculations.size_ants_around_springs,
                   calculations.springs_angles_to_nest,calculations.springs_angles_to_object]
    data_arrays_names = ["springs_length","N_ants_around_springs","size_ants_around_springs","springs_angles_to_nest",
                         "springs_angles_to_object"]
    for d,n in zip(data_arrays,data_arrays_names):
    # for dat in data_arrays_names:
        if first_save:
            with open(os.path.join(output_dir, str(n)+'.csv'), 'wb') as f:
                # exec(f"np.savetxt({f}, calculations.{dat}[:-1], delimiter=',')")
                np.savetxt(f, d[:-1], delimiter=',')
        else:
            with open(os.path.join(output_dir, str(n)+'.csv'), 'a') as f:
                # exec(f"np.savetxt({f}, calculations.{dat}[:-1], delimiter=',')")
                np.savetxt(f, d[:-1], delimiter=',')
    Calculation.clear_data(calculations)

def present_analysis_result(frame, springs, calculations, ants):
    # image_to_illustrate = copy.copy(frame)
    image_to_illustrate = frame
    for point_green in springs.green_centers:
        image_to_illustrate = cv2.circle(image_to_illustrate, point_green.astype(int), 1, (0, 255, 0), 2)
    for point_red in springs.red_centers:
        image_to_illustrate = cv2.circle(image_to_illustrate, point_red.astype(int), 1, (0, 0, 255), 2)
    # print(calculations.springs_angles_matrix[-1, :])
    for count_, angle in enumerate(calculations.springs_angles_matrix[-1, :]):
        if angle != 0:
            if angle in springs.fixed_ends_edges_bundles_labels:
                point = springs.fixed_ends_edges_centers[springs.fixed_ends_edges_bundles_labels.index(angle)]
                image_to_illustrate = cv2.circle(image_to_illustrate, point, 1, (255, 255, 255), 2)
                # springs.bundles_centers = springs.swap_columns(springs.bundles_centers)
                # print(springs.bundles_centers)
                # point = springs.bundles_centers[springs.fixed_ends_edges_bundles_labels.index(angle)]
                # print(point)
                image_to_illustrate = cv2.putText(image_to_illustrate, str(count_), point, cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0), 2)
                # image_to_illustrate = cv2.putText(image_to_illustrate, str(np.round(angle,2)), point, cv2.FONT_HERSHEY_SIMPLEX, 1,                                               (255, 0, 0), 2)
                image_to_illustrate = cv2.circle(image_to_illustrate, point, 1, (255, 255, 255), 2)

            if angle in springs.free_ends_edges_bundles_labels:
                point = springs.free_ends_edges_centers[springs.free_ends_edges_bundles_labels.index(angle)]
                image_to_illustrate = cv2.circle(image_to_illustrate, point, 1, (255, 255, 255), 2)
    image_to_illustrate = cv2.circle(image_to_illustrate, springs.object_center, 1, (255, 255, 255), 2)
    image_to_illustrate = cv2.circle(image_to_illustrate, springs.tip_point, 1, (255, 0, 0), 2)

    try:
        # print(np.unique(ants.corrected_labeled_image))
        # for label, point in zip(np.unique(ants.corrected_labeled_image)[1:],ants.corrected_labels_center_of_mass):
        #     point = np.array((point[1],point[0])).astype(int)
        #     # image_to_illustrate = cv2.circle(image_to_illustrate, point, 1, (0, 255, 0), 2)
        #     image_to_illustrate = cv2.putText(image_to_illustrate, str(label), point, cv2.FONT_HERSHEY_SIMPLEX, 1,
        #                                       (255, 0, 0), 2)
        image_to_illustrate = label2rgb(ants.labaled_ants, image=image_to_illustrate, bg_label=0)
        # image_to_illustrate = utils.draw_lines_on_image(image_to_illustrate, ants.ants_lines)
        # print(ants.ants_lines)
    except: pass

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
    # print(cap)
    # if starting_frame is not None:
    #     parameters['starting_frame'] = starting_frame
    if start_frame is not None:
        # print("start_frame: ", start_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, parameters["starting_frame"])
        # print("cap",  parameters["starting_frame"])
    previous_detections = None
    count = 0
    # images = []
    # joints = []
    # for count, x in enumerate(range(N_ITERATIONS)):
    balance_values = (1.5, 1.5, 1.5)
    while True:
        # currFrame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        # print("frame: ", frame)
        if frame is None:
            print("End of video")
            save_data(calculations,count,first_save=False)
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
            else:
                # ants.track_ants(ants.labaled_ants, previous_detections[2])
                calculations.make_calculations(springs, ants,previous_calculations=previous_detections)
            previous_detections = [springs.object_center,springs.mask_blue_full, ants.labaled_ants, calculations.springs_angles_to_nest]

            print("frame number:",count, end="\r")
            SAVE_GAP = 100
            if (count%SAVE_GAP == 0 and count != 0):# or count==(N_ITERATIONS-1):
                if count==SAVE_GAP: first_save = True
                else: first_save = False
                save_data(calculations, output_dir, first_save)
                # create_video(output_dir, images,"video")
                # create_video(output_dir, joints,"joints")

            # Presnting analysis:
            results_frame,results_joints = present_analysis_result(frame, springs, calculations, ants)
            # images.append(results_frame)
            # joints.append(results_joints)
            count += 1
        except:
            print("skipped frame ", end="\r")
            # images.append(frame*0)
            # joints.append(frame*0)
            calculations.add_blank_row()
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

