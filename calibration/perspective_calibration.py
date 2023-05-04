# This code takes a video, extract 1 frame (specified as a parameter),
# crop the frame, and finds all the blobs on the surface,using Otsu filter from skimage.
# Then the code finds the center of mass of each blob, and calculates the distances of each point to each point.
# For each point it finds the 4 distances that are the smallest,
# it calculates whether each of the 4 distances that are greater than 1.3*median of all distances,
# if so it will remove the that distance.
# Then it puts all distances in one pool, and finds the 3 largest distances.
# Then it takes the points which are assigned to those distances, and it finds their center of mass,
# and calls it camera_closets.
# Eventually it returns camera_closest coordinates, and the coordinates of all other points.

import cv2
from skimage import filters
import numpy as np
import utils
import copy
from scipy.ndimage import label, center_of_mass
from general_video_scripts import collect_color_parameters
from skimage import transform

def extract_image(video_path,frame_number):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    return frame

def crop_image(image):
    print("Please select the crop coordinates:")
    crop_ccordinates = collect_color_parameters.pick_points(image, "crop").flatten()
    crop_ccordinates = crop_ccordinates[[0, 2, 1, 3]]
    image_cropped = utils.crop_frame_by_coordinates(copy.copy(image), crop_ccordinates)
    return image_cropped

def find_blobs(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = filters.threshold_otsu(frame_gray)
    bool = frame_gray < thresh
    labeled, labels = label(bool)
    labeled_copy = copy.copy(labeled)
    # frame_labeled = label2rgb(labeled_copy, image=frame_gray)
    # cv2.imshow("frame_labeled", frame_labeled)
    # cv2.waitKey(0)
    # while True:
    #     labeled_copy = copy.copy(labeled)
    #     # filter blobs by size
    #     # min_size = int(input("Please enter the minimum size of the blobs"))
    #     # max_size = int(input("Please enter the maximum size of the blobs"))
    #     # for i in range(1, labels + 1):
    #     #     if (labeled == i).sum() < min_size or (labeled == i).sum() > max_size:
    #     #         labeled_copy[labeled == i] = 0
    #     # present the blobs
    #     frame_labeled = label2rgb(labeled_copy, image=frame_gray)
    #     cv2.imshow("frame_labeled", frame_labeled)
    #     cv2.waitKey(0)
    #     if str(input("Happy?[Y,n]:")) in ["y", "Y"]:
    #         break
    # convert bool to binary
    binary = bool.astype(int)
    # cv2.imshow("binary", binary.astype(np.uint8)*255)
    # cv2.waitKey(0)
    labels = np.unique(labeled_copy)[1:]
    centers = np.array(center_of_mass(binary, labeled, labels))
    return centers, labels


def points_distances(centers, labels):
    centers_num = len(centers)
    distances = np.zeros([centers_num, centers_num])
    index = np.zeros([centers_num, centers_num]).astype(int)
    for label_i, idx_i in zip(labels,range(centers_num)):
        for label_j, idx_j in zip(labels,range(centers_num)):
            distances[idx_i, idx_j] = np.linalg.norm(centers[idx_i] - centers[idx_j])
            index[idx_i, idx_j] = label_j
    return distances, index


def remove_outliers(distances):
    smallest_distances = np.array([])
    for i in range(distances.shape[0]):
        smallest_distances = np.append(smallest_distances, np.sort(distances[i])[0:4])
    smallest_median = np.median(smallest_distances)
    smallest_distances[smallest_distances > 1.3 * smallest_median] = np.nan
    smallest_distances[smallest_distances == 0] = np.nan
    distances[np.isin(distances, smallest_distances) == False] = np.nan
    return distances

def color_scale(numbers):
    max_num = max(numbers)
    min_num = min(numbers)
    color_map = {}
    for num in numbers:
        normalized_value = (num - min_num) / (max_num - min_num)
        red = int(255 * (1 - normalized_value))
        blue = int(255 * normalized_value)
        color_map[num] = (red, 0, blue)
    return color_map


def present_distances(frame,distances,centers,labels):
    frame_copy = copy.copy(frame)
    distances_without_nans = distances[~np.isnan(distances)]
    colors_distances = color_scale(distances_without_nans.flatten())
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            if np.isnan(distances[i,j]):
                continue
            else:
                cen_i = tuple(centers[i][[1,0]].astype(int))
                cen_j = tuple(centers[j][[1,0]].astype(int))
                color = colors_distances[distances[i,j]]
                cv2.line(frame_copy, cen_i, cen_j, color, 2)
    # write the number next to each center, with small font:
    for i in range(len(centers)):
        center = tuple(centers[i][[1,0]].astype(int))
        cv2.putText(frame_copy, str(i), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow("frame_copy", frame_copy)
    cv2.waitKey(0)




def closest_to_camera(distances,centers,labels):
    # find the 3 largest distances, and their index
    distances_flatten_without_nan = distances.flatten()
    distances_flatten_without_nan = distances_flatten_without_nan[~np.isnan(distances_flatten_without_nan)]
    largest_distances_index = np.where(np.isin(distances, np.sort(distances_flatten_without_nan)[-3:]))[0]
    largest_distances_labels = np.unique(largest_distances_index)+1
    largest_centers = centers[labels== largest_distances_labels]
    center_largest_centers = np.mean(largest_centers,axis=0)
    return center_largest_centers



def estimate_projective_transformation_parameters(image):
    print("Please select the co-linear  coordinates:")
    points = input("write the points numbers as a list: ").split(",")
    points = [int(p) for p in points]
    dst = centers[points]
    # dst = collect_color_parameters.collect_points(image,4)
    # dst = utils.swap_columns(dst)
    # print("dst",dst)
    # up_down_distance =  np.linalg.norm(dst[1]-dst[3])
    # left_right_distance = np.linalg.norm(dst[0]-dst[2])
    # dst[0,1] = dst[0,1] - up_down_distance/2
    # dst[1,0] = dst[1,0] - left_right_distance/2
    # dst[2,1] = dst[2,1] + up_down_distance/2
    # dst[3,0] = dst[3,0] + left_right_distance/2
    # print(dst)
    # print(up_down_distance,left_right_distance)
    print("dst",dst)
    print(centers)
    dst_centers = []
    for d in dst:
        dst_centers.append(find_closest_point(d,centers))
    dst = np.array(dst_centers)
    print("dst",dst)
    distance = np.linalg.norm(dst[0]-dst[1])
    start_point = dst[0]
    src = np.array([start_point, start_point + [distance, 0], start_point + [distance, distance], start_point + [0, distance]])
    dst = utils.swap_columns(dst)

    tform3 = transform.ProjectiveTransform()
    tform3.estimate(src, dst)
    warped = transform.warp(image, tform3, output_shape=image.shape)

    warped = (warped * 255).astype(np.uint8)
    return warped

def find_closest_point(point,points):
    distances = np.sqrt(np.sum((points - point) ** 2, axis=1))
    closest_point = points[np.argmin(distances)]
    return closest_point

def perpective(image):
    image = crop_image(image)
    centers, labels = find_blobs(image)
    distances, index = points_distances(centers, labels)
    distances = remove_outliers(distances)
    # min-max difference
    min_max_diff = np.nanmax(distances) - np.nanmin(distances)
    print("min_max_diff: ", min_max_diff)
    present_distances(image, distances, centers, labels)
    return image, centers

if __name__ =="__main__":
    video_path = "Z:\\Dor_Gabay\\ThesisProject\\data\\videos\\perspective_calibration\\S5260001.MP4"
    # video_path = "Z:\\Dor_Gabay\\ThesisProject\\data\\videos\\15.9.22\\plus0.3mm_force\\S5280003.MP4"
    frame_number = 43

    frame = extract_image(video_path,frame_number)
    image, centers = perpective(frame)
    warped = estimate_projective_transformation_parameters(frame)
    # perpective(warped)
    print("Please select the co-linear  coordinates:")
    dst = collect_color_parameters.collect_points(warped, 4)
    dst = utils.swap_columns(dst)
    distances = np.linalg.norm(dst[0]-dst[2]), np.linalg.norm(dst[1]-dst[3])
    print("distances",distances)
