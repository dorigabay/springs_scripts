# import numpy as np
from scipy.ndimage import label, binary_fill_holes
from scipy.spatial import ConvexHull
from skimage import draw
# import tkinter as tk
# from PIL import Image, ImageTk
# import pickle
# import uuid
import numpy as np
import cv2
# from collections import Counter
from skimage.morphology import erosion, convex_hull_image
from sklearn.cluster import DBSCAN
from skimage.morphology import binary_dilation, binary_erosion

def close_element(mask, structure):
  mask = convert_bool_to_binary(binary_dilation(mask, structure))
  mask = convert_bool_to_binary(binary_erosion(mask, structure))
  return mask

def convert_bool_to_binary(bool_mask):
  binary_mask = np.zeros(bool_mask.shape, "int")
  binary_mask[bool_mask] = 1
  return binary_mask


# def separate_blobs(labeled_image, threshold):
#   """
#   This function separates the blobs in the labeled image into different labels.
#   It calculates the angle between each point in the contour of the blob and its neighbors,
#   and if the angle is greater than the threshold, the blob is separated into two blobs.
#   :param labeled_image: a matrix with integer labels for each blob
#   :param threshold: the angle between a point in the contour and its neighbors that determines whether to separate the blob
#   :return: a matrix with the separated blobs labeled with different integers
#   """
#   # Create a copy of the labeled image to modify
#   label_image = labeled_image.copy()
#
#   # Get the unique labels in the labeled image
#   labels = np.unique(label_image)
#
#   # Iterate through each label
#   for label in labels:
#     # Get the binary image of the current label
#     blob = label_image == label
#
#     # Calculate the angle between each point in the contour and its neighbors
#     points = np.argwhere(blob)
#     x = points[:, 1]
#     y = points[:, 0]
#     dx = np.diff(x)
#     dy = np.diff(y)
#     angles = np.arctan2(dy, dx)
#
#     # If the blob is empty or contains only a single point, skip it
#     if angles.shape[0] == 0:
#       continue
#
#     # Find the "break point" in the contour where the angle is significantly different from its neighbors
#     break_point = None
#     for i in range(angles.shape[0]):
#       if abs(angles[i] - angles[(i + 1) % angles.shape[0]]) > threshold:
#         break_point = (x[i], y[i])
#         break
#
#     # If a break point was found, separate the blob along the line formed by the break point and its neighbor
#     if break_point is not None:
#       x1, y1 = break_point
#       x2, y2 = x[(i + 1) % angles.shape[0]], y[(i + 1) % angles.shape[0]]
#       y = np.linspace(y1, y2, num=abs(y1 - y2))
#       x = np.linspace(x1, x2, num=abs(x1 - x2))
#       for i in range(len(x)):
#         if blob[int(y[i]), int(x[i])]:
#           label_image[int(y[i]), int(x[i])] = label + 1
#   return label_image

# def find_lines_in_mask(mask):
#   mask = np.stack((mask, mask, mask), axis=2)
#   mask = mask.mean(axis=2)
#   y, x = np.where(mask)
#   points = np.column_stack((x, y))
#
#   # Use DBSCAN to cluster the points into groups
#   dbscan = DBSCAN(eps=1, min_samples=2)
#   labels = dbscan.fit_predict(points)
#
#   # Find the start and end points of each group
#   lines = []
#   for label in np.unique(labels):
#     if label == -1:
#       continue
#     group = points[labels == label]
#     x1, y1 = group[0]
#     x2, y2 = group[-1]
#     lines.append((x1, y1, x2, y2))
#   return lines

def cluster_vectors(vectors, threshold):
  """
  Clusters the given vectors into groups based on the given threshold.
  Vectors that are within the threshold distance of each other are considered
  to be in the same group.
  """
  # Convert the list of vectors to a numpy array
  vectors = np.array(vectors)

  # Use DBSCAN to cluster the vectors into groups
  from sklearn.cluster import DBSCAN
  dbscan = DBSCAN(eps=threshold, min_samples=2)
  labels = dbscan.fit_predict(vectors)

  # Return the clusters as a list of lists
  return [vectors[labels == label].tolist() for label in np.unique(labels)]

def find_lines_in_mask(mask):
  mask = np.stack((mask, mask, mask), axis=2)
  mask = mask.mean(axis=2)
  y, x = np.where(mask)
  points = np.column_stack((x, y))

  # Use DBSCAN to cluster the points into groups
  from sklearn.cluster import DBSCAN
  dbscan = DBSCAN(eps=1, min_samples=2)
  labels = dbscan.fit_predict(points)

  # Find the start and end points of each group
  lines = []
  for label in np.unique(labels):
    if label == -1:
      continue
    group = points[labels == label]
    for i in range(group.shape[0] - 1):
      x1, y1 = group[i]
      x2, y2 = group[i + 1]
      lines.append((x1, y1, x2, y2))

  lines = cluster_vectors(lines, 5)
  lines = [x[0] for x in lines]
  return lines

def draw_lines_on_image(image, lines):
  for line in lines:
    y1, x1, y2, x2 = line
    image = cv2.line(image, (y1, x1), (y2, x2), (0, 255, 0), 3)
    return image
# def separate_blobs(labeled, threshold):
#   # Create a new array with the same shape as the original labeled array
#   separated = np.zeros_like(labeled)
#   # Iterate over each label in the labeled array
#   for label in np.unique(labeled):
#     # Extract the mask for the current label
#     mask = labeled == label
#     # Erode the mask until the ratio between the space in the convex hull
#     # and the space of the blob is close enough to 1
#     while True:
#       # Compute the convex hull of the blob
#       hull = convex_hull_image(mask)
#       # Calculate the ratio between the space in the convex hull and the space of the blob
#       ratio = np.sum(hull) / np.sum(mask)
#       # If the ratio is close enough to 1, stop eroding and update the separated array
#       if np.abs(ratio - 1) < threshold:
#         separated[mask] = label
#         break
#       # Erode the mask
#       mask = erosion(mask)
#   # Return the separated array
#   return separated


# def separate_blobs(image, threshold):
#     # Check the number of dimensions of the image
#     if image.ndim == 3:
#         # Convert the image to grayscale
#         gray = image.mean(axis=2)
#     else:
#         gray = image
#
#     # Threshold the image to create a binary image
#     binary = gray > 127
#
#     # Fill the holes in the blobs
#     binary = binary_fill_holes(binary)
#
#     # Label the blobs in the image
#     _, labels = label(binary)
#
#     # Iterate over the labels
#     for blob_label in np.unique(labels)[1:]:
#         # Select the current blob
#         blob = labels == blob_label
#
#         # Calculate the area of the blob
#         area = blob.sum()
#
#         # Calculate the convex hull of the blob
#         points = np.argwhere(blob)
#         hull = ConvexHull(points)
#         hull_area = hull.area
#
#         # Calculate the ratio between the blob area and the convex hull area
#         ratio = area / hull_area
#
#         # If the ratio is bigger than the threshold, separate the blob into two blobs
#         if ratio > threshold:
#             # Find the longest line in the convex hull
#             longest_line = find_longest_line(hull, points)
#
#             # Split the blob into two blobs using the longest line
#             image1, image2 = split_blob(image, longest_line)
#
#             # Return the two separated blobs
#             return image1, image2
#
#
#
# def find_longest_line(hull, points):
#     # Initialize the longest line to be the first line in the convex hull
#     longest_line = (hull.vertices[0], hull.vertices[1])
#     longest_length = 0
#
#     # Iterate over the lines in the convex hull
#     for i in range(len(hull.vertices)):
#         # Get the starting and ending points of the line
#         p1 = points[hull.vertices[i]]
#         p2 = points[hull.vertices[(i+1) % len(hull.vertices)]]
#
#         # Calculate the length of the line
#         length = np.linalg.norm(p1 - p2)
#
#         # If the length of the line is longer than the current longest length, update the longest line
#         if length > longest_length:
#             longest_line = (p1, p2)
#             longest_length = length
#
#     # Return the longest line
#     return longest_line
#
#
# def split_blob(image, line):
#     # Create a mask with the same size as the image
#     mask = np.zeros_like(image, dtype=np.bool)
#
#     # Draw the line on the mask
#     rr, cc = draw.line(*line[0], *line[1])
#     mask[rr, cc] = True
#
#     # Split the image into two images using the mask
#     image1 = image[mask]
#     image2 = image[~mask]
#
#     # Return the two separated images
#     return image1, image2

# COLOR_CLOSING = np.ones((3, 3))
# BLUE_CLOSING = np.ones((5, 5))
# BLUE_SIZE_DEVIATION = 0.85
# LABELING_BINARY_STRUCTURE = generate_binary_structure(2, 2)
# MIN_GREEN_SIZE = 7
# MIN_RED_SIZE = 300
# BUNDLES_CLOSING = np.ones((10, 10))
# SPRINGS_PARTS_OVERLAP_SIZE = 10
# OBJECT_DILATION_SIZE = 3
# ANTS_OPENING_CLOSING_STRUCTURE = np.ones((4, 4))
# MIN_ANTS_SIZE = 60
# ANTS_SPRINGS_OVERLAP_SIZE = 10


# def close_element(mask, structure):
#     mask = convert_bool_to_binary(binary_dilation(mask, structure))
#     mask = convert_bool_to_binary(binary_erosion(mask, structure))
#     return mask
#
# def convert_bool_to_binary(bool_mask):
#     binary_mask = np.zeros(bool_mask.shape, "int")
#     binary_mask[bool_mask] = 1
#     return binary_mask
#
# class Springs:
#     def __init__(self, parameters, image, previous_detections):
#         # self.image = image
#         # self.contrast_mask = self.high_contrast_mask(image)
#         # self.contrast_mask = self.high_contrast_mask(image)
#         self.binary_color_masks = self.mask_object_colors(parameters, image)
#         self.whole_object_mask = self.close_element(self.combine_masks(list(self.binary_color_masks.values())),np.ones((15,15)))
#         # cv2.imshow("whole_object_mask", (self.whole_object_mask*255).astype(np.uint8))
#         # cv2.waitKey(1)
#         self.object_center, self.tip_point, self.mask_blue_full, self.blue_radius =\
#             self.detect_blue_stripe(self.binary_color_masks["b"], previous_detections = previous_detections)
#         self.green_mask = self.clean_mask(self.binary_color_masks["g"],MIN_GREEN_SIZE)
#         self.red_mask = self.clean_mask(self.binary_color_masks["r"],MIN_RED_SIZE)
#         self.red_labeled, self.green_labeled, self.fixed_ends_labeled, self.free_ends_labeled, self.red_centers, self.green_centers = \
#             self.get_spring_parts(self.object_center,self.binary_color_masks["r"],self.green_mask)
#         self.bundles_labeled, self.bundles_labels = self.create_bundles_labels()
#         self.fixed_ends_bundles_labels, self.free_ends_bundles_labels, self.real_springs_bundles_labels = \
#             self.assign_ends_to_bundles(self.bundles_labeled, self.fixed_ends_labeled,
#                                           self.free_ends_labeled,self.red_centers, self.green_labeled)
#         self.bundles_labeled_after_removal = self.remove_labels(self.real_springs_bundles_labels, self.bundles_labeled)
#         self.fixed_ends_edges_centers, self.fixed_ends_edges_bundles_labels = \
#             self.find_bounderies_touches(self.fixed_ends_labeled, self.red_labeled, self.bundles_labeled_after_removal)
#         self.free_ends_edges_centers, self.free_ends_edges_bundles_labels = \
#             self.find_bounderies_touches(self.free_ends_labeled, self.red_labeled, self.bundles_labeled_after_removal)
#
#     def combine_masks(self,list_of_masks):
#         # combined = [np.zeros(list_of_masks[0].shape,"int")+x for x in list_of_masks][0]
#         combined = list_of_masks[0]+list_of_masks[1]+list_of_masks[2]
#         combined = self.convert_bool_to_binary(combined.astype("bool"))
#         return combined
#
#     def numpy_masking(self, image, hsv_lower, hsv_upper):
#         image_zeros = np.zeros(image.shape[:-1])
#         bool_lower = (image[:,:,0] >= hsv_lower[0])&(image[:,:,1] >= hsv_lower[1])&(image[:,:,2] >= hsv_lower[2])
#         bool_upper = (image[:,:,0] <= hsv_upper[0])&(image[:,:,1] <= hsv_upper[1])&(image[:,:,2] <= hsv_upper[2])
#         image_zeros[bool_lower&bool_upper] = 1
#         return image_zeros
#
#     def mask_object_colors(self, parameters, image):
#         binary_color_masks = {x:None for x in parameters["colors_spaces"]}
#         blurred = cv2.GaussianBlur(image, (3, 3), 0)
#         hsv_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
#         for color in parameters["colors_spaces"]:
#             binary_mask = np.zeros(image.shape[:-1],"int")
#             for hsv_space in parameters["colors_spaces"][color]:
#                 mask1_np = self.numpy_masking(hsv_image, hsv_space[0], hsv_space[1])
#                 # mask1 = cv2.inRange(hsv_image, hsv_space[0], hsv_space[1])
#                 mask2_np = self.numpy_masking(hsv_image, hsv_space[2], hsv_space[3])
#                 # mask2 = cv2.inRange(hsv_image, hsv_space[2], hsv_space[3])
#                 # binary_mask[(mask1+mask2)!=0] = 1
#                 binary_mask[(mask1_np+mask2_np)!=0] = 1
#             binary_mask = self.close_element(binary_mask,COLOR_CLOSING)
#             binary_color_masks[color] = binary_mask
#         return binary_color_masks
#
#     def close_element(self, mask, structure):
#         # mask = self.convert_bool_to_binary(binary_closing(mask, structure))
#         mask = self.convert_bool_to_binary(binary_dilation(mask, structure))
#         # mask = self.convert_bool_to_binary(thin(mask, max_iter=1))
#         mask = self.convert_bool_to_binary(binary_erosion(mask, structure[:-2,:-2]))
#         return mask
#
#
#     def convert_bool_to_binary(self,bool_mask):
#         binary_mask = np.zeros(bool_mask.shape, "int")
#         binary_mask[bool_mask] = 1
#         return binary_mask
#
#     def detect_blue_stripe(self,mask_blue,closing_structure=BLUE_CLOSING,previous_detections=None):
#         # cv2.imshow("mask_blue",self.convert_bool_to_binary(mask_blue.astype("bool")).astype(np.uint8) * 255)
#         # cv2.waitKey(0)
#         mask_blue_empty_closed = self.close_element(mask_blue, closing_structure)
#         # cv2.imshow("mask_blue_empty_closed", self.convert_bool_to_binary(mask_blue_empty_closed.astype("bool")).astype(np.uint8)*255)
#         # cv2.waitKey(0)
#         lableled,_ = label(mask_blue_empty_closed)
#         bulb_prop = regionprops(lableled)
#         biggest_blub = bulb_prop[np.argmax([x.area for x in bulb_prop])].label
#         mask_blue_empty_closed[np.invert(lableled==biggest_blub)] = 0
#         mask_blue_full = np.zeros(mask_blue_empty_closed.shape,"int")
#         binary_fill_holes(mask_blue_empty_closed,output=mask_blue_full)
#         inner_mask = mask_blue_full - mask_blue_empty_closed
#         inner_mask_center = center_of_mass(inner_mask)
#
#         if not np.isnan(np.sum(inner_mask_center)):
#             object_center = [int(x) for x in inner_mask_center]
#             object_center = np.array([object_center[1],object_center[0]])
#         else: object_center = previous_detections[0]
#         if not previous_detections is None:
#             if np.sum(mask_blue_full) < np.sum(previous_detections[1])*BLUE_SIZE_DEVIATION:
#                 mask_blue_full = previous_detections[1]
#         # cv2.imshow("inner_mask", (mask_blue_empty_closed*255).astype(np.uint8))
#         # cv2.waitKey(1)
#         contour = (find_contours(mask_blue_full)[0]).astype(int)
#         farthest_point,blue_radius = self.find_farthest_point(object_center, contour)
#         farthest_point = np.array([farthest_point[1],farthest_point[0]])
#         return object_center, farthest_point, mask_blue_full, blue_radius
#
#     def find_farthest_point(self, point, contour):
#         point = np.array([point[1],point[0]])
#         distances = np.sqrt(np.sum(np.square(contour-point),1))
#         farthest_from_point = contour[np.argmax(distances),:]
#         return farthest_from_point, np.max(distances)
#
#     def clean_mask(self,mask,min_size):
#         inner_circle_mask = create_circular_mask(mask.shape, center=self.object_center, radius=self.blue_radius)
#         outer_circle_mask = create_circular_mask(mask.shape, center=self.object_center, radius=self.blue_radius*3)
#         mask[inner_circle_mask] = 0
#         mask[np.invert(outer_circle_mask)] = 0
#         mask = self.convert_bool_to_binary(binary_fill_holes(mask))
#         mask = self.remove_small_blobs(mask, min_size)
#         return mask
#
#     # def clean_red_mask(self,red_mask):
#     #     circle_mask = create_circular_mask(green_mask.shape, center=self.object_center, radius=self.blue_radius*3)
#     #     red_mask[np.invert(circle_mask)] = 0
#     #     red_mask = self.convert_bool_to_binary(binary_fill_holes(red_mask))
#     #     green_mask = self.remove_small_blobs(green_mask, MIN_GREEN_SIZE)
#     #     red = self.red_labeled
#     #     sizes = [np.sum(self.convert_bool_to_binary(red==x)) for x in np.unique(red)]
#     #     print(sizes)
#     def remove_small_blobs(self, binary_mask: np.ndarray, min_size: int = 0):
#         """
#         Removes from the input mask all the blobs having less than N adjacent pixels.
#         We set the small objects to the background label 0.
#         """
#         if min_size > 0:
#             dtype = binary_mask.dtype
#             binary_mask = remove_small_objects(binary_mask.astype(bool), min_size=min_size)
#             binary_mask = binary_mask.astype(dtype)
#         return binary_mask
#
#     def get_spring_parts(self,object_center,red_mask,green_mask):
#         red_labeled, red_num_features = label(red_mask, LABELING_BINARY_STRUCTURE)
#         red_centers = np.array(center_of_mass(red_labeled, labels=red_labeled, index=range(1, red_num_features + 1)))
#         red_centers = self.swap_columns(red_centers)
#         red_radii = np.sqrt(np.sum(np.square(red_centers - object_center), axis=1))
#
#         green_labeled, green_num_features = label(green_mask, LABELING_BINARY_STRUCTURE)
#         green_centers =np.array(center_of_mass(green_labeled,labels=green_labeled,index=range(1,green_num_features+1)))
#         green_centers = self.swap_columns(green_centers)
#         green_radii = np.sqrt(np.sum(np.square(green_centers - object_center), axis=1))
#         fixed_ends_labels = np.array([x for x in range(1, green_num_features + 1)])[green_radii < (np.mean(red_radii))]
#         free_ends_labels = np.array([x for x in range(1, green_num_features + 1)])[green_radii > np.mean(red_radii)]
#
#         fixed_ends_labeled = copy.copy(green_labeled)
#         fixed_ends_labeled[np.invert(np.isin(green_labeled, fixed_ends_labels))] = 0
#         free_ends_labeled = copy.copy(green_labeled)
#         free_ends_labeled[np.invert(np.isin(green_labeled, free_ends_labels))] = 0
#         return red_labeled, green_labeled, fixed_ends_labeled, free_ends_labeled, red_centers, green_centers
#
#     def swap_columns(self,array):
#         array[:, [0, 1]] = array[:, [1, 0]]
#         return array
#
#     def create_bundles_labels(self,closing_structure=BUNDLES_CLOSING):
#         all_parts_mask = self.red_mask + self.green_mask
#         all_parts_mask = self.close_element(all_parts_mask,closing_structure)
#         labeled_image, num_features = label(all_parts_mask, generate_binary_structure(2, 2))
#         fied_ends_centers = center_of_mass(self.fixed_ends_labeled, labels=self.fixed_ends_labeled,
#                                          index=np.unique(self.fixed_ends_labeled)[1:])
#         fied_ends_centers = np.array([np.array([x, y]).astype("int") for x, y in fied_ends_centers])
#         self.bundles_centers = fied_ends_centers
#         center = np.array([self.object_center[1],self.object_center[0]])
#         tipp = np.array([self.tip_point[1],self.tip_point[0]])
#         fied_ends_angles = self.calc_angles(fied_ends_centers, center, tipp)
#         labeled_image_sorted = np.zeros(labeled_image.shape)
#         for pnt,angle in zip(fied_ends_centers,fied_ends_angles):
#             bundle_label = labeled_image[pnt[0],pnt[1]]
#             if bundle_label != 0:
#                 labeled_image_sorted[labeled_image == bundle_label] = angle
#         bundels_labels_fixed_centers = [labeled_image[x[0],x[1]] for x in fied_ends_centers]
#         bad_bundels = np.unique(bundels_labels_fixed_centers)
#
#         counts = np.array([bundels_labels_fixed_centers.count(x) for x in bundels_labels_fixed_centers])
#         melted_bundles = np.unique(np.array(bundels_labels_fixed_centers)[counts > 1])
#         for bad_label in melted_bundles:
#             labeled_image_sorted[labeled_image==bad_label] = 0
#         return labeled_image_sorted, fied_ends_angles
#
#     def calc_angles(self, points_to_measure, object_center, tip_point):
#         ba = points_to_measure - object_center
#         bc = (tip_point - object_center)
#         ba_y = ba[:,0]
#         ba_x = ba[:,1]
#         dot = ba_y*bc[0] + ba_x*bc[1]
#         det = ba_y*bc[1] - ba_x*bc[0]
#         angles = np.arctan2(det, dot)
#         return angles
#
#     def assign_ends_to_bundles(self,bundles_labeled,fixed_ends_labeled,free_ends_labeled,red_centers,green_labeled):
#         fixed_ends_centers = self.swap_columns(np.array(
#             center_of_mass(green_labeled, labels=green_labeled, index=list(np.unique(fixed_ends_labeled))[1:]), "int"))
#         free_ends_centers = self.swap_columns(np.array(
#             center_of_mass(green_labeled, labels=green_labeled, index=list(np.unique(free_ends_labeled))[1:]), "int"))
#         fixed_ends_bundles_labels = []
#         free_ends_bundles_labels = []
#         red_bundles_labels = []
#         for x1, y1 in fixed_ends_centers:
#             fixed_ends_bundles_labels.append(bundles_labeled[y1, x1])
#         for x1, y1 in free_ends_centers:
#             free_ends_bundles_labels.append(bundles_labeled[y1, x1])
#         for x1, y1 in red_centers.astype("int"):
#             red_bundles_labels.append(bundles_labeled[y1, x1])
#         labels_to_keep = self.screen_bundles(fixed_ends_bundles_labels, free_ends_bundles_labels, red_bundles_labels)
#         return fixed_ends_bundles_labels, free_ends_bundles_labels, labels_to_keep
#
#     def screen_bundles(self, fixed_labels, free_labels, red_labels):
#         counts = np.array([fixed_labels.count(x) for x in fixed_labels])
#         melted_bundles = np.unique(np.array(fixed_labels)[counts>1])
#         for label in melted_bundles:
#             fixed_labels = list(filter((label).__ne__, fixed_labels))
#         ### Only bundles which has all 3 parts (free,fixed,red) will be consider real bundle
#         real_springs_labels=list(set(fixed_labels).intersection(set(free_labels)).intersection(set(red_labels)))
#         return real_springs_labels
#
#     def remove_labels(self, labels_to_keep, labeled_image):
#         labeled_image[np.isin(labeled_image, labels_to_keep, invert=True)] = 0
#         return labeled_image
#
#     def find_bounderies_touches(self, labeled1, labeled2, bundles_labeled):
#         maximum_filter_labeled1 = maximum_filter(labeled1, SPRINGS_PARTS_OVERLAP_SIZE)
#         maximum_filter_labeled2 = maximum_filter(labeled2, SPRINGS_PARTS_OVERLAP_SIZE)
#         overlap_labeled = np.zeros(maximum_filter_labeled1.shape, "int")
#         boolean_overlap = (maximum_filter_labeled1 != 0) * (maximum_filter_labeled2 != 0)
#         overlap_labeled[boolean_overlap] = maximum_filter_labeled1[boolean_overlap]
#         overlap_labels = list(np.unique(overlap_labeled))[1:]
#         overlap_centers = self.swap_columns(np.array(
#             center_of_mass(overlap_labeled, labels=overlap_labeled, index=overlap_labels)).astype("int"))
#         overlap_bundles_labels = [bundles_labeled[x[1], x[0]] for x in overlap_centers]
#         return overlap_centers, overlap_bundles_labels

# class Ants:
#     def __init__(self, image, springs):
#         self.object_mask = self.create_object_mask(image.shape,springs)
#         self.labaled_ants = self.label_ants(image,self.object_mask)
#
#     def create_object_mask(self,image_dim,springs):
#         circle_mask = self.create_circular_mask(image_dim, center=springs.object_center, radius=springs.blue_radius*1.5)
#         # mask_object = maximum_filter(springs.whole_object_mask,size=OBJECT_DILATION_SIZE) != 0  # inflation of the object and convertion to bool array
#         mask_object = springs.whole_object_mask != 0
#         # mask_object[circle_mask] = True
#         # cv2.imshow("str(color)", (springs.convert_bool_to_binary(mask_object)*255).astype(np.uint8))
#         # cv2.imshow("str(color)", (springs.convert_bool_to_binary(circle_mask)*255).astype(np.uint8))
#         # cv2.waitKey(1)
#         return mask_object
#
#     def create_circular_mask(self, image_dim, center=None, radius=None):
#         h,w = image_dim[0],image_dim[1]
#         if center is None:  # use the middle of the image
#             center = (int(w / 2), int(h / 2))
#         if radius is None:  # use the smallest distance between the center and image walls
#             radius = min(center[0], center[1], w - center[0], h - center[1])
#
#         Y, X = np.ogrid[:h, :w]
#         dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
#
#         mask = dist_from_center <= radius
#         return mask
#
#     def label_ants(self,image,mask_object):
#         # grayscale = rgb2gray(image)
#         grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#         # thresh = threshold_local(grayscale, block_size=51, offset=10)
#         thresh = threshold_otsu(grayscale)
#
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#         lower_val = np.array([0, 0, 0])
#         upper_val = np.array([179, 100, 100])
#         mask = cv2.inRange(hsv, lower_val, upper_val)
#         mask = mask>0
#         # thresh = threshold_yen(grayscale)
#         # mask = grayscale < thresh
#         mask[mask_object] = False
#
#
#         # mask = mask_black#+mask
#         mask = binary_opening(mask, ANTS_OPENING_CLOSING_STRUCTURE)
#         mask = binary_closing(mask, ANTS_OPENING_CLOSING_STRUCTURE )
#
#         mask = remove_small_objects(mask, MIN_ANTS_SIZE)
#         # mask = close_element(convert_bool_to_binary(mask), np.ones((2, 2)))
#         mask = convert_bool_to_binary(binary_dilation(convert_bool_to_binary(mask), np.ones((2, 2))))
#         # mask = morphology.remove_small_holes(mask, 50)
#
#
#         label_image, features = label(mask)
#
#         label_image = clear_border(label_image)
#         # label_image_binary =np.zeros(label_image.shape,"int")
#         # label_image_binary[label_image] = 1
#
#         ### WATERSDHED SEGMANTATION ###
#         # distance = ndi.distance_transform_edt(label_image)
#         # coords = peak_local_max(distance, footprint=np.ones((3,3)), labels=label_image)
#         # mask = np.zeros(distance.shape, dtype=bool)
#         # mask[tuple(coords.T)] = True
#         # markers, _ = ndi.label(mask)
#         # labels = watershed(-distance, markers, mask=label_image)
#         # image_label_overlay = label2rgb(labels, image=grayscale, bg_label=0)
#
#         # image_label_overlay = label2rgb(label_image, image=image, bg_label=0)
#         # cv2.imshow("image_label_overlay", image_label_overlay)
#         # cv2.waitKey(1)
#         self.ants_lines = []
#         for lab in range(1,features+1):
#             self.ants_lines += (utils.find_lines_in_mask(label_image==lab))
#             # self.lines += lines
#         # print("lines",self.lines)
#         # except: pass
#         # threshold = 0.5
#         # label_image = utils.separate_blobs(label_image, threshold)
#         return label_image
#
#     def ants_coordinates(self, labeled_image):
#         labels = list(np.unique(labeled_image))[1:]
#         centers = np.array(center_of_mass(labeled_image, labels=labeled_image, index=labels)).astype("int")
#         return centers, labels
#
#     def track_ants(self, image_labeled, previous_labeled):
#         """
#         Takes the labeled image and matches the labels to the previous image labels,
#         based on the proximity of the centers of mass. Then it corrects the labels,
#         provides new labels for the new ants, and deletes the labels of the ants that disappeared.
#         param image_labeled:
#         :param previous_image_labeled:
#         :return: The corrected labeled image.
#         """
#         labeled_image_center_of_mass = center_of_mass(image_labeled, labels=image_labeled,
#                                                       index=np.unique(image_labeled)[1:])
#         previous_labeled_center_of_mass = center_of_mass(previous_labeled, labels=previous_labeled,
#                                                          index=np.unique(previous_labeled)[1:])
#         labeled_image_center_of_mass = np.array(labeled_image_center_of_mass).astype("int")
#         previous_image_labeled_center_of_mass = np.array(previous_labeled_center_of_mass).astype("int")
#         # print("labeled_image_center_of_mass", labeled_image_center_of_mass)
#         labeled_image_center_of_mass = labeled_image_center_of_mass
#         previous_image_labeled_center_of_mass = previous_image_labeled_center_of_mass
#         closest_pair = self.closetest_pair_of_points(labeled_image_center_of_mass,
#                                                      previous_image_labeled_center_of_mass,previous_labeled)
#         self.corrected_labeled_image = self.replace_labels(image_labeled, closest_pair)
#         self.corrected_labels_center_of_mass = center_of_mass(self.corrected_labeled_image,
#                         labels=self.corrected_labeled_image, index=np.unique(self.corrected_labeled_image)[1:])
#
#     def replace_labels(self, image_labeled,closest_pair):
#         # print("closest_pair", closest_pair)
#         for pair in closest_pair:
#             image_labeled[image_labeled == pair[0]] = pair[1]
#         return image_labeled
#
#     def closetest_pair_of_points(self, centers_current,centers_previous,previous_labeled):
#         """
#         Finds the closest pair of points between two sets of points.
#         :param points1: The first set of points.
#         :param points2: The second set of points.
#         :return: The closest pair of points.
#         """
#         from scipy.spatial.distance import cdist
#         distances = cdist(centers_current, centers_previous)
#         # print("distances", distances)
#         #get the column of each minimum value in each row:
#         min_col = np.argmin(distances, axis=1)
#         previous_labels = np.take(np.unique(previous_labeled)[1:], min_col)
#         # print("len(min_col)", len(min_col))
#         # print(list(zip(np.arange(len(min_col))+1, min_col+1)))
#         return list(zip(np.arange(len(min_col))+1, previous_labels))


# class Calculation:
#     def __init__(self, springs, ants):
#         self.initiate_springs_angles_matrix(springs)
#         springs_order = self.springs_angles_matrix[-1, :]
#         self.springs_length = self.calc_springs_lengths(springs, springs_order)
#         self.N_ants_around_springs, self.size_ants_around_springs =\
#             self.occupied_springs(springs, ants, springs_order)
#         self.springs_angles_to_nest, self.springs_angles_to_object = self.calc_springs_angles(springs, springs_order)
#
#     def initiate_springs_angles_matrix(self,springs):
#         if len(springs.bundles_labels)!=20:
#             exit("First frame should have exactly 20 springs. Different number of springs were detect,"
#                  " please start the process from a different frame.")
#         self.springs_angles_matrix = np.sort(springs.bundles_labels).reshape(1,20)
#
#     def make_calculations(self,springs,ants):
#         self.springs_angles_matrix = np.vstack([self.springs_angles_matrix, self.match_springs(springs)])
#
#         springs_order = self.springs_angles_matrix[-1, :]
#         self.springs_length = np.vstack([self.springs_length, self.calc_springs_lengths(springs, springs_order)])
#         N_ants_around_springs, size_ants_around_springs =\
#             self.occupied_springs(springs,ants,springs_order)
#         self.N_ants_around_springs = np.vstack([self.N_ants_around_springs, N_ants_around_springs])
#         self.size_ants_around_springs = np.vstack([self.size_ants_around_springs, size_ants_around_springs])
#         angles_to_nest, angles_to_object = self.calc_springs_angles(springs, springs_order)
#         self.springs_angles_to_nest = np.vstack([self.springs_angles_to_nest, angles_to_nest])
#         self.springs_angles_to_object = np.vstack([self.springs_angles_to_object, angles_to_object])
#
#     def match_springs(self, springs):
#         current_springs_angles = list(springs.bundles_labels)
#         previous_springs_angles_mean = self.springs_angles_matrix[0, :]
#         assigned_angles_classes = []
#         for value in current_springs_angles:
#             cos_diff = np.cos(previous_springs_angles_mean - value)
#             assigned_class = np.argmax(cos_diff)
#             assigned_angles_classes.append(assigned_class)
#         new_springs_row = np.array([np.nan for x in range(20)])
#         for angle_class, angle in zip(assigned_angles_classes, current_springs_angles):
#             new_springs_row[angle_class] = angle
#         return new_springs_row
#
#     def calc_springs_lengths(self, springs, springs_order):# fixed_ends_coor, free_ends_coor, fixed_ends_labels, free_ends_labels):
#         springs_length = np.empty((20))
#         springs_length[:] = np.nan
#         found_in_both = list(set(springs.fixed_ends_edges_bundles_labels).
#                              intersection(set(springs.free_ends_edges_bundles_labels)))
#         for label in found_in_both:
#             if label != 0:
#                 coor_free = springs.free_ends_edges_centers[springs.free_ends_edges_bundles_labels.index(label)]
#                 coor_fixed = springs.fixed_ends_edges_centers[springs.fixed_ends_edges_bundles_labels.index(label)]
#                 distance = np.sqrt(np.sum(np.square(coor_free - coor_fixed)))
#                 index_springs = springs_order == label
#                 springs_length[index_springs] = distance
#         return springs_length.reshape(1,20)
#
#     def occupied_springs(self, springs, ants, springs_order):
#         dialated_ends = maximum_filter(springs.free_ends_labeled,ANTS_SPRINGS_OVERLAP_SIZE)
#         dialated_ants = maximum_filter(ants.labaled_ants,ANTS_SPRINGS_OVERLAP_SIZE)
#         joints = ((dialated_ends != 0) * (dialated_ants != 0))
#         ends_occupied = np.sort(np.unique(springs.bundles_labeled[joints*(springs.bundles_labeled!=0)]))
#         N_ants_around_springs = np.empty((20))
#         N_ants_around_springs[:] = np.nan
#         size_ants_around_springs = np.empty((20))
#         size_ants_around_springs[:] = np.nan
#         for end in ends_occupied:
#             index_springs = springs_order==end
#             ends_i = np.unique(dialated_ants[(springs.bundles_labeled==end)*(dialated_ends != 0)])[1:]
#             N_ants_around_springs[index_springs] = len(ends_i)
#             size_ants_around_springs[index_springs] = np.sum(np.isin(ants.labaled_ants,ends_i))
#         return N_ants_around_springs.reshape(1,20), size_ants_around_springs.reshape(1,20)
#
#     def calc_springs_angles(self,springs, springs_order):
#         nest_direction = np.array([springs.object_center[0], springs.object_center[1] - 100])
#         fixed_ends_angles_to_nest = springs.calc_angles(springs.fixed_ends_edges_centers,springs.object_center,nest_direction)
#         free_ends_angles_to_object = springs.calc_angles(springs.free_ends_edges_centers,springs.object_center,springs.tip_point)
#         angles_to_nest = np.empty((20))
#         angles_to_nest[:] = np.nan
#         angles_to_object = np.empty((20))
#         angles_to_object[:] = np.nan
#         for label in springs.bundles_labels:
#             if label !=0:
#                 index_springs = springs_order == label
#                 if label in springs.fixed_ends_edges_bundles_labels:
#                     # print(label)
#                     # print(springs.fixed_ends_edges_bundles_labels==label)
#                     angles_to_nest[index_springs] = fixed_ends_angles_to_nest[springs.fixed_ends_edges_bundles_labels==label]
#                 if label in springs.free_ends_edges_bundles_labels:
#                     angles_to_object[index_springs] = free_ends_angles_to_object[springs.free_ends_edges_bundles_labels==label]
#         angles_to_nest = angles_to_nest.reshape(1,20)
#         angles_to_object = angles_to_nest.reshape(1,20)
#         return angles_to_nest, angles_to_object
#
#     def add_blank_row(self):
#         empty_row = np.empty((20))
#         empty_row[:] = np.nan
#         self.springs_angles_matrix = np.vstack([self.springs_angles_matrix, empty_row])
#         self.springs_length = np.vstack([self.springs_length, empty_row])
#         self.N_ants_around_springs = np.vstack([self.N_ants_around_springs, empty_row])
#         self.size_ants_around_springs = np.vstack([self.size_ants_around_springs, empty_row])
#         self.springs_angles_to_nest = np.vstack([self.springs_angles_to_nest, empty_row])
#         self.springs_angles_to_object = np.vstack([self.springs_angles_to_object, empty_row])
#
#     def clear_data(calculations):
#         # data_arrays_names = ["springs_length", "N_ants_around_springs", "size_ants_around_springs",
#         #                      "springs_angles_to_nest", "springs_angles_to_object"]
#         # for dat in data_arrays_names:
#         #     exec(f"calculations.{dat} = calculations.{dat}[-1,:]")
#         calculations.springs_length = calculations.springs_length[-1,:]
#         calculations.N_ants_around_springs = calculations.N_ants_around_springs[-1,:]
#         calculations.size_ants_around_springs = calculations.size_ants_around_springs[-1,:]
#         calculations.springs_angles_to_nest = calculations.springs_angles_to_nest[-1,:]
#         calculations.springs_angles_to_object = calculations.springs_angles_to_object[-1,:]
