import cv2
import numpy as np

#local imports:
from video_analysis import utils


COLOR_CLOSING = 5


class PerspectiveSquares:
    def __init__(self, parameters, image, previous_detections):
        self.parameters = parameters
        prepared_images, perspective_squares_crop_coordinates = self.prepare_images(image, previous_detections[7], parameters["pcm"])
        boolean_masks_connected, boolean_masks_unconnected = self.mask_perspective_squares(parameters["colors_spaces"]["p"], prepared_images)
        squares_properties = self.get_squares_properties(boolean_masks_connected)
        self.perspective_squares_properties, self.all_perspective_squares_mask =\
            self.perpective_squares_coordinates_transformation(squares_properties, perspective_squares_crop_coordinates, boolean_masks_unconnected, parameters)

    def prepare_images(self, image, coordinates, box_margin):
        crop_coordinates = utils.create_box_coordinates(coordinates, box_margin)
        prepared_images = {}
        for count in range(len(crop_coordinates)):
            cropped_image = utils.crop_frame_by_coordinates(image, crop_coordinates[count])
            prepared_images[count] = utils.white_balance_bgr(utils.neutrlize_colour(cropped_image, alpha=2.5, beta=0))
        return prepared_images, crop_coordinates

    def mask_perspective_squares(self, color_spaces, images):
        boolean_masks_connected = {}
        boolean_masks_unconnected = {}
        for count, image in images.items():
            blurred = cv2.GaussianBlur(image, (3, 3), 0)
            hsv_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            boolean_mask = np.full(hsv_image.shape[:-1], False, dtype="bool")
            for hsv_space in color_spaces:
                mask1 = cv2.inRange(hsv_image, hsv_space[0], hsv_space[1]).astype(bool)
                mask2 = cv2.inRange(hsv_image, hsv_space[2], hsv_space[3]).astype(bool)
                boolean_mask[mask1+mask2] = True
            boolean_masks_unconnected[count] = boolean_mask
            boolean_mask = utils.connect_blobs(boolean_mask,COLOR_CLOSING)
            boolean_masks_connected[count] = boolean_mask
        return boolean_masks_connected, boolean_masks_unconnected

    def get_squares_properties(self, masks):
        squares_properties = np.array([[[0, 0], [0, 0]] for x in range(len(masks))])
        for count, mask in masks.items():
            mask = mask.astype("uint8")
            indices = np.argwhere(mask > 0)
            hull = cv2.convexHull(indices)
            rect = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            center_x = np.mean(box[:, 1])
            center_y = np.mean(box[:, 0])
            width = np.linalg.norm(box[0] - box[1])
            height = np.linalg.norm(box[1] - box[2])
            squares_properties[count] = [[center_x, center_y], [width, height]]
        return squares_properties

    def perpective_squares_coordinates_transformation(self, squares_properties, crop_coordinates, boolean_masks_unconnected, parameters):
        coordinates = squares_properties[:,0,:] # x, y
        addition = np.copy(crop_coordinates[:, [2,0]]) # x, y
        squares_properties[:, 0, :] = (coordinates + addition).astype(np.uint16)
        squares_mask_unconnected = np.full(parameters["resolution"], False, dtype="bool")
        for count, coordinates in enumerate(crop_coordinates):
            squares_mask_unconnected[coordinates[0]:coordinates[1], coordinates[2]:coordinates[3]] = boolean_masks_unconnected[count]
        return squares_properties, squares_mask_unconnected
