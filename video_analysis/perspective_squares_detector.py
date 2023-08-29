import cv2
import numpy as np

#local imports:
from video_analysis import utils


COLOR_CLOSING = 5
SQUARE_ON_BORDER_RATIO_THRESHOLD = 0.4


class PerspectiveSquares:
    def __init__(self, parameters, image, previous_detections):
        self.parameters = parameters
        self.previous_detections = previous_detections
        boolean_masks_connected, boolean_masks_unconnected, perspective_squares_crop_coordinates =\
            self.get_perspective_squares_masks(image)
        squares_properties = self.get_squares_properties(boolean_masks_connected)
        self.perspective_squares_properties, self.all_perspective_squares_mask =\
            self.perspective_squares_coordinates_transformation(squares_properties,
                 perspective_squares_crop_coordinates, boolean_masks_unconnected)

    def get_perspective_squares_masks(self, image):
        prepared_images, perspective_squares_crop_coordinates = self.prepare_images(image, self.parameters["pcm"])
        boolean_masks_connected, boolean_masks_unconnected = self.mask_perspective_squares(self.parameters["colors_spaces"]["p"], prepared_images)
        if self.are_squares_on_border(boolean_masks_unconnected):
            prepared_images, perspective_squares_crop_coordinates = self.prepare_images(image, self.parameters["pcm"]*5)
            boolean_masks_connected, boolean_masks_unconnected = self.mask_perspective_squares(
                self.parameters["colors_spaces"]["p"], prepared_images)
            if self.are_squares_on_border(boolean_masks_unconnected):
                raise ValueError("Perspective squares are on border.")
        return boolean_masks_connected, boolean_masks_unconnected, perspective_squares_crop_coordinates

    def are_squares_on_border(self, squares_boolean_masks):
        on_border = np.full((len(squares_boolean_masks),4), 0, dtype="uint16")
        for key, square_mask in squares_boolean_masks.items():
            on_border[key] = np.array([np.sum(square_mask[0, :]), np.sum(square_mask[-1, :]), np.sum(square_mask[:, 0]), np.sum(square_mask[:, -1])])
        return np.any(on_border/self.parameters["pcm"] > SQUARE_ON_BORDER_RATIO_THRESHOLD)

    def prepare_images(self, image, box_margin):
        if self.previous_detections["skipped_frames"] >= 25:
            box_margin = 500
        crop_coordinates = utils.create_box_coordinates(self.previous_detections["perspective_squares_coordinates"], box_margin)
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
            num_labels, labels = cv2.connectedComponents(boolean_mask.astype("uint8"))
            if num_labels > 1:
                blobs_sizes = np.bincount(labels.ravel())
                biggest_blob = np.argmax(blobs_sizes[1:]) + 1
                boolean_mask[labels != biggest_blob] = False
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

    def perspective_squares_coordinates_transformation(self, squares_properties, crop_coordinates, boolean_masks_unconnected):
        coordinates = squares_properties[:,0,:] # x, y
        addition = np.copy(crop_coordinates[:, [2,0]]) # x, y
        squares_properties[:, 0, :] = (coordinates + addition).astype(np.uint16)
        squares_mask_unconnected = np.full(self.parameters["resolution"], False, dtype="bool")
        for count, coordinates in enumerate(crop_coordinates):
            squares_mask_unconnected[coordinates[0]:coordinates[1], coordinates[2]:coordinates[3]] = boolean_masks_unconnected[count]
        return squares_properties, squares_mask_unconnected
