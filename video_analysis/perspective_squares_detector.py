import cv2
import numpy as np
# local imports:
import utils


class PerspectiveSquares:
    def __init__(self, parameters, frame, checkpoint):
        self.parameters = parameters
        self.frame = frame
        self.checkpoint = checkpoint
        masks_connected, masks_unconnected, crop_coordinates = self.mask_squares()
        try:
            squares_properties = self.get_properties(masks_connected)
            self.perspective_squares_properties, self.perspective_squares_mask = self.transform_coordinates(squares_properties, crop_coordinates, masks_unconnected)
        except Exception as e:
            self.perspective_squares_properties = np.full((4, 4), np.nan)
            self.perspective_squares_mask = np.full(self.parameters["RESOLUTION"], False, dtype="bool")
            for mask, coordinates in zip(masks_unconnected, crop_coordinates):
                self.perspective_squares_mask[coordinates[0]:coordinates[1], coordinates[2]:coordinates[3]] = mask

    def mask_squares(self):
        frame_crops, crop_coordinates = self.prepare_frame()
        masks_connected, masks_unconnected = self.mask_perspective_squares(self.parameters["COLOR_SPACES"]["p"], frame_crops)
        if self.ascertain_borderline(masks_unconnected):
            frame_crops, crop_coordinates = self.prepare_frame()
            masks_connected, masks_unconnected = self.mask_perspective_squares(self.parameters["COLOR_SPACES"]["p"], frame_crops)
        return masks_connected, masks_unconnected, crop_coordinates

    def ascertain_borderline(self, squares_boolean_masks):
        on_frame_border = np.full((len(squares_boolean_masks), 4), 0, dtype="uint16")
        for key, square_mask in squares_boolean_masks.items():
            on_frame_border[key] = np.array([np.sum(square_mask[0, :]), np.sum(square_mask[-1, :]), np.sum(square_mask[:, 0]), np.sum(square_mask[:, -1])])
        return np.any(on_frame_border/self.parameters["PCM"] > self.parameters["SQUARE_ON_BORDER_RATIO_THRESHOLD"])

    def prepare_frame(self, box_margin_factor=5):
        if (self.checkpoint.skipped_frames >= 25) or self.checkpoint.frame_count == 0:
            corners_coordinates = np.array([[0, 0],
                                            [0, self.parameters["RESOLUTION"][1]],
                                            [self.parameters["RESOLUTION"][0], self.parameters["RESOLUTION"][1]],
                                            [self.parameters["RESOLUTION"][0], 0]])
            crop_coordinates = utils.create_box_coordinates(corners_coordinates, self.parameters["OCM"] * box_margin_factor)
        else:
            crop_coordinates = utils.create_box_coordinates(self.checkpoint.perspective_squares_coordinates, self.parameters["OCM"])
        frame_crops = {}
        for count in range(len(crop_coordinates)):
            frame_cropped = utils.crop_frame(self.frame, crop_coordinates[count])
            frame_crops[count] = utils.process_image(frame_cropped,
                                                     alpha=self.parameters["NEUTRALIZE_COLOUR_ALPHA"],
                                                     blur_kernel=self.parameters["NEUTRALIZE_COLOUR_BETA"],
                                                     gradiant_threshold=self.parameters["GRADIANT_THRESHOLD"])
        return frame_crops, crop_coordinates

    def mask_perspective_squares(self, color_spaces, frame_crops):
        boolean_masks_connected = {}
        boolean_masks_unconnected = {}
        for count, image in frame_crops.items():
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
            boolean_mask = utils.connect_blobs(boolean_mask, self.parameters["PERSPECTIVE_SQUARES_COLOR_CLOSING"])
            boolean_masks_connected[count] = boolean_mask
        return boolean_masks_connected, boolean_masks_unconnected

    def get_properties(self, masks):
        squares_properties = np.full((len(masks), 4), 0, np.float64)
        # rectangle_similarity_scores = np.full(len(masks), np.nan)
        for count, mask in masks.items():
            mask = mask.astype("uint8")
            indices = np.argwhere(mask > 0)
            hull = cv2.convexHull(indices)
            contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rect = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rect)
            rectangle_similarity_score = cv2.matchShapes(utils.swap_columns(np.copy(box)), contour[0][0], 1, 0.0)
            length_side_a = np.linalg.norm(box[0] - box[1])
            length_side_b = np.linalg.norm(box[0] - box[3])
            squareness_score = (length_side_a - length_side_b) / (length_side_a * length_side_b)
            center_x = np.mean(box[:, 1])
            center_y = np.mean(box[:, 0])
            squares_properties[count] = [center_x, center_y, rectangle_similarity_score, squareness_score]
        return squares_properties

    def transform_coordinates(self, squares_properties, crop_coordinates, boolean_masks_unconnected):
        coordinates = squares_properties[:, 0:2]  # x, y
        addition = np.copy(crop_coordinates[:, [2, 0]])  # x, y
        squares_properties[:, 0:2] = (coordinates + addition)#.astype(np.uint16)
        squares_mask_unconnected = np.full(self.parameters["RESOLUTION"], False, dtype="bool")
        for count, coordinates in enumerate(crop_coordinates):
            squares_mask_unconnected[coordinates[0]:coordinates[1], coordinates[2]:coordinates[3]] = boolean_masks_unconnected[count]
        return squares_properties, squares_mask_unconnected


def save(checkpoint, perspective_squares=None):
    if perspective_squares is None:
        arrays = [np.full((1, 4), np.nan) for _ in range(4)]
    else:
        arrays = [perspective_squares.perspective_squares_properties[:, 0].reshape(1, 4),
                  perspective_squares.perspective_squares_properties[:, 1].reshape(1, 4),
                  perspective_squares.perspective_squares_properties[:, 2].reshape(1, 4),
                  perspective_squares.perspective_squares_properties[:, 3].reshape(1, 4)]
    names = ["perspective_squares_coordinates_x", "perspective_squares_coordinates_y", "perspective_squares_rectangle_similarity", "perspective_squares_squareness"]
    utils.save(arrays, names, checkpoint)
