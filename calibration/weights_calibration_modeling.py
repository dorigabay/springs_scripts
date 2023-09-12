import copy
import numpy as np
import pandas as pd
import os
import cv2
from data_analysis import utils
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pickle


class CalibrationModeling:
    def __init__(self, directories, weights, output_path, video_paths=None):
        self.angular_force = False
        self.directories = np.array(directories)
        self.weights = weights
        self.output_path = output_path
        self.num_of_springs = 1
        self.video_paths = np.array(video_paths)
        self.zero_weight_dir = self.directories[np.where(np.array(self.weights) == 0)[0][0]]
        self.get_bias_equations()
        self.combine_all_data()
        self.create_calibration_model()

    def get_bias_equations(self):
        self.post_processing(self.zero_weight_dir)
        # self.video_path = self.video_paths[np.where(np.array(self.weights) == 0)[0][0]]
        y_length = np.linalg.norm(self.free_ends_coordinates - self.fixed_ends_coordinates, axis=2)
        y_angle = utils.calc_pulling_angle_matrix(self.fixed_ends_coordinates, self.object_center_repeated, self.free_ends_coordinates)
        angles_to_nest = np.expand_dims(self.fixed_end_angle_to_nest, axis=2)
        # angles_to_nest = np.expand_dims(self.needle_part_angle_to_nest, axis=2)
        # fixed_end_distance = np.expand_dims(self.object_center_to_fixed_end_distance, axis=2)
        # fixed_to_tip_distance = np.expand_dims(self.object_needle_tip_to_fixed_end_distance, axis=2)
        # fixed_to_needle_angle_change = np.expand_dims(
        #     np.repeat(np.expand_dims(np.nanmedian(self.fixed_to_needle_angle_change, axis=1), axis=1),
        #               self.num_of_springs, axis=1), axis=2)
        # object_center = self.object_center_repeated
        # needle_length = np.expand_dims(np.repeat(np.expand_dims(np.linalg.norm(
        #     self.needle_tip_coordinates - self.object_center, axis=1), axis=1), self.num_of_springs,
        #     axis=1), axis=2)
        # X = np.concatenate((np.sin(angles_to_nest), np.cos(angles_to_nest), object_center, fixed_end_distance,
        X = np.concatenate((np.sin(angles_to_nest), np.cos(angles_to_nest)), axis=2)
        idx = self.rest_bool
        not_nan_idx = ~(np.isnan(y_angle) + np.isnan(X).any(axis=2)) * idx
        X_fit = X[not_nan_idx[:, 0], 0]
        y_length_fit = y_length[not_nan_idx[:, 0], 0]
        y_angle_fit = y_angle[not_nan_idx[:, 0], 0]
        self.model_length = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
        self.model_angle = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
        self.model_length.fit(X_fit, y_length_fit)
        self.model_angle.fit(X_fit, y_angle_fit)



    def get_nest_direction(self,video_path):
        import cv2
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        #collect two points from the user
        from video_analysis.collect_analysis_parameters import collect_points
        points = collect_points(frame, 2)
        upper_point = points[0]
        lower_point = points[1]
        lower_artificial_point = (lower_point[0], upper_point[1])
        ba = lower_artificial_point - upper_point
        bc = lower_point - upper_point
        ba_y = ba[0]
        ba_x = ba[1]
        dot = ba_y * bc[0] + ba_x * bc[1]
        det = ba_y * bc[1] - ba_x * bc[0]
        angle = np.arctan2(det, dot)
        self.calib_nest_angle = angle

    def combine_all_data(self):
        calibration_force_direction = np.array(())
        calibration_force_magnitude = np.array(())
        pulling_angle = np.array(())
        length = np.array(())
        # weights = np.array(())
        # nest_direction = np.array(())
        a = [0,1,7,9,10,11]
        # a = [0,1,2,3,4,5,6,7,8,9,10,11]
        # a = [0,1,7,11]
        # a = [0]
        # print(self.directories)
        for dir, weight in zip(self.directories[a], self.weights[a]):

        # for dir, weight in zip(self.directories, self.weights):
            self.post_processing(dir)
            # self.video_path = video_path

            self.calc_pulling_angle()
            self.calc_spring_length()
            self.calc_calibration_force(weight)
            # self.weight = weight
            slice = [x for x in range(0, 2200)] if weight == self.weights[-1] else [x for x in range(0, self.calibration_force_direction.shape[0])]
            # if weight == 0:
            #     self.calibration_force_direction = np.zeros(self.calibration_force_direction.shape)
            calibration_force_direction = np.append(calibration_force_direction, self.calibration_force_direction[slice])
            calibration_force_magnitude = np.append(calibration_force_magnitude, self.calibration_force_magnitude[slice])
            pulling_angle = np.append(pulling_angle, (self.pulling_angle[slice]).flatten())
            length = np.append(length, self.spring_length[slice].flatten())
            # weights = np.append(weights, np.repeat(weight, len(self.calibration_force_magnitude)))
            # if weight == 0:
            #     calibration_force_direction = np.append(calibration_force_direction, np.zeros(self.calibration_force_direction.shape))
            # else:
            # nest_direction = np.append(nest_direction, self.fixed_end_angle_to_nest.flatten())#+self.calib_nest_angle
        #
        data = np.vstack((pulling_angle*-1,
                          length,
                          calibration_force_direction,
                          calibration_force_magnitude,
                          )).transpose()
        # import matplotlib.pyplot as plt
        # plt.scatter(data[data[:, 3] == 0, 1],data[data[:, 3] == 0, 2], alpha=0.5)
        # plt.show()

        data = np.abs(data)
        # FD_negative = data[:, 2] < 0
        # data[FD_negative, 2] *= -1
        # data[FD_negative, 0] *= -1
        # pulling_angle_negative = data[:, 0] < 0
        # data[pulling_angle_negative, :] = np.nan

        data = data[~np.isnan(data).any(axis=1)]
        data = self.binning_to_nest_direction(data)
        self.X = data[:, 0:2]
        self.y = np.sin(data[:, 2]) * data[:, 3] if self.angular_force else data[:, 2:4]
        self.data = data
        self.plot()

    def binning_to_nest_direction(self, data, bins=300):
        max_value = np.max(data[~(data[:, 3] == 0), 2])
        # max_value = np.pi
        # max_value = np.pi/2
        bins = np.linspace(0, max_value, bins)
        for force_magnitude in np.unique(data[:, 3]):
            for bin in range(1, len(bins)):
                idx = (data[:, 2] > bins[bin - 1]) * (data[:, 2] < bins[bin]) * (data[:, 3] == force_magnitude)
                # idx = (nest_direction_binned == bin) * (data[:, 4] == weight)
                data[idx, :] = np.nanmedian(data[idx, :], axis=0)
        data[data[:,2] > max_value,:] = np.nan
        data = data[~np.isnan(data).any(axis=1)]
        # for force_magnitude in np.unique(data[:, 3]):
        #     idx = (data[:, 3] == force_magnitude)
        #     print(force_magnitude, data[idx, 2].shape, np.nanmin(data[idx, 1]), np.nanmax(data[idx, 1]), np.nanmin(data[idx, 0]), np.nanmax(data[idx, 0]))
        #remove duplicates
        data = pd.DataFrame(data)
        data = data.drop_duplicates()
        return np.array(data)

    def calc_pulling_angle(self):
        self.pulling_angle = utils.calc_pulling_angle_matrix(self.fixed_ends_coordinates, self.object_center_repeated, self.free_ends_coordinates)
        prediction_pulling_angle = self.norm_values(self.pulling_angle, self.model_angle)
        self.pulling_angle -= prediction_pulling_angle

    def calc_spring_length(self):
        self.spring_length = np.linalg.norm(self.free_ends_coordinates - self.fixed_ends_coordinates , axis=2)
        prediction_spring_length = self.norm_values(self.spring_length, self.model_length)
        self.spring_length /= prediction_spring_length
        self.spring_length /= self.norm_size

    def plot(self):
        import matplotlib.pyplot as plt
        # for weight in self.weights[[0, 4, 10]]:
        # for weight in np.unique(self.data[:, 4]):
        for force_magnitude in np.unique(self.data[:, 3]):
            # print(weight)
            data = self.data[self.data[:, 3] == force_magnitude]
            # print(weight, len(np.unique(weight_data[:, 1])))
            coordinates = np.array([np.sin(data[:, 0]), np.cos(data[:, 0])]).transpose() * (data[:, 1:2])#-self.median_spring_length_at_rest)
            plt.scatter(coordinates[:, 0], coordinates[:, 1], label=np.round(force_magnitude,3), alpha=0.5)
        plt.legend()
        plt.show()

    def create_calibration_model(self):
        number_degrees = list(range(1,2))
        plt_mean_squared_error = []
        plt_r_squared = []
        best_degrees = []
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        for degree in number_degrees:
            model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            plt_mean_squared_error.append(mean_squared_error(y_test, y_pred, squared=False))
            plt_r_squared.append(model.score(X_test, y_test))
        best_degrees.append(number_degrees[np.argmin(plt_mean_squared_error)])
        self.degree = max(set(best_degrees), key=best_degrees.count)
        print("Best polynomial degree is for force calibration equation: ", self.degree)
        self.model = make_pipeline(PolynomialFeatures(degree=self.degree), LinearRegression())
        self.model.fit(X_train, y_train)
        with open(os.path.join(self.output_path,"calibration_model.pkl"), "wb") as f:
            pickle.dump(self.model, f)
        y_pred = self.model.predict(X_test)
        if self.angular_force:
            y_pred = np.repeat(y_pred[:, np.newaxis], 3, axis=1)
        print("r squared (before removing out layers): ", self.model.score(X_test, y_test))
        if self.angular_force:
            y_test = np.repeat(y_test[:, np.newaxis], 3, axis=1)
        print("mean squared error (before removing out layers): ", mean_squared_error(y_test, y_pred, squared=False))
        self.ploting_fitting_results_data = (number_degrees, plt_mean_squared_error, plt_r_squared,y_test,y_pred)
        print("-"*20)

    def post_processing(self,directory):
        self.current_dir = directory
        self.load_data(self.current_dir)
        self.calc_distances()
        self.repeat_values()
        self.calc_angle()

    def load_data(self,directory):
        print(directory)
        # directory = os.path.join(directory, "sliced_data")+"\\"
        # self.norm_size = np.nanmedian(np.loadtxt(os.path.join(directory, "needle_area_sizes.csv"), delimiter=","))
        perspective_squares_sizes_height = np.loadtxt(os.path.join(directory, "perspective_squares_sizes_height.csv"), delimiter=",")
        perspective_squares_sizes_width = np.loadtxt(os.path.join(directory, "perspective_squares_sizes_width.csv"), delimiter=",")
        self.perspective_squares_sizes = np.stack((np.expand_dims(perspective_squares_sizes_height,1), np.expand_dims(perspective_squares_sizes_width,1)), axis=2)
        perspective_squares_coordinates_x = np.loadtxt(os.path.join(directory, "perspective_squares_coordinates_x.csv"), delimiter=",")
        perspective_squares_coordinates_y = np.loadtxt(os.path.join(directory, "perspective_squares_coordinates_y.csv"), delimiter=",")
        self.dst = np.stack((np.expand_dims(perspective_squares_coordinates_x,1), np.expand_dims(perspective_squares_coordinates_y,1)), axis=2)
        # self.PTM = self.create_PTM(self.dst)
        self.norm_size = np.nanmedian(np.sqrt(perspective_squares_sizes_height*perspective_squares_sizes_width))
        # print(self.norm_size)
        free_ends_coordinates_x = np.loadtxt(os.path.join(directory, "free_ends_coordinates_x.csv"), delimiter=",")
        free_ends_coordinates_y = np.loadtxt(os.path.join(directory, "free_ends_coordinates_y.csv"), delimiter=",")
        self.free_ends_coordinates = np.stack((np.expand_dims(free_ends_coordinates_x,1), np.expand_dims(free_ends_coordinates_y,1)), axis=2)
        # self.free_ends_coordinates = self.transform_projection(self.dst, self.free_ends_coordinates)
        fixed_ends_coordinates_x = np.loadtxt(os.path.join(directory, "fixed_ends_coordinates_x.csv"), delimiter=",")
        fixed_ends_coordinates_y = np.loadtxt(os.path.join(directory, "fixed_ends_coordinates_y.csv"), delimiter=",")
        self.fixed_ends_coordinates = np.stack((np.expand_dims(fixed_ends_coordinates_x,1), np.expand_dims(fixed_ends_coordinates_y,1)), axis=2)
        # self.fixed_ends_coordinates = self.transform_projection(self.dst, self.fixed_ends_coordinates)
        needle_part_coordinates_x = np.loadtxt(os.path.join(directory, "needle_part_coordinates_x.csv"), delimiter=",")
        needle_part_coordinates_y = np.loadtxt(os.path.join(directory, "needle_part_coordinates_y.csv"), delimiter=",")
        needle_part_coordinates = np.stack((needle_part_coordinates_x, needle_part_coordinates_y), axis=2)
        self.object_center = needle_part_coordinates[:, 0, :]
        # self.object_center = self.transform_projection(self.dst, self.object_center)
        self.needle_tip_coordinates = needle_part_coordinates[:, -1, :]
        # self.needle_tip_coordinates = self.transform_projection(self.dst, self.needle_tip_coordinates)
        self.num_of_frames = self.free_ends_coordinates.shape[0]
        self.rest_bool = np.ones((self.num_of_frames, 1), dtype=bool)


    def calc_distances(self):
        object_center = np.repeat(self.object_center[:, np.newaxis, :], self.num_of_springs, axis=1)
        needle_tip_coordinates = np.repeat(self.needle_tip_coordinates[:, np.newaxis, :], self.num_of_springs, axis=1)
        self.needle_length = np.nanmedian(np.linalg.norm(self.needle_tip_coordinates - self.object_center, axis=1))
        self.object_center_to_free_end_distance = np.linalg.norm(self.free_ends_coordinates - object_center, axis=2)
        self.object_center_to_fixed_end_distance = np.linalg.norm(self.fixed_ends_coordinates - object_center, axis=2)
        self.object_needle_tip_to_fixed_end_distance = np.linalg.norm(self.fixed_ends_coordinates - needle_tip_coordinates, axis=2)

    def repeat_values(self):
        nest_direction_to_center = np.stack((self.object_center[:, 0], self.object_center[:, 1]-500), axis=1)
        nest_direction = np.stack((self.fixed_ends_coordinates[:, 0, 0], self.fixed_ends_coordinates[:, 0, 1]-500), axis=1)
        self.nest_direction_repeated = np.repeat(nest_direction[:, np.newaxis, :], self.fixed_ends_coordinates.shape[1], axis=1)
        self.nest_direction_to_center_repeated = np.repeat(nest_direction_to_center[:, np.newaxis, :], self.fixed_ends_coordinates.shape[1], axis=1)
        self.object_center_repeated = copy.copy(np.repeat(self.object_center[:, np.newaxis, :], self.fixed_ends_coordinates.shape[1], axis=1))
        self.needle_tip_coordinates_repeated = np.repeat(self.needle_tip_coordinates[:, np.newaxis, :], self.free_ends_coordinates.shape[1], axis=1)

    def calc_angle(self):
        self.fixed_end_angle_to_nest = utils.calc_angle_matrix(self.object_center_repeated, self.fixed_ends_coordinates, self.nest_direction_repeated)+np.pi
        # self.free_end_angle_to_nest = utils.calc_angle_matrix(self.free_ends_coordinates, self.object_center_repeated, self.nest_direction_repeated)+np.pi
        # print(self.current_dir)
        # print(np.where((self.fixed_end_angle_to_nest <(np.pi*1.05))*self.fixed_end_angle_to_nest >(np.pi*0.95) ))
        # print(np.nanmin(self.fixed_end_angle_to_nest), np.nanmax(self.fixed_end_angle_to_nest), np.nanmedian(self.fixed_end_angle_to_nest))
        # print(np.where(self.fixed_end_angle_to_nest< -1.5 * np.pi))
        self.needle_part_angle_to_nest = utils.calc_angle_matrix(self.nest_direction_to_center_repeated, self.object_center_repeated, self.needle_tip_coordinates_repeated) + np.pi
        # self.free_end_angle_to_needle_part = utils.calc_angle_matrix(self.needle_tip_coordinates_repeated, self.object_center_repeated, self.free_ends_coordinates)# + np.pi
        # self.fixed_end_angle_to_needle_part = utils.calc_angle_matrix(self.needle_tip_coordinates_repeated, self.object_center_repeated, self.fixed_ends_coordinates)# + np.pi
        # self.fixed_to_needle_angle_change = utils.calc_pulling_angle_matrix(self.needle_tip_coordinates_repeated, self.object_center_repeated, self.fixed_ends_coordinates)

    def norm_values(self, matrix, model):
        angles_to_nest = np.expand_dims(self.fixed_end_angle_to_nest, axis=2)
        # angles_to_nest = np.expand_dims(self.needle_part_angle_to_nest, axis=2)
        # fixed_end_distance = np.expand_dims(self.object_center_to_fixed_end_distance, axis=2)
        # object_center = self.object_center_repeated
        # fixed_to_tip_distance = np.expand_dims(self.object_needle_tip_to_fixed_end_distance, axis=2)
        #
        # fixed_to_needle_angle_change = np.expand_dims(np.repeat(np.expand_dims(np.nanmedian(self.fixed_to_needle_angle_change, axis=1), axis=1), self.num_of_springs, axis=1), axis=2)
        # needle_length = np.expand_dims(np.repeat(np.expand_dims(np.linalg.norm(
        #     self.needle_tip_coordinates - self.object_center, axis=1), axis=1), self.num_of_springs, axis=1), axis=2)
        X = np.concatenate((np.sin(angles_to_nest), np.cos(angles_to_nest)), axis=2)
        not_nan_idx = ~(np.isnan(matrix) + np.isnan(X).any(axis=2))
        # print(not_nan_idx.shape)
        prediction_matrix = np.zeros(matrix.shape)
        for col in range(matrix.shape[1]):
            prediction_matrix[not_nan_idx[:, col], col] = model.predict(X[not_nan_idx[:, col], col, :])
        # print(prediction_matrix.shape)
        return prediction_matrix

    def create_PTM(self, dst):
        """
        Creates projective transformation matrix
        :param dst: destination points coordinates
        :return:
        """
        dst = np.median(dst[~np.isnan(dst[:, 0, 0, :]).any(axis=1),0,:,:], axis=0).transpose().astype(np.float32)
        # distances = np.concatenate([np.sqrt(np.sum((coordinates[:, :, :, 0] - coordinates[:, :, :, 1]) ** 2, axis=2)),
        #                         np.sqrt(np.sum((coordinates[:, :, :, 1] - coordinates[:, :, :, 2]) ** 2, axis=2)),
        #                         np.sqrt(np.sum((coordinates[:, :, :, 2] - coordinates[:, :, :, 3]) ** 2, axis=2)),
        # np.sqrt(np.sum((coordinates[:, :, :, 3] - coordinates[:, :, :, 0]) ** 2, axis=2))], axis=1)
        distances = np.linalg.norm(dst[0] - dst[1]), np.linalg.norm(dst[0] - dst[3])
        src = np.array(
            [dst[0], [dst[0][0] + distances[0], dst[0][1]], [dst[0][0] + distances[0], dst[0][1] + distances[1]],
             [dst[0][0], dst[0][1] + distances[1]]]).astype(np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        return M

    def transform_projection(self, dst, coordinates_to_project):
        """
        :param dst: destination coordinates - the perspective squares centers
        :param coordinates_to_project: the coordinates to project
        :return:
        """
        # dst = dst[:, 0, :, :]
        original_shape = coordinates_to_project.shape
        if len(original_shape) == 2:
            coordinates_to_project = np.expand_dims(coordinates_to_project, axis=1)
            # print(coordinates_to_project.shape)
        dst_median = np.median(dst[~np.isnan(dst[:, 0, 0, :]).any(axis=1), 0, :, :], axis=0).transpose().astype(np.float32)
        frames_changed = []
        for row in range(coordinates_to_project.shape[0]):
            curr_dst = dst[row, 0, :, :].transpose().astype(np.float32)
            curr_sizes = self.perspective_squares_sizes[row, 0, :]
            if not np.any(np.isnan(curr_dst))\
                and not np.any(np.isnan(coordinates_to_project[row]))\
                and not np.any(np.sqrt(curr_sizes[0]*curr_sizes[1]) < self.norm_size*0.95):
                # and not np.any(np.isnan(coordinates_to_project[row]))\
                # and not np.any((0.99 < curr_dst/dst_median)*(curr_dst/dst_median > 1.01))\
                frames_changed.append(row)
                # print(curr_dst, curr_sizes)
                distances = np.linalg.norm(curr_dst[0] - curr_dst[1]), np.linalg.norm(curr_dst[0] - curr_dst[3])
                src = np.array(
                    [curr_dst[0], [curr_dst[0][0] + distances[0], curr_dst[0][1]],
                        [curr_dst[0][0] + distances[0], curr_dst[0][1] + distances[1]],
                        [curr_dst[0][0], curr_dst[0][1] + distances[1]]]).astype(np.float32)
                M = cv2.getPerspectiveTransform(src, curr_dst)
                M_inv = np.linalg.inv(M)
                coordinates_to_project[row] = cv2.perspectiveTransform(coordinates_to_project[row:row+1], M_inv)
        coordinates_to_project = coordinates_to_project.reshape(original_shape)
        # print("frames changed: ", len(frames_changed))
        return coordinates_to_project

    def calc_calibration_force(self, mass1):
        self.calibration_force_direction = (self.fixed_end_angle_to_nest).flatten()-np.pi
        mass1 *= 1e-3  # convert to kg
        G = 6.67430e-11
        mass2 = 5.972e24  # Mass of the Earth (in kilograms)
        distance = 6371000  # Approximate radius of the Earth (in meters)
        # G = 9.81
        force = (G * mass1 * mass2) / (distance ** 2)
        force_mN = force * 1000
        self.calibration_force_magnitude = np.repeat(force_mN, self.num_of_frames)


# if __name__ == "__main__":
#
#     calibration_dir4 = "Z:\\Dor_Gabay\\ThesisProject\\data\\calibration\\post_slicing\\calibration_perfect2\\sliced_videos\\"
#     calibration_video_dir4 = "Z:\\Dor_Gabay\\ThesisProject\\data\\videos\\calibration_perfect2\\sliced_videos\\"
#     directories_4 = [os.path.join(calibration_dir4, o) for o in os.listdir(calibration_dir4)
#                      if os.path.isdir(os.path.join(calibration_dir4, o)) and "plots" not in os.path.basename(o)]# and "S5430006_sliced" not in os.path.basename(o)]  # not in os.path.basename(o) and "S5430003_sliced" not in os.path.basename(o)]
#     weights4 = list(np.array([0.11656, 0.16275, 0.21764, 0.28268, 0.10657]) - 0.10657)
#     # weights3 = list(np.array([0.41785, 0.2389, 0.18053, 0.15615, 0.11511, 0.12561, 0.10610]) - 0.10610)
#     video_paths4 = [os.path.join(calibration_video_dir4, o) for o in os.listdir(calibration_video_dir4)
#                     if "MP4" in os.path.basename(o) and "plots" not in os.path.basename(o)]#]# and "_sliced" in os.path.basename(o)]# and "S5430006_sliced" not in os.path.basename(o)]
#
#     CalibrationModeling(directories=directories_4, weights=weights4, output_path=calibration_dir4, video_paths=video_paths4)
#     # Calibration(directories=directories_3, weights=weights3, output_path=calibration_dir3, video_paths=video_paths3)
#     # Calibration(directories=directories_2, weights=weights2, output_path=calibration_dir2, video_paths=video_paths2)
#     print("-"*60)
#     print("Calibration model has been created.")
#     print("-"*60)

