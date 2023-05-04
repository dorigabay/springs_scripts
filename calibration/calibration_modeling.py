import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from data_analysis import utils
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pickle
# from calibration import utils


class Calibration:

    def __init__(self, directories, weights, output_path, video_paths=None):
        self.directories = directories
        self.weights = weights
        self.output_path = output_path
        self.video_paths = video_paths
        self.zero_weight_dir = self.directories[np.where(np.array(self.weights) == 0)[0][0]]
        self.get_bias_equations()
        self.combine_all_data()
        self.create_calibration_model()
        self.plot_pulling_angle_to_nest_angle()
        self.plot_fitting_results()

    def get_bias_equations(self):
        print("-"*60)
        print("Saving calibration model and plots to: ", self.output_path)
        print("Creating bias equations from: ", self.zero_weight_dir)
        self.post_processing(self.zero_weight_dir)
        self.video_path = self.video_paths[np.where(np.array(self.weights) == 0)[0][0]]
        # self.get_nest_direction(self.video_path)
        _, self.free_end_angle_to_blue_part_bias_equations = self.norm_values(self.free_end_angle_to_nest,
                                                                              self.free_end_angle_to_blue_part,
                                                                              bias_bool=self.rest_bool,
                                                                              find_boundary_column=True)
        _, self.fixed_end_angle_to_blue_part_bias_equations = self.norm_values(self.fixed_end_angle_to_nest,
                                                        self.fixed_end_angle_to_blue_part,
                                                        bias_bool=self.rest_bool,
                                                        find_boundary_column=True)
        self.find_fixed_coordinates(bias_equations_free=self.free_end_angle_to_blue_part_bias_equations,
                                            bias_equations_fixed=self.fixed_end_angle_to_blue_part_bias_equations)
        self.calc_pulling_angle()
        self.calc_spring_length()
        # self.zero_length = np.nanmedian(self.spring_length.flatten())

    def get_nest_direction(self,video_path):
        import cv2
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        #collect two points from the user
        from general_video_scripts.collect_color_parameters import collect_points
        points = collect_points(frame, 2)
        upper_point = points[0]
        lower_point = points[1]
        lower_artificial_point = (lower_point[0], upper_point[1])
        #calculate the angle between the three points
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
        extension = np.array(())
        weights = np.array(())
        nest_direction = np.array(())
        first = True
        b = -1
        self.angular_force = True
        for dir, weight, video_path in zip(self.directories[:], self.weights[:], self.video_paths[:]):
            # self.get_nest_direction(video_path)
            self.post_processing(dir)
            self.video_path = video_path
            self.find_fixed_coordinates(bias_equations_free=self.free_end_angle_to_blue_part_bias_equations,
                                            bias_equations_fixed=self.fixed_end_angle_to_blue_part_bias_equations)
            self.calc_pulling_angle(bias_equations=self.pulling_angle_bias_equation)
            self.calc_spring_length(bias_equations=self.length_bias_equation)
            self.calc_calibration_force(weight)
            self.weight = weight
            calibration_force_magnitude = np.append(calibration_force_magnitude, self.calibration_force_magnitude)
            pulling_angle = np.append(pulling_angle, (self.pulling_angle).flatten())
            extension = np.append(extension, self.spring_extension.flatten())
            if weight == 0:
                calibration_force_direction = np.append(calibration_force_direction, np.zeros(self.calibration_force_direction.shape))
            #     if first:
            #         first = False
            #     else:
            #         calibration_force_direction = np.append(calibration_force_direction,
            #                                                 self.calibration_force_direction)
            else:
                calibration_force_direction = np.append(calibration_force_direction, self.calibration_force_direction)
            weights = np.append(weights, np.repeat(weight, len(self.calibration_force_magnitude)))
            nest_direction = np.append(nest_direction, self.fixed_end_fixed_coordinates_angle_to_nest.flatten())#+self.calib_nest_angle

        data = np.vstack((pulling_angle,
                          extension,
                          calibration_force_direction,
                          calibration_force_magnitude,
                          weights,
                          nest_direction
                          )).transpose()
        # data[data[:, 0] < 0, 0] = np.nan
        data = data[~np.isnan(data).any(axis=1)]
        self.X = data[:, 0:2]
        negative = self.X[:, 0] > 0
        # print(self.X[negative, 0])
        self.X[negative, 0] *= -1
        # print("X shape: ", self.X.shape)
        self.X = np.array(pd.concat([pd.DataFrame(self.X), pd.DataFrame(self.X)*-1], axis=0))
        # print("X shape: ", self.X.shape)
        if self.angular_force:
            self.y = np.sin(data[:, 2]) * data[:, 3]
            # print(self.y[negative])
            self.y[negative] *= -1
            # print("y shape: ", self.y.shape)
            self.y = np.array(pd.concat([pd.DataFrame(self.y), pd.DataFrame(self.y)*-1], axis=0)[0])
            # print("y shape: ", self.y[0].shape)
            # print("y shape: ", self.y.shape)
        else:
            self.y = data[:, 2:4]

        self.weights_labels = data[:, 4]
        self.weights_labels = np.array(pd.concat([pd.DataFrame(self.weights_labels), pd.DataFrame(self.weights_labels)], axis=0)[0])
        self.nest_direction = data[:, 5]
        self.nest_direction = np.array(pd.concat([pd.DataFrame(self.nest_direction), pd.DataFrame(self.nest_direction)], axis=0)[0])
        self.weights = self.weights[:]

    def create_calibration_model(self):
        number_degrees = list(range(1, 3))
        plt_mean_squared_error = []
        plt_r_squared = []
        best_degrees = []
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(self.X, self.y, self.weights_labels, test_size=0.2, random_state=42)
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
        self.ploting_fitting_results_data = (number_degrees, plt_mean_squared_error, plt_r_squared,y_test,y_pred, weights_test)
        print("-"*20)
        print(f"min pulling_angel: {np.min(self.X[:,0])}, max pulling_angel: {np.max(self.X[:,0])}, mean pulling_angel: {np.mean(self.X[:,0])}")
        print(f"min extension: {np.min(self.X[:,1])}, max extension: {np.max(self.X[:,1])}, mean extension: {np.mean(self.X[:,1])}")
        if not self.angular_force:
            print(f"min force_direction: {np.min(self.y[:,0])}, max force_direction: {np.max(self.y[:,0])}, mean force_direction: {np.mean(self.y[:,0])}")
            print(f"min force_magnitude: {np.min(self.y[:,1])}, max force_magnitude: {np.max(self.y[:,1])}, mean force_magnitude: {np.mean(self.y[:,1])}")
        else:
            print(f"min force_magnitude: {np.min(self.y)}, max force_magnitude: {np.max(self.y)}, mean force_magnitude: {np.mean(self.y)}")


    def plot_pulling_angle_to_nest_angle(self):
        save_path = os.path.join(self.output_path, "plots","weights")
        os.makedirs(save_path, exist_ok=True)
        for weight in self.weights:
            title = np.round(weight, 3)
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

            x = self.nest_direction[self.weights_labels == weight]
            above = x > np.pi
            below = x < np.pi
            x[above] = x[above] - np.pi
            x[below] = x[below] + np.pi
            x -= np.pi

            if not self.angular_force:
                y1 = self.X[self.weights_labels == weight][:, 0]
                above = y1 > np.pi
                below = y1 < np.pi
                y1[above] = y1[above] - np.pi
                y1[below] = y1[below] + np.pi
                y1 -= np.pi
            else:
                y1 = self.X[self.weights_labels == weight]
            ax1.plot(x, y1, "o",alpha=0.5)
            ax1.set_xlabel("Direction to the nest")
            ax1.set_ylabel("Pulling direction")
            ax1.set_title(f"weight: {title}")

            if not self.angular_force:
                y2 = self.X[self.weights_labels == weight][:, 1]
            else:
                y2 = self.y[self.weights_labels == weight]
            ax2.plot(x, y2, "o",alpha=0.5)
            ax2.set_xlabel("Direction to the nest")
            ax2.set_ylabel("Spring extension")

            y_pred = self.model.predict(self.X[self.weights_labels == weight, :])
            if not self.angular_force:
                y3 = self.y[self.weights_labels == weight][:, 0]
                direction_pred = y_pred[:, 0]
                above = y3 > 0
                below = y3 < 0
                y3[above] = y3[above] - np.pi
                y3[below] = y3[below] + np.pi
                direction_pred[above] = direction_pred[above] - np.pi
                direction_pred[below] = direction_pred[below] + np.pi
            else:
                y3 = self.y[self.weights_labels == weight]
                direction_pred = y_pred
            ax3.plot(x, y3, "o",alpha=0.5)
            ax3.plot(x, direction_pred, "o", color="red",alpha=0.1)
            ax3.set_xlabel("Direction to the nest")
            ax3.set_ylabel("Force")

            if not self.angular_force:
                y4 = self.y[self.weights_labels == weight][:, 1]
                extent_pred = y_pred[:, 1]
                ax4.plot(x, y4, "o",alpha=0.5)
                ax4.plot(x, extent_pred, "o", color="red",alpha=0.1)
            ax4.plot()
            ax4.set_xlabel("Direction to the nest")
            ax4.set_ylabel("Force magnitude")

            fig.tight_layout()
            fig.set_size_inches(7.5, 5)
            fig.savefig(os.path.join(save_path, f"weight_{title}.png"))

    def plot_fitting_results(self):
        number_degrees, plt_mean_squared_error, plt_r_squared, y_test, y_pred, weights_test = self.ploting_fitting_results_data
        save_dir = os.path.join(self.output_path, "plots")
        os.makedirs(save_dir, exist_ok=True)
        for true,pred,name in \
                zip([np.abs(y_test[:, 0])*y_test[:, 1],y_test[:, 0],y_test[:, 1]],
                    [np.abs(y_pred[:, 0])*y_pred[:, 1],y_pred[:, 0],y_pred[:, 1]],
                    ["angle_times_extension","angle","extension"]):
            fig, ax = plt.subplots()
            ax.scatter(true,pred, c=weights_test, cmap="viridis")
            from matplotlib import cm
            cmap = cm.get_cmap('viridis', 10)
            sm = cm.ScalarMappable(cmap=cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm)
            cbar.set_label('Weight (g)')
            ax.set_xlabel("y_true")
            ax.set_ylabel("y_predicted")
            plt.savefig(os.path.join(save_dir, f"pred_true_comparison-{name}.png"))
            plt.clf()

        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Degree")
        ax1.set_ylabel("Mean Squared Error", color="red")
        ax1.plot(number_degrees, plt_mean_squared_error, color="red")
        ax1.scatter(number_degrees, plt_mean_squared_error, color="green")
        ax2 = ax1.twinx()
        ax2.set_ylabel("R squared", color="blue")
        ax2.plot(number_degrees, plt_r_squared, color="blue")
        from matplotlib.ticker import MaxNLocator
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(save_dir, "mean_squared_error.png"))
        plt.clf()

    def post_processing(self,directory):
        self.current_dir = directory
        self.load_data(self.current_dir)
        self.calc_distances()
        self.repeat_values()
        self.calc_angle()


    def load_data(self,directory):
        self.num_of_springs = 1
        self.norm_size = pickle.load(open(os.path.join(directory,"blue_median_area.pickle"), "rb"))
        directory = os.path.join(directory, "raw_analysis")+"\\"
        fixed_ends_coordinates_x = np.loadtxt(os.path.join(directory,"fixed_ends_coordinates_x.csv"), delimiter=",")
        fixed_ends_coordinates_y = np.loadtxt(os.path.join(directory,"fixed_ends_coordinates_y.csv"), delimiter=",")
        self.fixed_ends_coordinates = np.stack((np.expand_dims(fixed_ends_coordinates_x,1), np.expand_dims(fixed_ends_coordinates_y,1)), axis=2)
        free_ends_coordinates_x = np.loadtxt(os.path.join(directory,"free_ends_coordinates_x.csv"), delimiter=",")
        free_ends_coordinates_y = np.loadtxt(os.path.join(directory,"free_ends_coordinates_y.csv"), delimiter=",")
        self.free_ends_coordinates = np.stack((np.expand_dims(free_ends_coordinates_x,1), np.expand_dims(free_ends_coordinates_y,1)), axis=2)
        blue_part_coordinates_x = np.loadtxt(os.path.join(directory,"blue_part_coordinates_x.csv"), delimiter=",")
        blue_part_coordinates_y = np.loadtxt(os.path.join(directory,"blue_part_coordinates_y.csv"), delimiter=",")
        blue_part_coordinates = np.stack((blue_part_coordinates_x, blue_part_coordinates_y), axis=2)
        self.object_center = blue_part_coordinates[:, 0, :]
        self.blue_tip_coordinates = blue_part_coordinates[:, -1, :]
        self.num_of_frames = self.free_ends_coordinates.shape[0]
        self.rest_bool = np.ones((self.num_of_frames, self.num_of_springs), dtype=bool)

    def calc_distances(self):
        object_center = np.repeat(self.object_center[:, np.newaxis, :], self.num_of_springs, axis=1)
        self.blue_length = np.nanmedian(np.linalg.norm(self.blue_tip_coordinates - self.object_center, axis=1))
        self.object_center_to_free_end_distance = np.linalg.norm(self.free_ends_coordinates - object_center, axis=2)
        self.object_center_to_fixed_end_distance = np.linalg.norm(self.fixed_ends_coordinates - object_center, axis=2)

    def repeat_values(self):
        nest_direction = np.stack((self.object_center[:, 0], self.object_center[:, 1]-100), axis=1)
        self.nest_direction_repeated = np.repeat(nest_direction[:, np.newaxis, :], self.free_ends_coordinates.shape[1], axis=1)
        self.object_center_repeated = copy.copy(np.repeat(self.object_center[:, np.newaxis, :], self.free_ends_coordinates.shape[1], axis=1))
        self.blue_tip_coordinates_repeated = np.repeat(self.blue_tip_coordinates[:, np.newaxis, :], self.free_ends_coordinates.shape[1], axis=1)

    def calc_angle(self):
        self.free_end_angle_to_nest = utils.calc_angle_matrix(self.free_ends_coordinates, self.object_center_repeated, self.nest_direction_repeated)+np.pi
        self.fixed_end_angle_to_nest = utils.calc_angle_matrix(self.fixed_ends_coordinates, self.object_center_repeated, self.nest_direction_repeated)+np.pi
        self.blue_part_angle_to_nest = utils.calc_angle_matrix(self.nest_direction_repeated, self.object_center_repeated, self.blue_tip_coordinates_repeated)+np.pi
        self.free_end_angle_to_blue_part = utils.calc_angle_matrix(self.blue_tip_coordinates_repeated, self.object_center_repeated, self.free_ends_coordinates)+np.pi
        self.fixed_end_angle_to_blue_part = utils.calc_angle_matrix(self.blue_tip_coordinates_repeated, self.object_center_repeated,self.fixed_ends_coordinates)+np.pi

    def find_column_on_boundry(self,X):

        # columns_on_boundry = []
        # for s in range(X.shape[1]):
        #     no_nans_fixed_end_angle_to_blue_part = X[:, s][~np.isnan(X[:, s])]
        #     print(no_nans_fixed_end_angle_to_blue_part)
        #     print(np.sum(np.abs(np.diff(no_nans_fixed_end_angle_to_blue_part))>np.pi))
        #     if np.sum(np.abs(np.diff(no_nans_fixed_end_angle_to_blue_part))>np.pi) > 5:
        #         columns_on_boundry.append(s)
        # if len(columns_on_boundry) == 1:
        #     column_on_boundry = columns_on_boundry[0]
        # else:
        #     # print(X.shape)
        #     # print(len(columns_on_boundry))
        #     # return 0
        #     raise ValueError("more than one column on boundry")
        return 0

    def norm_values(self, X, Y, bias_bool=None, find_boundary_column=False, bias_equations=None):
        if bias_bool is not None:
            X_bias = copy.deepcopy(X)
            Y_bias = copy.deepcopy(Y)
            X_bias[np.invert(bias_bool)] = np.nan
            Y_bias[np.invert(bias_bool)] = np.nan
        else:
            X_bias = X
            Y_bias = Y
        if find_boundary_column:
            column_on_boundary = self.find_column_on_boundry(Y)
            above_nonan = Y_bias[:, column_on_boundary] > np.pi
            Y_bias[above_nonan, column_on_boundary] -= 2*np.pi
        Y_bias -= np.nanmedian(Y_bias, axis=0)
        normed_Y = np.zeros(Y.shape)
        if bias_equations is None:
            bias_equations = []
            for i in range(X.shape[1]):
                df = pd.DataFrame({"x": X_bias[:, i], "y": Y_bias[:, i]}).dropna()
                bias_equation = utils.deduce_bias_equation(df["x"], df["y"])
                bias_equations.append(bias_equation)
        if find_boundary_column:
            Y = copy.deepcopy(Y)
            above = Y[:, column_on_boundary] > np.pi
            Y[above, column_on_boundary] -= 2*np.pi
        for i in range(self.num_of_springs):
            bias_equation = bias_equations[i]
            normed_Y[:, i] = utils.normalize(Y[:, i], X[:, i], bias_equation)
        if find_boundary_column:
            below = normed_Y[:, column_on_boundary] < 0
            normed_Y[below, column_on_boundary] += 2*np.pi
        return normed_Y, bias_equations

    def find_fixed_coordinates(self,bias_equations_free=None,bias_equations_fixed=None):
        def calc_fixed(distance_to_object_center,end_type):
            median_distance = None  # removable line
            angle_to_blue = None  # removable line
            if end_type== "free":
                angle_to_blue, self.free_end_angle_to_blue_bias_equations = \
                    self.norm_values(self.free_end_angle_to_nest, self.free_end_angle_to_blue_part,
                                     bias_bool=self.rest_bool, bias_equations=bias_equations_free, find_boundary_column=True)
                distance_to_object_center[~self.rest_bool] = np.nan
                median_distance = np.nanmedian(distance_to_object_center, axis=0)
            elif end_type == "fixed":
                angle_to_blue, self.fixed_end_angle_to_blue_bias_equations = \
                    self.norm_values(self.fixed_end_angle_to_nest, self.fixed_end_angle_to_blue_part,
                                        bias_bool=self.rest_bool, bias_equations=bias_equations_fixed, find_boundary_column=True)

                median_distance = np.nanmedian(distance_to_object_center, axis=0)
            median_distance = np.repeat(median_distance[np.newaxis, :], self.num_of_frames, axis=0)
            angle_to_blue_part_normed = utils.bound_angle(self.blue_part_angle_to_nest+angle_to_blue-np.pi/2)
            fixed_coordinates = self.object_center_repeated + np.stack((np.cos(angle_to_blue_part_normed) * median_distance,
                                                                        np.sin(angle_to_blue_part_normed) * median_distance), axis=2)
            return fixed_coordinates
        self.fixed_end_fixed_coordinates = calc_fixed(self.object_center_to_fixed_end_distance,end_type="fixed")
        self.free_end_fixed_coordinates = calc_fixed(self.object_center_to_free_end_distance,end_type="free")

        # frame = utils.load_first_frame(self.video_path)
        # import cv2
        # for fix, free in zip(self.fixed_end_fixed_coordinates[0], self.free_end_fixed_coordinates[0]):
        #     print(fix, free)
        #     frame = cv2.circle(frame, tuple(fix.astype(int)), 5, (0, 0, 255), -1)
        #     frame = cv2.circle(frame, tuple(free.astype(int)), 5, (0, 255, 0), -1)
        # cv2.imshow("frame", frame)
        # cv2.waitKey(0)

        self.calc_fixed_coordinates_angles()

    def calc_fixed_coordinates_angles(self):
        self.free_end_fixed_coordinates_angle_to_nest = utils.calc_angle_matrix(self.free_end_fixed_coordinates, self.object_center_repeated, self.nest_direction_repeated)+np.pi
        self.fixed_end_fixed_coordinates_angle_to_nest = utils.calc_angle_matrix(self.fixed_end_fixed_coordinates, self.object_center_repeated, self.nest_direction_repeated)+np.pi

        self.free_end_fixed_coordinates_angle_to_blue_part = utils.calc_angle_matrix(self.blue_tip_coordinates_repeated,
                                                                   self.object_center_repeated,
                                                                   self.free_end_fixed_coordinates)+np.pi
        self.fixed_end_fixed_coordinates_angle_to_blue_part = utils.calc_angle_matrix(self.blue_tip_coordinates_repeated,
                                                                    self.object_center_repeated,
                                                                    self.fixed_end_fixed_coordinates)+np.pi

    def calc_pulling_angle(self,bias_equations=None):
        self.pulling_angle = utils.calc_angle_matrix(self.free_end_fixed_coordinates, self.fixed_end_fixed_coordinates,
                                                     self.object_center_repeated) + np.pi
        if bias_equations is None:
            self.pulling_angle, self.pulling_angle_bias_equation = \
                self.norm_values(self.free_end_fixed_coordinates_angle_to_nest, self.pulling_angle,
                                 bias_bool=self.rest_bool, find_boundary_column=True)
        else:
            self.pulling_angle, _ = self.norm_values(self.free_end_fixed_coordinates_angle_to_nest, self.pulling_angle,
                                                     bias_bool=self.rest_bool, bias_equations=bias_equations,
                                                     find_boundary_column=True)
        self.pulling_angle = self.pulling_angle - np.pi
        # self.pulling_angle = utils.bound_angle(self.pulling_angle)
        above = self.pulling_angle > 0
        below = self.pulling_angle < 0
        self.pulling_angle[above] = self.pulling_angle[above] - np.pi
        self.pulling_angle[below] = self.pulling_angle[below] + np.pi
        # find the index of top 5% closest angles to zero
        self.zero_angles_index = np.argsort(np.abs(self.pulling_angle), axis=0)[:int(self.num_of_frames * 0.05), :]
        # self.free_end_fixed_coordinates_angle_to_nest+=self.calib_nest_angle
        # print(np.median(self.free_end_fixed_coordinates_angle_to_nest[self.zero_angles_index]))
        # self.negative_pulling_angle = self.pulling_angle < 0
        # self.pulling_angle = utils.calc_angle_matrix(self.free_end_fixed_coordinates,self.fixed_end_fixed_coordinates,self.object_center_repeated)
        # above = self.pulling_angle > 0
        # below = self.pulling_angle < 0
        # self.pulling_angle[above] = self.pulling_angle[above] - np.pi
        # self.pulling_angle[below] = self.pulling_angle[below] + np.pi
        # if zero_angles:
        #     median_pulling_angle_at_rest = copy.copy(self.pulling_angle)
        #     median_pulling_angle_at_rest[np.invert(self.rest_bool)] = np.nan
        #     median_spring_length_at_rest = np.nanmedian(median_pulling_angle_at_rest, axis=0)
        #     self.pulling_angle -= median_spring_length_at_rest

    def calc_spring_length(self,bias_equations=None,zero_length=None):
        self.spring_length = np.linalg.norm(self.free_end_fixed_coordinates - self.fixed_end_fixed_coordinates , axis=2)
        self.spring_length = utils.interpolate_data(self.spring_length,
                                                       utils.find_cells_to_interpolate(self.spring_length))
        if bias_equations is None:
            self.spring_length, self.length_bias_equation = self.norm_values(self.fixed_end_angle_to_nest, self.spring_length,
                                                bias_bool=self.rest_bool, find_boundary_column=False)
            self.spring_length /= self.norm_size
            median_spring_length_at_rest = copy.copy(self.spring_length)
            median_spring_length_at_rest[np.invert(self.rest_bool)] = np.nan
            median_spring_length_at_rest = np.nanmedian(median_spring_length_at_rest, axis=0)
            self.zero_length = median_spring_length_at_rest
            # self.spring_length /= median_spring_length_at_rest
            # self.zero_length = np.nanmedian(self.spring_length.flatten())
            # self.spring_extension = self.spring_length -1
            # self.spring_extension = self.spring_length - median_spring_length_at_rest
        else:
            self.spring_length, bias_equations = self.norm_values(self.fixed_end_angle_to_nest, self.spring_length,
                                                  bias_bool=self.rest_bool, bias_equations=bias_equations,
                                                  find_boundary_column=False)
            self.spring_length /= self.norm_size
            # median_spring_length_at_rest = copy.copy(self.spring_length)
            # median_spring_length_at_rest[np.invert(self.rest_bool)] = np.nan
            # median_spring_length_at_rest = np.nanmedian(median_spring_length_at_rest, axis=0)
            # self.spring_length /= median_spring_length_at_rest
        self.spring_length /= self.zero_length
        self.spring_extension = self.spring_length - 1
            # self.spring_extension = self.spring_length.flatten() - self.zero_length
        return bias_equations

    def calc_calibration_force(self, calibration_weight):
        self.calibration_force_direction = (self.fixed_end_fixed_coordinates_angle_to_nest).flatten()
        above = self.calibration_force_direction > np.pi
        self.calibration_force_direction[above] = self.calibration_force_direction[above] - 2 * np.pi
        G = 9.81
        weight_in_Kg = calibration_weight*1e-3
        self.calibration_force_magnitude = np.repeat(weight_in_Kg * G, self.num_of_frames)


########################################################################################################################
# def iter_dirs(spring_type_directories,function):
#     springs_analysed = {}
#     for spring_type in spring_type_directories:
#         springs_analysed[spring_type] = {}
#         for video_data in spring_type_directories[spring_type]:
#             video_name = video_data.split("\\")[-2]
#             object = function(video_data)
#             # object.save_data(video_data)
#             # springs_analysed[spring_type][video_name] = object
#     return springs_analysed
#


if __name__ == "__main__":
    data_dir = "Z:\\Dor_Gabay\\ThesisProject\\data\\exp_collected_data\\15.9.22\\plus0.3mm_force\\S5280006\\"

    # calibration_dir1 = "Z:\\Dor_Gabay\\ThesisProject\\data\\calibration\\post_slicing\\calibration1\\"
    # directories_1 = [os.path.join(calibration_dir1, o) for o in os.listdir(calibration_dir1)
    #                  if os.path.isdir(os.path.join(calibration_dir1, o)) and "_sliced" in os.path.basename(o)]
    # weights1 = list(np.array([0.10606, 0.14144, 0.16995, 0.19042, 0.16056, 0.15082]) - 0.10506)
    #
    # calibration_dir2 = "Z:\\Dor_Gabay\\ThesisProject\\data\\calibration\\post_slicing\\calibration2\\"
    # calibration_video_dir2 = "Z:\\Dor_Gabay\\ThesisProject\\data\\videos\\calibration2\\"
    # directories_2 = [os.path.join(calibration_dir2, o) for o in os.listdir(calibration_dir2)
    #                  if os.path.isdir(os.path.join(calibration_dir2, o)) and "_sliced" in os.path.basename(o) and "6" not in os.path.basename(o)]# and "9" not in os.path.basename(o)]
    # # weights2 = list(np.array([0.10582,0.13206,0.15650,0.18405,0.21030,0.46612])-0.10582)
    # weights2 = list(np.array([0.10582,0.13206,0.15650,0.18405,0.46612])-0.10582)
    # video_paths2 = [os.path.join(calibration_video_dir2, o) for o in os.listdir(calibration_video_dir2)
    #                  if "MP4" in os.path.basename(o) and "_sliced" in os.path.basename(o) and "6" not in os.path.basename(o)]
    #
    # calibration_dir3 = "Z:\\Dor_Gabay\\ThesisProject\\data\\calibration\\post_slicing\\calibration3\\"
    # calibration_video_dir3 = "Z:\\Dor_Gabay\\ThesisProject\\data\\videos\\calibration3\\"
    # directories_3 = [os.path.join(calibration_dir3, o) for o in os.listdir(calibration_dir3)
    #                  if os.path.isdir(os.path.join(calibration_dir3, o)) and "_sliced" in os.path.basename(o) and "S5430006_sliced" not in os.path.basename(o)]# not in os.path.basename(o) and "S5430003_sliced" not in os.path.basename(o)]
    # # weights3 = list(np.array([0.41785, 0.36143, 0.3008, 0.2389, 0.18053, 0.15615, 0.11511, 0.12561, 0.10610]) - 0.10610)
    # weights3 = list(np.array([0.41785, 0.36143, 0.3008, 0.2389, 0.18053, 0.11511, 0.12561, 0.10610]) - 0.10610)
    # # weights3 = list(np.array([0.41785, 0.2389, 0.18053, 0.15615, 0.11511, 0.12561, 0.10610]) - 0.10610)
    # video_paths3 = [os.path.join(calibration_video_dir3, o) for o in os.listdir(calibration_video_dir3)
    #                if "MP4" in os.path.basename(o) and "_sliced" in os.path.basename(o) and "S5430006_sliced" not in os.path.basename(o)]## and "S5430002_sliced" not in os.path.basename(o) and "S5430003_sliced" not in os.path.basename(o)]

    calibration_dir4 = "Z:\\Dor_Gabay\\ThesisProject\\data\\calibration\\post_slicing\\calibration_perfect2\\sliced_videos\\"
    calibration_video_dir4 = "Z:\\Dor_Gabay\\ThesisProject\\data\\videos\\calibration_perfect2\\sliced_videos\\"
    directories_4 = [os.path.join(calibration_dir4, o) for o in os.listdir(calibration_dir4)
                     if os.path.isdir(os.path.join(calibration_dir4, o)) and "plots" not in os.path.basename(o)]# and "S5430006_sliced" not in os.path.basename(o)]  # not in os.path.basename(o) and "S5430003_sliced" not in os.path.basename(o)]
    weights4 = list(np.array([0.11656, 0.16275, 0.21764, 0.28268, 0.10657]) - 0.10657)
    # weights3 = list(np.array([0.41785, 0.2389, 0.18053, 0.15615, 0.11511, 0.12561, 0.10610]) - 0.10610)
    video_paths4 = [os.path.join(calibration_video_dir4, o) for o in os.listdir(calibration_video_dir4)
                    if "MP4" in os.path.basename(o) and "plots" not in os.path.basename(o)]#]# and "_sliced" in os.path.basename(o)]# and "S5430006_sliced" not in os.path.basename(o)]

    print(directories_4)

    Calibration(directories=directories_4, weights=weights4, output_path=calibration_dir4, video_paths=video_paths4)
    print("-"*60)
    print("Calibration model has been created.")
    print("-"*60)

