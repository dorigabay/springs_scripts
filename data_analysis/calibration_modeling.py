import os
import pickle
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# local imports:
from data_preparation import DataPreparation


class CalibrationModeling(DataPreparation):
    def __init__(self, video_path, output_path, weights=None, mm_per_pixel=0.1):
        self.video_path = video_path
        self.calibration_name = os.path.basename(os.path.normpath(self.video_path))
        self.output_path = os.path.join(output_path, self.calibration_name)
        self.figures_output_path = os.path.join(output_path, "calibration_figures", self.calibration_name)
        self.load_weights(weights)
        self.mm_per_pixel = mm_per_pixel
        self.num_of_springs = 1
        data_paths = np.array([os.path.normpath(root) for root, dirs, files in os.walk(self.output_path) if not dirs])
        # self.weights = self.weights[list(videos_idx)] if videos_idx is not None else self.weights
        # data_paths = data_paths[list(videos_idx)] if videos_idx is not None else data_paths
        super().__init__(video_path, data_paths, n_springs=1, calib_mode=True)
        self.zero_weight_dir = data_paths[np.where(np.array(self.weights) == 0)[0][0]]
        self.concat_calib_data()
        self.plot()
        self.create_calibration_model()

    def load_weights(self, weights=None):
        if weights is None:
            self.weights = pickle.load(open(os.path.join(self.output_path, self.calibration_name, "weights.pickle"), 'rb'))
            raise ValueError("No weights were given and no weights were found in the output path.")
        else:
            self.weights = np.array(weights)
            os.makedirs(self.output_path, exist_ok=True)
            pickle.dump(self.weights, open(os.path.join(self.output_path, "weights.pickle"), 'wb'))

    def concat_calib_data(self):
        weights_repeated = np.concatenate([np.repeat(weight, self.video_n_frames[count]) for count, weight in enumerate(self.weights)], axis=0)
        magnitude, direction = self.calc_calib_force(self.fixed_end_angle_to_wall, weights_repeated)
        self.calib_data = np.concatenate((self.pulling_angle, self.spring_length, direction.reshape(-1, 1), magnitude.reshape(-1, 1)), axis=1)
        self.calib_data[self.calib_data[:, 2] < 0, 0] *= -1
        self.calib_data[self.calib_data[:, 2] < 0, 2] *= -1
        self.calib_data[self.calib_data[:, 0] < 0, :] = np.nan
        # self.calib_data = self.remove_bin_outliers(self.calib_data, self.video_n_frames, bins=200, percentile=10, max_angle_value=np.pi*(13/24))
        self.calib_data = self.remove_bin_outliers(self.calib_data, self.video_n_frames, bins=200, percentile=10, max_angle_value=np.pi*(12/24))
        self.calib_data = self.calib_data[~np.isnan(self.calib_data).any(axis=1)]
        self.calib_data = np.concatenate((self.calib_data, self.calib_data), axis=0)
        self.calib_data[0:self.calib_data.shape[0]//2, [0, 2]] *= -1

    def remove_bin_outliers(self, data, frames_per_weight, bins=200, percentile=5, max_angle_value=np.pi*(13/24)):
        # max_angle_value = np.nanpercentile(data[~(data[:, 3] == 0), 2], 90)
        angle_bins = np.linspace(0, max_angle_value, bins)
        data[data[:, 2] > max_angle_value, :] = np.nan
        for count, n_frames in enumerate(frames_per_weight):
            s, e = frames_per_weight[:count].sum(), frames_per_weight[:count + 1].sum()
            for i in range(len(angle_bins) - 1):
                idx = (data[s:e, 2] > angle_bins[i]) * (data[s:e, 2] < angle_bins[i + 1])
                pa_upper, pa_lower = np.nanpercentile(data[s:e, 0][idx], 100-percentile), np.nanpercentile(data[s:e, 0][idx], percentile)
                l_upper, l_lower = np.nanpercentile(data[s:e, 1][idx], 100-percentile), np.nanpercentile(data[s:e, 1][idx], percentile)
                angle_bin_outliers = (data[s:e, 0] > pa_upper) * idx + (data[s:e, 0] < pa_lower) * idx + (data[s:e, 1] > l_upper) * idx + (data[s:e, 1] < l_lower) * idx
                data[s:e][angle_bin_outliers, :] = np.nan
        return data

    def plot(self, fig_size=(82, 70), font_size=6):
        magnitudes = np.unique(self.calib_data[:, 3])
        magnitudes_coordinates = []
        for mag in magnitudes:
            data = self.calib_data[self.calib_data[:, 3] == mag]
            magnitudes_coordinates.append(np.array([np.sin(data[:, 0]), np.cos(data[:, 0])]).transpose() * (data[:, 1:2]))
        fig, ax = plt.subplots()
        inch_per_mm = 0.0393701
        fig.set_size_inches(fig_size[0]*inch_per_mm, fig_size[1]*inch_per_mm)
        colors = sns.color_palette("magma", len(self.weights))
        mm_convert_factor = (self.max_extenstion * self.mm_per_pixel) / np.nanmax(magnitudes_coordinates[-1][:, 1])
        for count, (mag, coordinates) in enumerate(zip(magnitudes, magnitudes_coordinates)):
            coordinates *= mm_convert_factor
            df = pd.DataFrame({"x": coordinates[:, 0], "y": coordinates[:, 1]})
            sns.scatterplot(data=df, x="x", y="y", color=colors[count], label=np.round(mag, 3), ax=ax, s=10)
        legend = plt.legend(title="Force (mN)", fontsize=font_size)
        legend.get_title().set_fontsize(str(font_size))
        plt.setp(ax.get_xticklabels(), fontsize=font_size)
        plt.setp(ax.get_yticklabels(), fontsize=font_size)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        # plt.show()
        # print("Presenting data. Close the plot to continue.")
        # self.figures_output_path = "Z:\\Dor_Gabay\\ThesisProject\\results\\summer_2023\\plus_0.0\\calibration_figures\\"
        os.makedirs(self.figures_output_path, exist_ok=True)
        fig.savefig(os.path.join(self.figures_output_path, "calibration_data.svg"), format="svg")

    def create_calibration_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.calib_data[:, 0:2], self.calib_data[:, 2:4], test_size=0.2)
        print("X_train shape: ", X_train.shape)
        self.model = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
        self.model.fit(X_train, y_train)
        pickle.dump(self.model, open(os.path.join(self.output_path, "calibration_model.pkl"), 'wb'))
        print("Model saved to: ", os.path.join(self.output_path,"calibration_model.pkl"))
        y_prediction = self.model.predict(X_test)
        print("r squared: ", self.model.score(X_test, y_test))
        print("mean squared error: ", mean_squared_error(y_test, y_prediction, squared=False))
        with open(os.path.join(self.output_path, "calibration_model_statistics.txt"), "w") as f:
            f.write("r squared: " + str(self.model.score(X_test, y_test)) + "\n")
            f.write("mean squared error: " + str(mean_squared_error(y_test, y_prediction, squared=False)) + "\n")
        np.save(os.path.join(self.figures_output_path, "y_test.npy"), y_test)
        np.save(os.path.join(self.figures_output_path, "y_test_prediction.npy"), y_prediction)

    def calc_calib_force(self, angle_to_nest, mass1):
        mass1 *= 1e-3  # convert to kg
        gravitational_constant = 6.67430e-11  # Gravitational constant (in m^3 kg^-1 s^-2)
        mass2 = 5.972e24  # Mass of the Earth (in kilograms)
        distance = 6371000  # Approximate radius of the Earth (in meters)
        force = (gravitational_constant * mass1 * mass2) / (distance ** 2)
        force_millinewton = force * 1e3  # convert to mN
        magnitude = force_millinewton
        direction = angle_to_nest.flatten() - np.pi
        return magnitude, direction

