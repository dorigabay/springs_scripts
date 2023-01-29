import cv2
import numpy as np
import pickle
import os

def import_data(data_dir):
    data = []
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            data.append(np.loadtxt(os.path.join(data_dir, file), delimiter=','))
    return data

def transform_data(data):
    data = np.array(data)
    data = np.swapaxes(data, 0, 1)
    return data

def save_data_as_video(output_dir, data, vid_name):
    # create output directory if not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Creating video...")
    height, width = data[0].shape[:2]
    video = cv2.VideoWriter(os.path.join(output_dir, vid_name+'.MP4'), cv2.VideoWriter_fourcc(*'mp4v'), 50, (width, height))
    for image in data:
        video.write(image)
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    data_dir = "Z:/Dor_Gabay/ThesisProject/data/test6/pickle_files/"
    output_dir = "Z:/Dor_Gabay/ThesisProject/results/data_as_videos/"
    data = import_data(data_dir)
    data = transform_data(data)
    save_data_as_video(output_dir, data, "test")