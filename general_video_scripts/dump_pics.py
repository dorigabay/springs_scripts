import cv2
import os

def save_first_frame(video_path, output_dir):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not video.isOpened():
        raise Exception("Could not open video file")

    # Read the first frame
    success, frame = video.read()

    # Check if a frame was read successfully
    if not success:
        raise Exception("Could not read frame from video")

    # Save the frame to a file
    output_path = os.path.join(output_dir, "first_frame.jpg")
    cv2.imwrite(output_path, frame)


if __name__ == '__main__':
    # video_path = "Z:\\Dor_Gabay\\ThesisProject\\data\\videos\\10.9\\plus0_force\\S5200007.MP4"
    video_path = "Z:\\Tabea\\Videos\\01_09_2022_(CactusGarden Wall)\\S5150004_SSpecialT.MP4"
    output_dir = "Z:\\Dor_Gabay\\DeepLearningProject\\DL4CV_FinalProject\\example_pics\\"
    save_first_frame(video_path, output_dir)