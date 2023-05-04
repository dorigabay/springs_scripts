from moviepy.video.io.VideoFileClip import VideoFileClip

VIDEO_PATH = "Z:/Dor_Gabay/videos/10.9/plus0/S5200007.MP4"
OUTPUT_PATH = "Z:/Dor_Gabay/videos/for_test/S5200007.MP4"
t1 = 0
t2 = 20

with VideoFileClip(VIDEO_PATH) as video:
    new = video.subclip(t1, t2)
    new.write_videofile(OUTPUT_PATH, audio_codec='aac')
