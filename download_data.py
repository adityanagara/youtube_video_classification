#!/use/bin/env python

import os
import numpy as np
import requests
import pytube
import cv2
import scipy.misc
from PIL import Image
import glob

CAP_PROP_POS_MSEC = 0
def get_all_categories():
    r = requests.get('https://research.google.com/youtube8m/explore.html')
    print(r.json)
    print(r.content)

def get_videos_by_category():
    """Automate this later
    """

    indoor_videos = {"room": ["https://www.youtube.com/watch?v=N9a9abjsqbE",
                              "https://www.youtube.com/watch?v=N9a9abjsqbE"],
                     "BedRoom": ["https://www.youtube.com/watch?v=hFmXTgqJ98Q"],
                     "Restaurant": ["https://www.youtube.com/watch?v=pGCTn_UdTxI"]}

    outdoor_videos = {"oceans": ["https://www.youtube.com/watch?v=9ntinpHGlec",
                              "https://www.youtube.com/watch?v=IYePs7Q-se8"],
                   "mountains": ["https://www.youtube.com/watch?v=o1-TOwCaKBQ",
                                 "https://www.youtube.com/watch?v=2SaOEUZQ2G8"],
                   "building": ["https://www.youtube.com/watch?v=TDOU34ThXeY"],
                   "city": ["https://www.youtube.com/watch?v=UwlA4ZUkc-g"]}

    return {0: indoor_videos, 1: outdoor_videos}


def download_video():
    all_videos = get_videos_by_category()
    for category in all_videos:
        ctr = 0
        for type in all_videos[category]:
            for v_idx, video in enumerate(all_videos[category][type]):
                print("Downloading video {}".format(video))
                yt = pytube.YouTube(video)
                out_file = ".mp4".format(video)
                stream = yt.streams.filter(file_extension="mp4").first()
                stream.download("videos/", filename='video_{}_{}'.format(category, ctr))

                print(os.path.join("videos", "{}.mp4".format(yt.title)))
                ctr += 1
                # os.rename(os.path.join("videos", "{}.mp4".format(yt.title)), 'videos/video_{}_{}.mp4'.format(category, ctr))

def download_all_videos():
    download_video()


def get_all_frames(filename, sample_period=1.0):
    """A function that gets all the frames from a video every 1 second

    Args:
        filename (str) : The file name of the video
    """
    video_capture = cv2.VideoCapture()
    video_capture.open(filename)
    video_capture.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    max_seconds = video_capture.get(cv2.CAP_PROP_POS_MSEC)
    video_capture.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
    num_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    print("Total duration of video {}".format(max_seconds))
    print("Total number of frames is {}".format(num_frames))
    fps = video_capture.get(cv2.CAP_PROP_FPS)  # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    print("The FPS is {}".format(fps))
    num_seconds = num_frames / fps
    print("Duration of video {} in seconds".format(num_seconds))
    print("Duration of video {} in minutes".format(num_seconds / 60.0))
    second_number = 0
    video_number = filename.split(".")[0].split("_")[2]
    class_label = filename.split(".")[0].split("_")[1]
    for idx in range(0, int(num_frames)):
        frame_number = video_capture.get(1)
        is_frame, frame = video_capture.read()
        if not is_frame:
            break
        frame = np.array(frame)
        if idx % int(fps * sample_period) == 0:
            new_im = Image.fromarray(frame)
            new_im.save("images/Image_{}_{}_{}.png".format(class_label,
                                                     video_number,
                                                     second_number))
            second_number += 1
            yield frame

def get_all_images():
    all_videos = glob.glob('videos/*.mp4')
    print("All videos")
    print(all_videos)
    for video in all_videos:
        for i, frame in enumerate(get_all_frames(video, sample_period=30)):
            if i % 30 == 0:
                print("Shape of frame {}".format(frame.shape))


if __name__ == "__main__":
    get_all_images()
    # download_all_videos()
    # for i, frame in enumerate(get_all_frames("TBNRfrags Final Goodbye Apartment Tour.mp4")):
    #     if i % 30 == 0:
    #         print("Shape of frame {}".format(frame.shape))
