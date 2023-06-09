import os
import numpy as np
import cv2
from datetime import timedelta
from moviepy.editor import VideoFileClip

# Inspired by https://www.thepythoncode.com/article/extract-frames-from-videos-in-python.

def format_timedelta(td):
    """Utility function to format timedelta objects in a cool way (e.g 00:00:20.05) 
    omitting microseconds and retaining milliseconds"""
    result = str(td)
    try:
        result, _ = result.split(".")
    except ValueError:
        return (result + ".00").replace(":", "-")
    return f"{result}".replace(":", "-")


def get_saving_frames_durations(cap, saving_fps):
    """A function that returns the list of durations where to save the frames"""
    s = []
    # get the clip duration by dividing number of frames by the number of frames per second
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    # use np.arange() to make floating-point steps
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s

SAVING_FRAMES_PER_SECOND = 1

def generate_frames(video_file):
    filename, _ = os.path.splitext(video_file)
    filename = './data/' + os.path.basename(filename) + "-opencv"
    # make a folder by the name of the video file
    print(filename)
    if not os.path.isdir(filename):
        os.mkdir(filename)
    else:
        print("Frames are already generated. Moving to next file.")
        return
    # read the video file    
    cap = cv2.VideoCapture(video_file)
    # get the FPS of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
    saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)
    # get the list of duration spots to save
    saving_frames_durations = get_saving_frames_durations(cap, saving_frames_per_second)
    # start the loop
    count = 0
    while True:
        is_read, frame = cap.read()
        if not is_read:
            # break out of the loop if there are no frames to read
            break
        # get the duration by dividing the frame count by the FPS
        frame_duration = count / fps
        try:
            # get the earliest duration to save
            closest_duration = saving_frames_durations[0]
        except IndexError:
            # the list is empty, all duration frames were saved
            break
        if frame_duration >= closest_duration:
            # if closest duration is less than or equals the frame duration, 
            # then save the frame
            frame_duration_formatted = format_timedelta(timedelta(seconds=frame_duration))
            cv2.imwrite(os.path.join(filename, f"frame{frame_duration_formatted}.jpg"), frame) 
            # drop the duration spot from the list, since this duration spot is already saved
            try:
                saving_frames_durations.pop(0)
            except IndexError:
                pass
        # increment the frame count
        count += 1

AUDIO_CLIP_HALF_LENGTH = 2

def generate_audio(video_file):
    filename, _ = os.path.splitext(video_file)
    filename = './data/' + os.path.basename(filename) + "-audio"
    # make a folder by the name of the video file
    print(filename)
    if not os.path.isdir(filename):
        os.mkdir(filename)
    else:
        print("Audio file have been generated. Moving to next file.")
        return
    # read the video file    
    video = VideoFileClip(video_file)
    duration = int(video.duration)

    for time_stamp in range(AUDIO_CLIP_HALF_LENGTH, duration - AUDIO_CLIP_HALF_LENGTH):
        # get intervals
        start = time_stamp - AUDIO_CLIP_HALF_LENGTH
        end = time_stamp + AUDIO_CLIP_HALF_LENGTH
        # clip the audio
        clip = video.subclip(start, end)
        time_stamp_formatted = format_timedelta(timedelta(seconds=time_stamp))
        name = os.path.join(filename, f"audio{time_stamp_formatted}.wav")
        # store the audio file
        clip.audio.write_audiofile(name)

import sys
dir_list = os.listdir('./videos')
for video_file in dir_list:
    if not video_file.endswith('.mp4'):
        continue
    print(f"Generating frames from {video_file}.")
    generate_frames('./videos/' + video_file)
    generate_audio('./videos/' + video_file)