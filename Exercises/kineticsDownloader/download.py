import os
import cv2
import numpy as np
from ffpyplayer.player import MediaPlayer
import pafy

def download_video(video_id, download_path, video_format="mp4", log_file=None):
  """
  Download video from YouTube.
  :param video_id:        YouTube ID of the video.
  :param download_path:   Where to save the video.
  :param video_format:    Format to download.
  :param log_file:        Path to a log file for youtube-dl.
  :return:                Tuple: path to the downloaded video and a bool indicating success.
  """

  if log_file is not None:
    stderr = open(log_file, "a")

  url = 'https://www.youtube.com/watch?v=--6bJUbfpnQ'
  video = pafy.new(url)
  best = video.getbest(preftype=video_format)
  videocap = cv2.VideoCapture(best.url)
  player = MediaPlayer(best.url)

  while True:
    grabbed, frame = videocap.read()
    audio_frame, val = player.get_frame()
    if not grabbed:
      print("End of video")
      break
    if cv2.waitKey(28) & 0xFF == ord("q"):
      break
    cv2.imshow("Video", frame)
    if val != 'eof' and audio_frame is not None:
      # audio
      img, t = audio_frame

  if log_file is not None:
    stderr.close()

  videocap.release()
  cv2.destroyAllWindows()

  return True

download_video('','')