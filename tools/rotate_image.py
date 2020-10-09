import subprocess
import cv2

def get_rotation_with_ffprobe(vid_path):
  # requires ffprobe
  check_rotate_cmd = "ffprobe -loglevel error -select_streams v:0 -show_entries side_data=rotation -of default=nw=1:nk=1 {}".format(vid_path)
  check_rotate_result = subprocess.check_output(check_rotate_cmd, shell=True).decode("UTF-8")

  rotation = 0
  if check_rotate_result:
      rotation = int(check_rotate_result)

  return rotation

def rotate_frame(frame, rotation):
  # map the rotation value
  if rotation == -90:
      frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
  if rotation == 90:
      frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

  return frame