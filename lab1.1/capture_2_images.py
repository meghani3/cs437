"""
# from vilib import Vilib
# import os
# from time import sleep, time, strftime, localtime
# from picarx import Picarx

# px = Picarx()
# Vilib.camera_start(vflip=False,hflip=False)
# Vilib.display(local=False,web=True)

# sleep(1)

# path = os.getcwd()
# Vilib.take_photo("0", path)
# print('photo saved')

# sleep(0.5)

# angle = 5
# px.set_cam_pan_angle(angle)
# sleep(0.5)

# Vilib.take_photo(str(angle), path)
# print('photo saved')
"""


# Used the official docs to figure out the camera module
# https://github.com/sunfounder/vilib/blob/picamera2/vilib/vilib.py#L19
# https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf
# https://randomnerdtutorials.com/raspberry-pi-picamera2-python/

from picamera2 import Picamera2
import libcamera
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

picam2 = Picamera2()
config = picam2.create_still_configuration()
picam2.configure(config)
picam2.start()

time.sleep(2)

im = picam2.capture_array()

# plt.figure(figsize=(4, 4))
# plt.imshow(im, interpolation='nearest')
# plt.axis('off')  # Hide axes
# plt.savefig("file_name", dpi=300, bbox_inches='tight', pad_inches=0)
