from picamera2 import Picamera2
import libcamera
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import threading
from picarx import Picarx
import datetime

POWER = 5

def setup_camera():
    picam2 = Picamera2()
    config = picam2.create_still_configuration()
    picam2.configure(config)
    picam2.start()

    return picam2

def setup_robot():
    return Picarx()

def plot_im(im1, f):
    plt.figure(figsize=(4, 4))
    plt.imshow(im1, interpolation='nearest')
    plt.axis('off')  # Hide axes
    plt.savefig("captured_images/file_name"+f, dpi=300, bbox_inches='tight', pad_inches=0)

def 

def main():
    camera = setup_camera()
    robot = setup_robot()
    time.sleep(2)

    try:
        factor_flow_to_degrees = calibrate_camera(camera, robot)
        time.sleep(1)

        factor_seconds_to_flow = calibrate_servos(camera, robot)
        time.sleep(1)

        # Compute approx factor
        factor_seconds_to_degrees = factor_seconds_to_flow * factor_flow_to_degrees

        time_for_turn = 2 * np.pi / factor_seconds_to_degrees
        print("Time taken for a full turn:", time_for_turn, " seconds")

        origin_image = cv2.cvtColor(camera.capture_array(), cv2.COLOR_RGB2GRAY)  
        time.sleep(0.1)

        turn_clockwise(robot)
        time.sleep(time_for_turn)
        robot.forward(0)
        time.sleep(0.1)
        final_image = cv2.cvtColor(camera.capture_array(), cv2.COLOR_RGB2GRAY)  
        time.sleep(0.1)

        # Compute optical flow between the two images
        optical_flow_full_turn = compute_camera_pan(origin_image, final_image)
        print("Estimated offset", optical_flow_full_turn * factor_flow_to_degrees)


    finally:
        robot.forward(0)


if __name__ == "__main__":
    main()