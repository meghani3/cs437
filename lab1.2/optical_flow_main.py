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

def compute_camera_pan(img1, img2):

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches based on distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Compute affine transformation matrix
    matrix, _ = cv2.estimateAffinePartial2D(pts1, pts2)

    if matrix is not None:
        # Extract translation in the X-direction
        pan_x = matrix[0, 2]
        return pan_x
    else:
        return None

def setup_robot():
    return Picarx()

def plot_im(im1, f):
    plt.figure(figsize=(4, 4))
    plt.imshow(im1, interpolation='nearest')
    plt.axis('off')  # Hide axes
    plt.savefig("captured_images/file_name"+f, dpi=300, bbox_inches='tight', pad_inches=0)

def calibrate_camera(camera, robot):

    print("Calibrating camera mount")

    images = {}
    ANGLES = list(range(-25, 26, 5))

    for angle in ANGLES:
        robot.set_cam_pan_angle(angle)
        time.sleep(0.2)
        images[angle] = camera.capture_array() # cv2.cvtColor(camera.capture_array(), cv2.COLOR_RGB2GRAY)  
        time.sleep(0.2)

    pan_amounts = []

    for im1_idx, im2_idx in zip(ANGLES[:-1], ANGLES[1:]):
        im1 = images[im1_idx]
        im2 = images[im2_idx]

        pan_amounts.append(compute_camera_pan(im1, im2))
    
    avg_pan = sum(pan_amounts) / len(pan_amounts)
    factor_flow_to_degrees = 5 / avg_pan    # Interval of 5 degrees

    print("Average pan factor_flow_to_degrees: ", factor_flow_to_degrees)

    for pan_amount in pan_amounts:
        pan_degrees = pan_amount * factor_flow_to_degrees
        print("Computed pan in degrees: ", pan_degrees)
        if (pan_degrees < 2 or pan_degrees > 8):
            for angle in ANGLES:
                plot_im(images[angle], str(angle))
            raise Exception("Cannot use optical flow to compute rotation; please check lighting conditions/obstruction")

    return factor_flow_to_degrees

def turn_clockwise(robot):
    robot.set_motor_speed(1, POWER)
    robot.set_motor_speed(2, POWER)

def calibrate_servos(camera, robot):

    print("Calibrating clockwise rotation")

    robot.set_cam_pan_angle(0)
    origin_image = cv2.cvtColor(camera.capture_array(), cv2.COLOR_RGB2GRAY)  

    factor = 0
    number_of_samples = 0
    factor_sum_across_samples = 0

    time.sleep(1)

    for _ in range(5):
        prev_image = camera.capture_array()
        prev_time = datetime.datetime.now()
        
        # Cannot capture images while robot is in motion
        # Too much blur for optical flow to work correctly
        turn_clockwise(robot)
        time.sleep(0.1)
        robot.forward(0)
        time.sleep(0.1)

        current_time = datetime.datetime.now()
        current_image = camera.capture_array()

        elapsed_time = current_time - prev_time
        camera_pan = compute_camera_pan(prev_image, current_image)

        factor_sum_across_samples += camera_pan / elapsed_time.total_seconds()
        number_of_samples += 1

        print("Current optical flow:", camera_pan / elapsed_time.total_seconds())
        plot_im(prev_image, prev_time.strftime("%Y-%m-%d %H:%M:%S:%f"))
        plot_im(current_image, current_time.strftime("%Y-%m-%d %H:%M:%S:%f"))

        turn_clockwise(robot)
        time.sleep(1)
        robot.forward(0)
        time.sleep(1)
    
    robot.forward(0)
    factor_seconds_to_flow = factor_sum_across_samples / number_of_samples
    print("Average clockwise factor:", factor_seconds_to_flow, "across #", number_of_samples, "samples")

    return factor_seconds_to_flow


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