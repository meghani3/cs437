import cv2
import numpy as np

def compute_camera_pan(image1_path, image2_path):
    # Load images
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

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

# Example usage
image1_path = "0.jpg"
image2_path = "5.jpg"
pan_amount = compute_camera_pan(image1_path, image2_path)

if pan_amount is not None:
    print(f"Estimated camera pan (X direction): {pan_amount} pixels")
else:
    print("Could not compute pan.")
