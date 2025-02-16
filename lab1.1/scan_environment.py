from picarx import Picarx
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import ndimage

MAP_SIZE = 200
grid = np.array([[0 for i in range(MAP_SIZE)] for j in range(MAP_SIZE)], dtype=int)

pos = np.array([MAP_SIZE//2, 0], dtype=int)

def spherical_coordinates(r, theta):
    print(r, theta, end="\t")
    theta = np.deg2rad(theta)
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    return np.array([x, y], dtype=int)

def print_map(file_name):
    plt.figure(figsize=(4, 4))
    plt.imshow(grid, cmap='bone', interpolation='nearest')
    plt.axis('off')  # Hide axes
    plt.savefig(file_name, dpi=300, bbox_inches='tight', pad_inches=0)


px = Picarx()

ANGLE = 80
time.sleep(3)

camera_pan_angle = 0 

def gently_pan(final_angle:int):
    global camera_pan_angle
    for i in np.linspace(camera_pan_angle, final_angle, 100):
        px.set_cam_pan_angle(i)
        camera_pan_angle = i
        time.sleep(0.01)
    time.sleep(0.01)

gently_pan(-ANGLE)
for angle in np.linspace(-ANGLE, ANGLE, 2*ANGLE):
    px.set_cam_pan_angle(angle)
    time.sleep(0.1)
    camera_pan_angle = angle
    
    distance = px.ultrasonic.read() 
    distance += px.ultrasonic.read() 
    distance += px.ultrasonic.read() 

    distance /= 3

    if distance > 0 and distance < 100:
        coords = spherical_coordinates(distance, angle)
        loc = pos+coords

        print(loc)

        loc[0] = max(0, loc[0])
        loc[0] = min(MAP_SIZE-1, loc[0])

        loc[1] = max(0, loc[1])
        loc[1] = min(MAP_SIZE-1, loc[1])

        grid[*loc] = 2
    else:
        print("oopsie")

    time.sleep(0.01)

gently_pan(0)

print_map("char_array_image_before.png")
weights = [[0.5, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 0.5]]
grid = ndimage.convolve(grid, weights, mode='reflect')
grid = np.array(grid > 2, dtype=int)


for i in range(pos[0]%30, MAP_SIZE, 30):
    for j in range(pos[1]%30, MAP_SIZE, 30):
        snapshot = grid[i-2:i+2, j-2:j+2]
        if snapshot.sum() > 3:
            grid[i, j] = 3
        else:
            grid[i, j] = 2


grid[*pos] = 3

print_map("char_array_image.png")