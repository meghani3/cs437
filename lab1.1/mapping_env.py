from picarx import Picarx
import numpy as np
import time
import datetime
import os
import threading
import sys
import matplotlib.pyplot as plt
from scipy import ndimage

# print(sys.maxsize)


# np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(threshold=20)

POWER = 5

MAP_SIZE = 1000
grid = np.array([[0 for i in range(MAP_SIZE)] for j in range(MAP_SIZE)], dtype=int)


# np.empty(shape=(MAP_SIZE, MAP_SIZE), dtype=str)
pos = np.array([MAP_SIZE//2, MAP_SIZE//2], dtype=int)

grid[*pos] = 1

px = Picarx()

def turn_clockwise():
    px.set_motor_speed(1, POWER)
    px.set_motor_speed(2, POWER)

def time_to_theta(start_time):
    elapsed_time = datetime.datetime.now() - start_time
    return 2 * np.pi * (elapsed_time.total_seconds()) / 9  # time for one rotation = 9sec

def spherical_coordinates(r, theta):
    print(r, theta)
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    return np.array([x, y], dtype=int)

def print_map(file_name):
    # color_map = {' ': 0, 'x': 1, 'o': 2}  # Assign numerical values

    # # Convert char array to numerical values
    # num_array = np.vectorize(color_map.get)(grid)

    # Plot and save the image
    plt.figure(figsize=(4, 4))
    plt.imshow(grid, cmap='plasma', interpolation='nearest')
    plt.axis('off')  # Hide axes
    plt.savefig(file_name, dpi=300, bbox_inches='tight', pad_inches=0)

start_time = datetime.datetime.now()
try:
    
    # t = threading.Thread(target=log_map, daemon=True)
    # t.start()

    angle = 0

    while (angle < 2 * np.pi): 
        turn_clockwise()
        distance = px.ultrasonic.read() 
        if (distance > 10):
            angle = time_to_theta(start_time)
            coords = spherical_coordinates(distance, angle)
            print (pos + coords)
            grid[*(pos + coords)] = 2
        time.sleep(.01)
finally:
    end_time = datetime.datetime.now()
    # np.savetxt('test.txt', grid)
    # print(grid)
    print("Total time, ", (end_time - start_time))
    px.forward(0)

    print_map("char_array_image_before.png")
    # Apply gaussian image to clean up

    weights = [[0.5, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 0.5]]
    
    grid = ndimage.convolve(grid, weights, mode='reflect')
    # grid = grid[grid > 2]

    # print

    grid = grid > 2

    print_map("char_array_image.png")