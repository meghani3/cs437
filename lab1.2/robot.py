from picarx import Picarx
import time
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from path_finding.chatgpt import a_star

POWER = 30
TURN_STEPS = 50

STEERING_ANGLE = 30
PAN_ANGLE = 80

MAP_SIZE = 200
IMAGE_ROBOT_POS = np.array([MAP_SIZE//2, 0], dtype=int) 

STEPS_BETWEEN_RESCANS = 3

class Robot:
    
    def __init__(self):
        self.px  = Picarx()    
        self.camera_pan_angle = 0 
        
        time.sleep(5)

    def gently_turn(self, final_angle:int):
        for i in np.linspace(self.px.dir_current_angle, final_angle, TURN_STEPS):
            self.px.set_dir_servo_angle(i)
            time.sleep(0.01)
        time.sleep(0.1)

    def gently_move_forward(self, time_to_run:int):
        for power in np.linspace(0, POWER, TURN_STEPS):
            self.px.forward(power)
            time.sleep(time_to_run/2/TURN_STEPS)

        for power in np.linspace(POWER, 0, TURN_STEPS):
            self.px.forward(power)
            time.sleep(time_to_run/2/TURN_STEPS)

    def gently_move_backward(self, time_to_run:int):
        for power in np.linspace(0, POWER, TURN_STEPS):
            self.px.backward(power)
            time.sleep(time_to_run/2/TURN_STEPS)

        for power in np.linspace(POWER, 0, TURN_STEPS):
            self.px.backward(power)
            time.sleep(time_to_run/2/TURN_STEPS)

    def turn_left(self):
        TIME_TO_TURN_90_DEG = .915
        self.gently_turn(-STEERING_ANGLE)                
        self.gently_move_forward(TIME_TO_TURN_90_DEG)
        self.gently_turn(STEERING_ANGLE)
        self.gently_move_backward(TIME_TO_TURN_90_DEG)
    
    def turn_right(self):
        TIME_TO_TURN_90_DEG = 1.06
        self.gently_turn(STEERING_ANGLE)                
        self.gently_move_forward(TIME_TO_TURN_90_DEG)
        self.gently_turn(-STEERING_ANGLE)
        self.gently_move_backward(TIME_TO_TURN_90_DEG)

    def gently_pan(self, final_angle:int):
        for i in np.linspace(self.camera_pan_angle, final_angle, 100):
            self.px.set_cam_pan_angle(i)
            self.camera_pan_angle = i
            time.sleep(0.01)
        time.sleep(0.01)

    def spherical_coordinates(self, r, theta):
        print(r, theta, end="\t")
        theta = np.deg2rad(theta)
        x = r * np.sin(theta)
        y = r * np.cos(theta)
        return np.array([x, y], dtype=int)
    
    def raw_scan_environment(self):
        image_grid = np.array([[0 for i in range(MAP_SIZE)] for j in range(MAP_SIZE)], dtype=int)

        self.gently_pan(-PAN_ANGLE)
        for angle in np.linspace(-PAN_ANGLE, PAN_ANGLE, 2*PAN_ANGLE):
            self.px.set_cam_pan_angle(angle)
            time.sleep(0.1)
            self.camera_pan_angle = angle
            
            # Average out 3 reads
            distance = self.px.ultrasonic.read() 
            distance += self.px.ultrasonic.read() 
            distance += self.px.ultrasonic.read() 

            distance /= 3

            if distance > 0 and distance < 100:
                coords = self.spherical_coordinates(distance, angle)
                loc = IMAGE_ROBOT_POS + coords

                print(loc)

                loc[0] = max(0, loc[0])
                loc[0] = min(MAP_SIZE-1, loc[0])

                loc[1] = max(0, loc[1])
                loc[1] = min(MAP_SIZE-1, loc[1])

                image_grid[*loc] = 2

            else:
                print("nothing in line of sight")

            time.sleep(0.01)
        self.gently_pan(0)
        return image_grid

    def print_map(self, file_name, image_grid):
        plt.figure(figsize=(4, 4))
        plt.imshow(image_grid, cmap='bone', interpolation='nearest')
        plt.axis('off')  # Hide axes
        plt.savefig(file_name, dpi=300, bbox_inches='tight', pad_inches=0)

    def clean_up_scan(self, image_grid):

        self.print_map("char_array_image_before.png", image_grid)
        weights = [[0.5, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 0.5]]
        cleaned_grid = ndimage.convolve(image_grid, weights, mode='reflect')
        cleaned_grid = np.array(cleaned_grid > 2, dtype=int)

        points = []

        for i_idx, i in enumerate(range(IMAGE_ROBOT_POS[0]%30, MAP_SIZE, 30)):
            for j_idx, j in enumerate(range(IMAGE_ROBOT_POS[1]%30, MAP_SIZE, 30)):
                # Take a 5x5 snapshot of the area around point
                # If more than 3 points, we set
                snapshot = cleaned_grid[i-2:i+2, j-2:j+2]
                if snapshot.sum() > 3:
                    cleaned_grid[i, j] = 3
                    points.append([i_idx, j_idx])
                else:
                    cleaned_grid[i, j] = 2


        cleaned_grid[*IMAGE_ROBOT_POS] = 3
        self.print_map("char_array_image.png", cleaned_grid)
        return points
    
    def scan(self):
        points = self.clean_up_scan(self.raw_scan_environment())
        y_points_length = len(range(IMAGE_ROBOT_POS[0]%30, MAP_SIZE, 30))
        x_points_length = len(range(IMAGE_ROBOT_POS[1]%30, MAP_SIZE, 30))

        consolidated_map = np.zeros((y_points_length, x_points_length), dtype=int)
        for point in points:
            consolidated_map[*point] = 1

        return consolidated_map, points
    
    def plot_scan(self, map, current_pos, dest):
        consolidated_map = np.copy(map)
        consolidated_map[*current_pos] = 2
        consolidated_map[*dest] = 3
        self.print_map("map.png", consolidated_map)

    def a_star_search(self, map, current_pos, dest_pos):
        g = map == 0
        return a_star(g, current_pos, dest_pos)

    def move_forward(self):
        if self.px.dir_current_angle != 0:
            self.gently_turn(0)
        
        self.px.forward(POWER)
        time.sleep(1.5)
        self.px.forward(0)

    def move_to(self, paces_in_x_coord, paces_in_y_coord):
        if paces_in_x_coord == 0 and paces_in_y_coord == 0:
            return
    
        current_pos = ((IMAGE_ROBOT_POS - (IMAGE_ROBOT_POS % 30)) // 30)
        dest_pos = current_pos + [paces_in_y_coord, paces_in_x_coord]
        
        map, walls = self.scan()
        self.plot_scan(map, current_pos, dest_pos)

        path = self.a_star_search(map, current_pos, dest_pos)

        # TODO: Current implementation does not rescan
        
        # 0: Facing +ve x-axis
        # 1: Facing -ve y-axis
        # -1: Facing +ve y-axis
        orientation = 0 
        for idx in range(1, len(path)):
            # where_to_move = path[idx] - path[idx-1]
            # future_orientation = where_to_move[0] # We assume that the robot only needs to move forward
            future_orientation = path[idx][0] - path[idx-1][0]

            if (future_orientation == 1):
                # If we are not oriented already, fix that
                if (orientation != 1):
                    self.turn_right()
                self.move_forward()
                orientation = future_orientation

            if (future_orientation == 0):
                # If we are not oriented already, fix that
                if (orientation == 1):
                    self.turn_left()
                if orientation == -1:
                    self.turn_right()
                self.move_forward()
                orientation = future_orientation

            if (future_orientation == -1):
                # If we are not oriented already, fix that
                if (orientation != -1):
                    self.turn_left()
                self.move_forward()
                orientation = future_orientation

        self.px.forward(0)        

    def __del__(self):
        self.px.forward(0)
        self.gently_turn(0)
        self.gently_pan(0)


def main():
    robot = Robot()

    robot.move_to(2, 1)

    # for i in range(4):
    #     robot.turn_left()
    # time.sleep(5)
    # for i in range(4):
    #     robot.turn_right()

    # robot.scan_environment()
    # robot.clean_up_scan()

if __name__ == "__main__":
    main()