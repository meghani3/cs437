from picarx import Picarx
import time
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from path_finding.chatgpt import a_star
from object_detection import ObjectDetector

POWER = 15
TURN_STEPS = 50

STEERING_ANGLE = 30
PAN_ANGLE = 80

MAP_SIZE = 300
# IMAGE_ROBOT_POS = np.array([MAP_SIZE // 2, 0], dtype=int)

STEPS_BETWEEN_RESCANS = 3
SCAN_NO = 0


class Robot:

    def __init__(self):
        self.px = Picarx()
        self.camera_pan_angle = 0
        self.object_detector = ObjectDetector()

        self.robot_pos = np.array([5, 0])
        self.map = np.zeros((11, 11), dtype=int)  # Odd grid makes it easy

        self.dest_pos = np.array([5, 0])  # This will change

        time.sleep(5)

    def gently_turn(self, final_angle: int):
        for i in np.linspace(self.px.dir_current_angle, final_angle, TURN_STEPS):
            self.px.set_dir_servo_angle(i)
            time.sleep(0.01)
        time.sleep(0.1)

    def gently_move_forward(self, time_to_run: int):
        for power in np.linspace(0, POWER, TURN_STEPS):
            self.px.forward(power)
            time.sleep(time_to_run / 2 / TURN_STEPS)

        for power in np.linspace(POWER, 0, TURN_STEPS):
            self.px.forward(power)
            time.sleep(time_to_run / 2 / TURN_STEPS)

    def gently_move_backward(self, time_to_run: int):
        for power in np.linspace(0, POWER, TURN_STEPS):
            self.px.backward(power)
            time.sleep(time_to_run / 2 / TURN_STEPS)

        for power in np.linspace(POWER, 0, TURN_STEPS):
            self.px.backward(power)
            time.sleep(time_to_run / 2 / TURN_STEPS)

    def turn_left(self):
        TIME_TO_TURN_90_DEG = 1  # 0.915
        self.gently_turn(-STEERING_ANGLE)
        self.gently_move_forward(TIME_TO_TURN_90_DEG)
        self.gently_turn(STEERING_ANGLE)
        self.gently_move_backward(TIME_TO_TURN_90_DEG)

    def turn_right(self):
        TIME_TO_TURN_90_DEG = 1.4
        self.gently_turn(STEERING_ANGLE)
        # self.gently_move_forward(TIME_TO_TURN_90_DEG)
        self.px.forward(POWER)
        time.sleep(TIME_TO_TURN_90_DEG)
        self.px.forward(0)
        self.gently_turn(-STEERING_ANGLE)
        TIME_TO_TURN_90_DEG = 1.4
        self.gently_move_backward(TIME_TO_TURN_90_DEG)

    def gently_pan(self, final_angle: int):
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
        # return np.loadtxt(str(SCAN_NO) + "t.txt")
        image_grid = np.array(
            [[0 for i in range(MAP_SIZE)] for j in range(MAP_SIZE)], dtype=int
        )
        IMAGE_ROBOT_POS = np.array([MAP_SIZE // 2, 0], dtype=int)

        self.gently_pan(-PAN_ANGLE)
        for angle in np.linspace(-PAN_ANGLE, PAN_ANGLE, 2 * PAN_ANGLE):
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
                loc[0] = min(MAP_SIZE - 1, loc[0])

                loc[1] = max(0, loc[1])
                loc[1] = min(MAP_SIZE - 1, loc[1])

                image_grid[*loc] = 2

            else:
                print("nothing in line of sight")

            time.sleep(0.01)
        self.gently_pan(0)

        np.savetxt(str(SCAN_NO) + "t.txt", image_grid)

        return image_grid

    def print_map(self, file_name, image_grid):
        plt.figure(figsize=(4, 4))
        plt.imshow(image_grid, cmap="bone", interpolation="nearest")
        plt.axis("off")  # Hide axes
        plt.savefig(file_name, dpi=300, bbox_inches="tight", pad_inches=0)

    def clean_up_scan(self, image_grid):

        # IMAGE_ROBOT_POS = np.array([MAP_SIZE // 2, 0], dtype=int)

        self.print_map("char_array_image_before" + str(SCAN_NO) + ".png", image_grid)
        weights = [[0.5, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 0.5]]
        cleaned_grid = ndimage.convolve(image_grid, weights, mode="reflect")
        cleaned_grid = np.array(cleaned_grid > 2, dtype=int)

        cleaned_grid = np.array(
            ndimage.binary_dilation(cleaned_grid, iterations=4), dtype=int
        )

        points = []
        # Imagine we are setting up map at start position
        for y_idx in range(11):
            for x_idx in range(11):
                y = y_idx * 30
                x = x_idx * 30

                # if y - 3 < 0 or y + 3 > MAP_SIZE or x - 3 < 0 or x + 3 > MAP_SIZE:
                #     continue
                if y >= MAP_SIZE or x >= MAP_SIZE:
                    continue

                is_occupied = cleaned_grid[y - 3 : y + 3, x - 3 : x + 3].sum() > 3
                translated_idx = [
                    (y_idx - 5) + self.robot_pos[0],
                    x_idx + self.robot_pos[1],
                ]

                if is_occupied:
                    cleaned_grid[y, x] = 3
                    points.append(translated_idx)
                else:
                    cleaned_grid[y, x] = 2

        print(points)
        self.print_map("char_array_image" + str(SCAN_NO) + ".png", cleaned_grid)
        return points

    def scan(self):
        points = self.clean_up_scan(self.raw_scan_environment())

        for point in points:
            self.map[*point] = 1

        # Current and dest pos will always be available
        self.map[*self.robot_pos] = 0
        self.map[*self.dest_pos] = 0

        # return consolidated_map, points

    def plot_scan(self):
        consolidated_map = np.copy(self.map)
        consolidated_map[*self.robot_pos] = 2
        consolidated_map[*self.dest_pos] = 3
        self.print_map("map" + str(SCAN_NO) + ".png", consolidated_map)

    def a_star_search(self):
        g = self.map == 0
        return a_star(g, self.robot_pos, self.dest_pos)

    def move_forward(self):
        if self.px.dir_current_angle != 0:
            self.gently_turn(0)

        self.px.forward(POWER)
        time.sleep(1.8)
        self.px.forward(0)

    def orient(self, current_orientation, dest_orientation):
        if dest_orientation == 1:
            # If we are not oriented already, fix that
            if current_orientation != 1:
                self.turn_right()

        if dest_orientation == 0:
            # If we are not oriented already, fix that
            if current_orientation == 1:
                self.turn_left()
            if current_orientation == -1:
                self.turn_right()

        if dest_orientation == -1:
            # If we are not oriented already, fix that
            if current_orientation != -1:
                self.turn_left()

    def _move_to(self):
        while True:
            global SCAN_NO

            print("current robot pos", self.robot_pos)

            if (
                self.robot_pos[0] == self.dest_pos[0]
                and self.robot_pos[1] == self.dest_pos[1]
            ):
                return

            self.scan()
            self.plot_scan()

            path = self.a_star_search()

            # 0: Facing +ve x-axis
            # 1: Facing -ve y-axis
            # -1: Facing +ve y-axis
            orientation = 0
            for idx in range(1, min(len(path), STEPS_BETWEEN_RESCANS + 1)):

                # Repeat this loop this the blocking object is removed

                object_blocking_path = True
                while object_blocking_path:
                    object_blocking_path = False

                    time.sleep(1)

                    # Note: It seems like the object needs to be close to camera for model to work
                    objects_detected = self.object_detector.detect()
                    for object_dected in objects_detected.detections:
                        if object_dected.categories[0].category_name == "stop sign":
                            object_blocking_path = True

                    # Wait 3 seconds before rescanning
                    if object_blocking_path:
                        time.sleep(3)

                # We assume that the robot only needs to move forward and never backwards
                future_orientation = path[idx][0] - path[idx - 1][0]
                self.orient(orientation, future_orientation)
                orientation = future_orientation
                self.move_forward()
                self.robot_pos = path[idx]

                print("self.robot_pos:", self.robot_pos)

            self.px.forward(0)

            if (
                self.robot_pos[0] == self.dest_pos[0]
                and self.robot_pos[1] == self.dest_pos[1]
            ):
                return

            self.orient(orientation, 0)

            SCAN_NO += 1

    def move_to(self, paces_in_x_coord, paces_in_y_coord):
        self.dest_pos = self.robot_pos + [paces_in_y_coord, paces_in_x_coord]
        self._move_to()

    def __del__(self):
        self.px.forward(0)
        self.gently_turn(0)
        self.gently_pan(0)


def main():
    robot = Robot()

    time.sleep(5)

    try:
        robot.move_to(2, 1)
    finally:
        robot.__del__()

    # for i in range(2):
    #     robot.turn_right()
    #     robot.gently_turn(0)
    # time.sleep(3)
    # for i in range(2):
    #     robot.turn_right()

    # robot.scan_environment()
    # robot.clean_up_scan()


if __name__ == "__main__":
    main()
