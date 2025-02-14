from picarx import Picarx
import time
import numpy as np

POWER = 30
STEPS = 50

STEERING_ANGLE = 30

class Robot:
    
    def __init__(self):
        self.px  = Picarx()    
        time.sleep(5)

    def gently_turn(self, final_angle:int):
        for i in np.linspace(self.px.dir_current_angle, final_angle, STEPS):
            self.px.set_dir_servo_angle(i)
            time.sleep(0.01)
        time.sleep(0.1)

    def gently_move_forward(self, time_to_run:int):
        for power in np.linspace(0, POWER, STEPS):
            self.px.forward(power)
            time.sleep(time_to_run/2/STEPS)

        for power in np.linspace(POWER, 0, STEPS):
            self.px.forward(power)
            time.sleep(time_to_run/2/STEPS)

    def gently_move_backward(self, time_to_run:int):
        for power in np.linspace(0, POWER, STEPS):
            self.px.backward(power)
            time.sleep(time_to_run/2/STEPS)

        for power in np.linspace(POWER, 0, STEPS):
            self.px.backward(power)
            time.sleep(time_to_run/2/STEPS)

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

    def __del__(self):
        self.px.forward(0)
        self.gently_turn(0)


def main():
    robot = Robot()
    for i in range(4):
        robot.turn_left()
    time.sleep(5)
    for i in range(4):
        robot.turn_right()

if __name__ == "__main__":
    main()