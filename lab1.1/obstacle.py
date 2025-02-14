# Based on: https://github.com/sunfounder/picar-x/blob/v2.0/example/4.avoiding_obstacles.py

from picarx import Picarx
import time

POWER = 5
DangerDistance = 20 

def main():
    try:
        picar = Picarx()

        while True:
            distance = round(picar.ultrasonic.read(), 2)
            print("distance: ",distance)
            if distance >= DangerDistance:
                picar.set_dir_servo_angle(0)
                picar.forward(POWER)
            else:
                picar.set_dir_servo_angle(-30)
                picar.backward(POWER)
                time.sleep(0.5)

    finally:
        picar.forward(0)


if __name__ == "__main__":
    main()

