from picarx import Picarx
import time
import readchar
import datetime
import numpy as np

POWER = 30
STEPS = 50

def gently_turn(px:Picarx, final_angle:int):
    for i in np.linspace(px.dir_current_angle, final_angle, STEPS):
        px.set_dir_servo_angle(i)
        time.sleep(0.01)

def gently_move_forward(px:Picarx, time_to_run:int):
    for power in np.linspace(0, POWER, STEPS):
        px.forward(power)
        time.sleep(time_to_run/2/STEPS)

    for power in np.linspace(POWER, 0, STEPS):
        px.forward(power)
        time.sleep(time_to_run/2/STEPS)

def gently_move_backward(px:Picarx, time_to_run:int):
    for power in np.linspace(0, POWER, STEPS):
        px.backward(power)
        time.sleep(time_to_run/2/STEPS)

    for power in np.linspace(POWER, 0, STEPS):
        px.backward(power)
        time.sleep(time_to_run/2/STEPS)

if __name__ == "__main__":

    start_time = None

    # try:
    #     px = Picarx()    
    #     time.sleep(3)
    #     # px.set_dir_servo_angle(-30)
    #     px.backward(POWER)
    #     time.sleep(.5)
    #     px.backward(POWER/2)
    #     time.sleep(.5)
    #     px.forward(0)
    #     time.sleep(.5)
    #     px.forward(POWER/2)
    #     time.sleep(.5)

    #     px.forward(POWER)
    #     # time.sleep(10.178484)
    #     while True:
    #         reading = px.get_grayscale_data()
    #         # print(reading)
    #         if (reading[0]<200 and reading[1]<200 and reading[2]<200):
    #             if (start_time == None):
    #                 # px.set_dir_servo_angle(-30)
    #                 start_time = time.time_ns()
    #                 time.sleep(0.5) # Clear tape fully
    #             else:
    #                 end_time = time.time_ns()
    #                 delta = end_time - start_time
    #                 print(delta /(10**9))
    #                 raise Exception("Stop this nonsense")

    try:

        px = Picarx()    
        time.sleep(5)

        for i in range(4):
            time_to_run = .92

            gently_turn(px, -30)                
            gently_move_forward(px, time_to_run)
            gently_turn(px, 30)
            gently_move_backward(px, time_to_run)

            time.sleep(1)
        
        
        
        
        px.forward(0)
        gently_turn(px, 0)
        # time.sleep(0.1)
        # px.backward(POWER)
        # time.sleep(.374*2)
        # px.forward(0)
        
        # end_time = datetime.datetime.now()
        # delta = end_time - start_time
        # print(delta.total_seconds())

    finally:
        # end_time = datetime.datetime.now()
        
        px.set_cam_tilt_angle(0)
        px.set_cam_pan_angle(0)  
        # px.set_dir_servo_angle(0)  
        px.stop()
        time.sleep(.2)


