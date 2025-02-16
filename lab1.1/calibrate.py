from picarx import Picarx
import time

px = Picarx()
px.forward(30)
time.sleep(4)
px.forward(0)