import numpy as np 
import cv2
import matplotlib.pyplot as plt

from robot import *
from particles import *

NUM_PARTICLES = 100

room = room = cv2.imread("room.png")
HEIGHT, WIDTH,_ = room.shape

# This function returns the next steps and turns of the robot
def get_input(TURN = np.radians(20), STEP = 5):
  fwd = 0
  turn = 0
  halt = False
  k = cv2.waitKey(0)&0xFF

  if k == 109:
    fwd = STEP
  elif k == 33:
    turn = TURN
  elif k == 58:
    turn = -TURN
  elif k == 0:
    halt = True

  return fwd, turn, halt


halt = False     
pos = (3*WIDTH//4, 4*HEIGHT//5)
rot = 180
fwd, turn = 0, 0
error_pos = list()

robot_ = Robot(pos, rot)

particles_ = Particles(NUM_PARTICLES, HEIGHT, WIDTH)

while not(halt):

    lroom = np.array(room)

    fwd, turn, halt = get_input()
    robot_.move(lroom, fwd, turn, noisy = True)

    sensor = robot_.sense_lidar(lroom)

    particles_.put_particles(lroom)
    particles_.move_particles(fwd, turn)

    particles_.get_weights(lroom, sensor)
    particles_.resample()
    particles_.add_noise()

    if particles_.particles_exist:
        particle_mean = particles_.estimation()
        cv2.circle(lroom, (int(particle_mean[0]), int(particle_mean[1])), 8, (0, 0, 255), -1)

    lroom = robot_.put_robot(lroom)
    cv2.imshow("room", lroom)

    error_pos.append(np.abs(robot_.pos - np.array([int(particle_mean[0]), int(particle_mean[1])])))

plt.plot(error_pos)
plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()