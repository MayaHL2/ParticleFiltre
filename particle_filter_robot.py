#!/usr/bin/env python
# coding: utf-8
 
# In[2]:


import numpy as np 
import cv2

map = cv2.imread("map.png", 0)

#percent by which the image is resized
scale_percent = 300

#calculate the 50 percent of original dimensions
width = int(map.shape[1] * scale_percent / 100)
height = int(map.shape[0] * scale_percent / 100)

# dsize
dsize = (width, height)

# resize image
map = cv2.resize(map, dsize)
cv2.imshow("map_i", map)

HEIGHT, WIDTH = map.shape

rx, ry, rtheta = (WIDTH/4, HEIGHT/4, 0)


# In[3]:


STEP = 5
TURN = np.radians(25)
def get_input():
  fwd = 0
  turn = 0
  halt = 0
  k = cv2.waitKey(0)&0xFF


  if k == 109:
    fwd = STEP
  elif k == 33:
    turn = TURN
  elif k == 58:
    turn = -TURN
  else: 
    halt = True

  return fwd, turn, halt


# In[4]:


SIGMA_STEP = 0.5
SIGMA_TURN = np.radians(5)

def move_robots(rx, ry, rtheta, fwd, turn):
  fwd_noisy = np.random.normal(fwd, SIGMA_STEP, 1)
  rx += fwd_noisy*np.cos(rtheta)
  ry += fwd_noisy*np.sin(rtheta)

  turn_noisy = np.random.normal(turn, SIGMA_TURN, 1)
  rtheta += turn_noisy
  return rx, ry, rtheta


# In[5]:


NUM_PARTICLES = 3000

def init():
  particles = np.random.rand(NUM_PARTICLES, 3)
  particles *= np.array((WIDTH, HEIGHT, np.radians(360)))
  return particles


# In[6]:


def move_particles(particles, fwd, turn):
  particles[:, 0] += fwd*np.cos(rtheta)
  particles[:, 1] += fwd*np.sin(rtheta)
  particles[:, 2] += turn

  particles[:, 0] = np.clip(particles[:, 0], 0.0, WIDTH-1)
  particles[:, 1] = np.clip(particles[:, 1], 0.0, HEIGHT-1)

  return particles


# In[7]:


SIGMA_SENSOR = 2

def sense(x, y, noisy = False):
  x = int(x)
  y = int(y)
  if noisy: 
    return np.random.normal(map[y,x], SIGMA_SENSOR, 1)
  return map[y,x]


# In[8]:


def compute_weights(particles, robot_sensor):
  errors = np.zeros(NUM_PARTICLES)
  for i in range(NUM_PARTICLES):
    elevation = sense(particles[i,0], particles[i,1], noisy = False)
    errors[i] = abs(robot_sensor - elevation)
  weights = np.max(errors) - errors

  print(np.max(weights), np.min(weights))

  weights[np.where(
            (particles[:,0] == 0)|
            (particles[:,0] == WIDTH-1)|
            (particles[:,1] == 0)|
            (particles[:,1] == HEIGHT-1)
          )[0]
          ] = 0.0

  weights = weights**3
  return weights


# In[9]:


def resample(particles, weights):
  probabilities = weights/np.sum(weights)
  new_index = np.random.choice(
      NUM_PARTICLES, 
      size = NUM_PARTICLES, 
      p = probabilities
  )

  particles = particles[new_index, :]

  return particles


# In[10]:


SIGMA_POS = 2
SIGMA_TURN = np.radians(10)
def add_noise(particles):
  noise = np.concatenate(
      (
          np.random.normal(0,SIGMA_POS, (NUM_PARTICLES, 1)),
          np.random.normal(0,SIGMA_POS, (NUM_PARTICLES, 1)),
          np.random.normal(0,SIGMA_TURN, (NUM_PARTICLES, 1))  
      ),
      axis = 1
  )
  particles += noise 
  return particles


# In[1]:


def display(map, rx, ry, particles):
  lmap = cv2.cvtColor(map, cv2.COLOR_GRAY2BGR)

  # Display particles
  if len(particles) > 0:
    for i in range(NUM_PARTICLES):
      cv2.circle(lmap, 
                 (int(particles[i,0]),(int(particles[i,1]))),
                  1, 
                  (255, 0, 0),
                  1)
                 
  # Display robot 
  cv2.circle(lmap, (int(rx), int(ry)), 5, (0,255,0), 10)

  # Display best guess
  if len(particles) > 0:
    px = np.mean(particles[:,0])
    py = np.mean(particles[:,1])
    cv2.circle(lmap, (int(px), int(py)), 5, (0,0,255), 5)
  
  cv2.imshow("map", lmap)


# In[14]:


particles = init()

while True:
    display(map, rx, ry, particles)
    fwd, turn, halt = get_input()
    if halt: 
        break
    rx, ry, rtheta = move_robots(rx, ry, rtheta, fwd, turn) 
    particles = move_particles(particles, fwd, turn)
    if fwd != 0:
        robot_sensor = sense(rx, ry, noisy = True)
        weights = compute_weights(particles, robot_sensor)
        particles = resample(particles, weights)
        particles = add_noise(particles)

cv2.destroyAllWindows() 
