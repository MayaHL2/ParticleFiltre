import numpy as np 
import cv2
import time

NUM_PARTICLES = 200

def create_room(door, thikness = 50):
    room = cv2.imread("piece.png")
    # HEIGHT, WIDTH,_ = room.shape
    # room = cv2.line(room, (0,0), (0,HEIGHT), (0,0,0), thikness)
    # room = cv2.line(room, (0,0), (WIDTH,0), (0,0,0), thikness)
    # room = cv2.line(fond, (WIDTH,HEIGHT), (0,HEIGHT), (0,0,0), thikness)
    # room = cv2.line(room, (WIDTH,HEIGHT), (WIDTH,0), (0,0,0), thikness)

    # if door:
    #     door_1 = np.random.randint(2*thikness, HEIGHT-5*thikness)
    #     door_2 = np.random.randint(door_1+3*thikness, HEIGHT-2*thikness)
    #     # print(door_1, door_2, HEIGHT)
    #     room = cv2.line(room, (WIDTH,door_1), (WIDTH,door_2), (255,255,255), thikness)

    
    # room = cv2.line(room, (WIDTH//2,HEIGHT), (WIDTH//2,0), (0,0,0), thikness//2)

    # door_1 = np.random.randint(2*thikness, HEIGHT-5*thikness)
    # door_2 = np.random.randint(door_1+3*thikness, HEIGHT-2*thikness)

    # room = cv2.line(room, (WIDTH//2,door_1), (WIDTH//2,door_2), (255,255,255), thikness//2)

    return room


room = create_room(door = True)
HEIGHT, WIDTH,_ = room.shape
print(HEIGHT, WIDTH)
# The position of the robot is relative to the room
# its rotation is relative to its center with x,y <=> width, height
def put_robot(room, pos, rot_angle, robot_radius = 25, thikness = 10, robot_color = (255,0,0)):

    cv2.circle(room, pos, robot_radius,robot_color , -1)
    cv2.line(room, pos, (int(pos[0]+robot_radius*np.cos(rot_angle)), int(pos[1]+robot_radius*np.sin(rot_angle))),(0,0,0), thikness//2)

    return room



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


# This function changes the position of the robot with a small noise
def move(room, pos, rot_angle, fwd, turn, noisy = True, STD_STEP = 2, STD_TURN = np.radians(3)):
  if noisy: 
      fwd_noisy = np.random.normal(fwd, STD_STEP, 1)
      turn_noisy = np.random.normal(turn, STD_TURN, 1)
  else: 
      fwd_noisy = fwd
      turn_noisy = turn

  pos0 = pos[0] + int(fwd_noisy*np.cos(rot_angle))
  pos1 = pos[1] + int(fwd_noisy*np.sin(rot_angle))
    
  rot_angle += turn_noisy

  if room[pos1][pos0][0] == 0:
    pos0 = pos[0]
    pos1 = pos[1]
    

  return (pos0, pos1), rot_angle

# Senses the position of the robot like a lidar 
def sense_lidar(room, pos, rot_angle, robot, nbr_angle_accuracy = 15, step_xy = 13, sensor_angle = np.radians(90), max_distance_sensor = 300):

    start = time.time()
    sensor = np.zeros((nbr_angle_accuracy,2))

    lroom = cv2.cvtColor(room, cv2.COLOR_BGR2GRAY)
    if not(nbr_angle_accuracy == 1):
        step_angle = sensor_angle/(nbr_angle_accuracy-1)
    else : 
        sensor_angle = step_angle = 0

    for i in range(nbr_angle_accuracy):
        dist = 0 
        posx = pos[1]
        posy = pos[0]
        
        if posx<lroom.shape[0]  and posy<lroom.shape[1] :
            pix = lroom[posx, posy]
        else :
            pix = -1

        while pix>0 and dist<max_distance_sensor and posx+step_xy<lroom.shape[0] and posy+step_xy< lroom.shape[1]:

                posy += int(step_xy*np.cos(rot_angle-sensor_angle/2 + step_angle*i))
                posx += int(step_xy*np.sin(rot_angle-sensor_angle/2 + step_angle*i))

                dist += step_xy

                pix = lroom[posx, posy]
        
                # print((step_xy*np.cos(rot_angle-sensor_angle/2 + step_angle*i)), (step_xy*np.sin(rot_angle-sensor_angle/2 + step_angle*i)))

        

        
        if pix == 0 :
            sensor[i,0] = posx
            sensor[i,1] = posy

            if robot:
                cv2.circle(room, (int(posy), int(posx)), 5, (255,0,255), -1)
        else:
            sensor[i,0] = -1
            sensor[i,1] = -1
            if robot:
                cv2.circle(room, (int(posy), int(posx)), 5, (0,255,255), -1)

    return sensor, time.time()-start


def init_particles(num_particles = NUM_PARTICLES):
    particles = np.random.rand(num_particles, 3)
    particles *= np.array((WIDTH, HEIGHT, np.radians(360)))
    return particles

def move_particles(particles, fwd, turn):
    particles[:, 0] += fwd*np.cos(particles[:, 2])
    particles[:, 1] += fwd*np.sin(particles[:, 2])
    particles[:, 2] += turn

    particles[:, 0] = np.clip(particles[:, 0], 0.0, WIDTH-1)
    particles[:, 1] = np.clip(particles[:, 1], 0.0, HEIGHT-1)

    return particles


def put_particles(room, particles, num_particles = NUM_PARTICLES):
    if len(particles) > 0:
        for i in range(num_particles):
            cv2.circle(room, (int(particles[i,0]),(int(particles[i,1]))),1, (255, 255, 0), 1)


def get_weights(room, sensor_robot, particles):

    weights = np.zeros((2,len(particles)))

    for i in range(len(particles)):
        sensor_particle, timing = sense_lidar(room, (int(particles[i][0]), int(particles[i][1])), particles[i][2], robot = False)
        # print(sensor_robot.shape, sensor_particle.shape)
        weight = (sensor_robot-sensor_particle)**2
        weights[0,i] = np.sum(weight, axis=1)[0]
        weights[1,i] = np.sum(weight, axis=1)[1]

    weights = np.sum(weights, axis = 0)

    weights = (np.max(weights) - weights)

    weights[np.where(
            (particles[:,0] == 0)|
            (particles[:,0] == WIDTH-1)|
            (particles[:,1] == 0)|
            (particles[:,1] == HEIGHT-1)
          )[0]
          ] = 0.0
    
    weights = weights**3

    return(weights, timing)


def resample(particles, weights):
  probabilities = weights/np.sum(weights)

  new_index = np.random.choice(
      NUM_PARTICLES, 
      size = NUM_PARTICLES, 
      p = probabilities
  )

  particles = particles[new_index, :]

  return particles

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

halt = False     
pos = (3*WIDTH//4, 4*HEIGHT//5)
rot = 180
fwd, turn = 0, 0

particles = init_particles(num_particles= NUM_PARTICLES)

while not(halt):

    lroom = np.array(room)

    fwd, turn, halt = get_input()
    pos, rot = move(lroom, pos, rot, fwd, turn, noisy = True)
    sensor, timing = sense_lidar(lroom, pos, rot, robot = True)
    print(sensor)

    # print(timing)
    # put_particles(lroom, particles, num_particles = NUM_PARTICLES)
    # move_particles(particles, fwd, turn)

    # weights, timing = get_weights(lroom, sensor, particles)
    # particles = resample(particles, weights)
    # particles = add_noise(particles)

    # if len(particles>0):
    #     particle_mean = np.mean(particles, axis = 0)
    #     cv2.circle(lroom, (int(particle_mean[0]), int(particle_mean[1])), 8, (0, 0, 255), -1)

    put_robot(lroom, pos, rot)
    cv2.imshow("room", lroom)


cv2.waitKey(0)
cv2.destroyAllWindows()