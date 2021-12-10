import numpy as np 
import cv2
from sewar.full_ref import mse
import time

def gasuss_noise(image, mean=0, var=0.000001):

    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    
    kernel = np.ones((5,5),np.float32)/25
    out = cv2.filter2D(out, -1, kernel)
    #cv.imshow("gasuss", out)
    return out

def create_room(door, thikness = 50):
    fond = cv2.imread("fond.png")
    HEIGHT, WIDTH,_ = fond.shape
    room = cv2.line(fond, (0,0), (0,HEIGHT), (0,0,0), thikness)
    room = cv2.line(room, (0,0), (WIDTH,0), (0,0,0), thikness)
    room = cv2.line(fond, (WIDTH,HEIGHT), (0,HEIGHT), (0,0,0), thikness)
    room = cv2.line(room, (WIDTH,HEIGHT), (WIDTH,0), (0,0,0), thikness)

    if door:
        door_1 = np.random.randint(2*thikness, HEIGHT-5*thikness)
        door_2 = np.random.randint(door_1+3*thikness, HEIGHT-2*thikness)
        # print(door_1, door_2, HEIGHT)
        room = cv2.line(room, (WIDTH,door_1), (WIDTH,door_2), (255,255,255), thikness)

    return room
    # cv2.imshow("room", room)


room = create_room(door = True)
HEIGHT, WIDTH,_ = room.shape
print(HEIGHT, WIDTH)

# The position of the robot is relative to the room
# its rotation is relative to its center with x,y <=> width, height
def put_robot(room, pos, rot_angle, robot_radius = 25, thikness = 10):
    # lroom = cv2.cvtColor(room, cv2.COLOR_GRAY2BGR)
    # rot_angle_rad = np.radians(rot_angle)
    cv2.circle(room, pos, robot_radius, (255,0,0), -1)
    cv2.line(room, pos, (int(pos[0]+robot_radius*np.cos(rot_angle)), int(pos[1]+robot_radius*np.sin(rot_angle))),(0,0,0), thikness//2)
    # cv2.imshow("room with robot", room)
    return room




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




def move(pos, rot_angle, fwd, turn, noisy = True, STD_STEP = 2, STD_TURN = np.radians(3)):
  if noisy: 
      fwd_noisy = np.random.normal(fwd, STD_STEP, 1)
      turn_noisy = np.random.normal(turn, STD_TURN, 1)
  else: 
      fwd_noisy = fwd
      turn_noisy = turn

  pos0 = pos[0]+int(fwd_noisy*np.cos(rot_angle))
  pos1 = pos[1]+int(fwd_noisy*np.sin(rot_angle))
  
  rot_angle += turn_noisy
  return (pos0, pos1), rot_angle


def sense_all_around(room, pos, rot_angle, noisy, sensor_angle = 30, max_distance_sensor = 150):
    # rot_angle_rad = np.radians(rot_angle)
    sensor_angle_rad = np.radians(sensor_angle)
    max_ = int(max_distance_sensor/np.cos(sensor_angle_rad/2))

    mask = np.zeros((2*(max_), 2*(max_)))
    cv2.circle(mask, (mask.shape[1]//2, mask.shape[0]//2), (max_), (255,255,255), -1)
    # cv2.imshow("mask", mask)
    
    # M = cv2.getRotationMatrix2D(pos, rot_angle, 1)
    # lroom = cv2.warpAffine(lroom, M, (lroom.shape[1], lroom.shape[0]))
 
    sensor = cv2.copyMakeBorder(room, max_, max_, max_, max_,  cv2.BORDER_CONSTANT, value = (0, 0, 0))
    
    if noisy:
        x = np.random.randint(0, 5)
        y = np.random.randint(0, 5)
        # theta = np.random.randn()*3
        sensor = sensor[pos[1] + y:pos[1]+2*max_+ y, pos[0] +x:pos[0]+2*max_+x,:]
        # M = cv2.getRotationMatrix2D(pos, theta, 1)
        # print(theta)
        PEAK = np.random.randint(850000,1000000)/1000000
        sensor = gasuss_noise(cv2.cvtColor(sensor, cv2.COLOR_BGR2GRAY))
        sensor = np.random.poisson(sensor / 255.0 * PEAK) / PEAK * 255
        cv2.circle(sensor, (sensor.shape[1]//2, sensor.shape[0]//2), sensor.shape[1], (0,0,0), sensor.shape[1])
        # print(np.shape(sensor))
        # sensor = cv2.warpAffine(sensor, M, (sensor.shape[1], sensor.shape[0]))
        
    else:
        sensor = cv2.cvtColor(sensor, cv2.COLOR_BGR2GRAY)
        sensor = sensor[pos[1]:pos[1]+2*max_, pos[0]:pos[0]+2*max_]
        cv2.circle(sensor, (sensor.shape[1]//2, sensor.shape[0]//2), sensor.shape[1], (0,0,0), sensor.shape[1])
        sensor = sensor.astype(np.float64)

    # cv2.imshow("room", sensor)
    M = cv2.getRotationMatrix2D((sensor.shape[1]//2, sensor.shape[0]//2), rot_angle, 1)
    sensor = cv2.warpAffine(sensor, M, (sensor.shape[1], sensor.shape[0]))

    return sensor


def sense_lidar(room, pos, rot_angle, noisy, nbr_angle_accuracy = 15, step_xy = 13, sensor_angle = np.radians(60), max_distance_sensor = 1000):

    sensor = np.zeros((nbr_angle_accuracy,2))

    lroom = cv2.cvtColor(room, cv2.COLOR_BGR2GRAY)
    step_angle = sensor_angle/(nbr_angle_accuracy-1)

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

        cv2.circle(room, (int(posy), int(posx)), 5, (0,255,255), -1)

        if pix == 0:
            sensor[i,0] = posx
            sensor[i,1] = posy
        else:
            sensor[i,0] = -1
            sensor[i,1] = -1

    return sensor


def init_particles(num_particles = 3000):
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


def put_particles(room, particles, num_particles = 3000):
    if len(particles) > 0:
        for i in range(num_particles):
            cv2.circle(room, (int(particles[i,0]),(int(particles[i,1]))),1, (255, 0, 0), 1)


def get_weights(room, sensor_robot, particles):
    weights = np.zeros(len(particles))
    for i in range(len(particles)):
        sensor_particle = sense_lidar(room, (int(particles[i][0]), int(particles[i][1])), particles[i][2], noisy = False)
        cv2.resize(sensor_particle, (31,31))
        weights[i] = mse(sensor_robot, sensor_particle)
    return(weights)


halt = False     
pos = (WIDTH//2, HEIGHT//2)
rot = 0
fwd, turn = 0, 0


while not(halt):

    lroom = np.array(room)

    fwd, turn, halt = get_input()
    pos, rot = move(pos, rot, fwd, turn, noisy = True)
    sensor = sense_lidar(lroom, pos, rot, noisy = False)

    # print(sensor)

    put_robot(lroom, pos, rot)
    cv2.imshow("room", lroom)


cv2.waitKey(0)
cv2.destroyAllWindows()