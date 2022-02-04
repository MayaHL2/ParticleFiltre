import numpy as np 
import cv2

class Robot:
  def __init__(self, pos, rot_angle):
    # super().__init__(room, pos, rot_angle)
    self.pos = pos
    self.rot_angle = rot_angle
    self.robot_radius = 25
    self.thikness = 10
    self.robot_color = (255,0,0)

  # The position of the robot is relative to the room
  # its rotation is relative to its center with x,y <=> width, height
  def put_robot(self, room):

    cv2.circle(room, self.pos, self.robot_radius, self.robot_color , -1)
    cv2.line(room, self.pos, (int(self.pos[0]+self.robot_radius*np.cos(self.rot_angle)), int(self.pos[1]+self.robot_radius*np.sin(self.rot_angle))),(0,0,0), self.thikness//2)
    cv2.imshow("room", room)

    return room

  # This function changes the position of the robot with a small noise
  def move(self, room, fwd, turn, noisy = True, STD_STEP = 2, STD_TURN = np.radians(3)):
    if noisy: 
        fwd_noisy = np.random.normal(fwd, STD_STEP, 1)
        turn_noisy = np.random.normal(turn, STD_TURN, 1)
    else: 
        fwd_noisy = fwd
        turn_noisy = turn

    pos0 = self.pos[0] + int(fwd_noisy*np.cos(self.rot_angle))
    pos1 = self.pos[1] + int(fwd_noisy*np.sin(self.rot_angle))
      
    self.rot_angle += turn_noisy

    if room[pos1][pos0][0] == 0:
      pos0 = self.pos[0]
      pos1 = self.pos[1]
      
    self.pos = (pos0, pos1)

  # Senses the position of the robot like a lidar 
  def sense_lidar(self, room, nbr_angle_accuracy = 5, step_xy = 13, sensor_angle = np.radians(360), max_distance_sensor = 300):

    sensor = np.zeros((nbr_angle_accuracy,1))

    lroom = cv2.cvtColor(room, cv2.COLOR_BGR2GRAY)
    if not(nbr_angle_accuracy == 1):
        step_angle = sensor_angle/(nbr_angle_accuracy-1)
    else : 
        sensor_angle = step_angle = 0

    for i in range(nbr_angle_accuracy):
        dist = 0 
        posx = self.pos[1]
        posy = self.pos[0]
        
        if posx<lroom.shape[0]  and posy<lroom.shape[1] :
            pix = lroom[posx, posy]
        else :
            pix = -1

        while pix>0 and dist<max_distance_sensor and posx+step_xy<lroom.shape[0] and posy+step_xy< lroom.shape[1]:

                posy += int(step_xy*np.cos(self.rot_angle-sensor_angle/2 + step_angle*i))
                posx += int(step_xy*np.sin(self.rot_angle-sensor_angle/2 + step_angle*i))

                dist += step_xy

                pix = lroom[posx, posy]

        if pix == 0 :
            sensor[i] = np.sqrt(posx**2 + posy**2)
            cv2.circle(room, (int(posy), int(posx)), 5, (255,0,255), -1)
        else:
            sensor[i] = -1
            cv2.circle(room, (int(posy), int(posx)), 5, (0,255,255), -1)

    return sensor
