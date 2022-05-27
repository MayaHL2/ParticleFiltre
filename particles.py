import numpy as np 
import cv2
from constant import *

class Particles:
    def __init__(self, num_particles, HEIGHT, WIDTH):
        self.num_particles = num_particles
        self.HEIGHT, self.WIDTH = HEIGHT, WIDTH
        self.particles = np.random.rand(num_particles, 3)
        self.particles *= np.array((self.WIDTH, self.HEIGHT, np.radians(360)))
        self.weights =  np.zeros(self.num_particles)
        self.SIGMA_POS = 2
        self.SIGMA_TURN = np.radians(10)

    def move_particles(self, fwd, turn):
        self.particles[:, 0] += fwd*np.cos(self.particles[:, 2])
        self.particles[:, 1] += fwd*np.sin(self.particles[:, 2])
        self.particles[:, 2] += turn

        self.particles[:, 0] = np.clip(self.particles[:, 0], 0.0, self.WIDTH-1)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0.0, self.HEIGHT-1)


    def put_particles(self, room):
        if len(self.particles) > 0:
            for i in range(self.num_particles):
                cv2.circle(room, (int(self.particles[i,0]),(int(self.particles[i,1]))),1, (255, 255, 0), 1)


    def get_weights(self, room, sensor_robot):

        for i in range(len(self.particles)):
            sensor_particle = self.sense_lidar(room, (int(self.particles[i][0]), int(self.particles[i][1])), self.particles[i][2])
            weight = np.abs(sensor_robot-sensor_particle)
            self.weights[i] = np.sum(weight)

        self.weights = (NBR_ANGLE_ACCURACY*max(self.WIDTH//15, self.HEIGHT//15) - self.weights)
        # self.wieghts = (np.max(self.weights)- self.weights)

        self.weights[np.where(
                (self.particles[:,0] == 0)|
                (self.particles[:,0] == self.WIDTH-1)|
                (self.particles[:,1] == 0)|
                (self.particles[:,1] == self.HEIGHT-1)
            )[0]
            ] = 0.0
        self.weights[np.where(self.weights <0)] = 0.0
        self.weights = self.weights**3


    def resample(self):
        probabilities = self.weights/np.sum(self.weights)

        new_index = np.random.choice(
            self.num_particles, 
            size = self.num_particles, 
            p = probabilities
        )

        self.particles = self.particles[new_index, :]

        # self.particles[np.where(self.weights == 0)]  = [0,0,0]

        # hey = self.particles[np.where(self.weights == 0)] 




    def add_noise(self):
        noise = np.concatenate(
            (
                np.random.normal(0,self.SIGMA_POS, (self.num_particles, 1)),
                np.random.normal(0,self.SIGMA_POS, (self.num_particles, 1)),
                np.random.normal(0,self.SIGMA_TURN, (self.num_particles, 1))  
            ),
            axis = 1
        )
        self.particles += noise 

    def sense_lidar(self, room, pos, rot_angle, nbr_angle_accuracy = NBR_ANGLE_ACCURACY, step_xy = 13, sensor_angle = np.radians(SENSOR_ANGLE), max_distance_sensor = 300):

        sensor = np.zeros((nbr_angle_accuracy,1))

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
        
            

            
            if pix == 0 :
                sensor[i] = np.sqrt(posx**2 + posy**2)
            else:
                sensor[i] = -1

        return sensor

    def particles_exist(self):
        return len(self.particles)>0

    def estimation(self):
        return np.mean(self.particles, axis = 0)

    def number_particles(self):
        return len(self.particles)