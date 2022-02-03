import numpy as np 
import cv2

class Room:
  def __init__(self, door, thikness = 50):
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

    def create_wall(self):
        return 0

