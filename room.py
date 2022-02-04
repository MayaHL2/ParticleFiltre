import cv2 
import numpy as np


def create_room(file_name, door, thikness = 50):
    room = cv2.imread(file_name)
    HEIGHT, WIDTH,_ = room.shape
    cv2.resize(room, (HEIGHT//2, WIDTH//2))
    HEIGHT, WIDTH,_ = room.shape
    room = cv2.line(room, (0,0), (0,HEIGHT), (0,0,0), thikness)
    room = cv2.line(room, (0,0), (WIDTH,0), (0,0,0), thikness)
    room = cv2.line(room, (WIDTH,HEIGHT), (0,HEIGHT), (0,0,0), thikness)
    room = cv2.line(room, (WIDTH,HEIGHT), (WIDTH,0), (0,0,0), thikness)

    if door:
        door_1 = np.random.randint(2*thikness, HEIGHT-5*thikness)
        door_2 = np.random.randint(door_1+3*thikness, HEIGHT-2*thikness)
        # print(door_1, door_2, HEIGHT)
        room = cv2.line(room, (WIDTH,door_1), (WIDTH,door_2), (255,255,255), thikness)

    
    room = cv2.line(room, (WIDTH//2,HEIGHT), (WIDTH//2,0), (0,0,0), thikness//2)

    door_1 = np.random.randint(2*thikness, HEIGHT-5*thikness)
    door_2 = np.random.randint(door_1+3*thikness, HEIGHT-2*thikness)

    room = cv2.line(room, (WIDTH//2,door_1), (WIDTH//2,door_2), (255,255,255), thikness//2)

    return room

