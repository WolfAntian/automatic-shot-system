import cv2
import numpy as np
import dlib
from math import sqrt
from time import time
from time import sleep
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)
GPIO.setup(11,GPIO.OUT)
GPIO.setup(13,GPIO.OUT)
servoX = GPIO.PWM(11,50)
servoY = GPIO.PWM(13,50)
servoX.start(0)
servoY.start(0)

cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)
capWidth = cap.get(3)
capHeight = cap.get(4)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

font = cv2.FONT_HERSHEY_SIMPLEX

last_frame_time = time()

def draw_circle(frame, x, y):
    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
    
def draw_cross(frame, x, y):
    cv2.line(frame,(x-5,y),(x+5,y),(255,255,255),1)
    cv2.line(frame,(x,y-5),(x,y+5),(255,255,255),1)

def drawText(frame, textArr):
    textArr.reverse()
    frame_height, frame_width, frame_channels = frame.shape
    for i in range(len(textArr)):
        height = frame_height - ((i + 1) * 11)
        cv2.putText(frame,textArr[i],(10,height), font, 0.4,(255,255,255),1,cv2.LINE_AA, False)


def get_lips(landmarks):
    return {
        "top": {
            "x": landmarks.part(62).x,
            "y": landmarks.part(62).y
        },
        "bottom": {
            "x": landmarks.part(66).x,
            "y": landmarks.part(66).y
        },
        "left": {
            "x": landmarks.part(60).x,
            "y": landmarks.part(60).y
        },
        "right": {
            "x": landmarks.part(64).x,
            "y": landmarks.part(64).y
        }
    }

def get_centre(lips):
    return {
        "x": round((lips['top']['x'] + lips['bottom']['x'] + lips['left']['x'] + lips['right']['x']) / 4),
        "y": round((lips['top']['y'] + lips['bottom']['y'] + lips['left']['y'] + lips['right']['y']) / 4)
    }

def get_is_open(lips):
    height = round(sqrt((abs(lips['top']['x'] - lips['bottom']['x']) ** 2) + (abs(lips['top']['y'] - lips['bottom']['y']) ** 2)))
    width = round(sqrt((abs(lips['left']['x'] - lips['right']['x']) ** 2) + (abs(lips['left']['y'] - lips['right']['y']) ** 2)))
    return height > width * 0.55

def get_fps():
    global last_frame_time
    current_time = time()
    fps = round(1 / (current_time - last_frame_time), 2)
    last_frame_time = current_time
    return fps


################################
##### run loop starts here #####
################################

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    textArr = []
    
    faces = detector(gray)
    if len(faces) > 0:
        face = faces[0]
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        landmarks = predictor(gray, face)
        lips = get_lips(landmarks)
        centre = get_centre(lips)
        
        #servos
        if get_is_open(lips) :
            angleX = 180 - (180 * (centre['x'] / capWidth))
            servoX.ChangeDutyCycle(2+(angleX/18))
            
            angleY = 180 - (180 * (centre['y'] / capHeight))
            servoY.ChangeDutyCycle(2+(angleY/18))
            
            sleep(0.05)
            servoY.ChangeDutyCycle(0)
            servoX.ChangeDutyCycle(0)

        #draw
        for pos in lips.values():
            x = pos['x']
            y = pos['y']
            draw_circle(frame, x, y)
        draw_cross(frame, centre['x'], centre['y'])

        textArr.append("centre: " + str(centre))
        textArr.append("is_open: " + str(get_is_open(lips)))
       
    textArr.append("fps: " + str(get_fps()))
    drawText(frame, textArr)
    cv2.imshow('test1.py',frame)

    if cv2.waitKey(1) & 0xFF == ord('q') :
        break



cap.release()
cv2.destroyAllWindows()
servoX.stop()
servoY.stop()
GPIO.cleanup()
print("Goodbye!")
