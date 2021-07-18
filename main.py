import cv2
from cv2 import bgsegm
import numpy as np
import time

cap = cv2.VideoCapture('DRONE-SURVEILLANCE-CONTEST-VIDEO.mp4')

#counterLine = 450
offset = 6
counter = 0
minimum_w = 80
minimum_h = 80
algorithm = cv2.bgsegm.createBackgroundSubtractorMOG()


def count_center(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx,cy


detect = []

while True:
    ret, frame = cap.read()
    resized = cv2.resize(frame, (1280, 690), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    frameGray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    frameBlur = cv2.GaussianBlur(frameGray,(3,3),6)
    subtract_img = algorithm.apply(frameBlur)
    dilate = cv2.dilate(subtract_img, np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilate1 = cv2.morphologyEx(dilate,cv2.MORPH_CLOSE, kernel)
    dilate2 = cv2.morphologyEx(dilate1, cv2.MORPH_CLOSE, kernel)
    contour,h = cv2.findContours(dilate2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.line(resized,(0,counterLine), (1290, counterLine),(255,0,0),2)
    for (i,c) in enumerate(contour):
        (x,y,w,h) = cv2.boundingRect(c)
        counter_val = (w>=minimum_w) and (h>=minimum_h)
        if not counter_val:
            continue
        cv2.rectangle(resized, (x,y), (x+w,y+h),(0,0,255),3)
        #cv2.putText(resized, f"Vechile {str(counter)}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        center = count_center(x,y,w,h)
        detect.append(center)
        cv2.circle(resized,center,1,(0,0,255),0)

        for (x,y) in detect:
            if y<(500 + offset) and y>(500 - offset):
                counter += 1
                detect.remove((x,y))
                print("car counter: "+str(counter))

    cv2.putText(resized, f"{str(counter)}",(1105,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)


    #cv2.imshow("output1", dilate2)
    cv2.imshow("output", resized)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

