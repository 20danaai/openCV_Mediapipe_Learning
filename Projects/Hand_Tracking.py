import cv2 as cv
import numpy as np
from collections import deque
cap=cv.VideoCapture(0)
pts=deque(maxlen=64) # A smart storage device that holds only 64 points to save the pen's path and automayically deletes the old one.
while True:
    success,frame=cap.read()
    if not success:
        break
    frame=cv.flip(frame,1)
    blurred=cv.GaussianBlur(frame,(11,11),0)
    hsv_frame=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    lower_blue=np.array([100,100,50]) # Blue level range from 100 to 140
    upper_blue=np.array([140,255,255])
    # lower_green=np.array([35,100,50]) # green level range from 35 to 85
    # upper_green=np.array([85,255,255])
    # lower_red=np.array([0,100,50]) # red level range from 0 to 10
    # upper_red=np.array([10,255,255])
    mask=cv.inRange(hsv_frame,lower_blue,upper_blue)
    mask=cv.erode(mask,None,iterations=2) # Remove small white spots and background blur.
    mask=cv.dilate(mask,None,iterations=2) # Restore the original color spot to its original size.
    contours,_=cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contours=max(contours,key=cv.contourArea) # cv.max : error ( comparison) , It selects only the largest blue spot and ignores the extra small spots.
        x,y,w,h=cv.boundingRect(largest_contours)
        center=(int(x+w/2)),int(y+(h+2))
        circle=cv.circle(frame,(center),7,(255,255,255),-1)
        pts.appendleft(center)

        for i in range(1,len(pts)):
            if pts[i-1] is None or pts[i] is None:
                continue
            line=cv.line(frame,pts[i-1],pts[i],(255,255,255),4)

    cv.imshow('Original  with drawing',frame)
    #cv.imshow('Blue Mask',mask)
    key=cv.waitKey(1)& 0xFF
    if key==ord('c'):
        pts.clear()
    elif key==ord('q'):
        break
cap.release()
cv.destroyAllWindows()
