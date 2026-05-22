import cv2 as cv
import numpy as np
from collections import deque
cap=cv.VideoCapture(0)
blue_pts=deque(maxlen=32)
green_pts=deque(maxlen=32)
red_pts=deque(maxlen=32) # A smart storage device that holds only 64 points to save the pen's path and automayically deletes the old one.
while True:
    success,frame=cap.read()
    if not success:
        break
    frame=cv.flip(frame,1)
    blurred=cv.GaussianBlur(frame,(11,11),0)
    hsv_frame=cv.cvtColor(blurred,cv.COLOR_BGR2HSV)
    lower_blue=np.array([100,100,50]) # Blue level range from 100 to 140
    upper_blue=np.array([140,255,255])
    lower_green=np.array([35,100,50]) # green level range from 35 to 85
    upper_green=np.array([85,255,255])
    lower_red=np.array([170,150,120]) # red level range from 0 to 10
    upper_red=np.array([180,255,255])
    mask1=cv.inRange(hsv_frame,lower_blue,upper_blue)
    mask2=cv.inRange(hsv_frame,lower_green,upper_green)
    mask3=cv.inRange(hsv_frame,lower_red,upper_red)
    mask1=cv.erode(mask1,None,iterations=2) # Remove small white spots and background blur.
    mask1=cv.dilate(mask1,None,iterations=2) # Restore the original color spot to its original size.
    mask2=cv.erode(mask2,None,iterations=2)
    mask2=cv.dilate(mask2,None,iterations=2)
    mask3=cv.erode(mask3,None,iterations=2)
    mask3=cv.dilate(mask3,None,iterations=2)
    contours1,_=cv.findContours(mask1,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    contours2,_=cv.findContours(mask2,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    contours3,_=cv.findContours(mask3,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    all_contours=[(contours1,(255,0,0)),(contours2,(0,255,0)),(contours3,(0,0,255))]
    for contours,color in all_contours:
        if contours:
            largest_contours=max(contours,key=cv.contourArea)  # cv.max : error ( comparison) , It selects only the largest blue spot and ignores the extra small spots.
            x,y,w,h=cv.boundingRect(largest_contours)
            center=(int(x+w/2)),int(y+(h+2))
            circle=cv.circle(frame,(center),7,(color),-1)
            all_pts=[blue_pts,green_pts,red_pts]
            for pts in all_pts:
                 pts.appendleft(center)

                 for i in range(1,len(pts)):
                    if pts[i-1] is None or pts[i] is None:
                      continue
                    line=cv.line(frame,pts[i-1],pts[i],(color),4)
                    np.save('Hand',line)
                    
    cv.putText(frame,' Press C to clear',(50,100),cv.FONT_HERSHEY_COMPLEX_SMALL,1,(0,250,250),thickness=3)
    cv.putText(frame,' Press Q to quit',(50,140),cv.FONT_HERSHEY_COMPLEX_SMALL,1,(0,250,250),thickness=3)
    cv.imshow(' Multi Color Tracking',frame)
    # cv.imshow('Blue Mask',mask1)
    # cv.imshow('green Mask',mask2)
    # cv.imshow('Red Mask',mask3)
    key=cv.waitKey(1)& 0xFF
    if key==ord('c'):
        pts.clear()
    elif key==ord('q'):
        break
cap.release()
cv.destroyAllWindows()
