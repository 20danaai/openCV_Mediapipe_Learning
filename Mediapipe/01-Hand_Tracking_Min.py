import mediapipe as mp
import cv2 as cv
import time
cap=cv.VideoCapture(0)
mphands=mp.solutions.hands
hands=mphands.Hands()
mpDraw=mp.solutions.drawing_utils
pTime=0
cTime=0
while True:
    sucess,img=cap.read()
    if not sucess:
        break
    img=cv.flip(img,1)
    imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    if results.multi_hand_landmarks:
         for handlms in results.multi_hand_landmarks:
             for id,lm in enumerate(handlms.landmark):
                 h,w,c,=img.shape
                 cx,cy=int(lm.x*w),int(lm.y*h)
                 print(id,cx,cy)
                 if id==0:
                     cv.circle(img,(cx,cy),10,(255,0,255),6)
                 elif id==16:
                     cv.circle(img,(cx,cy),20,(255,0,255),-1)
             mpDraw.draw_landmarks(img,handlms,mphands.HAND_CONNECTIONS)
    cTime=time.time()
    fps=1/(cTime-pTime) 
    pTime=cTime
    
    cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_COMPLEX,3,(255,255,255),3)
    cv.imshow('img',img)

    key=cv.waitKey(1)
    if key==ord('q'):
        break

cap.release() 
cv.destroyAllWindows()  
