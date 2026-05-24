import mediapipe as mp
import cv2 as cv
import time

class handDetector():
    def __init__(self,mode=False,maxHands=2,detectionCon=0.5,trackCon=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionCon=detectionCon
        self.trackCon=trackCon
        self.mphands=mp.solutions.hands
        self.hands=self.mphands.Hands()
        self.mpDraw=mp.solutions.drawing_utils
    def findHands(self,img,draw=True):
        imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:

             for handlms in self.results.multi_hand_landmarks:
                 if draw:
                     
                     self.mpDraw.draw_landmarks(img,handlms,self.mphands.HAND_CONNECTIONS)
        return img
        
    def findposition(self,img,handNo=0,draw=True):
             lmlist=[]
             
             if self.results.multi_hand_landmarks:

                myHand=self.results.multi_hand_landmarks[handNo]
                for id,lm in enumerate(myHand.landmark):
                  h,w,c,=img.shape
                  cx,cy=int(lm.x*w),int(lm.y*h)
                  lmlist.append([id,cx,cy])
                if draw:
                 
                     cv.circle(img,(cx,cy),10,(255,0,255),6)
             return lmlist
             
def main():
    pTime=0
    cTime=0
    cap=cv.VideoCapture(0)
    detector=handDetector()
    
    while True:
      sucess,img=cap.read()
      if not sucess:
          break
      img=detector.findHands(img)
      lmlist=detector.findposition(img)
      if len(lmlist)!=0:
        print(lmlist[4])
      img=cv.flip(img,1)
      cTime=time.time()
      fps=1/(cTime-pTime) 
      pTime=cTime
    
      cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_COMPLEX,3,(255,255,255),3)
      cv.imshow('img',img)
      if cv.waitKey(1)==ord('q'):
          break

if __name__=='__main__':
    main()
