import cv2 as cv
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, mode=False, maxHands = 2, complexity = 1, detection_confidence = 0.75, track_confidence = 0.5): #dCon around 0.75 gives best general results
        self.mode = mode
        self.maxHands = maxHands
        self.dCon = detection_confidence
        self.tCon = track_confidence

        self.mpHands = mp.solutions.hands
        #self.mpHands = mp.solutions.mediapipe.python.solutions.hands #Does not work, but shows the classes and such.
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, min_detection_confidence= self.dCon, min_tracking_confidence=self.tCon)
        self.mpDraw = mp.solutions.drawing_utils
    
    def findHands(self, img, draw = True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks: #multi_hand_landmarks
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self, img, handNO = 0, draw = True):
        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNO]
            for id, lm in enumerate(myHand.landmark):
                        h, w, c = img.shape
                        cx , cy = int(lm.x*w), int(lm.y*h)
                        lmList.append([id, cx, cy])
                        #Draws on a single landmark
                        if draw:
                            cv.circle(img, (cx, cy), 15, (0,255,0), cv.FILLED)
                        #Drawing points on the hands, and connection lines between them
        return lmList   

def main():
    pTime = 0
    CTime = 0
    cap = cv.VideoCapture(0)

    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        LmList = detector.findPosition(img)
        print(LmList[4])
        
        #Calculating fps
        CTime = time.time()
        fps = 1/(CTime-pTime)
        pTime = CTime
        #Displaying
        cv.putText(img, str(int(fps)), (10, 70), cv.QT_FONT_NORMAL, 3, (0,255,0), 3)

        #Showing the cam
        cv.imshow("Image", img)
        cv.waitKey(1)
        

if __name__ == '__main__':
    main()