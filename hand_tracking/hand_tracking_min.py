import cv2 as cv
import mediapipe as mp
import time

#Capturing webcam n,0
cap = cv.VideoCapture(0)

#Hand class
mpHands = mp.solutions.hands
#mpHands = mp.solutions.mediapipe.python.solutions.hands #Does not work, but shows the classes and such.
hands = mpHands.Hands(1,)
mpDraw = mp.solutions.drawing_utils

#Time managment
pTime = 0
CTime = 0


while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB) #Class object that contains the information of the hands. It takes only RGB format of images.
    #print(results.multi_hand_landmarks)

    #Tracking the hands
    if results.multi_hand_landmarks: #multi_hand_landmarks
        for handLms in results.multi_hand_landmarks:
            #Taking the index number and landmark of the landmarks. Then finding the pixel values for the landmark. Usefull for tracking thumb for example
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx , cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                #Draws on a single landmark
                if id == 0:
                    cv.circle(img, (cx, cy), 15, (0,255,0), cv.FILLED)
            #Drawing points on the hands, and connection lines between them
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


    #Calculating fps
    CTime = time.time()
    fps = 1/(CTime-pTime)
    pTime = CTime
    #Displaying
    cv.putText(img, str(int(fps)), (10, 70), cv.QT_FONT_NORMAL, 3, (0,255,0), 3)

    #Showing the cam
    cv.imshow("Image", img)
    cv.waitKey(1)