import cv2
import sys
import face_recognition

import logging as log
import datetime as dt
from time import sleep

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture("rtsp://des-cave-kitchen:xxxxxx-2014@192.168.0.248:554/stream2")
#video_capture = cv2.VideoCapture(1)
anterior = 0

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame
    ret, frame = video_capture.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = frame[:, :, ::-1]
    faces = faceCascade.detectMultiScale(
        rgb_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))

    dim = (1550,850)
    #Resize image for tapo
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break

    if key == ord('s'):
        cv2.imwrite('desmond1.jpg',rgb_frame)
        print("image saved as 'desmond.jpg'")

    # Display the resulting frame
    #cv2.imshow('Video', frame)

# When all shit is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
