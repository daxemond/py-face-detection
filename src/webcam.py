import cv2
import sys
import face_recognition

import logging as log
import datetime as dt
from time import sleep

import numpy

image = face_recognition.load_image_file("desmond.jpg")
face_encoding = face_recognition.face_encodings(image)[0]
known_face_encodings = [face_encoding]
known_face_names = ["Desmond"]

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture("rtsp://des-cave-kitchen:xxxxxx-2014@192.168.0.247:554/stream2")
#video_capture = cv2.VideoCapture(1)
anterior = 0

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame
    ret, frame = video_capture.read()

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = numpy.ascontiguousarray(frame[:, :, ::-1])
    face_locations = face_recognition.face_locations(rgb_frame)
    if (len(face_locations) > 0):
        face_encodings = face_recognition.face_encodings(rgb_frame,face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)


       # Display the resulting frame
    cv2.imshow('Video', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

# When all shit is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
