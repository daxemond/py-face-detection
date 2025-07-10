import cv2
import sys
import face_recognition

import logging as log
import datetime as dt
from time import sleep


# Load the image file
image = face_recognition.load_image_file("desmond.jpg")

# Find all face locations in the image
face_locations = face_recognition.face_locations(image)

# Load the image into a numpy array for OpenCV
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Loop through each face found in the image
for (top, right, bottom, left) in face_locations:
    # Draw a box around the face
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

# Display the image with the face boxes
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

