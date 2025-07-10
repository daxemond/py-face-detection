import multiprocessing
from time import sleep, time
from datetime import datetime, timezone
import cv2
import numpy
import sys
import face_recognition

def clear_queue(q):
    while not q.empty():
        q.get()
        q.task_done()


def receive_messages(tq,notify):
    image = face_recognition.load_image_file("desmond.jpg")
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings = [face_encoding]
    known_face_names = ["Desmond"]
    while True:
        if not tq.empty():
            #print(f"count: {tq.qsize()}")
            while not tq.empty():
                rgb_frame = tq.get()
            clear_queue(tq)
            #print(f"count after: {tq.qsize()}")
            face_locations = face_recognition.face_locations(rgb_frame)
            if (len(face_locations) > 0):
                name = "Unknown"
                face_encodings = face_recognition.face_encodings(rgb_frame,face_locations)
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                now = datetime.now()
                current_time = now.strftime("%Y-%m-%d %H:%M:%S")
                print(f"{current_time},find name={name}")
                notify.put(name)
            #print(f"Received: {face_locations}")
        sleep(2)

if __name__ == "__main__":
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    video_capture = cv2.VideoCapture("rtsp://des-cave-kitchen:xxxxxx-xxxx@192.168.0.245:554/stream2")
    #video_capture = cv2.VideoCapture(1)
    anterior = 0

    queue = multiprocessing.JoinableQueue(maxsize=0)
    notify = multiprocessing.JoinableQueue(maxsize=0)
    receiver = multiprocessing.Process(target=receive_messages, args=(queue,notify,))
    receiver.start()
    #receiver.join()

    start_time = time()
    name = "Unknown"
    while True:
        if not video_capture.isOpened():
            print('Unable to load camera.')
            sleep(5)
            pass

        # Capture frame
        ret, frame = video_capture.read()
        if ret == False:
            pass

        if frame is None:
            pass

        t_frame = frame[:, :, ::-1]
        rgb_frame = numpy.ascontiguousarray(t_frame)
        current_time = time()
        if current_time - start_time >= 2:
            queue.put(rgb_frame)
            start_time = current_time

        faces = faceCascade.detectMultiScale(
            t_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        while not notify.empty():
            name = notify.get()

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x + 6, y + h + 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        # Display the resulting frame
        cv2.imshow('Video', frame)
        key = cv2.waitKey(1) & 0xFF   
        if key == ord('q'):
            break

    # When all is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

