import cv2
import dlib
import numpy as np

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('.\shape_predictor_68_face_landmarks.dat')
face_rec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Load your reference image
ref_image = cv2.imread('desmond.jpg')
ref_image_rgb = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

# Detect faces in the reference image
ref_faces = detector(ref_image_rgb)
if len(ref_faces) == 0:
    print("No face found in the reference image.")
    exit()

# Extract face descriptor for the reference face
ref_shape = predictor(ref_image_rgb, ref_faces[0])
ref_descriptor = face_rec.compute_face_descriptor(ref_image_rgb, ref_shape)

# Open the default camera
cap = cv2.VideoCapture("rtsp://des-cave-kitchen:xxxxxx-2014@192.168.0.247:554/stream2")
#cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the current frame
    faces = detector(frame_rgb)

    for face in faces:
        # Extract face descriptor for the current face
        shape = predictor(frame_rgb, face)
        descriptor = face_rec.compute_face_descriptor(frame_rgb, shape)

        # Compute the Euclidean distance between the reference and current face descriptors
        distance = np.linalg.norm(np.array(ref_descriptor) - np.array(descriptor))

        # A smaller distance indicates more similarity
        if distance < 0.6:  # Threshold for similarity
            match_text = "Match"
            color = (0, 255, 0)  # Green for a match
        else:
            match_text = "No Match"
            color = (0, 0, 255)  # Red for no match

        # Draw a box around the face and display the match status
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, match_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
