import cv2

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the reference image
ref_image = cv2.imread('desmond.jpg')

# Convert the reference image to grayscale
gray_ref = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)

# Detect faces in the reference image
ref_faces = face_cascade.detectMultiScale(gray_ref, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Ensure a face is found in the reference image
if len(ref_faces) == 0:
    print("No face found in the reference image.")
    exit()

# Extract the face region from the reference image
(x, y, w, h) = ref_faces[0]
ref_face = gray_ref[y:y+h, x:x+w]

# Resize the reference face to a fixed size
ref_face_resized = cv2.resize(ref_face, (100, 100))
# Open the default camera
cap = cv2.VideoCapture("rtsp://des-cave-kitchen:xxxxxx-2014@192.168.0.247:554/stream2")
#cap = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the current frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region from the current frame
        face = gray_frame[y:y+h, x:x+w]

        # Resize the face to the same size as the reference face
        face_resized = cv2.resize(face, (100, 100))

        # Calculate the Mean Squared Error (MSE) between the faces
        mse = ((ref_face_resized - face_resized) ** 2).mean()
        print(f"Mean Squared Error between faces: {mse}")

        # A smaller MSE value indicates more similarity
        if mse < 50:  # Arbitrary threshold for similarity
            match_text = "Match"
            color = (0, 255, 0)  # Green for a match
        else:
            match_text = "No Match"
            color = (0, 0, 255)  # Red for no match

        # Draw a box around the face and display the match status
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, match_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
