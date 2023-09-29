import cv2
import face_recognition
import os

known_faces = []
known_names = []

# Load the known images
for name in os.listdir("C:\\test"):
    for filename in os.listdir(f"C:\\test\\{name}"):
        image = face_recognition.load_image_file(f"C:\\test\\{name}\\{filename}")
        face_encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(face_encoding)
        known_names.append(name)

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_faces, face_encoding)

        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]

        print(f"Found {name} in the video!")

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
