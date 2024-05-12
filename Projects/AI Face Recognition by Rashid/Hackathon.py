import cv2
import face_recognition
import time
from datetime import datetime

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
known_face_encodings = []
known_face_names = []
face_counter = 0
unique_face_counter = 0
last_seen_time = None
total_time_between_faces = 0

# Open the log file
with open('log.txt', 'w') as log_file:
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face detection processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(small_frame, model="cnn")
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            else:
                known_face_encodings.append(face_encoding)
                name = f"Person {unique_face_counter}"
                known_face_names.append(name)
                unique_face_counter += 1

            # Calculate the time between faces
            current_time = time.time()
            if last_seen_time is not None:
                time_between_faces = current_time - last_seen_time
                total_time_between_faces += time_between_faces
            last_seen_time = current_time

            # Write a log entry to the file
            log_file.write(f"{datetime.now()}: Detected face, ID: {name}\n")

            face_names.append(name)

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        # Draw the face counter at the bottom of the frame
        cv2.putText(frame, f"Faces seen: {unique_face_counter}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

# Calculate the average time between faces
average_time_between_faces = total_time_between_faces / unique_face_counter if unique_face_counter != 0 else 0

# Write the total number of unique people and the average time between faces to a text file
with open('output.txt', 'w') as f:
    f.write(f"Total number of unique people: {unique_face_counter}\n")
    f.write(f"Average time between new faces: {average_time_between_faces} seconds\n")

