import face_recognition
import cv2
import numpy as np
import pickle
import os
import os.path

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Initialize some variables
process_this_frame = True

def predict(frame, svm_clf=None, model_path=None):
    if svm_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh svm_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if svm_clf is None:
        with open(model_path, 'rb') as f:
            svm_clf = pickle.load(f)

    # Load image file and find face locations
    X_face_locations = face_recognition.face_locations(frame)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(frame, known_face_locations=X_face_locations)

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if pred else ("unknown", loc) for pred, loc in zip(svm_clf.predict(faces_encodings), X_face_locations)]

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all people in the image using a trained classifier model
        # Note: You can pass in either a classifier file name or a classifier model instance
        predictions = predict(small_frame, model_path="./models/trained_svm_model.clf")

    process_this_frame = not process_this_frame

    # Print results on the console
    for name, (top, right, bottom, left) in predictions:
        print("- Found {} at ({}, {})".format(name, left, top))

        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        # name = name.encode("UTF-8")

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
