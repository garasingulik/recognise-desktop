import os
import re
import os.path
import pickle
import face_recognition
import cv2
import math

from sklearn import svm
from PIL import Image, ImageDraw
from tools.files import image_files_in_folder, video_files_in_folder
from tools.rotate_image import get_rotation_with_ffprobe, rotate_frame

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            print("Processing image: {}".format(img_path))

            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(
                        face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(
                    image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

        # Loop through each training video for the current person
        for vid_path in video_files_in_folder(os.path.join(train_dir, class_dir)):
            print("Processing video: {}".format(vid_path))

            # requires ffprobe
            rotation = get_rotation_with_ffprobe(vid_path)

            input_movie = cv2.VideoCapture(vid_path)
            length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_number = 0

            while True:
                # Grab a single frame of video
                ret, frame = input_movie.read()
                frame_number += 1
                
                # Quit when the input video file ends
                if not ret:
                    break

                # Process every 15 frames
                if (frame_number % 15 != 0):
                  continue

                print("Processing video: {} frame {} of {}".format(vid_path, frame_number, length))

                # Rotate video if rotation value != 0
                frame = rotate_frame(frame, rotation)
                face_bounding_boxes = face_recognition.face_locations(frame)

                # We can remove next line after we validate the frame extraction is good
                cv2.imwrite("{}-frame_{}.jpg".format(vid_path, frame_number), frame)

                if len(face_bounding_boxes) != 1:
                    # If there are no people (or too many people) in a training image, skip the image.
                    if verbose:
                        print("Frame {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(
                            face_bounding_boxes) < 1 else "Found more than one face"))
                else:
                    # Add face encoding for current image to the training set
                    X.append(face_recognition.face_encodings(frame, known_face_locations=face_bounding_boxes)[0])
                    y.append(class_dir)

            input_movie.release

    # Create and train the SVC classifier
    svm_clf = svm.SVC(gamma='scale',class_weight='balanced')
    svm_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(svm_clf, f)

    return svm_clf


if __name__ == "__main__":
    # Train the SVM classifier and save it to disk
    # Once the model is trained and saved, you can skip this step next time.
    print("Training SVM classifier...")
    classifier = train(
        "./blob/train", model_save_path="./models/trained_svm_model.clf")
    print("Training complete!")
