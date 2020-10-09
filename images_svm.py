# Train multiple images per person
# Find and recognize faces in an image using a SVC with scikit-learn

import os
import os.path
import pickle
import face_recognition

from sklearn import svm
from PIL import Image, ImageDraw
from train_svm import train, image_files_in_folder

def predict(X_img_path, svm_clf=None, model_path=None):
    # Throw exception when the model is empty
    if svm_clf is None and model_path is None:
        raise Exception("Must supply svm classifier either thourgh svm_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if svm_clf is None:
        with open(model_path, 'rb') as f:
            svm_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if pred else ("unknown", loc) for pred, loc in zip(svm_clf.predict(faces_encodings), X_face_locations)]

def show_prediction_labels_on_image(img_path, predictions):
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    pil_image.show()    

if __name__ == "__main__":
    # Using the trained classifier, make predictions for unknown images
    for image_file in image_files_in_folder("./blob/test"):
        print("Looking for faces in {}".format(image_file))

        # Find all people in the image using a trained classifier model
        # Note: You can pass in either a classifier file name or a classifier model instance
        predictions = predict(image_file, model_path="./models/trained_svm_model.clf")

        # Print results on the console
        for name, (top, right, bottom, left) in predictions:
            print("- Found {} at ({}, {})".format(name, left, top))

        # Display results overlaid on an image
        show_prediction_labels_on_image(image_file, predictions)
