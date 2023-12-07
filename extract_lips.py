import cv2
import dlib
import numpy as np
import os
from tqdm import tqdm

# Initialize dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def extract_feature(image, landmarks, feature_indices):
    # Create a mask for the feature
    mask = np.zeros_like(image)
    points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in feature_indices], np.int32)
    cv2.fillPoly(mask, [points], (255, 255, 255))

    # Extract the feature
    feature = cv2.bitwise_and(image, mask)

    return feature

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Initialize a progress bar
    pbar = tqdm(total=len(os.listdir(input_folder)))

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = detector(gray)
            for face in faces:
                landmarks = predictor(gray, face)

                # Define feature indices (example: extracting lips)
                lips_indices = list(range(48, 61))
                lips = extract_feature(image, landmarks, lips_indices)

                # Save the feature image
                feature_filename = os.path.join(output_folder, 'lips_' + filename)
                cv2.imwrite(feature_filename, lips)

        pbar.update(1)

    pbar.close()

# Paths to your input and output folders
input_folder = 'processed_faces'  # Folder with 256x256 face images
output_folder = 'extracted_features'  # Folder to save extracted features

process_images(input_folder, output_folder)
