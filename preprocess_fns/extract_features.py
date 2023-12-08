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

def extract_lips(input_folder, output_folder, img):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Initialize a progress bar
    image_filenames = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and img.replace('.mp4', '') in f]

    pbar = tqdm(total=len(image_filenames), desc="Extracting Lips")


    for filename in image_filenames:
        if filename:
        # if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and img.replace('.mp4', '') in filename:
            # print(filename)
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
    print()


def extract_eyes(input_folder, output_folder, img):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Initialize a progress bar
    image_filenames = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and img.replace('.mp4', '') in f]

    pbar = tqdm(total=len(image_filenames), desc="Extracting Eyes")

    for filename in image_filenames:
        if filename:
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = detector(gray)
            for face in faces:
                landmarks = predictor(gray, face)

                # Define eye feature indices
                left_eye_indices = list(range(36, 42))
                right_eye_indices = list(range(42, 48))

                # Extract eyes
                left_eye = extract_feature(image, landmarks, left_eye_indices)
                right_eye = extract_feature(image, landmarks, right_eye_indices)

                # Save the feature images
                left_eye_filename = os.path.join(output_folder, 'left_eye_' + filename)
                right_eye_filename = os.path.join(output_folder, 'right_eye_' + filename)

                cv2.imwrite(left_eye_filename, left_eye)
                cv2.imwrite(right_eye_filename, right_eye)

        pbar.update(1)

    pbar.close()
    print()


def extract_eyebrows(input_folder, output_folder, img):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Initialize a progress bar
    image_filenames = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and img.replace('.mp4', '') in f]

    pbar = tqdm(total=len(image_filenames), desc="Extracting Eyebrows")

    for filename in image_filenames:
        if filename:
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = detector(gray)
            for face in faces:
                landmarks = predictor(gray, face)

                # Define eyebrow feature indices
                left_eyebrow_indices = list(range(17, 22))
                right_eyebrow_indices = list(range(22, 27))

                # Extract eyebrows
                left_eyebrow = extract_feature(image, landmarks, left_eyebrow_indices)
                right_eyebrow = extract_feature(image, landmarks, right_eyebrow_indices)

                # Save the feature images
                left_eyebrow_filename = os.path.join(output_folder, 'left_eyebrow_' + filename)
                right_eyebrow_filename = os.path.join(output_folder, 'right_eyebrow_' + filename)

                cv2.imwrite(left_eyebrow_filename, left_eyebrow)
                cv2.imwrite(right_eyebrow_filename, right_eyebrow)

        pbar.update(1)

    pbar.close()
    print()


def main():
    pass


if __name__ == "__main__":
    main()