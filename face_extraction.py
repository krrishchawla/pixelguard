import cv2
import dlib
import os
from tqdm import tqdm

# Initialize face detector
detector = dlib.get_frontal_face_detector()

# Function to extract and save faces from a frame
def extract_and_save_faces(frame, frame_number, video_path, output_folder="extracted_faces"):
    # Convert frame to grayscale (Dlib works with grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = detector(gray)
    for i, face in enumerate(faces):
        # Get the bounding box of the face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_region = frame[y:y+h, x:x+w]

        # Save the face image
        video_name = os.path.basename(video_path).replace('.mp4', '')
        face_filename = f"{output_folder}/{video_name}_{frame_number}_{i}.jpg"
        cv2.imwrite(face_filename, face_region)

# Process a video file
def process_video(video_path):
    # Create a directory to save face images
    if not os.path.exists("extracted_faces"):
        os.mkdir("extracted_faces")

    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Process each frame with a progress bar
    with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Extract and save faces
            extract_and_save_faces(frame, frame_number, video_path)

            frame_number += 1
            pbar.update(1)

    cap.release()

# Replace with the path to your video
process_video("aaqaifqrwn.mp4")