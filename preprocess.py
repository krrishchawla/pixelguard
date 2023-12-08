from preprocess_fns import extract_features, extract_faces, resize_faces
import time
import os

# start = time.time()

# extract_faces.extract_faces_from_video("aaqaifqrwn.mp4")

# resize_faces.resize('extracted_faces', 'processed_faces')

# input_folder = 'processed_faces'  # Folder with 256x256 face images

# extract_features.extract_eyes(input_folder, 'extracted_eyes')
# extract_features.extract_eyebrows(input_folder, 'extracted_eyebrows')
# extract_features.extract_lips(input_folder, 'extracted_lips')

# end = time.time()

# time_taken = end - start
# print(f'{time_taken=}')

def process_real_videos():
    start = time.time()
    for filename in os.listdir('./real_videos'):
        file_path = os.path.join('./real_videos', filename)
        if os.path.isfile(file_path):
            extract_faces.extract_faces_from_video(file_path, 'extracted_faces_real')
            resize_faces.resize('extracted_faces_real', 'processed_faces_real')
            extract_features.extract_lips('processed_faces_real', 'extracted_lips_real')
    end = time.time()
    time_taken = end - start
    print(f'{time_taken=}')

process_real_videos()



