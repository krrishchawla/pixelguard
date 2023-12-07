from preprocess_fns import extract_features, extract_faces, resize_faces
import time

start = time.time()

extract_faces.extract_faces_from_video("aaqaifqrwn.mp4")

resize_faces.resize('extracted_faces', 'processed_faces')

input_folder = 'processed_faces'  # Folder with 256x256 face images

extract_features.extract_eyes(input_folder, 'extracted_eyes')
extract_features.extract_eyebrows(input_folder, 'extracted_eyebrows')
extract_features.extract_lips(input_folder, 'extracted_lips')

end = time.time()

time_taken = end - start
print(f'{time_taken=}')



