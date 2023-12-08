from preprocess_fns import extract_features, extract_faces, resize_faces
import time
import os


def process_fake_videos():
    start = time.time()
    lst = os.listdir('./fake_videos')
    try:
        lst.remove('.DS_Store')
    except ValueError:
        pass
    skipped = []
    for i, f in enumerate(lst):
        print(f'File {i+1} of {len(lst)}')
        print(f'{skipped=}')
        file_path = os.path.join('./fake_videos', f)
        if os.path.isfile(file_path):
            extract_faces.extract_faces_from_video(file_path, 'extracted_faces_fake', skipped)
            resize_faces.resize('extracted_faces_fake', 'processed_faces_fake', f)
            extract_features.extract_lips('processed_faces_fake', 'extracted_lips_fake', f)
            extract_features.extract_eyes('processed_faces_fake', 'extracted_eyes_fake', f)
            extract_features.extract_eyebrows('processed_faces_fake', 'extracted_eyebrows_fake', f)
    end = time.time()
    time_taken = end - start
    print(f'{time_taken=}')

process_fake_videos()