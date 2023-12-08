import json
import os
import random

data_path = "" #Replace with data path 
real_video_path = ""
fake_video_path = ""
output_path = ""
def extract_video():
    metadata1_path = os.path.join(data_path, "metadata.json")   
    with open(metadata1_path, "r") as f:
        data = json.load(f)
    real_videos = []    
    fake_videos = []
    for filename, video_data in data.items():
        video_path = os.path.join(data_path, filename)
        label = video_data["label"]
        if label == "REAL":
            real_videos.append((filename, label))
            os.rename(video_path, os.path.join(real_video_path, filename))
        else:
            fake_videos.append((filename, label))

    random.shuffle(fake_videos)
    selected_fake_videos = fake_videos[:200]
    processed_videos = real_videos + selected_fake_videos

    output_data = {}
    for filename, label in processed_videos:
        output_data[filename] = {"label": label}

    with open(os.path.join(output_path, "processed_data.json"), "a") as f:
        json.dump(output_data, f, indent=4)

    for filename, _ in selected_fake_videos:
        video_path = os.path.join(data_path, filename)
        output_video_path = os.path.join(fake_video_path, filename)
        os.rename(video_path, output_video_path)
extract_video()
