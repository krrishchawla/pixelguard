import cv2
import os
import numpy as np
from tqdm import tqdm

def resize_and_pad_image(image_path, output_size=(256, 256)):
    # Read the image
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # Calculate padding
    top_pad = (output_size[1] - h) // 2
    bottom_pad = output_size[1] - h - top_pad
    left_pad = (output_size[0] - w) // 2
    right_pad = output_size[0] - w - left_pad

    # Pad and/or resize the image
    if h > output_size[1] or w > output_size[0]:
        # Resize if larger than output size
        image = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)
    else:
        # Pad with zeros (black) if smaller
        image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return image

def resize(input_folder, output_folder, img):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Get all image filenames
    image_filenames = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and img.replace('.mp4', '') in f]
    # print(image_filenames)

    # Process each image in the folder with a progress bar
    with tqdm(total=len(image_filenames), desc="Resizing Images", unit="image") as pbar:
        for filename in image_filenames:
            image_path = os.path.join(input_folder, filename)
            processed_image = resize_and_pad_image(image_path)

            # Save the processed image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, processed_image)

            # Update progress bar
            pbar.update(1)
    print()

# Define your input and output folders
# input_folder = 'extracted_faces'  # Folder where extracted faces are saved
# output_folder = 'processed_faces'  # Folder to save resized/padded faces

# process_folder(input_folder, output_folder)


def main():
    pass

if __name__ == "__main__":
    main()
