import os
from PIL import Image
import shutil

def convert_to_grayscale(input_folder, output_folder):
    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Open an image file
            with Image.open(os.path.join(input_folder, filename)) as img:
                # Convert the image to grayscale
                gray_img = img.convert('L')
                # Save the grayscale image to the output folder with the same filename
                gray_img.save(os.path.join(output_folder, filename))
                print(f"Converted {filename} to grayscale and saved to {output_folder}")

# Define the input and output folders
train_input_folder = 'DIV2K_train_HR'
valid_input_folder = 'DIV2K_valid_HR'
train_output_folder = 'DIV2K_train_HR_bw'
valid_output_folder = 'DIV2K_valid_HR_bw'

# Convert images in both folders
convert_to_grayscale(train_input_folder, train_output_folder)
convert_to_grayscale(valid_input_folder, valid_output_folder)
