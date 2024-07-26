import glob
import os
from PIL import Image

def create_gif_from_kernel_images(folder_path,pattern, target_dir ,output_filename='output.gif', duration=0.1):
    # Change to the specified directory
    os.chdir(folder_path)
    
    # Get all files matching the pattern 'kernel_*.png'
    file_pattern = pattern
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"No files found matching the pattern '{file_pattern}' in folder: {folder_path}")
        return

    # Sort files based on the numeric value in the filename
    sorted_files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Open all images
    images = []
    for file in sorted_files:
        try:
            img = Image.open(file)
            img = img.resize((512,512), Image.Resampling.NEAREST)
            images.append(img)
        except IOError:
            print(f"Error opening file {file}")

    if not images:
        print("No valid images found")
        return

    # Save the GIF
    try:
        images[0].save(
            f'{target_dir}/{output_filename}',
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
        print(f"GIF created successfully: {os.path.join(target_dir, output_filename)}")
    except Exception as e:
        print(f"Error creating GIF: {str(e)}")

# Specify the folder path and call the function
root =r'C:\Users\exhik\workspace\TANAKA-s_Lab'
target_dir = r'C:\Users\exhik\workspace\TANAKA-s_Lab\gifs_0712'
pairs = {'1ch_sparse_mask': r'checkpoints\ResNet_PCA_Sparse_Mask_1_ch_lambda_0.001\train_result', 
         '1_ch_normal' : r'checkpoints\ResNet_PCA_1_ch_lambda_0.001_new\train_result',
         '4_ch_sparse_mask': r'checkpoints\ResNet_PCA_Sparse_Mask_4_ch_lambda_0.001\train_result',
         '4_ch_normal': r'checkpoints\ResNet_PCA_4_ch_lambda_0.001\train_result'
         }
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
for name, path in pairs.items():
    create_gif_from_kernel_images(f'{root}\{path}',pattern = 'pca_kernel_*.png',target_dir=target_dir,output_filename=f'{name}_pca_kernels.gif', duration=100)
    create_gif_from_kernel_images(f'{root}\{path}',pattern = 'kernel_*.png',target_dir=target_dir, output_filename=f'{name}_kernels.gif', duration=100)