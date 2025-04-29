import os

import torch
from PIL import Image
import tifffile
# Define directories
input_dir = "Fluo-N2DL-HeLa/01"  # Replace with your input directory
output_dir = "SingleParticleImages/interphase-control"  # Replace with your output directory

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define crop regions (list of tuples: (image_index, left, upper, right, lower))
# image_index starts at 0 for the first image
crop_regions = [
    (0, 440,490,150,210),  # xlow,xhigh,ylow,yhigh
    (0, 264,300,280,320),  # xlow,xhigh,ylow,yhigh
    (0, 355,440,610,660),  # xlow,xhigh,ylow,yhigh
    (0, 330,380,110,170),  # xlow,xhigh,ylow,yhigh
    (1,490,530,130,180),
    (1,100,140,440,190),
    (1,360,410,620,660),
    (3,510,550,410,455)
]


# Function to process and crop images
def process_images(input_dir, output_dir, crop_regions):
    # Get all .tif files in the input directory
    tif_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.tif')]
    # Process each image with its index
    for img_idx, file_name in enumerate(tif_files):
        # Load the image
        image_path = os.path.join(input_dir, file_name)
        try:
            #img = Image.open(image_path)
            array = tifffile.imread(image_path)
            print(f"Processing {file_name} (index {img_idx})")

            # Find crop regions for this image
            relevant_crops = [region for region in crop_regions if region[0] == img_idx]

            # Crop each relevant region
            for crop_idx, region in enumerate(relevant_crops):
                # Extract crop coordinates (skip the image_index)

                left,right,lower, upper = region[1:5]


                # Crop the image
                #cropped_img = img.crop((left, upper, right, lower))
                cropped_img=array[lower:upper,left:right]
                # Define output file name
                base_name = os.path.splitext(file_name)[0]

                output_file = os.path.join(output_dir, f"{base_name}_crop_{crop_idx + 1}.pt")

                # Save the cropped image
                torch.save(cropped_img,output_file)
                #cropped_img.save(output_file, format="TIFF")
                print(f"Saved crop {crop_idx + 1} for {file_name} at {output_file}")

            #img.close()  # Close the image to free memory

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

process_images(input_dir,output_dir,crop_regions)