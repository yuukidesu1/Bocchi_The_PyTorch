from PIL import Image, ImageOps
import os

# Create folders to save segmented images
output_directory_raster = "output_images/raster"
output_directory_label = "output_images/label"
os.makedirs(output_directory_raster, exist_ok=True)
os.makedirs(output_directory_label, exist_ok=True)

# Get the base paths for input images
base_path = "F:/PyCharm/Projects/UNet/DATASET/data_CitySegmentation/Berlin/raster/Berlin_{}_{}_raster.png"
base_label_path = "F:/PyCharm/Projects/UNet/DATASET/data_CitySegmentation/Berlin/label/Berlin_{}_{}_label.png"

num = 1
num_s = 1

# Iterate and split the images
for i in range(5):  # Rows
    for j in range(6):  # Columns
        # Build the paths for image and label
        image_path = base_path.format(i, j)
        label_path = base_label_path.format(i, j)

        # Open input image and label
        image = Image.open(image_path)
        label = Image.open(label_path)

        # Get the size of the input image
        width, height = image.size

        # Calculate the width and height of each block
        block_width = width // 4
        block_height = height // 4

        # Calculate the number of pixels needed for padding
        pad_width = (4 - width % 4) % 4
        pad_height = (4 - height % 4) % 4

        # Mirror pad the image
        image = ImageOps.expand(image, border=(pad_width // 2, pad_height // 2, pad_width - pad_width // 2, pad_height - pad_height // 2))
        label = ImageOps.expand(label, border=(pad_width // 2, pad_height // 2, pad_width - pad_width // 2, pad_height - pad_height // 2))

        # Get the new image size
        new_width, new_height = image.size

        # Initialize filename variable
        """if i == j == 0:
            filename_number = num
        else:
            filename_number = num_s"""
        filename_number = num_s
        num_s += 16

        # Iterate and split the image
        for m in range(4):  # Rows
            for n in range(4):  # Columns
                # Calculate the region for the split
                left = m * (new_width // 4)
                upper = n * (new_height // 4)
                right = left + (new_width // 4)
                lower = upper + (new_height // 4)

                # Crop the image block
                image_block = image.crop((left, upper, right, lower))
                label_block = label.crop((left, upper, right, lower))

                # Build the filename
                filename = f"{filename_number}.png"

                # Save the segmented image and label to their respective folders
                image_block.save(os.path.join(output_directory_raster, filename))
                label_block.save(os.path.join(output_directory_label, f"label_{filename}"))

                # Increment the filename variable for the next iteration
                filename_number += 1

print("Splitting complete and saved in folders:", output_directory_raster, "and", output_directory_label)
