"""Facilitates visualization of images from a list of image file paths."""

import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def visualize_images(image_paths, k):
    """
    Extract and visualize k images from the list of image paths.

    :param image_paths: List of image file paths.
    :param k: Number of images to visualize.
    """
    # Ensure k is not greater than the number of images available
    # Randomly select k images from the list
    images = extract_images_from_folder(image_paths)
    selected_images = random.sample(images, k)

    # Set up the grid for displaying images
    cols = min(k, 3)  # Set a maximum of 3 columns
    rows = (k // cols) + (k % cols > 0)

    plt.figure(figsize=(15, 5 * rows))

    for i, img_path in enumerate(selected_images):
        # Load the image
        img = Image.open(img_path)

        # Display the image in a subplot
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Image {i + 1}")

    # Show the grid of images
    plt.tight_layout()
    plt.show()


def extract_images_from_folder(folder_path, output_folder=None):
    """Extract images from a folder and optionally copy them to an output folder."""
    if not Path(folder_path).exists():
        msg = f"The folder '{folder_path}' does not exist."
        raise FileNotFoundError(msg)

    if output_folder and not Path(output_folder).exists():
        Path(output_folder).mkdir(parents=True)

    image_files = []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        image_files.append(file_path)
        if output_folder:
            # Optionally, copy the image to the output folder
            output_path = os.path.join(output_folder, file_name)
            with Path.open(file_path, "rb") as fsrc, Path.open(output_path, "wb") as fdst:
                fdst.write(fsrc.read())

    return image_files


def plot_description_image_target(folder_path_images, descritpion_path, index):
    """Visualize the description of a image with the target."""
    images = extract_images_from_folder(folder_path_images)
    dataset = pd.read_csv(descritpion_path)
    descritpions = dataset["description"].values
    objs = dataset["object"].values
    descritpion = descritpions[index]
    image_name = dataset["image_name"].values[index]
    obj = objs[index]
    i = 0
    # Check if the image name are the same
    while images[i][21:] != image_name:
        i += 1
        image = images[i]
    plt.figure(figsize=(20, 20))
    img = Image.open(image)
    print(f"Object:{obj}\nDescription: {descritpion}")
    plt.imshow(img)

    # Show the grid of images
    plt.tight_layout()
    plt.show()
