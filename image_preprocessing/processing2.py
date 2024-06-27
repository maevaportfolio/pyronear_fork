####### Import Librairies ######

import math

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms as T
from torchvision import tv_tensors

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches

####### Fonctions ####

def load_labels(label_path):

    """
    Loads object detection labels and bounding box coordinates from a specified text file and converts them to PyTorch tensors.

    Args:
    label_path (str): The file path to the text file containing the labels and bounding box coordinates.
                      Each line in the file should contain class id, x_center, y_center, width, and height, all separated by spaces.

    Returns:
    tuple: A tuple containing two elements (labels, boxes). 
           'labels' is a tensor of integer class ids.
           'boxes' is a tensor of bounding box coordinates with each box represented by [x_center, y_center, width, height].
           Returns (None, None) if there is an error during file loading or processing.
    """

    labels = []
    boxes = []
    try:
        with open(label_path, 'r') as file:
            for line in file.readlines():
                cls, x_center, y_center, width, height = map(float, line.split())
                labels.append(int(cls))
                boxes.append([x_center, y_center, width, height])
    except Exception as e:
        print(f"Error loading labels: {e}")
        return None, None
    return torch.tensor(labels), torch.tensor(boxes)


def show_image_with_boxes(image, boxes, labels, class_names):
    """
    Displays the image with bounding boxes and labels.

    Args:
    image (PIL.Image or similar): The image object on which the bounding boxes and labels will be drawn.
    boxes (list of lists): A list of bounding boxes, each defined as [x_center, y_center, width, height].
    labels (list of int): List of label indices corresponding to the bounding boxes.
    class_names (list of str): List of class names indexed according to labels.

    Returns:
    None. Displays the image with annotated bounding boxes and labels.
    """

    # Error handling for invalid inputs
    if not hasattr(image, 'width') or not hasattr(image, 'height'):
        raise ValueError("Invalid image format: Image must have 'width' and 'height' attributes.")
    if len(boxes) != len(labels):
        raise ValueError("Mismatch between the number of boxes and labels.")
    if any(len(box) != 4 for box in boxes):
        raise ValueError("Each box must be a list of four floats.")
    if any(label >= len(class_names) or label < 0 for label in labels):
        raise ValueError("Label indices must be valid indices into class_names.")

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for box, label in zip(boxes, labels):
        try:
            # Convert the center x, y, width, height to xmin, ymin, xmax, ymax
            xmin = (box[0] - box[2] / 2) * image.width
            ymin = (box[1] - box[3] / 2) * image.height
            xmax = (box[0] + box[2] / 2) * image.width
            ymax = (box[1] + box[3] / 2) * image.height
            
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(xmin, ymin, class_names[label], color='blue', fontsize=12)
        except Exception as e:
            print(f"Error processing box {box} with label {label}: {e}")
            continue

    plt.show()




def rotate_box(box: torch.Tensor, angle: float, image_width: int, image_height: int) -> torch.Tensor:
    """Rotates a bounding box by 'angle' degrees.

    Args:
        box (tensor): The bounding box, format [x_center, y_center, width, height].
        angle (float): The rotation angle in degrees, counter-clockwise.
        image_width (int): Width of the image.
        image_height (int): Height of the image.

    Returns:
        tensor: The rotated bounding box.
    """
    # Convert angle to radians
    angle_rad = math.radians(angle)

    # Calculate the rotation matrix
    cos_val = math.cos(angle_rad)
    sin_val = math.sin(angle_rad)
    rotation_matrix = np.array([[cos_val, -sin_val], [sin_val, cos_val]])

    # Convert box to corners (top-left and bottom-right)
    x_center, y_center, width, height = box
    x_center_abs = x_center * image_width
    y_center_abs = y_center * image_height
    width_abs = width * image_width
    height_abs = height * image_height
    corners = np.array([
        [x_center_abs - width_abs / 2, y_center_abs - height_abs / 2],  # Top-left
        [x_center_abs + width_abs / 2, y_center_abs + height_abs / 2]   # Bottom-right
    ])

    # Move corners to origin (for rotation), rotate, and move back
    corners_centered = corners - np.array([image_width / 2, image_height / 2])
    corners_rotated = np.dot(corners_centered, rotation_matrix)
    corners_moved_back = corners_rotated + np.array([image_width / 2, image_height / 2])

    # Convert rotated corners back to box format
    x_min, y_min = corners_moved_back[0]
    x_max, y_max = corners_moved_back[1]
    new_width = x_max - x_min
    new_height = y_max - y_min
    new_x_center = x_min + new_width / 2
    new_y_center = y_min + new_height / 2

    # Normalize the coordinates
    new_box = [
        new_x_center / image_width, new_y_center / image_height,
        new_width / image_width, new_height / image_height
    ]
    return torch.tensor(new_box)


def apply_transforms(image, boxes):
    """Applies transformations to the image and adjusts the bounding boxes.

    Args:
    image (PIL.Image or similar): The image to which the transformations will be applied.
    boxes (list of lists): A list of bounding boxes associated with the image, where each box is defined as
                           a list [x_center, y_center, width, height].

    Returns:
    tuple: A tuple containing the transformed image and a tensor of the transformed bounding boxes.
    """

    # image rotation
    angle =  15 # we define the angle here
    rotated_image = TF.rotate(image, angle)

    # Ajust the bounding boxes
    transformed_boxes = [rotate_box(box, angle, image.width, image.height) for box in boxes]

    return rotated_image, torch.stack(transformed_boxes)



def process_image(image_path, label_path, class_names):
    """Process the image and display results.
    
     Args:
    image_path (str): Path to the image file.
    label_path (str): Path to the text file containing labels and bounding box data.
    class_names (list): A list of class names corresponding to label indices.

    Returns:
    None: This function does not return any value but displays images using matplotlib.
    """
    try:
        # Load the image
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Load the labels
    labels, boxes = load_labels(label_path)
    if labels is None or boxes is None:
        return

    # Display original image
    print("Original Image:")
    plt.imshow(image)
    plt.show()

    # Apply the transforms and show the image with the bounding boxes adjusted
    transformed_image, transformed_boxes = apply_transforms(image, boxes)
    print("Transformed Image with Bounding Boxes:")
    show_image_with_boxes(transformed_image, transformed_boxes, labels, class_names)