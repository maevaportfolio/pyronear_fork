## 📝 Table of Contents
- [About the Project](#about-the-project)
- [Getting Started](#getting-started)
  - [Import Library](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Upcoming Features](#upcoming-features)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

## 🌐 About the Project

This project incorporates a variety of advanced image processing techniques, specifically designed to enhance the capabilities of the Pyronear project for forest fire detection. Here’s an overview of what each part of the project involves:

- **Image Preprocessing : Transformation and Augmentation**: These techniques artificially expand the dataset with modified versions of existing images through operations such as rotations, scaling, and flipping. This is crucial for training robust machine learning models. The pipeline can make rotation on the images but also the labels.

- **Super-resolution**:Enhances the resolution of input images, which is particularly beneficial for improving the quality of images in the Roboflow dataset that includes video game captures from low-resolution devices in forested areas. This improvement helps make our models more robust, enhancing their ability to detect smoke and other indicators of forest fires effectively

- **Hyperparameter Tuning**: Optimizes the performance of machine learning models by systematically searching for the most effective combination of parameters. This process ensures that the models perform optimally under various conditions.
Here, we are going to use Random search.

These components are integrated into the Pyronear project to enhance its effectiveness in detecting forest smoke, ultimately aiming to provide faster and more reliable fire detection solutions.

---

## **1️⃣ Image Preprocessing**:
This file contains code for data transformation and augmentation, including rotations and transformations on images and labels.

This section explains how to use the image preprocessing pipeline. Ensure you have followed the installation instructions provided in the [Getting Started](#Getting-Started) section before proceeding.

### Running the Image Preprocessing Pipeline

The image preprocessing pipeline is designed to transform and augment images using specific operations such as rotations. This functionality is encapsulated in the `process_image` function, which is part of the `processing` module.

### Setting Up and Running the Pipeline

Follow these steps to get the image preprocessing pipeline up and running:

## 🚀 Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Installation: Clone the Repository ⚙️
First, clone the repository to your local machine to get the necessary files. Run the following command in your terminal:

```
git https://github.com/hi-paris/pyronear.git
```

#### Import Library 📦
What things you need to install the software and how to install them:

```
pip install -r requirements.txt
```

## 🛠 Usage
Run the following notebook to visualize your images with data augmentation and transformation!

```
# Import the process_image function from the processing module
from processing import process_image

# Define the paths for your image and label
image_path = "images/aiformankind_v1_000007.jpg"
label_path = "images/aiformankind_v1_000007.txt"

# Define the class names associated with your labels
class_names = ["smoke"]

# Process the image and possibly display it, depending on your function's implementation
process_image(image_path, label_path, class_names)
```
---

## **2️⃣ Super-resolution Model**: ESRGAN
Utilizes a pre-trained ESRGAN model and a from-scratch ESRGAN model for enhancing image resolution.

### From Scratch 


### Pre-trained (Tensorflow)

