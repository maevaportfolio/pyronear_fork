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
The project includes advanced image processing techniques with a focus on machine learning models for image enhancement. Here’s what each part of the project involves:

- **Image Preprocessing**: This file contains code for data transformation and augmentation, including rotations and transformations on images and labels.

This section explains how to use the image preprocessing functionality in the project. Ensure you have followed the installation instructions provided in the [Getting Started](#getting-started) section before proceeding.

### Running the Image Preprocessing Pipeline

The image preprocessing pipeline is designed to transform and augment images using specific operations such as rotations. This functionality is encapsulated in the `process_image` function, which is part of the `processing` module.

### Setting Up and Running the Pipeline

Follow these steps to get the image preprocessing pipeline up and running:

- **Super-resolution Model**: Utilizes a pre-trained ESRGAN model and a from-scratch ESRGAN model for enhancing image resolution.

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
Run on a notebook to visualize your images with data augmentation and transformation !

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


