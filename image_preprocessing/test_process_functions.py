import pytest
from image_preprocessing.processing2 import process_image, load_labels, apply_transforms, show_image_with_boxes
from PIL import Image
import torch

def test_load_labels():
    # Assuming load_labels returns labels and boxes
    labels, boxes = load_labels("images/aiformankind_v1_000007.txt")
    assert labels is not None
    assert boxes is not None
    assert isinstance(labels, torch.Tensor)
    assert isinstance(boxes, torch.Tensor)
    # Add more specific checks, like checking for the expected number of labels and boxes

def test_load_labels_with_invalid_path():
    # Test error handling
    labels, boxes = load_labels("non_existent_file.txt")
    assert labels is None
    assert boxes is None

def test_apply_transforms():
    # Create a dummy image and dummy boxes
    image = Image.new('RGB', (100, 100))
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    transformed_image, transformed_boxes = apply_transforms(image, boxes)
    assert transformed_image is not None
    assert transformed_boxes is not None
    assert len(transformed_boxes) == len(boxes)
    # More detailed checks can include verifying specific transformations

@pytest.fixture
def sample_image():
    return Image.new('RGB', (100, 100))

def test_show_image_with_boxes(sample_image):
    # This is mainly for demonstration as show_image_with_boxes does not return anything
    # Could use a library like pytest-mock or unittest.mock to check calls and effects
    show_image_with_boxes(sample_image, torch.tensor([[0.5, 0.5, 0.2, 0.2]]), [0], ['class1'])
    # Assertions would normally check side effects or use mock objects


