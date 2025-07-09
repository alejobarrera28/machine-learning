import os
import numpy as np
from glob import glob
from PIL import Image


def load_image(path, as_gray=True):
    """
    Load an image and return it as a NumPy array.

    Args:
        path: Path to the image file as a string.
        as_gray: If True, convert to grayscale.

    Returns:
        Array of shape (height, width) with values normalized to [0, 1].
    """
    img = Image.open(path)
    if as_gray:
        img = img.convert('L')
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def load_all_images(as_gray=True):
    """
    Load all PNG images in the dataset folder (located alongside this file) and return data with labels.

    Assumes the script lives in 'coli20/' with 20 subfolders, each containing 72 images named 'obj{i}_{j}.png'.
    Labels are extracted from the filename prefix (e.g., 'obj3').

    Args:
        as_gray: If True, convert images to grayscale.

    Returns:
        X: NumPy array of shape (n_samples, height, width).
        y: List of labels (e.g., ['obj1', 'obj2', ...]).
    """

    base_dir = os.path.dirname(__file__)
    pattern = os.path.join(base_dir, '*', 'obj*__*.png')
    image_paths = sorted(glob(pattern))

    images = []
    labels = []
    for img_path in image_paths:
        images.append(load_image(img_path, as_gray))
        fname = os.path.basename(img_path)
        label = fname.split('_')[0]
        labels.append(label)

    X = np.stack(images, axis=0)
    return X, labels


def flatten_images(X):
    """
    Flatten images for clustering algorithms.

    Args:
        X: Array of images with shape (n_samples, height, width).

    Returns:
        Array of shape (n_samples, height * width).
    """
    n, h, w = X.shape
    return X.reshape(n, h * w)


def sample_images(X, y, n_samples=10, random_seed=None):
    """
    Randomly sample images and their labels.

    Args:
        X: Image array of shape (n_samples, height, width).
        y: Corresponding list of labels.
        n_samples: Number of images to sample.
        random_seed: Seed for reproducibility.

    Returns:
        Tuple of (X_sample, y_sample).
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    idx = np.random.choice(len(X), size=n_samples, replace=False)
    return X[idx], [y[i] for i in idx]
