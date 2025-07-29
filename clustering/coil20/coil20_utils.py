import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load_image(path: str, as_gray: bool = True) -> np.ndarray:
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
        img = img.convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def load_all_images(as_gray: bool = True) -> tuple[np.ndarray, list[str]]:
    """
    Load all PNG images in the 'data/' folder and return data with labels.

    Labels are extracted from the filename (e.g., 'obj3_45.png' -> 'obj3').

    Args:
        as_gray: If True, convert images to grayscale.

    Returns:
        Tuple containing:
            - X: NumPy array of images with shape (n_samples, height, width).
            - y: List of labels (e.g., ['obj1', 'obj2', ...]).
    """
    image_paths = sorted(glob("coil20/data/*.png"))

    images = []
    labels = []
    for path in image_paths:
        img = Image.open(path)
        if as_gray:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        images.append(np.array(img))

        fname = os.path.basename(path)
        label = fname.split("_")[0]
        labels.append(label)

    X = np.stack(images, axis=0)
    return X, labels


def flatten_images(X: np.ndarray) -> np.ndarray:
    """
    Flatten images for clustering algorithms.

    Args:
        X: Array of images with shape (n_samples, height, width).

    Returns:
        Array of shape (n_samples, height * width).
    """
    n, h, w = X.shape
    return X.reshape(n, h * w)


def extract_images_pca(X: np.ndarray, n_components: int = 50, whiten: bool = False) -> tuple[np.ndarray, PCA]:
    """
    Apply PCA to a stack of images.

    Args:
        X: NumPy array of images with shape (n_samples, height*width).
        n_components: Number of principal components to keep.
        whiten: Whether to whiten the components (scales components to unit variance).

    Returns:
        Tuple containing:
            - X_pca: Array of shape (n_samples, n_components) containing the PCA-transformed data.
            - pca: The fitted sklearn.decomposition.PCA object.
    """

    if X.ndim == 3:
        n, h, w = X.shape
        X = X.reshape(n, h * w)

    pca = PCA(
        n_components=n_components,
        whiten=whiten,
        svd_solver="randomized",
        random_state=0,
    )
    X_pca = pca.fit_transform(X)

    return X_pca, pca


def extract_images_tsne(X: np.ndarray, n_components: int = 2, perplexity: int = 30, random_state: int = 0) -> np.ndarray:
    """
    Apply t-SNE to a stack of images or feature vectors (e.g., HOG features).

    Args:
        X: Array of shape (n_samples, n_features) or (n_samples, height, width).
        n_components: Number of dimensions for the embedding (must be <=3 for 'barnes_hut').
        perplexity: t-SNE perplexity.
        random_state: Random seed for reproducibility.

    Returns:
        Array of shape (n_samples, n_components).
    """
    if X.ndim == 3:
        n, h, w = X.shape
        X = X.reshape(n, h * w)

    method = "exact" if n_components > 3 else "barnes_hut"

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        method=method,
    )

    return tsne.fit_transform(X)
