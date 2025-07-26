import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def unpickle(file_path):
    """
    Load a CIFAR-10 batch file using pickle.

    Parameters:
        file_path (str): Path to the batch file.

    Returns:
        dict: Dictionary containing data and labels.
    """
    with open(file_path, "rb") as fo:
        data_dict = pickle.load(fo, encoding="bytes")
    return data_dict


def get_data(batch_number):
    """
    Load a single training batch of the CIFAR-10 dataset.

    Parameters:
        batch_number (int): Number of the batch to load (1 through 5).

    Returns:
        tuple:
            - data (np.ndarray): Array of shape (10000, 32, 32, 3) containing image data.
            - labels (list): List of 10000 integer labels.
    """
    if not (1 <= batch_number <= 5):
        raise ValueError("Batch number must be between 1 and 5.")

    current_dir = os.path.dirname(__file__)
    batch_path = os.path.join(current_dir, f"data_batch_{batch_number}")
    batch = unpickle(batch_path)

    data = batch[b"data"]  # Raw image data (10000, 3072)
    labels = batch[b"labels"]  # List of labels

    # Convert to shape (10000, 3, 32, 32)
    data = data.reshape((10000, 3, 32, 32))
    # Transpose to (10000, 32, 32, 3) for standard image format
    data = data.transpose(0, 2, 3, 1)

    return data, np.array(labels)


def get_all_data():
    """
    Load all five training batches and concatenate them.

    Returns:
        tuple:
            - all_data (np.ndarray): Array of shape (50000, 32, 32, 3).
            - all_labels (list): List of 50000 labels.
    """
    data_list = []
    label_list = []

    for i in range(1, 6):
        data, labels = get_data(i)
        data_list.append(data)
        label_list.extend(labels)

    all_data = np.concatenate(data_list, axis=0)
    return all_data, np.array(label_list)


def get_test_data():
    """
    Load the test batch of the CIFAR-10 dataset.

    Returns:
        tuple:
            - data (np.ndarray): Array of shape (10000, 32, 32, 3).
            - labels (list): List of 10000 labels.
    """
    current_dir = os.path.dirname(__file__)
    test_batch_path = os.path.join(current_dir, "test_batch")
    batch = unpickle(test_batch_path)

    data = batch[b"data"]
    labels = batch[b"labels"]

    data = data.reshape((10000, 3, 32, 32))
    data = data.transpose(0, 2, 3, 1)

    return data, np.array(labels)


def flatten_images(X):
    """
    Flatten images for clustering algorithms.

    Args:
        X: Array of images with shape (n_samples, height, width, colors).

    Returns:
        Array of shape (n_samples, height * width * colors).
    """
    n, h, w, c = X.shape
    return X.reshape(n, h * w * c)


def extract_images_pca(X, n_components=50, whiten=False):
    """
    Apply PCA to a stack of images.

    Args:
        X: NumPy array of images with shape (n_samples, height*width).
        n_components: Number of principal components to keep.
        whiten: Whether to whiten the components (scales components to unit variance).

    Returns:
        X_pca: Array of shape (n_samples, n_components) containing the PCA-transformed data.
        pca: The fitted sklearn.decomposition.PCA object (useful if you want explained variance, inverse transform, etc.).
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


def extract_images_tSNE(X, n_components=2, perplexity=30, random_state=0):
    """
    Apply t-SNE to a stack of images or feature vectors (e.g., HOG features).

    Args:
        X: Array of shape (n_samples, n_features) or (n_samples, height, width).
        n_components: Number of dimensions for the embedding (must be <=3 for 'barnes_hut').
        perplexity: t-SNE perplexity.
        random_state: Random seed for reproducibility.

    Returns:
        X_tsne: Array of shape (n_samples, n_components).
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


def normalize_data(X, mean=None, std=None):
    """
    Normalize data values by subtracting the mean and dividing by the standard deviation.

    Args:
        X (np.ndarray): Data array of shape (n_samples, n_features).
        mean (np.ndarray, optional): If provided, use this mean for normalization.
        std (np.ndarray, optional): If provided, use this std for normalization.

    Returns:
        np.ndarray or Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - If mean and std are provided: returns normalized data only.
            - If mean and std are not provided: returns normalized data, mean, and std.
    """
    if mean is None or std is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X_norm = (X - mean) / (std + 1e-7)
        return X_norm, mean, std
    else:
        X_norm = (X - mean) / (std + 1e-7)
        return X_norm

