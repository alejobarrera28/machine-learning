import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def unpickle(file_path: str) -> dict[bytes, any]:
    """
    Load a CIFAR-10 batch file using pickle.

    Parameters:
        file_path: Path to the batch file.

    Returns:
        Dictionary containing data and labels.
    """
    with open(file_path, "rb") as fo:
        data_dict = pickle.load(fo, encoding="bytes")
    return data_dict


def get_data(batch_number: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a single training batch of the CIFAR-10 dataset.

    Parameters:
        batch_number: Number of the batch to load (1 through 5).

    Returns:
        Tuple containing:
            - data: Array of shape (10000, 32, 32, 3) containing image data.
            - labels: Array of 10000 integer labels.
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


def get_all_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Load all five training batches and concatenate them.

    Returns:
        Tuple containing:
            - all_data: Array of shape (50000, 32, 32, 3).
            - all_labels: Array of 50000 labels.
    """
    data_list = []
    label_list = []

    for i in range(1, 6):
        data, labels = get_data(i)
        data_list.append(data)
        label_list.extend(labels)

    all_data = np.concatenate(data_list, axis=0)
    return all_data, np.array(label_list)


def get_test_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Load the test batch of the CIFAR-10 dataset.

    Returns:
        Tuple containing:
            - data: Array of shape (10000, 32, 32, 3).
            - labels: Array of 10000 labels.
    """
    current_dir = os.path.dirname(__file__)
    test_batch_path = os.path.join(current_dir, "test_batch")
    batch = unpickle(test_batch_path)

    data = batch[b"data"]
    labels = batch[b"labels"]

    data = data.reshape((10000, 3, 32, 32))
    data = data.transpose(0, 2, 3, 1)

    return data, np.array(labels)


def flatten_images(X: np.ndarray) -> np.ndarray:
    """
    Flatten images for clustering algorithms.

    Args:
        X: Array of images with shape (n_samples, height, width, colors).

    Returns:
        Array of shape (n_samples, height * width * colors).
    """
    n, h, w, c = X.shape
    return X.reshape(n, h * w * c)


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


def normalize_data(X: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize data values by subtracting the mean and dividing by the standard deviation.

    Args:
        X: Data array of shape (n_samples, n_features).
        mean: If provided, use this mean for normalization.
        std: If provided, use this std for normalization.

    Returns:
        If mean and std are provided: returns normalized data only.
        If mean and std are not provided: returns normalized data, mean, and std.
    """
    if mean is None or std is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X_norm = (X - mean) / (std + 1e-7)
        return X_norm, mean, std
    else:
        X_norm = (X - mean) / (std + 1e-7)
        return X_norm

