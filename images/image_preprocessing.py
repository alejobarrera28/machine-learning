import numpy as np
import cv2
from skimage.feature import (
    hog,
    haar_like_feature,
    local_binary_pattern,
)
from skimage.filters import gabor
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans


# Utility to handle single vs batch
def _apply_batch(fn, imgs: np.ndarray | list[np.ndarray], *args, **kwargs) -> np.ndarray:
    """
    Apply a feature-extraction function to a single image or a batch of images.

    Parameters:
        fn: Function to apply to each image. Should take (img, *args, **kwargs).
        imgs: Single image array (2D or 3D) or batch (list or 4D array).
        *args: Positional arguments to pass to fn.
        **kwargs: Keyword arguments to pass to fn.

    Returns:
        Feature array for a single image, or stacked array for a batch.
    """
    # Single image: 2D gray or 3D with channel dim 1,3,4
    if isinstance(imgs, np.ndarray) and (
        imgs.ndim == 2 or (imgs.ndim == 3 and imgs.shape[-1] in (1, 3, 4))
    ):
        return fn(imgs, *args, **kwargs)

    # Otherwise assume a batch
    outputs = [fn(im, *args, **kwargs) for im in imgs]
    return np.stack(outputs, axis=0)


def extract_raw_pixels(imgs: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """
    Flatten raw pixel values into a normalized feature vector.

    Parameters:
        imgs: Single image or batch of images.

    Returns:
        1D or 2D array of normalized pixel values in [0,1].
    """

    def _single(img):
        # Flatten and scale to [0,1]
        return img.flatten().astype(np.float32) / 255.0

    return _apply_batch(_single, imgs)


def extract_color_histogram(imgs: np.ndarray | list[np.ndarray], bins: tuple[int, int, int] = (16, 16, 16)) -> np.ndarray:
    """
    Compute a 3D color histogram over BGR channels.

    Parameters:
        imgs: Single image or batch of images.
        bins: Number of bins per channel (B, G, R).

    Returns:
        Flattened and normalized histogram feature vector.
    """

    def _single(img, bins):
        # Calculate 3D histogram and normalize
        hist = cv2.calcHist([img], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    return _apply_batch(_single, imgs, bins)


def extract_hog(imgs: np.ndarray | list[np.ndarray], pixels_per_cell: tuple[int, int] = (8, 8), cells_per_block: tuple[int, int] = (2, 2), orientations: int = 9) -> np.ndarray:
    """
    Extract Histogram of Oriented Gradients (HOG) features.

    Parameters:
        imgs: Single image or batch of images.
        pixels_per_cell: Size (in pixels) of a cell.
        cells_per_block: Number of cells in each block.
        orientations: Number of orientation bins.

    Returns:
        HOG descriptor vector or batch of descriptors.
    """

    def _single(img, pp_cell, cp_block, orient):
        # Convert to grayscale if needed
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        return hog(
            gray,
            orientations=orient,
            pixels_per_cell=pp_cell,
            cells_per_block=cp_block,
            block_norm="L2-Hys",
            transform_sqrt=True,
            feature_vector=True,
        )

    return _apply_batch(_single, imgs, pixels_per_cell, cells_per_block, orientations)


# 4. Haar-like Features
def extract_haar_features(imgs: np.ndarray | list[np.ndarray], feature_types: list[str] = None) -> np.ndarray:
    """
    Compute Haar-like features on the image.

    Parameters:
        imgs: Single image or batch of images.
        feature_types: Haar feature types, e.g. ['type-2-x'].

    Returns:
        Array of Haar-like feature values.
    """

    def _single(img):
        types = feature_types or ["type-2-x", "type-2-y"]
        feats = haar_like_feature(
            img, 0, 0, img.shape[0], img.shape[1], feature_types=types
        )
        return feats

    return _apply_batch(_single, imgs)


def extract_lbp(imgs: np.ndarray | list[np.ndarray], P: int = 8, R: float = 1) -> np.ndarray:
    """
    Compute Local Binary Pattern (LBP) histogram features.

    Parameters:
        imgs: Single image or batch of images.
        P: Number of circularly symmetric neighbor set points.
        R: Radius of circle.

    Returns:
        Normalized histogram of LBP patterns.
    """

    def _single(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        lbp = local_binary_pattern(gray, P, R, method="uniform")
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype("float")
        hist /= hist.sum() + 1e-6
        return hist

    return _apply_batch(_single, imgs)


def extract_gabor_features(
    imgs: np.ndarray | list[np.ndarray], frequencies: list[float] = [0.1, 0.2, 0.3], thetas: list[float] = [0, np.pi / 4, np.pi / 2]
) -> np.ndarray:
    """
    Extract Gabor filter response statistics (mean & variance).

    Parameters:
        imgs: Single image or batch of images.
        frequencies: Gabor frequencies.
        thetas: Gabor orientations.

    Returns:
        Feature vector of mean and variance for each filter.
    """

    def _single(img):
        feats = []
        for freq in frequencies:
            for theta in thetas:
                real, imag = gabor(img, frequency=freq, theta=theta)
                mag = np.sqrt(real**2 + imag**2)
                feats.extend([mag.mean(), mag.var()])
        return np.array(feats)

    return _apply_batch(_single, imgs)


def extract_haralick_features(imgs: np.ndarray | list[np.ndarray], distances: list[int] = [1], angles: list[float] = [0]) -> np.ndarray:
    """
    Compute Haralick texture features from Gray-Level Co-occurrence Matrix (GLCM).

    Parameters:
        imgs: Single image or batch of images.
        distances: Pixel-pair distance offsets.
        angles: Pixel-pair angles in radians.

    Returns:
        Array of Haralick feature values (contrast, dissimilarity,
        homogeneity, energy, correlation).
    """

    def _single(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        glcm = graycomatrix(
            gray,
            distances=distances,
            angles=angles,
            levels=256,
            symmetric=True,
            normed=True,
        )
        props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
        feats = [graycoprops(glcm, prop=p).mean() for p in props]
        return np.array(feats)

    return _apply_batch(_single, imgs)


def extract_sift_bovw(imgs: np.ndarray | list[np.ndarray], codebook: KMeans) -> np.ndarray:
    """
    Extract Bag-of-Visual-Words histogram using SIFT descriptors.

    Parameters:
        imgs: Single image or batch of images.
        codebook: Trained KMeans model as visual vocabulary.

    Returns:
        Normalized histogram of visual words.
    """

    def _single(img, cb):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kps, descs = sift.detectAndCompute(gray, None)
        if descs is None or len(descs) == 0:
            return np.zeros(cb.n_clusters)
        labels = cb.predict(descs)
        hist, _ = np.histogram(labels, bins=np.arange(cb.n_clusters + 1))
        hist = hist.astype("float")
        hist /= hist.sum() + 1e-6
        return hist

    return _apply_batch(_single, imgs, codebook)


def extract_surf_bovw(imgs: np.ndarray | list[np.ndarray], codebook: KMeans) -> np.ndarray:
    """
    Extract Bag-of-Visual-Words histogram using SURF descriptors.

    Parameters:
        imgs: Single image or batch of images.
        codebook: Trained KMeans model as visual vocabulary.

    Returns:
        Normalized histogram of visual words.
    """

    def _single(img, cb):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        surf = cv2.xfeatures2d.SURF_create(400)
        kps, descs = surf.detectAndCompute(gray, None)
        if descs is None or len(descs) == 0:
            return np.zeros(cb.n_clusters)
        labels = cb.predict(descs)
        hist, _ = np.histogram(labels, bins=np.arange(cb.n_clusters + 1))
        hist = hist.astype("float")
        hist /= hist.sum() + 1e-6
        return hist

    return _apply_batch(_single, imgs, codebook)


def extract_orb_bovw(imgs: np.ndarray | list[np.ndarray], codebook: KMeans) -> np.ndarray:
    """
    Extract Bag-of-Visual-Words histogram using ORB descriptors.

    Parameters:
        imgs: Single image or batch of images.
        codebook: Trained KMeans model as visual vocabulary.

    Returns:
        Normalized histogram of visual words.
    """

    def _single(img, cb):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create()
        kps, descs = orb.detectAndCompute(gray, None)
        if descs is None or len(descs) == 0:
            return np.zeros(cb.n_clusters)
        # ORB descriptors are uint8; convert to float for KMeans predict
        descs_float = descs.astype("float")
        labels = cb.predict(descs_float)
        hist, _ = np.histogram(labels, bins=np.arange(cb.n_clusters + 1))
        hist = hist.astype("float")
        hist /= hist.sum() + 1e-6
        return hist

    return _apply_batch(_single, imgs, codebook)
