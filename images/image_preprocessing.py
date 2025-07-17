import numpy as np
import cv2
from skimage.feature import (
    hog,  # 3
    haar_like_feature,  # 4
    local_binary_pattern,  # 5
)
from skimage.filters import gabor
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans


# Utility to handle single vs batch
def _apply_batch(fn, imgs, *args, **kwargs):
    # If imgs is a single image (2D gray, or 3D with 1/3/4 channels), just apply fn
    if isinstance(imgs, np.ndarray) and (
        imgs.ndim == 2 or (imgs.ndim == 3 and imgs.shape[-1] in (1, 3, 4))
    ):
        return fn(imgs, *args, **kwargs)
    # Otherwise assume batch: list or 4D‐array (N,H,W[,C])
    outputs = [fn(im, *args, **kwargs) for im in imgs]
    return np.stack(outputs, axis=0)


# 1. Raw Pixels
def extract_raw_pixels(imgs):
    def _single(img):
        return img.flatten().astype(np.float32) / 255.0

    return _apply_batch(_single, imgs)


# 2. Color Histogram
def extract_color_histogram(imgs, bins=(16, 16, 16)):
    def _single(img):
        hist = cv2.calcHist([img], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    return _apply_batch(_single, imgs, bins)


# 3. HOG
def extract_hog(imgs, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
    def _single(img, pixels_per_cell, cells_per_block, orientations):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        return hog(
            gray,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm="L2-Hys",
            transform_sqrt=True,
            feature_vector=True,
        )

    return _apply_batch(_single, imgs, pixels_per_cell, cells_per_block, orientations)


# 4. Haar-like Features
def extract_haar_features(imgs, feature_types=None):
    def _single(img):
        types = feature_types or ["type-2-x", "type-2-y"]
        feats = haar_like_feature(
            img, 0, 0, img.shape[0], img.shape[1], feature_types=types
        )
        return feats

    return _apply_batch(_single, imgs)


# 5. LBP
def extract_lbp(imgs, P=8, R=1):
    def _single(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        lbp = local_binary_pattern(gray, P, R, method="uniform")
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype("float")
        hist /= hist.sum() + 1e-6
        return hist

    return _apply_batch(_single, imgs, P, R)


# 6. Gabor Filters
def extract_gabor_features(
    imgs, frequencies=[0.1, 0.2, 0.3], thetas=[0, np.pi / 4, np.pi / 2]
):
    def _single(img):
        feats = []
        for freq in frequencies:
            for theta in thetas:
                real, imag = gabor(img, frequency=freq, theta=theta)
                mag = np.sqrt(real**2 + imag**2)
                feats.extend([mag.mean(), mag.var()])
        return np.array(feats)

    return _apply_batch(_single, imgs, frequencies, thetas)


# 7. Haralick Texture Features
def extract_haralick_features(imgs, distances=[1], angles=[0]):
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

    return _apply_batch(_single, imgs, distances, angles)


# 8. SIFT + BoVW
def extract_sift_bovw(imgs, codebook: KMeans):
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


# 9. SURF + BoVW
def extract_surf_bovw(imgs, codebook: KMeans):
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


# 10. ORB + BoVW
def extract_orb_bovw(imgs, codebook: KMeans):
    def _single(img, cb):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create()
        kps, descs = orb.detectAndCompute(gray, None)
        if descs is None or len(descs) == 0:
            return np.zeros(cb.n_clusters)
        descs_float = descs.astype("float")
        labels = cb.predict(descs_float)
        hist, _ = np.histogram(labels, bins=np.arange(cb.n_clusters + 1))
        hist = hist.astype("float")
        hist /= hist.sum() + 1e-6
        return hist

    return _apply_batch(_single, imgs, codebook)
