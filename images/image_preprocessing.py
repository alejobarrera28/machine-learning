import numpy as np
import cv2
from skimage.feature import (
    hog,  # 3
    haar_like_feature,  # 4
    local_binary_pattern,  # 5
    # greycomatrix,  # 7
    # greycoprops,  # 7
)
from skimage.filters import gabor

# import mahotas
from sklearn.cluster import KMeans


# 1. Raw Pixels
def extract_raw_pixels(img):
    """Flatten the img into a 1D array."""
    return img.flatten().astype(np.float32) / 255.0


# 2. Color Histogram
def extract_color_histogram(img, bins=(16, 16, 16)):
    """Compute a 3D color histogram and flatten it."""
    # img assumed in BGR (OpenCV) or RGB
    hist = cv2.calcHist([img], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


# def color_histogram(img, bins=16):
#     """Compute normalized histogram for each channel and concatenate."""
#     chans = cv2.split(img)
#     hist = []
#     for ch in chans:
#         h, _ = np.histogram(ch, bins=bins, range=(0, 255))
#         hist.extend(h)
#     hist = np.array(hist, dtype=np.float32)
#     return hist / (hist.sum() + 1e-6)


# 3. HOG (Histogram of Oriented Gradients)
def extract_hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
    """Compute HOG feature vector for a grayscale img."""
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


# def hog_features(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
#     """Extract HOG (Histogram of Oriented Gradients) features."""
#     if img.ndim == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     feats = hog(
#         img,
#         orientations=orientations,
#         pixels_per_cell=pixels_per_cell,
#         cells_per_block=cells_per_block,
#         block_norm='L2-Hys',
#         transform_sqrt=True
#     )
#     return feats.astype(np.float32)


# 4. Haar-like Features
def extract_haar_features(img, feature_types=None):
    """Extract basic Haar-like features using skimg."""
    # Define default feature types
    if feature_types is None:
        feature_types = ["type-2-x", "type-2-y"]
    regions = None  # all regions
    feats = haar_like_feature(
        img,
        0,
        0,
        img.shape[0],
        img.shape[1],
        feature_types=feature_types,
        feature_region=regions,
    )
    return feats


# def haar_features(img, rects):
#     """
#     Compute Haar-like features given a list of rectangles.
#     rects: list of tuples (x, y, w, h)
#     """
#     if img.ndim == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     ii = cv2.integral(img)
#     feats = []
#     for (x, y, w, h) in rects:
#         A = ii[y, x]
#         B = ii[y, x + w]
#         C = ii[y + h, x]
#         D = ii[y + h, x + w]
#         feats.append(D - B - C + A)
#     return np.array(feats, dtype=np.float32)


# 5. LBP (Local Binary Patterns)
def extract_lbp(img, P=8, R=1):
    """Compute Local Binary Pattern histogram."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype("float")
    hist /= hist.sum() + 1e-6
    return hist


# def lbp_features(img, P=8, R=1, method='uniform'):
#     """Compute Local Binary Pattern histogram."""
#     if img.ndim == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     lbp = local_binary_pattern(img, P, R, method)
#     n_bins = P + 2 if method == 'uniform' else 2 ** P
#     hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
#     hist = hist.astype(np.float32)
#     return hist / (hist.sum() + 1e-6)


# 6. Gabor Filters
def extract_gabor_features(
    img, frequencies=[0.1, 0.2, 0.3], thetas=[0, np.pi / 4, np.pi / 2]
):
    """Apply a bank of Gabor filters and return the mean magnitude responses."""
    feats = []
    for freq in frequencies:
        for theta in thetas:
            real, imag = gabor(img, frequency=freq, theta=theta)
            magnitude = np.sqrt(real**2 + imag**2)
            feats.append(magnitude.mean())
            feats.append(magnitude.var())
    return np.array(feats)


# def gabor_features(img, frequencies=[0.1, 0.3, 0.5]):
#     """Extract Gabor filter responses (mean real & imag) for each frequency."""
#     if img.ndim == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     feats = []
#     for freq in frequencies:
#         real, imag = gabor(img, frequency=freq)
#         feats.append(np.mean(real))
#         feats.append(np.mean(imag))
#     return np.array(feats, dtype=np.float32)


# 7. Haralick Texture Features
def extract_haralick_features(img, distances=[1], angles=[0]):
    """Compute Haralick features (contrast, dissimilarity, homogeneity, energy, correlation)."""
    # Ensure grayscale uint8
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = img
    glcm = greycomatrix(
        img, distances=distances, angles=angles, levels=256, symmetric=True, normed=True
    )
    feats = []
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    for p in props:
        feats.append(greycoprops(glcm, prop=p).mean())
    return np.array(feats)


# def haralick_features(img, distances=[1], angles=[0], levels=256):
#     """Compute Haralick (GLCM) texture features."""
#     if img.ndim == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     glcm = greycomatrix(
#         img,
#         distances=distances,
#         angles=angles,
#         levels=levels,
#         symmetric=True,
#         normed=True
#     )
#     props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
#     feats = [greycoprops(glcm, prop)[0, 0] for prop in props]
#     return np.array(feats, dtype=np.float32)


# 8. SIFT + Bag of Visual Words
def extract_sift_bovw(img, codebook: KMeans):
    """Compute SIFT descriptors and encode as BoVW histogram via provided KMeans codebook."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kps, descs = sift.detectAndCompute(gray, None)
    if descs is None or len(descs) == 0:
        return np.zeros(codebook.n_clusters)
    labels = codebook.predict(descs)
    hist, _ = np.histogram(labels, bins=np.arange(codebook.n_clusters + 1))
    hist = hist.astype("float")
    hist /= hist.sum() + 1e-6
    return hist


# 9. SURF + Bag of Visual Words
def extract_surf_bovw(img, codebook: KMeans):
    """Compute SURF descriptors and encode as BoVW histogram via provided KMeans codebook."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create(400)
    kps, descs = surf.detectAndCompute(gray, None)
    if descs is None or len(descs) == 0:
        return np.zeros(codebook.n_clusters)
    labels = codebook.predict(descs)
    hist, _ = np.histogram(labels, bins=np.arange(codebook.n_clusters + 1))
    hist = hist.astype("float")
    hist /= hist.sum() + 1e-6
    return hist


# 10. ORB + Bag of Visual Words
def extract_orb_bovw(img, codebook: KMeans):
    """Compute ORB descriptors and encode as BoVW histogram via provided KMeans codebook."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    kps, descs = orb.detectAndCompute(gray, None)
    if descs is None or len(descs) == 0:
        return np.zeros(codebook.n_clusters)
    # convert binary descriptors to float for clustering
    descs_float = descs.astype("float")
    labels = codebook.predict(descs_float)
    hist, _ = np.histogram(labels, bins=np.arange(codebook.n_clusters + 1))
    hist = hist.astype("float")
    hist /= hist.sum() + 1e-6
    return hist


# class BoVW:
#     """
#     Bag-of-Visual-Words for SIFT, SURF, ORB.
#     Usage:
#       bovw = BoVW(method='SIFT', n_clusters=500)
#       bovw.fit(list_of_imgs)
#       hist = bovw.transform(img)
#     """
#     def __init__(self, method='SIFT', n_clusters=500):
#         self.n_clusters = n_clusters
#         self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0)
#         if method == 'SIFT':
#             self.detector = cv2.SIFT_create()
#         elif method == 'SURF':
#             self.detector = cv2.xfeatures2d.SURF_create()
#         elif method == 'ORB':
#             self.detector = cv2.ORB_create()
#         else:
#             raise ValueError("Unsupported method: choose 'SIFT', 'SURF', or 'ORB'")

#     def _descriptors(self, img):
#         if img.ndim == 3:
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         _, des = self.detector.detectAndCompute(img, None)
#         return des

#     def fit(self, imgs):
#         """Fit K-Means on descriptors from a list of imgs."""
#         all_desc = []
#         for img in imgs:
#             des = self._descriptors(img)
#             if des is not None:
#                 all_desc.append(des)
#         all_desc = np.vstack(all_desc)
#         self.kmeans.fit(all_desc)

#     def transform(self, img):
#         """Compute BoVW histogram for a single img."""
#         des = self._descriptors(img)
#         hist = np.zeros(self.n_clusters, dtype=np.float32)
#         if des is not None:
#             idx = self.kmeans.predict(des)
#             for i in idx:
#                 hist[i] += 1
#         return hist / (hist.sum() + 1e-6)
