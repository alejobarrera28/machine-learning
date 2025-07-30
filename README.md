# Machine Learning Algorithms - From Scratch Implementation

A comprehensive collection of machine learning algorithms implemented from scratch using NumPy, with extensive mathematical explanations and hands-on experiments using real datasets.

> **Focus:** Educational understanding over performance optimization. Each algorithm is built from the ground up to demonstrate core mathematical concepts.

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/alejobarrera28/machine-learning
cd machine-learning

# Install dependencies
pip install -r requirements.txt
```

### Usage
Navigate to any category folder and open the Jupyter notebooks to explore algorithm implementations with step-by-step explanations.

---

## 📚 Implemented Algorithms

###  Classification
- **Logistic Regression** - Multi-class with softmax, regularization (L1/L2)
- **Naive Bayes** - Probabilistic classification
- **Support Vector Machine** - Linear and kernel methods
- **Random Forest** - Ensemble decision trees
- **Gradient Boosting Machines** - Advanced boosting techniques

**Dataset:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) - 60,000 32x32 color images in 10 classes

### Regression
- **Linear Regression** - OLS with closed-form solution
- **Ridge Regression** - L2 regularization
- **Lasso Regression** - L1 regularization with feature selection

**Dataset:** [California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) - 20,640 housing samples with 8 features

### Clustering
- **K-Means** - Lloyd's algorithm with k-means++ initialization
- **Gaussian Mixture Models** - EM algorithm with full covariance
- **DBSCAN** - Density-based clustering with noise detection

**Dataset:** [COIL-20](http://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php) - 1,440 grayscale images of 20 objects

### Dimensionality Reduction
- **Principal Component Analysis (PCA)** - Eigenvalue decomposition
- **t-SNE** - Non-linear manifold learning

**Dataset:** [Olivetti Faces](https://scikit-learn.org/stable/datasets/real_world.html#olivetti-faces) - 400 face images (40 people × 10 images)

---

## 🛠️ Key Features

### **Comprehensive Image Processing**
Advanced feature extraction pipeline in `images/image_preprocessing.py`:
- **Basic:** Raw pixels, color histograms
- **Classical:** HOG, LBP, Gabor filters, Haralick texture
- **Advanced:** SIFT/SURF/ORB with Bag-of-Visual-Words

### **Standardized Algorithm Structure**
All algorithm notebooks follow a consistent educational template:
- **Algorithm Name & Description** - Clear introduction with key concepts
- **Visual Overview** - Diagrams illustrating the algorithm's approach
- **Mathematical Foundation** - Core equations and theoretical background
- **Algorithm Steps** - Step-by-step breakdown of the implementation
- **Advanced Topics** - Extensions and deeper mathematical concepts
- **Key Characteristics** - Advantages, limitations, and use cases


### **Real Dataset Integration**
- **Preprocessing pipelines** tailored to each dataset
- **Feature scaling and normalization** utilities
- **Train/test splits** with proper evaluation metrics
- **Visualization outputs** saved for each algorithm

---

## 🔍 Algorithm Details

### Implementation Philosophy
- **From-scratch implementations** using only NumPy for core algorithms
- **Educational focus** - prioritizes understanding over efficiency
- **Mathematical rigor** - includes derivations and geometric intuitions
- **Practical integration** - works with real-world datasets

### Evaluation Metrics
- **Classification:** Accuracy, cross-entropy loss
- **Clustering:** Adjusted Rand Index (ARI)
- **Regression:** Mean Squared Error (MSE), R²

---

## 📖 Additional Resources

- **`optimizers.md`** - Comprehensive guide to optimization algorithms (SGD, Adam, L-BFGS, etc.)
- **Visualization outputs** - Each algorithm saves plots showing results

---

## 🤝 Contributing

This is an educational project focused on understanding ML fundamentals. Feel free to:
- Add new algorithm implementations
- Improve mathematical explanations
- Enhance visualizations
- Add more comprehensive examples

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).