import os
import pickle
import numpy as np


def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def get_data(n):
    if not (1 <= n <= 5):
        raise ValueError("Batch number 'n' must be between 1 and 5 (inclusive).")

    current_dir = os.path.dirname(__file__)
    batch_file = os.path.join(current_dir, f"data_batch_{n}")


    batch = unpickle(batch_file)

    data = batch[b"data"]  # Shape: (10000, 3072)
    labels = batch[b"labels"]  # List of 10000 labels

    # Reshape data to (10000, 3, 32, 32)
    data = data.reshape((10000, 3, 32, 32))
    # Transpose to (10000, 32, 32, 3) for visualization
    data = data.transpose(0, 2, 3, 1)

    return data, labels

def get_all_data():
    all_data = []
    all_labels = []
    for i in range(1, 6):
        data, labels = get_data(i)
        all_data.append(data)
        all_labels.extend(labels)

    all_data = np.concatenate(all_data, axis=0)
    return all_data, all_labels



def get_test_data():
    current_dir = os.path.dirname(__file__)
    batch_file = os.path.join(current_dir, f"test_batch")

    batch = unpickle(batch_file)

    data = batch[b"data"]  # Shape: (10000, 3072)
    labels = batch[b"labels"]  # List of 10000 labels

    # Reshape data to (10000, 3, 32, 32)
    data = data.reshape((10000, 3, 32, 32))
    # Transpose to (10000, 32, 32, 3)
    data = data.transpose(0, 2, 3, 1)

    return data, labels


# data, labels = get_data(1)
# i = 0
# plt.imshow(data[i])
# plt.title(f'Label: {labels[i]}')
# plt.axis('off')
# plt.show()
