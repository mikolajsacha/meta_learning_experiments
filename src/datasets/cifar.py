"""
CIFAR100 dataset
"""
from typing import Tuple
import numpy as np
from keras.datasets import cifar100


def load_cifar100() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # The data, shuffled and split between train and test sets:
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # we don't normalize data here. We normalize it separately for each Learner dataset using mean/std of its samples

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    return X_train, y_train, X_test, y_test


img_rows, img_cols = 32, 32  # input image dimensions
img_channels = 3  # The CIFAR100 images are RGB.

cifar_input_shape = (img_channels, img_rows, img_cols)
