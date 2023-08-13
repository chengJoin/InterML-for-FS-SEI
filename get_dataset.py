import torch
import numpy as np
import random
from sklearn.model_selection import train_test_split


def load_val_dataset(num):
    x_val = np.load(f"../dataset/X_val_{num}Class.npy")
    y_val = np.load(f"../dataset/Y_val_{num}Class.npy")
    y_val = y_val.astype(np.uint8)
    return x_val, y_val


def load_train_dataset_k_shot(num, k_shot):
    x = np.load(f"../dataset/X_train_{num}Class.npy")
    y = np.load(f"../dataset/Y_train_{num}Class.npy")
    y = y.astype(np.uint8)
    random_index_shot = []
    for i in range(num):
        index_shot = [index for index, value in enumerate(y) if value == i]
        random_index_shot += random.sample(index_shot, k_shot)
    random.shuffle(random_index_shot)
    x_train_k_shot = x[random_index_shot, :, :]
    y_train_k_shot = y[random_index_shot]
    return x_train_k_shot, y_train_k_shot


def load_test_dataset(num):
    x = np.load(f"../dataset/X_test_{num}Class.npy")
    y = np.load(f"../dataset/Y_test_{num}Class.npy")
    y = y.astype(np.uint8)
    return x, y
