import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np


def Draw_plot(x, y, figsize, x_label, y_label, title, color, marker, marksize, alpha, save_path):
    fig = plt.figure(figsize=figsize)
    plt.plot(x, y, marker=marker, color=color, alpha=alpha, markersize=marksize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if save_path:
        if '\\' in save_path:
            save_path = '/'.join(save_path.split('\\'))
        save_dir = '/'.join(save_path.split('/')[:-1])
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, format='svg')
    plt.show()

def image_to_histogram(image_path, figsize=(6, 4), title='Histogram of Original Image', color='gray', save_path=None):
    image_array = []
    if isinstance(image_path, list):
        for path in image_path:
            image_array.append(np.array(Image.open(path).getdata()).flatten())
        image_array = np.concatenate(image_array)
    elif isinstance(image_path, str):
        image_array = np.array(Image.open(image_path).getdata()).flatten()
    fig = plt.figure(figsize=figsize)
    plt.hist(image_array, bins=256, color=color)
    plt.title(title)
    if save_path:
        if '\\' in save_path:
            save_path = '/'.join(save_path.split('\\'))
        save_dir = '/'.join(save_path.split('/')[:-1])
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, format='svg')
    plt.show()


def images_embedding_rate_chi_square_p(chi_test: list | np.ndarray, embedding_rate: list | np.ndarray,
                                       figsize=(6, 4),
                                       x_label='embedding_rate',
                                       y_label='p-value',
                                       title='P of LSB Image', color='gray', marker='x', marksize=10, alpha=0.7, save_path=None):
    Draw_plot(embedding_rate, chi_test, figsize, x_label, y_label, title, color, marker, marksize, alpha, save_path)

def images_h_2i_histogram(image_paths: list,
                          embedding_rate: list | np.ndarray,
                          figsize=(6, 4),
                          x_label='embedding_rate',
                          y_label='|h_{2i} - h_{2i+1}|',
                          title='Mean Difference of LSB Image', color='gray',  marker='x', marksize=10, alpha=0.7, save_path=None):
    q = []  # mean(h_2i - h_2i+1), i from 0 to 127
    for path in image_paths:
        image_array = np.array(Image.open(path).getdata()).flatten()
        his, _ = np.histogram(image_array, bins=256)
        his = his.reshape(-1, 2)
        diff = np.abs(his[:, 0] - his[:, 1])
        q.append(np.mean(diff))

    Draw_plot(embedding_rate, q, figsize, x_label, y_label, title, color, marker, marksize, alpha, save_path)