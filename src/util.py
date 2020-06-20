import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src import data as dt
import os


def highlight_max(data, color='#5fba7d'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)


def highlight_min(data, color='#d65f5f'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_min = data == data.min()
        return [attr if v else '' for v in is_min]
    else:  # from .apply(axis=None)
        is_max = data == data.min().min()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)


def get_image_set(path, image_prefix='all'):
    all_images = dt.load_data(path)
    if image_prefix is 'all':
        return all_images
    else:
        image_set = {}
        for key, value in all_images.items():
            if image_prefix in key:
                image_set[key] = value
        return image_set


def show_image_set(images):
    fig, axs = plt.subplots(len(images) // 3, 3, figsize=(10, 5), sharex=True, sharey=True)
    i = 0
    for img in images.values():
        axs[i // 3, i % 3].imshow(img)
        i += 1
    plt.show()