import pickle as pkl
import cv2
import os
import numpy as np
from src import detector_descriptor as dd
from timeit import default_timer

# Change current directory to the dataset folder
# os.chdir('..')
# os.chdir(os.path.join(os.getcwd(), 'dataset'))
# print(os.getcwd())


def get_paths(dataset_path, extension):
    """
    Returns a list of file paths ending in specified file extension(s) (`extension`)

    Args:
        dataset_path(`str`): Path to the dataset/folder.
        extension(`str` or `str tuple`): File extension or a tuple of file extensions.

    Returns:
        (`list`): A list of file paths ending in specified file extension(s).

    Examples:

        .. code-block:: python

        In[1]: from src.data import get_paths_by_extension
        In[2]: get_paths_by_extension('oxford', ('.pgm', '.ppm'))
        Out[3]: ['D:\\Programming Projects\\python projects\\state-of-the-binary-descriptor\\dataset\\oxford\\bark_img1.ppm',
                 'D:\\Programming Projects\\python projects\\state-of-the-binary-descriptor\\dataset\\oxford\\bark_img2.ppm',
                 'D:\\Programming Projects\\python projects\\state-of-the-binary-descriptor\\dataset\\oxford\\bark_img3.ppm',
                 'D:\\Programming Projects\\python projects\\state-of-the-binary-descriptor\\dataset\\oxford\\bark_img4.ppm',
                 'D:\\Programming Projects\\python projects\\state-of-the-binary-descriptor\\dataset\\oxford\\bark_img5.ppm',
                 'D:\\Programming Projects\\python projects\\state-of-the-binary-descriptor\\dataset\\oxford\\bark_img6.ppm',
                 'D:\\Programming Projects\\python projects\\state-of-the-binary-descriptor\\dataset\\oxford\\bikes_img1.ppm',
                 'D:\\Programming Projects\\python projects\\state-of-the-binary-descriptor\\dataset\\oxford\\bikes_img2.ppm']

    """
    path_list = list()
    for file in os.listdir(dataset_path):
        if file.endswith(extension):
            path_list.append(os.path.join(dataset_path, file))
    return path_list


def load_images(dataset_path, extension):
    """
    Loads all the images of type "extension" e.g. .ppm. .jpg, etc. from the given path to the dataset.
    Args:
        dataset_path(`str`): Path to the dataset directory.
        extension(`str`): Image extension.

    Returns:
        (`list`): A list of numpy arrays generated from the images.

    """
    image_paths = get_paths(dataset_path, extension)
    image_dataset = dict()
    for image_path in image_paths:
        _, file_name = os.path.split(image_path)
        image_np = cv2.imread(image_path)
        image_dataset[file_name.split('.')[0]] = image_np
    return image_dataset


def load_labels(dataset_path, extension):
    """
    Loads the labels e.g. homography between images.
    Args:
        dataset_path(`str`): Path to the dataset directory.
        extension(`str`): Image extension.

    Returns:
        (`list`): A list of numpy arrays containing the labels.
    """
    label_paths = get_paths(dataset_path, extension)
    label_dataset = dict()
    for label_path in label_paths:
        _, file_name = os.path.split(label_path)
        label_np = np.loadtxt(label_path)
        image_set_name = file_name.split('_')[0]
        image_num = file_name.split('to')[1][0]
        label_dataset[f'{image_set_name}_img{image_num}'] = label_np
    return label_dataset

def dump_data(data, path):
    """
    Saves data as a `.pckl` file in the given directory.
    Args:
        data(`any variable type`): Contains the data.
        path(`str`): Path to the file.

    """
    with open(path, 'wb') as file:
        pkl.dump(data, file, protocol=pkl.HIGHEST_PROTOCOL)


def load_data(path):
    """
    Loads data from a file path.
    Args:
        path(`str`): Path of the file.
    """
    with open(path, 'rb') as file:
        return pkl.load(file)



