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


def get_image_paths(dataset_path, extension):
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
    image_paths = get_image_paths(dataset_path, extension)
    image_dataset = dict()
    for image_path in image_paths:
        _, file_name = os.path.split(image_path)
        image_np = cv2.imread(image_path)
        image_dataset[file_name.split('.')[0]] = image_np
    return image_dataset


def dump_data(data, path):
    with open(path, 'wb') as file:
        pkl.dump(data, file, protocol=pkl.HIGHEST_PROTOCOL)


def load_data(path):
    with open(path, 'rb') as file:
        return pkl.load(file)


def kp_obj2np(all_keypoints):
    kp_np = dict()
    for detector, keypoints in all_keypoints.items():
        keypoints_to_list = list()
        for keypoint in keypoints:
            pt = (round(keypoint.pt[0]), round(keypoint.pt[1]))
            keypoints_to_list.append(pt)
        kp_np[detector] = np.array(keypoints_to_list)
        # keypoints_to_list.clear()
    return kp_np


def get_exec_time_keypoints(img):
    keypoints_by_detector = dict()
    execution_time = dict()
    all_detectors = dd.get_all_detectors()

    for name, _ in all_detectors.items():
        detector = dd.initialize_detector(name)
        start_time = default_timer()
        keypoints = detector.detect(img)
        execution_time[name] = default_timer() - start_time
        keypoints_by_detector[name] = keypoints
    return execution_time, keypoints_by_detector


# dd.print_dictionary(execution_time)

def get_exec_time_keypoints_det(image_set, detector_name):
    detector = dd.initialize_detector(detector_name)
    keypoints_by_image = dict()
    execution_time = dict()
    i = 0
    for image in image_set.values():
        start_time = default_timer()
        keypoints = detector.detect(image)
        execution_time[i] = default_timer() - start_time
        keypoints_by_image[i] = keypoints
        i += 1
    return execution_time, keypoints_by_image


def get_avg_exec_time_total_kp(image_set):
    avg_keypoints_by_detector = dict()
    avg_execution_time = dict()
    all_detectors = dd.get_all_detectors()
    num_images = len(image_set.values())
    for detector_name in all_detectors:
        avg_keypoints_by_detector[detector_name] = 0
        avg_execution_time[detector_name] = 0
    for img in image_set.values():
        execution_time, keypoints_by_detector = get_exec_time_keypoints(img)
        for detector_name in all_detectors:
            avg_keypoints_by_detector[detector_name] += len(keypoints_by_detector[detector_name])
            avg_execution_time[detector_name] += execution_time[detector_name]
    for detector_name in all_detectors:
        avg_keypoints_by_detector[detector_name] = avg_keypoints_by_detector[detector_name] // num_images
        avg_execution_time[detector_name] = avg_execution_time[detector_name] / num_images

    return avg_execution_time, avg_keypoints_by_detector
