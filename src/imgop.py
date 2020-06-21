# from skimage import data
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from timeit import default_timer
import src.detector_descriptor as dd

def cvkp2np(all_keypoints):

    kp_np = dict()
    for detector, keypoints in all_keypoints.items():
        keypoints_to_list = list()
        for keypoint in keypoints:
            pt = (round(keypoint.pt[0]), round(keypoint.pt[1]))
            keypoints_to_list.append(pt)
        kp_np[detector] = np.array(keypoints_to_list)

    return kp_np


def get_alldet_kp_et(img):
    keypoints_by_detector = dict()
    execution_time = dict()
    all_detectors = dd.get_all_detectors()

    for detector_name, _ in all_detectors.items():

        if detector_name is 'GFTT':
            detector = dd.initialize_detector(detector_name, additional_args='maxCorners=100000')
        elif detector_name is 'ORB':
            detector = dd.initialize_detector(detector_name, additional_args='nfeatures=100000')
        else:
            detector = dd.initialize_detector(detector_name)

        start_time = default_timer()
        keypoints = detector.detect(img)
        execution_time[detector_name] = default_timer() - start_time
        keypoints_by_detector[detector_name] = keypoints
    return execution_time, keypoints_by_detector


# dd.print_dictionary(execution_time)

def get_det_kp_et(image_set, detector_name):
    """
    Returns
    Args:
        image_set:
        detector_name:

    Returns:

    """
    if detector_name is 'GFTT':
        detector = dd.initialize_detector(detector_name, additional_args='maxCorners=100000')
    elif detector_name is 'ORB':
        detector = dd.initialize_detector(detector_name, additional_args='nfeatures=100000')
    else:
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


def get_det_avg_numkp_et(image_set):
    avg_keypoints_by_detector = dict()
    avg_execution_time = dict()
    all_detectors = dd.get_all_detectors()
    num_images = len(image_set.values())
    for detector_name in all_detectors:
        avg_keypoints_by_detector[detector_name] = 0
        avg_execution_time[detector_name] = 0
    for img in image_set.values():
        execution_time, keypoints_by_detector = get_alldet_kp_et(img)
        for detector_name in all_detectors:
            avg_keypoints_by_detector[detector_name] += len(keypoints_by_detector[detector_name])
            avg_execution_time[detector_name] += execution_time[detector_name]
    for detector_name in all_detectors:
        avg_keypoints_by_detector[detector_name] = avg_keypoints_by_detector[detector_name] // num_images
        avg_execution_time[detector_name] = avg_execution_time[detector_name] / num_images

    return avg_execution_time, avg_keypoints_by_detector
