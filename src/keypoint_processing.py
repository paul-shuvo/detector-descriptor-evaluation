import numpy as np
from timeit import default_timer
import src.detector_descriptor as dd
from src import util
from itertools import chain
import cv2

def get_unique_kpnp(kp_all):
    """
    Returns all unique numpy keypoints from a dictionary containing groups of cv keypoints.
    Args:
        kp_all(`dict`):  A dictionary containing groups of Opencv keypoint object.

    Returns:
        (`ndarray`): All unique numpy keypoints extracted from a dictionary containing groups of keypoints.

    Examples:
        .. code-block:: python


    See Also:
        :func:`~src.keypoint_processing`

    """
    kpnp_all = cvkp2np_all(kp_all)
    kpnp_unique = np.array(list(chain(*[value.tolist() for value in kpnp_all.values()])))
    kpnp_unique = np.unique(kpnp_unique, axis=0)
    return kpnp_unique


def get_kpnp_frequency(kpnp_all, kpnp_unique):
    """
    Returns frequency for all numpy type keypoints.
    Args:
        kpnp_all(`dict`): Contains grouped numpy keypoints. These groups are formed either by detectors, or images.
        kpnp_unique(`ndarray`): A set of all the unique keypoints found in `kpnp_all`.

    Returns:
        (`ndarray`): Contains keypoint locations and corresponding frequency for all numpy type keypoints.
        Shape is `(nx3) -> (keypoint_x, keypoint_y, frequency)`.
    """
    # pt_freq contains frequency value for each keypoint
    pt_freq = np.zeros((kpnp_unique.shape[0], 1))
    for i in range(0, kpnp_unique.shape[0]):
        for key in kpnp_all.keys():
            if kpnp_unique[i] in kpnp_all[key]:
                pt_freq[i] += 1
    # pt_freq is added as a column to the numpy keypoint array.
    kpnp_unique_freq = np.hstack((kpnp_unique, pt_freq))
    return kpnp_unique_freq


# def get_kpnp_by_frequency(kpnp_freq, kpnp_unique, frequency):
#     # kpnp_freq = get_kpnp_frequency(kpnp_all, kpnp_unique)
#     print(kpnp_freq.keys())
#     index_matched = np.where(kpnp_freq[:, 2] == frequency)
#     kpnp_by_frequency = kpnp_unique[index_matched]
#     return kpnp_by_frequency

def get_kpnp_by_frequency(kpnp_all, kpnp_unique, frequency):
    """
    Get all the numpy keypoints based on it frequency value.
    Args:
        kpnp_all(`dict`): Contains grouped numpy keypoints. These groups are formed either by detectors, or images.
        kpnp_unique(`ndarray`): A set of all the unique keypoints found in `kpnp_all`.
        frequency(`int`): The query frequency value.

    Returns:
        (`ndarray`): Contains keypoints for that corresponds to a certain frequency.
    """
    # get frequency for all numpy type keypoints
    if isinstance(kpnp_all, dict):
        kpnp_all_freq = get_kpnp_frequency(kpnp_all, kpnp_unique)
    else:
        kpnp_all_freq = kpnp_all
    # find index of numpy keypoints where frequency is matched.
    # print(type(kpnp_all_freq))
    index_matched = np.where(kpnp_all_freq[:,2] == frequency)
    # retrieve numpy keypoints by indices found in previous step.
    kpnp_filtered_by_frequency = kpnp_unique[index_matched]
    return kpnp_filtered_by_frequency


def get_kpnp_frequency_det_sigma(image, detector_name, sigma_values):
    kp_all = {}
    for sigma in sigma_values:
        if sigma is 0:
            image_blur = image
        else:
            ksize = np.int(np.round(((((sigma - 0.8)/0.3) + 1)/0.5)+1))
            if ksize % 2 == 0:
                ksize += 1
            image_blur = cv2.GaussianBlur(image,(ksize, ksize),0)
        kp_all[sigma] = get_kp(image_blur, detector_name)
    kpnp_all = cvkp2np_all(kp_all)
    kpnp_unique = get_unique_kpnp(kp_all)
    kpnp_all_frequency = get_kpnp_frequency(kpnp_all, kpnp_unique)
    return kpnp_unique, kpnp_all_frequency


def cvkp2np(keypoints, round_=True):
    """
    Converts an array of opencv keypoint object to numpy ndarray that contains
    all the keypoint location.

    .. important:
        The locations of the keypoints are rounded.

    Args:
        keypoints(`obj`): OpenCV keypoint object.

    Returns:
        (`ndarray`): An ndarray of `dtype=int` conatining the `x, y` locations of the
        keypoints.

        Examples:
        .. code-block:: python

            In[1]: from src.keypoint_processing import *
            In[2]: import cv2
            In[3]: image = cv2.imread('.\\tests\\lena.jpg')
            In[4]: kp = get_kp(image, 'AGAST')
            In[5]: type(kp[0])
            Out[5]: cv2.KeyPoint
            In[6]: kpnp = cvkp2np(kp)
            In[7]: type(kpnp)
            Out[7]: numpy.ndarray

    """
    keypoints_to_list = list()
    for keypoint in keypoints:
        if round_:
            pt = (round(keypoint.pt[0]), round(keypoint.pt[1]))
        else:
            pt = (keypoint.pt[0], keypoint.pt[1])
        keypoints_to_list.append(pt)
    return np.array(keypoints_to_list)


def cvkp2np_all(keypoints_all):
    """
    Converts a list of opencv keypoint object to `ndarray` and returns it.
    Args:
        keypoints_all(`dict`): Contains grouped numpy keypoints. These groups are formed either by detectors, or images.

    Returns:
        (`dict`): A `dict` containing converted OpenCV keypoints to `ndarray`s
        of `dtype=int` for different groups(can be different detector or image).
    """
    kp_np = dict()
    for key, keypoints in keypoints_all.items():
        # key can be either detector name or image name
        kp_np[key] = cvkp2np(keypoints)

    return kp_np

def get_kp(image, detector_name):
    """
    Extracts keypoints from an image using a specified detector.
    Args:
        image(`ndarray`): The query image.
        detector_name(`str`): The name of the detector that'll be used to extract keypoints.

    Examples:
        .. code-block:: python

            In[1]: from src.keypoint_processing import *
            In[2]: import cv2
            In[3]: image = cv2.imread('.\\tests\\lena.jpg')
            In[4]: kp = get_kp(image, 'AGAST')
            In[5]: type(kp[0])
            Out[5]: cv2.KeyPoint
    Returns:
        (`obj`): Opencv keypoint objects.
    """
    if detector_name is 'GFTT':
        # maxCorners is set high enough, so that all the keypoints can be retrieved.
        detector = dd.initialize_detector(detector_name, additional_args='maxCorners=15000')
    elif detector_name is 'ORB':
        # nfeatures is set high enough, so that all the keypoints can be retrieved.
        detector = dd.initialize_detector(detector_name, additional_args='nfeatures=100000')
    else:
        detector = dd.initialize_detector(detector_name)
    keypoints = detector.detect(image)
    return keypoints


def get_det_kp_et(image_set, detector_name):
    """
    Extract keypoints and compute execution time for all images in image set for a specified feature detector.
    Args:
        image_set(`dict`): Contains all the images for an image sequence.
        detector_name(`str`): The name of the detector.

    Returns:
        (`list`): Execution time, extracted keypoints for all the images
    """

    keypoints_by_image = dict()
    execution_time = dict()
    i = 0
    for image in image_set.values():
        start_time = default_timer()
        keypoints = get_kp(image, detector_name)
        execution_time[i] = default_timer() - start_time
        keypoints_by_image[i] = keypoints
        i += 1
    return execution_time, keypoints_by_image


def get_desc_by_det(image, detector_name, descriptor_name):
    """
    
    Args:
        image:
        detector_name:
        descriptor_name:

    Returns:

    """
    descriptor = dd.initialize_descriptor(descriptor_name)
    kp = get_kp(image, detector_name)
    desc = descriptor.compute(image, kp)
    return desc


def get_desc(image, kp, descriptor_name):
    descriptor = dd.initialize_descriptor(descriptor_name)
    desc = descriptor.compute(image, kp)
    return desc


def get_alldet_kp_et(image):
    keypoints_by_detector = dict()
    execution_time = dict()
    all_detectors = dd.get_all_detectors()

    for detector_name, _ in all_detectors.items():

        start_time = default_timer()
        keypoints = get_kp(image, detector_name)
        execution_time[detector_name] = default_timer() - start_time
        keypoints_by_detector[detector_name] = keypoints
    # change it to dict return type
    return execution_time, keypoints_by_detector


def get_alldes_desc_et(image, detector_name):
    kp = get_kp(image, detector_name)
    # print(image)
    # print(f'{detector_name}: {image_name}: {len(kp)}')
    descriptors = dict()
    execution_time = dict()
    for descriptor_name in dd.all_descriptors:
        # descriptor = dd.initialize_descriptor(descriptor_name)
        if descriptor_name is 'AKAZE' and detector_name is not 'AKAZE':
            continue
        start_time = default_timer()
        desc = get_desc(image, kp, descriptor_name)
        execution_time[descriptor_name] = default_timer() - start_time
        descriptors[descriptor_name] = desc
    return {'Execution Time': execution_time, 'Descriptors': descriptors, 'Number of Keypoints': len(kp)}

# dd.print_dictionary(execution_time)
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


def get_matched_kp_ratio_det(kpnp_det, frequency):
    kpnp_unique= np.unique(np.array(list(chain(*[value.tolist() for value in kpnp_det.values()]))), axis=0,)
    kpnp_filtered_frequency = get_kpnp_by_frequency(kpnp_det, kpnp_unique, frequency)
    matched_kp_ratio_det = dict()

    for detector, kp in kpnp_det.items():
        kp = kp.tolist()
        total_kp_det = len(kp)
        matched_kp_total = 0
        for kp_filtered in kpnp_filtered_frequency.tolist():
            if kp_filtered in kp:
                matched_kp_total += 1
        matched_kp_ratio_det[detector] = matched_kp_total/total_kp_det

    return matched_kp_ratio_det


def get_matched_kpnp_ratio_det(kpnp_det, frequency):
    kpnp_unique= np.unique(np.array(list(chain(*[value.tolist() for value in kpnp_det.values()]))), axis=0,)
    kpnp_filtered_by_frequency = get_kpnp_by_frequency(kpnp_det, kpnp_unique, frequency)
    matched_kpnp_ratio_det = dict()

    for detector, kp in kpnp_det.items():
        kpnp = kp.tolist()
        total_kpnp_det = len(kp)
        matched_kpnp_total = 0
        for kpnp_filtered in kpnp_filtered_by_frequency.tolist():
            if kpnp_filtered in kpnp:
                matched_kpnp_total += 1
        matched_kpnp_ratio_det[detector] = matched_kpnp_total/total_kpnp_det

    return matched_kpnp_ratio_det

def get_kpnp_det_arr(image_set_name, pckl_path):
    image_set_ = util.get_image_set(pckl_path, image_set_name)
    kpnp_det_arr = []
    # image_num = 4
    for image_num in range(1,7):
    #     print(1)
        image = image_set_['{0}_img{1}'.format(image_set_name, image_num)]
        execution_time, all_keypoints = get_alldet_kp_et(image)
        kpnp_det_arr.append(cvkp2np_all(all_keypoints))
    return kpnp_det_arr