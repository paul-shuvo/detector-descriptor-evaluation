import os
import numpy as np
from src import detector_descriptor as dd
from src import data as dt
from src import util
from src import keypoint_processing as kpp
import yaml
import cv2

"""
A module for experimental methods.
"""
image_set_variance = {
    'bark': 'zoom_rotation',
    'boat': 'zoom_rotation',
    'leuven': 'light',
    'bikes': 'blur',
    'trees': 'blur',
    'wall': 'viewpoint',
    'graf': 'viewpoint',
    'ubc': 'jpeg-compression'
}
os.chdir('..')

with open(os.path.join('./', 'config.yml'), 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    cfg = cfg['default']
print(os.getcwd())
dataset_path = cfg['path']['dataset']
dataset = cfg['current_dataset']
# labels = dt.load_labels(os.path.join(dataset_path, dataset), '.txt')
pckl_path = cfg['path']['pckl']
pckl_name = ''.join([dataset, '_label.pckl'])
label_path = os.path.join(pckl_path, pckl_name)
# dt.dump_data(labels, dump_path)
labels = dt.load_data(label_path)
# dataset_pckl_name = cfg['dataset']['dataset_type']['oxford']['pckl_name']
data_path = os.path.join(pckl_path, ''.join([dataset, '.pckl']))

# image = image_set['bikes_img6']
# ex.experiment_1_df(image)

# def get_alldes_desc_et(image, detector_name):
#     kp = ip.get_kp(image, detector_name)
#     descriptors = dict()
#     execution_time = dict()
#     for descriptor_name in dd.get_all_descriptors():
#         # descriptor = dd.initialize_descriptor(descriptor_name)
#         start_time = default_timer()
#         desc = ip.get_desc(image, kp, descriptor_name)
#         execution_time[descriptor_name] = default_timer() - start_time
#         descriptors[descriptor_name] = desc
#     return {'Execution Time': execution_time, 'Descriptors': descriptors}

# all = ip.get_alldes_desc_et(image, 'KAZE')
# dd.print_dictionary(all['Execution Time'])
# v = all["Descriptors"]
# for key, val in all['Descriptors'].items():
#     print(f'{key}: {val[1].shape[1]}')
#
# for key, val in all['Execution Time'].items():
#     print(f'{key}: {val}')


# def get_inlier_ratio(
#         image_tuple,
#         label_homography,
#         detector_name,
#         descriptor_name,
#         matcher_type=cv2.DescriptorMatcher_BRUTEFORCE,
#         nn_match_ratio=0.8,
#         inlier_threshold=2.5):
#
#     kp1, desc1 = kpp.get_desc_by_det(image_tuple[0], detector_name, descriptor_name)
#     kp2, desc2 = kpp.get_desc_by_det(image_tuple[1], detector_name, descriptor_name)
#
#     matcher = cv2.DescriptorMatcher_create(matcher_type)
#     nn_matches = matcher.knnMatch(desc1, desc2, 2)
#
#     matched1 = []
#     matched2 = []
#     # nn_match_ratio = 0.8  # Nearest neighbor matching ratio
#     for m, n in nn_matches:
#         if m.distance < nn_match_ratio * n.distance:
#             matched1.append(kp1[m.queryIdx])
#             matched2.append(kp2[m.trainIdx])
#
#     inliers1 = []
#     inliers2 = []
#     good_matches = []
#     # inlier_threshold = 2.5  # Distance threshold to identify inliers with homography check
#     for i, m in enumerate(matched1):
#         col = np.ones((3, 1), dtype=np.float64)
#         col[0:2, 0] = m.pt
#         col = np.dot(label_homography, col)
#         col /= col[2, 0]
#         # dist = sqrt(pow(col[0, 0] - matched2[i].pt[0], 2) + \
#         #             pow(col[1, 0] - matched2[i].pt[1], 2))
#         dist = np.sqrt(np.power(col[0, 0] - matched2[i].pt[0], 2) + \
#                        np.power(col[1, 0] - matched2[i].pt[1], 2))
#         if dist < inlier_threshold:
#             good_matches.append(cv2.DMatch(len(inliers1), len(inliers2), 0))
#             inliers1.append(matched1[i])
#             inliers2.append(matched2[i])
#
#     res_image = cv2.drawMatches(image1, inliers1, image2, inliers2, good_matches, None, flags=2)
#     try:
#         inlier_ratio = len(inliers1) / float(len(matched1))
#     except ZeroDivisionError:
#         inlier_ratio = 0
#     # print(f'{descriptor_name}: {inlier_ratio}')
#     # print(f'{descriptor_name} Matching Results')
#     # print('*******************************')
#     # print('# Keypoints 1:                        \t', len(kp1))
#     # print('# Keypoints 2:                        \t', len(kp2))
#     # print('# Matches:                            \t', len(matched1))
#     # print('# Inliers:                            \t', len(inliers1))
#     # print('# Inliers Ratio:                      \t', inlier_ratio)
#     return res_image, inlier_ratio


def get_alldet_matching_results(
        image_tuple,
        label_homography,
        descriptor_name,
        excluded_det=None,
        matcher_type=cv2.DescriptorMatcher_BRUTEFORCE,
        nn_match_ratio=0.8,
        inlier_threshold=2.5):
    if excluded_det is None:
        excluded_det = []
    alldet_inlier_ratio = dict()
    for detector_name in dd.all_detectors:
        if detector_name in excluded_det:
            continue
        alldet_inlier_ratio[detector_name] = get_matching_results(image_tuple, label_homography, detector_name,
                                                              descriptor_name, matcher_type, nn_match_ratio,
                                                              inlier_threshold)
    return alldet_inlier_ratio


def get_alldes_matching_results(
        image_tuple,
        label_homography,
        detector_name,
        excluded_des=None,
        matcher_type=cv2.DescriptorMatcher_BRUTEFORCE,
        nn_match_ratio=0.8,
        inlier_threshold=2.5):
    if excluded_des is None:
        excluded_des = []
    if detector_name is not 'AKAZE':
        excluded_des.append('AKAZE')
    alldes_inlier_ratio = dict()
    for descriptor_name in dd.get_all_descriptors():
        if descriptor_name in excluded_des:
            continue
        alldes_inlier_ratio[descriptor_name] = get_matching_results(image_tuple, label_homography, detector_name,
                                                                descriptor_name, matcher_type, nn_match_ratio,
                                                                inlier_threshold)
    return alldes_inlier_ratio



# for descriptor_name in dd.all_descriptors:
#     if descriptor_name in ['DAISY', 'KAZE']:
#         continue
#     res_image, inlier_ratio = get_inlier_ratio((image1, image2), label_homography, detector_name,
#                                                descriptor_name)
#     print(f'{descriptor_name}: {inlier_ratio}')
#     plt.title(descriptor_name)
#     plt.imshow(res_image)
#     plt.show()

# alldet_inlier_ratio = get_alldet_inlier_ratio((image1, image2), label_homography, descriptor_name)
#
# for detector_name, values in alldet_inlier_ratio.items():
#     res_image, inlier_ratio = values
#     print(f'{detector_name}: {inlier_ratio}')
#     plt.title(detector_name)
#     plt.imshow(res_image)
#     plt.show()

# alldes_inlier_ratio = get_alldes_inlier_ratio((image1, image2), label_homography, detector_name, excluded_des=['KAZE', 'AKAZE', 'DAISY'])
#
# for descriptor_name, values in alldes_inlier_ratio.items():
#     res_image, inlier_ratio = values
#     print(f'{descriptor_name}: {inlier_ratio}')
#     plt.title(descriptor_name)
#     plt.imshow(res_image)
#     plt.show()
#
# s = 1


def get_matching_results(
        image_tuple,
        label_homography,
        detector_name,
        descriptor_name,
        matcher_type=cv2.DescriptorMatcher_BRUTEFORCE,
        nn_match_ratio=0.8,
        inlier_threshold=2.5):
    # print(detector_name +'-'+ descriptor_name)

    kp1, desc1 = kpp.get_desc_by_det(image_tuple[0], detector_name, descriptor_name)
    kp2, desc2 = kpp.get_desc_by_det(image_tuple[1], detector_name, descriptor_name)

    matcher = cv2.DescriptorMatcher_create(matcher_type)
    nn_matches = matcher.knnMatch(desc1, desc2, 2)

    matched1 = []
    matched2 = []
    unmatched = []
    # nn_match_ratio = 0.8  # Nearest neighbor matching ratio
    for m, n in nn_matches:
        if m.distance < nn_match_ratio * n.distance:
            matched1.append(kp1[m.queryIdx])
            matched2.append(kp2[m.trainIdx])
    for kp in kp1:
        if kp not in matched1:
            unmatched.append(kp)

    true_positive = []
    true_negative = []
    false_positive = []
    false_negative = []
    inliers1 = []
    inliers2 = []
    good_matches = []
    # inlier_threshold = 2.5  # Distance threshold to identify inliers with homography check

    for i, m in enumerate(unmatched):
        col = np.ones((3, 1), dtype=np.float64)
        col[0:2, 0] = m.pt
        col = np.dot(label_homography, col)
        col /= col[2, 0]
        if 0 <= col[0] <= image_tuple[1].shape[0] and 0 <= col[1] <= image_tuple[1].shape[1]:
            false_negative.append(m)
        else:
            false_positive.append(m)

    for i, m in enumerate(matched1):
        col = np.ones((3, 1), dtype=np.float64)
        col[0:2, 0] = m.pt
        col = np.dot(label_homography, col)
        col /= col[2, 0]
        # dist = sqrt(pow(col[0, 0] - matched2[i].pt[0], 2) + \
        #             pow(col[1, 0] - matched2[i].pt[1], 2))
        dist = np.sqrt(np.power(col[0, 0] - matched2[i].pt[0], 2) + \
                       np.power(col[1, 0] - matched2[i].pt[1], 2))

        if dist < inlier_threshold:
            good_matches.append(cv2.DMatch(len(inliers1), len(inliers2), 0))
            inliers1.append(matched1[i])
            inliers2.append(matched2[i])
            true_positive.append(m)
        else:
            true_negative.append(m)

    res_image = cv2.drawMatches(image1, inliers1, image2, inliers2, good_matches, None, flags=2)
    try:
        inlier_ratio = len(inliers1) / float(len(matched1))
    except ZeroDivisionError:
        inlier_ratio = 0
    # print(f'{descriptor_name}: {inlier_ratio}')
    # print(f'{descriptor_name} Matching Results')
    # print('*******************************')
    # print('# Keypoints 1:                        \t', len(kp1))
    # print('# Keypoints 2:                        \t', len(kp2))
    # print('# Matches:                            \t', len(matched1))
    # print('# Inliers:                            \t', len(inliers1))
    # print('# Inliers Ratio:                      \t', inlier_ratio)
    accuracy = (len(true_positive) + len(true_negative))/(len(true_positive) + len(true_negative) + len(false_positive) + len(false_negative))
    precision = len(true_positive) / (len(true_positive) + len(false_positive))
    recall =  len(true_positive) / (len(true_positive) + len(true_negative))
    # print(f'Descriptor: {descriptor_name}, TP: {len(true_positive)}, FP: {len(false_positive)}, TN: {len(true_negative)}, FN: {len(false_negative)}, Total: {len(kp1)}')
    return {'Resultant image': res_image,
            'Inlier ratio': inlier_ratio,
            'TP': true_positive,
            'FN': false_negative,
            'TN': true_negative,
            'FP': false_positive,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall}


image_set_name = 'bikes'
image_set = util.get_image_set(data_path, image_set_name)
image_nums = (1, 2)
excluded_des=['KAZE', 'DAISY']
comb_result_avg = dict()
for detector_name in dd.all_detectors:
    for descriptor_name in dd.all_descriptors:
        if descriptor_name in excluded_des:
            continue
        if descriptor_name is 'AKAZE' and detector_name is not ('AKAZE' or 'KAZE'):
            comb_result_avg[f'{detector_name}-{detector_name}'] = None
            continue
        avg_accuracy = 0
        avg_precision = 0
        avg_recall = 0
        avg_inlier_ratio = 0
        for image_num in range(2, 7):
            image1 = image_set[f'{image_set_name}_img{1}']
            image2 = image_set[f'{image_set_name}_img{image_num}']
            label_name = f'{image_set_name}_img{image_num}'
            label_homography = labels[label_name]
            matching_results = get_matching_results((image1, image2), label_homography, detector_name, descriptor_name)

            avg_accuracy += matching_results['Accuracy']
            avg_precision += matching_results['Precision']
            avg_recall += matching_results['Recall']
            avg_inlier_ratio += matching_results['Inlier ratio']
        comb_result_avg[f'{detector_name}-{descriptor_name}'] = {'Accuracy': avg_accuracy/5,
                                                               'Precision': avg_precision/5,
                                                               'Recall': avg_recall/5,
                                                               'Inlier ratio': avg_inlier_ratio/5}
# s, t = get_inlier_ratio((image1, image2),label_homography,detector_name,descriptor_name,matcher_type=cv2.DescriptorMatcher_BRUTEFORCE,nn_match_ratio=0.8,inlier_threshold=2.5)
dd.print_dictionary(comb_result_avg)