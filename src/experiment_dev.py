import os
from matplotlib import pyplot as plt
import numpy as np
from timeit import default_timer
import pandas as pd
import seaborn as sns
from src import detector_descriptor as dd
from src import data as dt
from src import util
from src import imgop as ip
from src import experiments as ex
import yaml
import cv2
from math import sqrt

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

# def get_paths(dataset_path, extension):
#     path_list = list()
#     for file in os.listdir(dataset_path):
#         if file.endswith(extension):
#             path_list.append(os.path.join(dataset_path, file))
#     return path_list


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
image_set_name = 'bikes'
image_set = util.get_image_set(data_path, image_set_name)
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
image_nums = [1, 3]
image1 = image_set[f'{image_set_name}_img{image_nums[0]}']
image2 = image_set[f'{image_set_name}_img{image_nums[1]}']
label_name = f'{image_set_name}_img{image_nums[1]}'
label_homography = labels[label_name]
detector_name = 'KAZE'


for descriptor_name in dd.all_descriptors:
    # print(descriptor_name)
    # if descriptor_name in ['DAISY', 'KAZE']:
    #     continue
    kp1, desc1 = ip.get_desc_by_det(image1, detector_name, descriptor_name)
    # print(desc1.dtype)
    kp2, desc2 = ip.get_desc_by_det(image2, detector_name, descriptor_name)

    # bfm = cv2.BFMatcher(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING, crossCheck=True)
    # nn_matches = bfm.match(desc1, desc2, 2)
    # matches = sorted(matches, key=lambda x: x.distance)

    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE)
    nn_matches = matcher.knnMatch(desc1, desc2, 2)
    # matches = matcher.knnMatch(desc1, desc2, 2)

    matched1 = []
    matched2 = []
    nn_match_ratio = 0.8  # Nearest neighbor matching ratio
    for m, n in nn_matches:
        if m.distance < nn_match_ratio * n.distance:
            matched1.append(kp1[m.queryIdx])
            matched2.append(kp2[m.trainIdx])

    inliers1 = []
    inliers2 = []
    good_matches = []
    inlier_threshold = 2.5  # Distance threshold to identify inliers with homography check
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
    # res = np.empty((max(image1.shape[0], image2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    # img3 = cv2.drawMatches(image1, kp1, image2, kp2, matches[:10], None, flags=2)
    res_image = cv2.drawMatches(image1, inliers1, image2, inliers2, good_matches, None, flags=2)
    # cv.imwrite("akaze_result.png", res)
    inlier_ratio = len(inliers1) / float(len(matched1))
    print(f'{descriptor_name}: {inlier_ratio}')
    # print(f'{descriptor_name} Matching Results')
    # print('*******************************')
    # print('# Keypoints 1:                        \t', len(kp1))
    # print('# Keypoints 2:                        \t', len(kp2))
    # print('# Matches:                            \t', len(matched1))
    # print('# Inliers:                            \t', len(inliers1))
    # print('# Inliers Ratio:                      \t', inlier_ratio)
    # cv.imshow('result', res)
    # cv.waitKey()
    d = 1
    # matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    # nn_matches = matcher.knnMatch(desc1, desc2, 2)
    # img3 = cv2.drawMatches(image1, kp1, image2, kp2, matches[:10], None, flags=2)
    plt.title(descriptor_name)
    plt.imshow(res_image)
    plt.show()
# ax = ex.exp_desc_et_plt(image_set)
# ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
# ax.set_xlabel("Image num")
# ax.set_ylabel("Execution Time")
# plt.show()
s = 1
