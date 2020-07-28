import os
from matplotlib import pyplot as plt
import numpy as np
from timeit import default_timer
import pandas as pd
import seaborn as sns
from src import detector_descriptor as dd
from src import data as dt
from src import util
from src import keypoint_processing as kpp
from src import experiments as ex
import yaml
import cv2
# from math import round

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

def get_repeatability_by_det(detector_name, image_set_name, image_set, labels):
    image1 = image_set[f'{image_set_name}_img{image_nums[0]}']
    kp = kpp.get_kp(image1, detector_name)
    # kpl = [kp_.pt for kp_ in kp]
    # kpnp = kpp.cvkp2np(kp)
    repeatability = dict()
    for image_num in range(2, 7):
        repeated = 0
        not_repeated = 0
        image2 = image_set[f'{image_set_name}_img{image_num}']
        kp2 = kpp.get_kp(image2, detector_name)
        # kpl2 = [kp_.pt for kp_ in kp2]
        kpnp2 = kpp.cvkp2np(kp2)

        label_name = f'{image_set_name}_img{image_num}'
        label_homography = labels[label_name]

        for kp_ in kp:
            col = np.ones((3, 1), dtype=np.float64)
            col[0:2, 0] = kp_.pt
            col = np.dot(label_homography, col)
            col /= col[2, 0]
            col = np.rint(col)[0:2].astype(int).reshape((1, 2))[0]
            # np.rint(arr).astype(int)
            # if [255,32] in kpl2:
            #     s = 1
            if 0 <= col[0] <= image1.shape[0] and 0 <= col[1] <= image1.shape[1]:
                # any((a[:] == [1, 2]).all(1))
                if (kpnp2 == col).all(1).any():
                    # val = col.tolist()[0]
                    # if val not in kpl2:
                    #     s = 1
                    repeated += 1
                else:
                    not_repeated += 1
        repeatability[image_num] = {'repeated': repeated, 'not-repeated': not_repeated, 'ratio': (repeated/(repeated + not_repeated))}
    return repeatability

labels = dt.load_data(label_path)
# dataset_pckl_name = cfg['dataset']['dataset_type']['oxford']['pckl_name']
data_path = os.path.join(pckl_path, ''.join([dataset, '.pckl']))
image_set_name = 'leuven'
image_set = util.get_image_set(data_path, image_set_name)

image_nums = (1, 2)
image1 = image_set[f'{image_set_name}_img{image_nums[0]}']
image2 = image_set[f'{image_set_name}_img{image_nums[1]}']
# label_name = f'{image_set_name}_img{image_nums[1]}'
# label_homography = labels[label_name]
detector_name = 'ORB'
descriptor_name = 'BRISK'


    
dd.print_dictionary(get_repeatability_by_det(detector_name, image_set_name, image_set, labels))

# if np.array([111, 30]) in kpnp:
#     print('yes')

# {2: {'not-repeated': 105, 'repeated': 6865},
#  3: {'not-repeated': 245, 'repeated': 6967},
#  4: {'not-repeated': 482, 'repeated': 6656},
#  5: {'not-repeated': 761, 'repeated': 6355},
#  6: {'not-repeated': 1570, 'repeated': 5549}}

# {2: {'not-repeated': 6600, 'repeated': 370},
#  3: {'not-repeated': 7010, 'repeated': 202},
#  4: {'not-repeated': 7109, 'repeated': 29},
#  5: {'not-repeated': 7108, 'repeated': 8},
#  6: {'not-repeated': 7119, 'repeated': 0}}

# {2: {'not-repeated': 5700, 'repeated': 1270},
#  3: {'not-repeated': 6377, 'repeated': 835},
#  4: {'not-repeated': 6792, 'repeated': 346},
#  5: {'not-repeated': 6918, 'repeated': 198},
#  6: {'not-repeated': 7031, 'repeated': 88}}