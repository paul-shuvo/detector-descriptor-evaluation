import os
from matplotlib import pyplot as plt
from src import keypoint_processing as kpp
from src import data as dt
from src import util
import yaml
import numpy as np
from src import detector_descriptor as dd

color_map = plt.get_cmap('tab20').colors
image_set_variance = {
    'leuven': 'light',
    'bikes': 'blur',
    # 'trees': 'blur',
    # 'wall': 'viewpoint',
    'graf': 'viewpoint',
    # 'bark': 'zoom and rotation',
    'boat': 'zoom and rotation',
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

labels = dt.load_data(label_path)
# dataset_pckl_name = cfg['dataset']['dataset_type']['oxford']['pckl_name']
data_path = os.path.join(pckl_path, ''.join([dataset, '.pckl']))
image_set_name = 'boat'
image_set_name_list = [key for key in image_set_variance.keys()]
image_set = util.get_image_set(data_path, image_set_name)

image_nums = (1, 2)
# image1 = image_set[f'{image_set_name}_img{image_nums[0]}']
# image2 = image_set[f'{image_set_name}_img{image_nums[1]}']
# label_name = f'{image_set_name}_img{image_nums[1]}'
# label_homography = labels[label_name]
detector_name = 'ORB'
descriptor_name = 'BRISK'
colors = ['olive', 'red', 'cyan', 'blue', 'purple', 'green', 'grey', 'orange', 'indigo', 'black']
# plt.style.use('ggplot')

# plt.legend()
# plt.show()

# fig, axs = plt.subplots(len(image_set_name_list), 1, figsize=(12, len(image_set_name_list) * 6.5 + 1))





for detector_name in dd.all_detectors:
    print(f'{detector_name}:\n{get_frequency_ratio(data_path, detector_name, image_set_name)}')