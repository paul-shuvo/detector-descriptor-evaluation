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
image_set = util.get_image_set(data_path, 'leuven')
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


ax = ex.exp_desc_et_plt(image_set)
ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
ax.set_xlabel("Image num")
ax.set_ylabel("Execution Time")
plt.show()
s = 1


