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
# a = cmaps['Qualitative']
csmap = plt.get_cmap('tab20').colors
s = 1
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


def exp_repeatability_plt(image_set_name,
                          labels,
                          ax):
    plot_dict = dict()
    image_set = util.get_image_set(data_path, image_set_name)
    for detector in dd.all_detectors:
        print(detector)
        plot_dict[detector] = get_repeatability_by_det(detector, image_set_name, image_set, labels)

    bar_width = 0.1
    bars = dict()

    for detector, val_dict in plot_dict.items():
        bar = list()
        for image_num, val in val_dict.items():
            bar.append(val['ratio'])
        bars[detector] = bar

    # Set position of bar on X axis
    pos = dict()
    r = list(range(len(list(bars.values())[0])))
    # pos.append(r)
    for i, detector in enumerate(bars):
        if i is 0:
            pos[detector] = r
        else:
            r = [x + bar_width for x in r]
            pos[detector] = r

    # Make the plot
    for i, detector in enumerate(bars):
        ax.bar(pos[detector], bars[detector], color=csmap[i], alpha=0.75, width=bar_width, edgecolor='white', label=detector)
    # plt.bar(r2, bars2, color='#557f2d', width=bar_width, edgecolor='white', label='var2')
    # plt.bar(r3, bars3, color='#2d7f5e', width=bar_width, edgecolor='white', label='var3')
    ax.grid()
    # Add xticks on the middle of the group bars
    # plt.xlabel('group', fontweight='bold')
    xticks_pos = list()
    xticks_labels = list()
    for i in range(len(list(pos.values())[0])):
        xticks_labels.extend([detector for detector in pos.keys()])
        for detector_name, pos_val in pos.items():
            xticks_pos.append(pos_val[i])
    # for detector_name, pos_val in pos.items():
    #     for p in val:
    #         xticks_pos.append(p)
    # p = [p_ for p_ in p]
    # r_ = [r + bar_width*3.5 for r in list(range(len(list(bars.values())[0])))]
    ax.set_xticks(xticks_pos)
    # ax.set_xticks([r + bar_width*3.5 for r in list(range(len(list(bars.values())[0])))])
    # ax.set_xticklabels(list(range(1, len(list(bars.values())[0]) + 1)))
    # ax.set_xticklabels()
    # p = [p for p in pos.keys()]
    # p = [p_ for p_ in p]
    ax.set_xticklabels(xticks_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Create legend & Show graphic


labels = dt.load_data(label_path)
# dataset_pckl_name = cfg['dataset']['dataset_type']['oxford']['pckl_name']
data_path = os.path.join(pckl_path, ''.join([dataset, '.pckl']))
image_set_name = 'leuven'
image_set_name_list = [key for key in image_set_variance.keys()]
image_set = util.get_image_set(data_path, image_set_name)

image_nums = (1, 2)
image1 = image_set[f'{image_set_name}_img{image_nums[0]}']
image2 = image_set[f'{image_set_name}_img{image_nums[1]}']
# label_name = f'{image_set_name}_img{image_nums[1]}'
# label_homography = labels[label_name]
detector_name = 'HarrisLaplace'
descriptor_name = 'BRISK'
colors = ['olive', 'red', 'cyan', 'blue', 'purple', 'green', 'grey', 'orange', 'indigo', 'black']

# plt.legend()
# plt.show()

fig, axs = plt.subplots(len(image_set_name_list), 1, figsize=(12, len(image_set_name_list) * 6 + 1))

for i in range(len(image_set_name_list)):
    exp_repeatability_plt(image_set_name_list[i], labels, axs[i])

    axs[i].set_xlabel(f'Degree of change', fontsize=12)
    axs[i].set_ylabel('Repeatability ratio', fontsize=12)
    axs[i].set_title(f'Change: {image_set_variance[image_set_name_list[i]]}', fontsize=14)

handles, labels = axs[0].get_legend_handles_labels()
# handles2, labels2 = axs2[0].get_legend_handles_labels()
plt.legend(handles, labels, bbox_to_anchor=(0.75, -0.2), ncol=4)
fig.subplots_adjust(hspace=0.4)
# plt.show()

# plt.grid()
plt.show()