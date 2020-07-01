from src import data as dt
from src import util
from src import detector_descriptor as dd
import matplotlib.pyplot as plt
import src.imgop as ip
import pandas as pd
import numpy as np


def exp_det_kpet_plt(image, ax):
    # fig, ax = plt.subplots(1,2)
    execution_time, keypoints_by_detector = ip.get_alldet_kp_et(image)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    plot_data = {}
    for name in execution_time.keys():
        total_keypoints = len(keypoints_by_detector[name])
        plot_data[name] = [execution_time[name], total_keypoints]


    colors = ['olive', 'green', 'red', 'cyan', 'blue', 'purple', 'green', 'grey', 'orange', 'indigo']
    i = 0
    for key, values in plot_data.items():
        x, y = values
        ax.scatter(x, y, c=colors[i], s=10, label=key)
        ax.annotate(key, xy=(x+0.02, y), textcoords='data')
        i += 1
    # ax.grid(True)
    # ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    ax.set_xlabel("Execution Time")
    ax.set_ylabel("Number of Keypoints")
    # plt.show()
    # return ax

def experiment_1_df(image):
    execution_time, keypoints_by_detector = ip.get_alldet_kp_et(image)
    plot_data = {}
    for name in execution_time.keys():
        total_keypoints = len(keypoints_by_detector[name])
        plot_data[name] = [execution_time[name], total_keypoints]

    df = pd.DataFrame()
    df['Detector'] = plot_data.keys()
    df['Execution Time'] = [values[0] for values in plot_data.values()]
    df['Number of Keypoints'] = [values[1] for values in plot_data.values()]
    # df=df.sort_values(by=['Execution Time'])
    df.style. \
        apply(util.highlight_max, subset=['Execution Time', 'Number of Keypoints']). \
        apply(util.highlight_min, subset=['Execution Time', 'Number of Keypoints'])

    return df


def exp_desc_et_plt(image_set, detector_name, ax):
    # f, axs = plt.subplots(6,1)
    des_et_kp = dict()
    image_num = 0
    # print(detector_name)
    for image_name, image in image_set.items():
        des_et_kp[image_num] = ip.get_alldes_desc_et(image, detector_name)
        image_num += 1

    kp_size_arr = []
    image_num = 1


    for values in des_et_kp.values():
        kp_size_arr.append('{0} \nImage: \n{1} {2}'.format(str(values['Number of Keypoints']),
                                                          list(image_set.keys())[0].split('_')[0],
                                                          image_num))
        # val = values['Descriptors']['ORB'][1].shape[0]
        # print(f'vaue is:{val}')
        # for descriptor_name in dd.all_descriptors:
        #     if descriptor_name is 'AKAZE' and detector_name is not 'AKAZE':
        #         continue
        #     v = values['Descriptors'][descriptor_name][1].shape[0]
        #     print(f'{descriptor_name}: {v}')
        # print('------------')
        image_num += 1

        # print(values['Descriptors']['LATCH'][1].shape[0])

    plot_data_desc = dict()
    for descriptor_name in dd.all_descriptors:
        if descriptor_name is 'AKAZE' and detector_name is not 'AKAZE':
            continue
        plot_data_desc[descriptor_name] = list()
        for image_num, values in des_et_kp.items():
            plot_data_desc[descriptor_name].append(values['Execution Time'][descriptor_name])

    colors = ['olive', 'green', 'red', 'cyan', 'blue', 'purple', 'green', 'grey', 'orange', 'indigo']
    markers = ['+', '^', 'o', 's', '*', 'x', '+', '^', 'o', 's', '*', 'x']
    i = 0
    for descriptor_name, execution_times in plot_data_desc.items():
        if descriptor_name is 'AKAZE':
            ax.scatter(kp_size_arr, execution_times, c=colors[i], marker='p', label=descriptor_name)
        else:
            ax.scatter(kp_size_arr, execution_times, c=colors[i], marker=markers[i], label=descriptor_name)
            i += 1
    # return axs


def exp_kp_freq_frac_plt(image_set_name, pckl_path, frequencies, axs, row, col):
    kpnp_det_arr = ip.get_kpnp_det_arr(image_set_name, pckl_path)
    #     plot_data_arr = get_plot_data_arr(kp_np_det_arr, frequencies)
    plot_data_arr = []
    for kpnp_det in kpnp_det_arr:
        matched_kpnp_ratio_det_freq = dict()
        for frequency in frequencies:
            matched_kpnp_ratio_det_freq[frequency] = ip.get_matched_kpnp_ratio_det(kpnp_det, frequency)

        plot_data = dict()
        for detector in dd.all_detectors:
            plot_data_det = list()
            for frequency in matched_kpnp_ratio_det_freq.keys():
                plot_data_det.append(matched_kpnp_ratio_det_freq[frequency][detector])
            plot_data[detector] = np.array(plot_data_det)
        plot_data_arr.append(plot_data)

    #     plt.style.use('bmh')
    colors = ['olive', 'green', 'red', 'cyan', 'blue', 'purple', 'green', 'black', 'orange', 'indigo']
    markers = ['+', '^', 'o', 's', '*', 'x', '+', '^', 'o', 's', '*', 'x']
    linestyle_ = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':']
    row = row
    col = col
    #     fig, axs = plt.subplots(row, col, figsize=(12,8))
    axs_count = 0
    for plot_data in plot_data_arr:
        i = 0
        for detector, data in plot_data.items():
            axs[axs_count // col, axs_count % col].plot(frequencies, data, label=detector, linestyle=linestyle_[i],
                                                        linewidth=1.5, marker=markers[i], c=colors[i])
            i += 1
        axs[axs_count // col, axs_count % col].set_title(f"{image_set_name}_img{axs_count + 1}")
        axs[axs_count // col, axs_count % col].set_xlabel('Frequency of keypoints')
        axs[axs_count // col, axs_count % col].set_ylabel('Fraction of keypoints')
        #         plt.plot(frequencies, data, label=detector)

        axs_count += 1