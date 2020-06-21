from src import data as dt
from src import util
import matplotlib.pyplot as plt
import src.imgop as ip
import pandas as pd


def experiment_1_plt(image):
    execution_time, keypoints_by_detector = ip.get_alldet_kp_et(image)
    fig = plt.figure()
    ax = fig.add_subplot(111)
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
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.xlabel("Execution Time")
    plt.ylabel("Number of Keypoints")
    plt.show()


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
