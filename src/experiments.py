from src import data as dt
from src import util
import matplotlib.pyplot as plt


def experiment_1(path, image_set='trees', image_num=1):
    image_set_ = util.get_image_set(path, image_set)
    image = image_set_['{0}_img{1}'.format(image_set, image_num)]
    execution_time, keypoints_by_detector = dt.get_exec_time_keypoints(image)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_data = {}
    for name in execution_time.keys():
        total_keypoints = len(keypoints_by_detector[name])
        plot_data[name] = [execution_time[name], total_keypoints]

    fig = plt.plot()

    colors = ['olive', 'green', 'red', 'cyan', 'blue', 'purple', 'green', 'grey', 'orange', 'brown']
    i = 0
    for key, values in plot_data.items():
        x, y = values
        ax.scatter(x, y, c=colors[i], s=10, label=key)
        #     ax.annotate(key, xy=(x+0.02, y), textcoords='data')
        i += 1
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.xlabel("Execution Time")
    plt.ylabel("Number of Keypoints")
    plt.show()