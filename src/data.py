import pickle as pkl
import cv2
import os

# Change current directory to the dataset folder
os.chdir('..')
os.chdir(os.path.join(os.getcwd(), 'dataset'))


# print(os.getcwd())


def get_file_paths(dataset_name, extension):
    """
    Returns a list of file paths ending in specified file extensionension(s) (`extension`)
    Args:
        dataset_name(`str`): Dataset/folder name.
        extension(`str` or `str tuple`): File extensionension or a tuple of file extensionensions.

    Returns:
        (`list`): A list of file paths ending in specified file extensionension(s).

    Examples:

        .. code-block:: python

        In[1]: from src.data import get_paths_by_extension
        In[2]: get_paths_by_extension('oxford', ('.pgm', '.ppm'))
        Out[3]: ['D:\\Programming Projects\\python projects\\state-of-the-binary-descriptor\\dataset\\oxford\\bark_img1.ppm',
                 'D:\\Programming Projects\\python projects\\state-of-the-binary-descriptor\\dataset\\oxford\\bark_img2.ppm',
                 'D:\\Programming Projects\\python projects\\state-of-the-binary-descriptor\\dataset\\oxford\\bark_img3.ppm',
                 'D:\\Programming Projects\\python projects\\state-of-the-binary-descriptor\\dataset\\oxford\\bark_img4.ppm',
                 'D:\\Programming Projects\\python projects\\state-of-the-binary-descriptor\\dataset\\oxford\\bark_img5.ppm',
                 'D:\\Programming Projects\\python projects\\state-of-the-binary-descriptor\\dataset\\oxford\\bark_img6.ppm',
                 'D:\\Programming Projects\\python projects\\state-of-the-binary-descriptor\\dataset\\oxford\\bikes_img1.ppm',
                 'D:\\Programming Projects\\python projects\\state-of-the-binary-descriptor\\dataset\\oxford\\bikes_img2.ppm']

    """
    os.chdir(os.path.join(os.getcwd(), dataset_name))
    path_list = list()
    for file in os.listdir(os.getcwd()):
        if file.endswith(extension):
            path_list.append(os.path.join(os.getcwd(), file))
    return path_list


def load_images(dataset_name, extension):
    image_paths = get_file_paths(dataset_name, extension)
    image_dataset = dict()
    for image_path in image_paths:
        _, file_name = os.path.split(image_path)
        image_np = cv2.imread(image_path)
        image_dataset[file_name] = image_np
    return image_dataset


def dump_data(data, path):
    with open(path, 'wb') as file:
        pkl.dump(data, file, protocol=pkl.HIGHEST_PROTOCOL)


def load_data(path):
    with open(path, 'rb') as file:
        return pkl.load(file)

# print(get_paths_by_extension('oxord', ('.pgm', '.ppm')))
# load_images('oxford', ('.pgm', '.ppm'))
# os.chdir('..')
a = load_data('D:\Programming Projects\python projects\state-of-the-binary-descriptor\dataset\pickle_dump\oxford.pckl')
s = 1
# dump_data(load_images('oxford', ('.pgm', '.ppm')), os.path.join(os.getcwd(), 'oxford.pckl'))