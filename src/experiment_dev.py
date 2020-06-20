import os
from matplotlib import pyplot as plt
import numpy as np
from timeit import default_timer
import pandas as pd
import seaborn as sns
from src import detector_descriptor as dd
from src import data as dt
from src import util
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
with open(os.path.join('./', 'config.yml'), 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    cfg = cfg['default']
print(os.getcwd())
dataset_path = cfg['dataset']['pckl_path']
dataset_pckl_name = cfg['dataset']['dataset_type']['oxford']['pckl_name']
path = os.path.join(dataset_path, dataset_pckl_name)


