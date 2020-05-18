import sys
from os.path import dirname
from os.path import abspath, join

# print(__file__)
# print(abspath(__file__))
# print(dirname(abspath(__file__)))
# print(dirname(dirname(abspath(__file__))))

root_dir = dirname(dirname(abspath(__file__)))
sys.path.append(root_dir)