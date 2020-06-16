from src.data import *
import cv2
import pytest


def test_get_file_paths():
    with pytest.raises(FileNotFoundError):
        paths = get_file_paths('cambridge', '.ggm')