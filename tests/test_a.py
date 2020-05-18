# from __future__ import print_function, division, absolute_import

from src.a import Point
import pytest 
import os 


def test_a():
    p1 = Point("Dakar", 14.54, 14.65)
    assert p1.get_lat_long() == (14.54, 14.65)

def test_invalid_point_generation():
    with pytest.raises(ValueError) as exp:
        Point("Buenos", 12.11, -555.67)
    assert str(exp.value) == "Invalid latitude"
    
def test_invalid_type_for_name_attribute():
    with pytest.raises(TypeError) as exp:
        Point(12, 56, 76)
    assert str(exp.value) == "name must be a string"