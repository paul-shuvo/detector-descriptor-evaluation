from src.detector_descriptor import *
import cv2
import pytest


# import os
# import pytest
# os.chdir(os.path.join(os.getcwd(),'src') )
# os.chdir('..')

from contextlib import contextmanager

@contextmanager
def not_raises(exception):
  try:
    yield
  except exception:
    raise pytest.fail("DID RAISE {0}".format(exception))

def test_get_all_detectors():
    detectors = get_all_detectors()
    img = cv2.imread('lena.jpg')
    with not_raises(Exception):
        for key, value in detectors.items():
            key_point = value.create().detect(img)



def test_get_all_desriptors():
    with not_raises(Exception):
        for detector_name, detector_class in get_all_detectors().items():
            for descriptor_name, descriptor_class in get_all_descriptors().items():
                if descriptor_name == 'KAZE':
                    if detector_name not in ('KAZE','AKAZE'):
                        continue
                elif descriptor_name == 'AKAZE' and detector_name != 'AKAZE':
                    continue
                    # print(f'Detector:{detector_name}, Descriptor:{descriptor_name}\n')
                detector = detector_class.create()
                descriptor = descriptor_class.create()
                img = cv2.imread('lena.jpg')
                kp = detector.detect(img)
                desc = descriptor.compute(img, kp)


def test_get_all_variants():
    with not_raises(Exception):
        for class_, value in get_all_variants().items():
            if value is None:
                continue

            if class_ in get_all_detectors().keys():
                class_obj = select_detector(class_)
            else:
                class_obj = select_descriptor(class_)
            for variant_type, variant_values in value.items():
                for variant in variant_values.values():
                    exec_string = "instance = {0}.create({1}={2})".format(str(class_obj).split('\'')[1],
                                                                          variant_type, variant)
                    # `exec` is used to dynamically create the instances
                    exec(exec_string)


def test_initialize_detector():

    with pytest.raises(ModuleNotFoundError):
        instance = initialize_detector('DAISY')

    with pytest.raises(ValueError):
        instance = initialize_detector('FAST', 'type')

    with pytest.raises(ValueError):
        instance = initialize_detector('FAST', variant=2)

    with pytest.raises(ValueError):
        instance = initialize_detector('FAST', 'norm', 2)

    with pytest.raises(ValueError):
        instance = initialize_detector('FAST', 'type', 9)


def test_initialize_descriptor():

    with pytest.raises(ModuleNotFoundError):
        instance = initialize_descriptor('FAST')

    with pytest.raises(ValueError):
        instance = initialize_descriptor('AKAZE', 'diffusivity')

    with pytest.raises(ValueError):
        instance = initialize_descriptor('AKAZE', variant=1)

    with pytest.raises(ValueError):
        instance = initialize_descriptor('AKAZE', 'norm', 2)

    with pytest.raises(ValueError):
        instance = initialize_descriptor('AKAZE', 'diffusivity', 9)


def test_available_attributes():
    with pytest.raises(ModuleNotFoundError):
        attributes = available_attributes('THANOS')


def test_get_attribute():
    with pytest.raises(AttributeError):
        obj = initialize_detector('FAST', 'type', 2)
        get_attribute(obj, 'typ')


def test_set_attribute():
    with pytest.raises(AttributeError):
        obj = initialize_detector('FAST', 'type', 2)
        set_attribute(obj, 'typ', 1)

    obj = initialize_detector('FAST', 'type', 1)
    set_attribute(obj, 'Type', 2)
    assert obj.getType() == 2
