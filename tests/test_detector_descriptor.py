from src.detector_descriptor import *
import cv2
import pytest


# import os
# import pytest
# os.chdir(os.path.join(os.getcwd(),'src') )
# os.chdir('..')
def test_get_all_detectors():
    detectors = get_all_detectors()
    img = cv2.imread('lena.jpg')
    test_status = "All detector module found"
    for key, value in detectors.items():
        try:
            key_point = value.create().detect(img)
        except:
            test_status = "{0} module not found".format(key)
            break
    assert test_status == "All detector module found"


def test_get_all_desriptors():
    test_status = "All detector and descriptor combination works (Unique cases: KAZE and AKAZE)"
    error = ''
    for detector_name, detector_class in get_all_detectors().items():
        for descriptor_name, descriptor_class in get_all_descriptors().items():
            if descriptor_name == 'KAZE':
                if detector_name not in ('KAZE','AKAZE'):
                    continue
            elif descriptor_name == 'AKAZE' and detector_name != 'AKAZE':
                continue

            try:
                # print(f'Detector:{detector_name}, Descriptor:{descriptor_name}\n')
                detector = detector_class.create()
                descriptor = descriptor_class.create()
                img = cv2.imread('lena.jpg')
                kp = detector.detect(img)
                desc = descriptor.compute(img, kp)
            except:
                error += f'Error in the following combination: Detector:{detector_name}, Descriptor:{descriptor_name}\n'
    if error != '':
        test_status = error
    assert test_status == "All detector and descriptor combination works (Unique cases: KAZE and AKAZE)"


def test_get_all_variants():
    test_status = 'All the detector and descriptor classes were successfully initialized using respective variant types'
    error = ''

    for class_, value in get_all_variants().items():
        if value is None:
            continue

        if class_ in get_all_detectors().keys():
            class_obj = select_detector(class_)
        else:
            class_obj = select_descriptor(class_)
        for variant_type, variant_values in value.items():
            for variant in variant_values.values():
                try:
                    exec_string = "instance = {0}.create({1}={2})".format(str(class_obj).split('\'')[1],
                                                                          variant_type, variant)
                    # `exec` is used to dynamically create the instances
                    exec(exec_string)
                except:
                    error += 'Error in the following combination: class: {0}, variant_type: {1}, variant: {2}'.format(
                        str(class_obj).split('\'')[1], variant_type, variant)
    if error != '':
        test_status = error
    assert test_status == 'All the detector and descriptor classes were successfully initialized using respective variant types'


def test_initialize_detector():
    error_count = 0

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