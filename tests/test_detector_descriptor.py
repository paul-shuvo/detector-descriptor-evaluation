from src.detector_descriptor import *
import cv2


# import os
# import pytest
# os.chdir(os.path.join(os.getcwd(),'src') )
# os.chdir('..')
def test_get_all_detectors():
    detectors = get_all_detectors()
    img = cv.imread('lena.jpg')
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
                if detector_name != 'KAZE' and detector_name != 'AKAZE':
                    continue
            elif descriptor_name == 'AKAZE' and detector_name != 'AKAZE':
                continue

            try:
                # print(f'Detector:{detector_name}, Descriptor:{descriptor_name}\n')
                detector = detector_class.create()
                descriptor = descriptor_class.create()
                img = cv.imread('lena.jpg')
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
        if value == None:
            continue
        else:
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