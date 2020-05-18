import cv2 as cv
from pprint import pformat


def get_all_descriptors():
    """Returns a collection of descriptor classes in OpenCV (`version 4.2.0`_).

    Returns:
        (dict):A dictionary containing a collection of descriptor classes.

    .. note::
        **KAZE** descriptor only supports *KAZE* keypoints and
        **AKAZE** descriptor only supports *KAZE* or *AKAZE* keypoints.

    Examples:
        >>> get_all_descriptors()
        {'LATCH': <class 'cv2.xfeatures2d_LATCH'>, 'LUCID': <class 'cv2.xfeatures2d_LUCID'>,
        'FREAK': <class 'cv2.xfeatures2d_FREAK'>, 'DAISY': <class 'cv2.xfeatures2d_DAISY'>,
        'BOOSTDESC': <class 'cv2.xfeatures2d_BoostDesc'>, 'KAZE': <class 'cv2.KAZE'>,
        'AKAZE': <class 'cv2.AKAZE'>, 'BRIEF': <class 'cv2.xfeatures2d_BriefDescriptorExtractor'>,
        'BRISK': <class 'cv2.BRISK'>, 'ORB': <class 'cv2.ORB'>}

        .. parsed-literal::

            descriptors = get_all_descriptors()
            :func:`print_dictionary(descriptors) <src.detector_descriptor.print_dictionary>`

        Outputs:
            .. code::

                {'AKAZE': <class 'cv2.AKAZE'>,
                'BOOSTDESC': <class 'cv2.xfeatures2d_BoostDesc'>,
                'BRIEF': <class 'cv2.xfeatures2d_BriefDescriptorExtractor'>,
                'BRISK': <class 'cv2.BRISK'>,
                'DAISY': <class 'cv2.xfeatures2d_DAISY'>,
                'FREAK': <class 'cv2.xfeatures2d_FREAK'>,
                'KAZE': <class 'cv2.KAZE'>,
                'LATCH': <class 'cv2.xfeatures2d_LATCH'>,
                'LUCID': <class 'cv2.xfeatures2d_LUCID'>,
                'ORB': <class 'cv2.ORB'>}
    See Also:
        :func:`~src.detector_descriptor.get_all_detectors`

    .. _version 4.2.0:
        https://docs.opencv.org/4.2.0/modules.html
    """
    return {
        'LATCH': cv.xfeatures2d_LATCH,
        'LUCID': cv.xfeatures2d_LUCID,
        'FREAK': cv.xfeatures2d_FREAK,
        'DAISY': cv.xfeatures2d_DAISY,
        'BOOSTDESC': cv.xfeatures2d_BoostDesc,
        'KAZE': cv.KAZE,
        'AKAZE': cv.AKAZE,
        'BRIEF': cv.xfeatures2d_BriefDescriptorExtractor,
        'BRISK': cv.BRISK,
        'ORB': cv.ORB
    }


def get_all_detectors():
    """Returns a collection of detector classes in OpenCV (`version 4.2.0`_).

    Returns:
        (dict): A dictionary containing a collection of detector classes.

    Examples:
        >>> get_all_detectors()
        {'AGAST': <class 'cv2.AgastFeatureDetector'>, 'KAZE': <class 'cv2.KAZE'>,
        'AKAZE': <class 'cv2.AKAZE'>, 'FAST': <class 'cv2.FastFeatureDetector'>,
        'BRISK': <class 'cv2.BRISK'>, 'ORB': <class 'cv2.ORB'>, 'GFTT': <class 'cv2.GFTTDetector'>,
        'HarrisLaplace': <class 'cv2.xfeatures2d_HarrisLaplaceFeatureDetector'>,
        'StarDetector': <class 'cv2.xfeatures2d_StarDetector'>}


        .. parsed-literal::

            detectors = get_all_detectors()
            :func:`print_dictionary(detectors) <src.detector_descriptor.print_dictionary>`

        Outputs:
            .. code::

                {'AGAST': <class 'cv2.AgastFeatureDetector'>,
                 'AKAZE': <class 'cv2.AKAZE'>,
                 'BRISK': <class 'cv2.BRISK'>,
                 'FAST': <class 'cv2.FastFeatureDetector'>,
                 'GFTT': <class 'cv2.GFTTDetector'>,
                 'HarrisLaplace': <class 'cv2.xfeatures2d_HarrisLaplaceFeatureDetector'>,
                 'KAZE': <class 'cv2.KAZE'>,
                 'ORB': <class 'cv2.ORB'>,
                 'StarDetector': <class 'cv2.xfeatures2d_StarDetector'>}

    See Also:
        :func:`~src.detector_descriptor.get_all_descriptors`

    .. _version 4.2.0:
        https://docs.opencv.org/4.2.0/modules.html
    """
    return {
        'AGAST': cv.AgastFeatureDetector,
        'KAZE': cv.KAZE,
        'AKAZE': cv.AKAZE,
        'FAST': cv.FastFeatureDetector,
        'BRISK': cv.BRISK,
        'ORB': cv.ORB,
        'GFTT': cv.GFTTDetector,
        'HarrisLaplace': cv.xfeatures2d_HarrisLaplaceFeatureDetector,
        'StarDetector': cv.xfeatures2d_StarDetector
    }


def select_detector(detector_name):
    """Selects the detector class from the detector collection and returns it.

    Args:
        detector_name (str): Name of the detector that would be selected.

    Returns:
        (class): A OpenCV detector class.
    """
    return get_all_detectors().get(detector_name)


def select_descriptor(descriptor_name):
    """Selects the descriptor class from the descriptor collection and returns it.

    Args:
        descriptor_name (str): Name of the descriptor that would be selected.

    Returns:
        (class): A OpenCV descriptor class.
    """
    return get_all_descriptors().get(descriptor_name)


# need to test
# TODO: implement get_all_variants_by_class
# TODO: implement get_variants_by_class_ans_variant_type
def get_all_variants():
    """Returns all the variant types and corresponding variants for all the detector and descriptor classes.

    Returns:
        (dict): A dictionary containing all the variant types and corresponding variants for all the detector and descriptor classes.
    """
    return {
        'BOOSTDESC': {
            'desc': {
                'BGM': 100,
                'BGM_HARD': 101,
                'BGM_BILINEAR': 102,
                'LBGM': 200,
                'BINBOOST_64': 300,
                'BINBOOST_128': 301,
                'BINBOOST_256': 302
            }
        },
        'BRIEF': None,
        'DAISY': {
           'norm': {
               'NRM_NONE': 100,
               'NRM_PARTIAL': 101,
               'NRM_FULL': 102,
               'NRM_SIFT': 103
           }
        },
        'FREAK': None,
        'HarrisLaplace': None,
        'LATCH': None,
        'LUCID': None,
        'STAR': None,
        'AGAST': {
            'type': {
                'AGAST_5_8': 0,
                'AGAST_7_12d': 1,
                'AGAST_7_12s': 2,
                'OAST_9_16': 3
            }
        },
        'AKAZE': {
            'diffusivity': {
                'DIFF_PM_G1': 0,
                'DIFF_PM_G2': 1,
                'DIFF_WEICKERT': 2,
                'DIFF_CHARBONNIER': 3
            },
            'descriptor_type': {
                'DESCRIPTOR_KAZE_UPRIGHT': 2,
                'DESCRIPTOR_KAZE': 3,
                'DESCRIPTOR_MLDB_UPRIGHT': 4,
                'DESCRIPTOR_MLDB': 5
            }
        },
        'BRISK': None,
        'FAST': {
            'type': {
                'TYPE_5_8': 0,
                'TYPE_7_12': 1,
                'TYPE_9_16': 2
            }
        },
        'GFTT': None,
        'KAZE': {
            'diffusivity': {
                'DIFF_PM_G1': 0,
                'DIFF_PM_G2': 1,
                'DIFF_WEICKERT': 2,
                'DIFF_CHARBONNIER': 3
            }
        },
        'ORB': {
            'scoreType': {
                'HARRIS_SCORE': 0,
                'FAST_SCORE': 1
            }
        }

    }


def print_dictionary(dict_obj):
    """Pretty prints the dict obj

    Args:
        dict_obj (dict): A python `dict` object

    """
    print(pformat(dict_obj))
