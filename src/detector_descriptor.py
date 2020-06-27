import cv2
from pprint import pformat


def get_all_detectors():
    """
    Returns a collection of detector classes in OpenCV (`version 4.2.0`_).

    Returns:
        (`dict`): A dictionary containing a collection of detector classes.

    Examples:
        .. code-block:: python

            In[1]: from src.detector_descriptor import *
            In[2]: get_all_detectors()
            Out[2]:
            {'AGAST': cv2.AgastFeatureDetector,
             'KAZE': cv2.KAZE,
             'AKAZE': cv2.AKAZE,
             'FAST': cv2.FastFeatureDetector,
             'BRISK': cv2.BRISK,
             'ORB': cv2.ORB,
             'GFTT': cv2.GFTTDetector,
             'HarrisLaplace': cv2.xfeatures2d_HarrisLaplaceFeatureDetector,
             'StarDetector': cv2.xfeatures2d_StarDetector}


    See Also:
        :func:`~src.detector_descriptor.get_all_descriptors`

    .. _version 4.2.0:
        https://docs.opencv2.org/4.2.0/modules.html
    """
    return {
        'AGAST': cv2.AgastFeatureDetector,
        'KAZE': cv2.KAZE,
        'AKAZE': cv2.AKAZE,
        'FAST': cv2.FastFeatureDetector,
        'BRISK': cv2.BRISK,
        'ORB': cv2.ORB,
        'GFTT': cv2.GFTTDetector,
        'HarrisLaplace': cv2.xfeatures2d_HarrisLaplaceFeatureDetector,
        'StarDetector': cv2.xfeatures2d_StarDetector
    }


def get_all_descriptors():
    """
    Returns a collection of descriptor classes in OpenCV (`version 4.2.0`_).

    Returns:
        (`dict`):A dictionary containing a collection of descriptor classes.

    .. attention::
        **KAZE** descriptor only supports *KAZE* keypoints and
        **AKAZE** descriptor only supports *KAZE* or *AKAZE* keypoints.

    Examples:
        .. code-block:: python

            In[1]: from src.detector_descriptor import *
            In[2]: get_all_descriptors()
            Out[2]:
            {'LATCH': cv2.xfeatures2d_LATCH,
             'LUCID': cv2.xfeatures2d_LUCID,
             'FREAK': cv2.xfeatures2d_FREAK,
             'DAISY': cv2.xfeatures2d_DAISY,
             'BOOSTDESC': cv2.xfeatures2d_BoostDesc,
             'KAZE': cv2.KAZE,
             'AKAZE': cv2.AKAZE,
             'BRIEF': cv2.xfeatures2d_BriefDescriptorExtractor,
             'BRISK': cv2.BRISK,
             'ORB': cv2.ORB}


    See Also:
        :func:`~src.detector_descriptor.get_all_detectors`

    .. _version 4.2.0:
        https://docs.opencv2.org/4.2.0/modules.html
    """
    return {
        'LATCH': cv2.xfeatures2d_LATCH,
        'LUCID': cv2.xfeatures2d_LUCID,
        'FREAK': cv2.xfeatures2d_FREAK,
        'DAISY': cv2.xfeatures2d_DAISY,
        'BOOSTDESC': cv2.xfeatures2d_BoostDesc,
        'KAZE': cv2.KAZE,
        'AKAZE': cv2.AKAZE,
        'BRIEF': cv2.xfeatures2d_BriefDescriptorExtractor,
        'BRISK': cv2.BRISK,
        'ORB': cv2.ORB
    }


def select_detector(detector_name):
    """
    Selects the detector class from the detector collection and returns it.

    Args:
        detector_name (`str`): Name of the detector that would be selected.

    Returns:
        (`class`): A OpenCV detector class.

    Examples:

        .. code-block:: python

            In[1]: from src.detector_descriptor import *
            In[2]: select_detector('FAST')
            Out[2]: cv2.FastFeatureDetector

    See Also:
        :func:`~src.detector_descriptor.select_descriptor`

    """
    return get_all_detectors().get(detector_name)


def select_descriptor(descriptor_name):
    """
    Selects the descriptor class from the descriptor collection and returns it.

    Args:
        descriptor_name (`str`): Name of the descriptor that would be selected.

    Returns:
        (`class`): A OpenCV descriptor class.

    Examples:

        .. code-block:: python

            In[3]: from src.detector_descriptor import *
            In[4]: select_descriptor('BRISK')
            Out[4]: cv2.BRISK

    See Also:
        :func:`~src.detector_descriptor.select_detector`
    """
    return get_all_descriptors().get(descriptor_name)


def get_all_variants():
    """
    Returns all the variant types and corresponding variants for all the detector and descriptor classes.

    Returns:
        (`dict`): A dictionary containing all the variant types and corresponding variants for all the detector and descriptor classes.

    Examples:

        .. code-block:: python

            In[1]: from src.detector_descriptor import *
            In[2]: get_all_variants()
            Out[2]:
            {'BOOSTDESC': {'desc': {'BGM': 100,
               'BGM_HARD': 101,
               'BGM_BILINEAR': 102,
               'LBGM': 200,
               'BINBOOST_64': 300,
               'BINBOOST_128': 301,
               'BINBOOST_256': 302}},
             'BRIEF': None,
             'DAISY': {'norm': {'NRM_NONE': 100,
               'NRM_PARTIAL': 101,
               'NRM_FULL': 102,
               'NRM_SIFT': 103}},
             'FREAK': None,
             'HarrisLaplace': None,
             'LATCH': None,
             'LUCID': None,
             'STAR': None,
             'AGAST': {'type': {'AGAST_5_8': 0,
               'AGAST_7_12d': 1,
               'AGAST_7_12s': 2,
               'OAST_9_16': 3}},
             'AKAZE': {'diffusivity': {'DIFF_PM_G1': 0,
               'DIFF_PM_G2': 1,
               'DIFF_WEICKERT': 2,
               'DIFF_CHARBONNIER': 3},
              'descriptor_type': {'DESCRIPTOR_KAZE_UPRIGHT': 2,
               'DESCRIPTOR_KAZE': 3,
               'DESCRIPTOR_MLDB_UPRIGHT': 4,
               'DESCRIPTOR_MLDB': 5}},
             'BRISK': None,
             'FAST': {'type': {'TYPE_5_8': 0, 'TYPE_7_12': 1, 'TYPE_9_16': 2}},
             'GFTT': None,
             'KAZE': {'diffusivity': {'DIFF_PM_G1': 0,
               'DIFF_PM_G2': 1,
               'DIFF_WEICKERT': 2,
               'DIFF_CHARBONNIER': 3}},
             'ORB': {'scoreType': {'HARRIS_SCORE': 0, 'FAST_SCORE': 1}}}

    See Also:
        :func:`~src.detector_descriptor.get_all_detectors`
        :func:`~src.detector_descriptor.get_all_descriptors`

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


def get_variants(class_name, variant_type=None):
    """
    Returns all the variants for a specific detector or descriptor class.

    Args:
        class_name (`str`): Specified detector or descriptor class.
        variant_type (`str`, optional): Type of variant. Defaults to None.

    Examples:
        .. attention::
            If variant_type is not specified (None), then the function returns all the variant types and their corresponding variants.

            .. code-block:: python

                In[1] from src.detector_descriptor import *
                In[2] get_variants('AKAZE')
                Out[2]:
                {'diffusivity': {'DIFF_PM_G1': 0,
                  'DIFF_PM_G2': 1,
                  'DIFF_WEICKERT': 2,
                  'DIFF_CHARBONNIER': 3},
                 'descriptor_type': {'DESCRIPTOR_KAZE_UPRIGHT': 2,
                  'DESCRIPTOR_KAZE': 3,
                  'DESCRIPTOR_MLDB_UPRIGHT': 4,
                  'DESCRIPTOR_MLDB': 5}}

            If it is specified then it returns the variants only for the corresponding variant type.

            .. code-block:: python

                In[3] get_variants('AKAZE', 'diffusivity')
                Out[3]:
                {'diffusivity': {'DIFF_PM_G1': 0,
                  'DIFF_PM_G2': 1,
                  'DIFF_WEICKERT': 2,
                  'DIFF_CHARBONNIER': 3},
                 'descriptor_type': {'DESCRIPTOR_KAZE_UPRIGHT': 2,
                  'DESCRIPTOR_KAZE': 3,
                  'DESCRIPTOR_MLDB_UPRIGHT': 4,
                  'DESCRIPTOR_MLDB': 5}}


    .. note::
        All the available detector and descriptor can be retrieved using :func:`~src.detector_descriptor.get_all_detectors` and :func:`~src.detector_descriptor.get_all_descriptors`.

    Returns:
        (:obj:`dict`): A dictionary containing all the variants for a specific detector or descriptor class.

    See Also:
        :func:`~src.detector_descriptor.get_all_detectors`
        :func:`~src.detector_descriptor.get_all_descriptors`


    """
    if class_name not in get_all_variants():
        raise ModuleNotFoundError(f"{class_name} don't exist.")

    if variant_type is None:
        return get_all_variants().get(class_name)

    return get_all_variants().get(class_name).get(variant_type)


def print_dictionary(dict_obj):
    """
    Pretty prints a python dict obj

    Args:
        dict_obj (`dict`): A python `dict` object

    """
    print(pformat(dict_obj))


def initialize_detector(detector_name, variant_type=None, variant=None, additional_args=''):
    """
    Initializes a detector instance and returns it.

    Args:
        detector_name (`str`): Name of the detector.
        variant_type (`str`, optional): Variant type for the specified detector. Defaults to None.
        variant (`int`, optional): Variant of the specified variant type. Defaults to None.
        variant (`str`, optional): Additional argument passed to `.create()`. Defaults to ''.

    Returns:
        (:obj:`cv2`): A `cv2` object instance.

    Important:
        - If `variant_type` is `None`, `variant` also has to be `None`, and vice versa.
        - When `variant_type` and `variant` is `None` the detector would be initialized with default values.

    .. note::
        This method uses `exec`_.

    Examples:
        .. code-block:: python

            In[1]: from src.detector_descriptor import *
            In[2]: initialize_detector('FAST')
            Out[2]: <FastFeatureDetector 000002269365FC70>
            In[3]: initialize_detector('FAST','type',2)
            Out[3]: <FastFeatureDetector 000002269365FC50>

    Raises:
        ModuleNotFoundError: If the `detector_name` doesn't match with any of the available detectors.
        ValueError: If the `variant_type` doesn't match in any of the available variant type (`variant_type`) for the
            specified detector class (`detector_name`).
        ValueError: If the `variant` doesn't match any of the available variant for specified variant type
            (`variant_type`) and detector class (`detector_name`).
    .. _exec:
        https://docs.python.org/3/library/functions.html#exec
    """
    if detector_name not in get_all_detectors():
        raise ModuleNotFoundError(f"{detector_name} don't exist.")

    if variant_type is None and variant is None:
        # `temp` is a temporary dictionary that holds the object instance
        # Attention: `exec()` function has been used to dynamically create the desired
        #   object instance.
        # `exec_string` is the expression that is being executed by the `exec()` function.

        temp = {}
        exec_string = "temp['instance'] = {0}.create({1})".format(str(select_detector(detector_name)).split('\'')[1],
                                                                  additional_args)
        exec(exec_string)

        return temp['instance']

    if bool(variant_type) ^ bool(variant) and variant != 0:
        raise ValueError("Either one of the parameters variant_type or variant is None")

    if variant_type not in get_all_variants().get(detector_name):
        raise ValueError(f"The variant type {variant_type} for class {detector_name} doesn't exist.")

    if variant not in get_variants(detector_name, variant_type).values():
        raise ValueError(f"The variant {variant} of type {variant_type} for class {detector_name} doesn't exist.")

    # `temp` is a temporary dictionary that holds the object instance
    # Attention: `exec()` function has been used to dynamically create the desired
    #   object instance.
    # `exec_string` is the expression that is being executed by the `exec()` function.

    temp = {}
    exec_string = "temp['instance'] = {0}.create({1}={2})".format(str(select_detector(detector_name)).split('\'')[1],
                                                                          variant_type, variant)
    exec(exec_string)

    return temp['instance']


def initialize_descriptor(descriptor_name, variant_type=None, variant=None):
    """
    Initializes a descriptor instance and returns it.

    Args:
        descriptor_name (`str`): Name of the descriptor.
        variant_type (`str`, optional): Variant type for the specified descriptor. Defaults to None.
        variant (`int`, optional): Variant of the specified variant type. Defaults to None.

    Returns:
        (:obj:`cv2`): A `cv2` object instance.

    Important:
        - If `variant_type` is `None`, `variant` also has to be `None`, and vice versa.
        - When `variant_type` and `variant` is `None` the descriptor would be initialized with default values.

    .. Note::
        This method uses `exec`_.

    Examples:
        .. code-block:: python

            In[1]: from src.descriptor_descriptor import *
            In[2]: initialize_descriptor('AKAZE')
            Out[2]: <AKAZE 000002269365FD10>
            In[3]: initialize_descriptor('AKAZE','diffusivity', 3)
            Out[3]: <AKAZE 000002269365FDB0>

    Raises:
        ModuleNotFoundError: If the `descriptor_name` doesn't match with any of the available descriptors.
        ValueError: If the `variant_type` doesn't match in any of the available variant type (`variant_type`) for the
            specified descriptor class (`descriptor_name`).
        ValueError: If the `variant` doesn't match any of the available variant for specified variant type
            (`variant_type`) and descriptor class (`descriptor_name`).

    .. _exec:
        https://docs.python.org/3/library/functions.html#exec
    """
    if descriptor_name not in get_all_descriptors():
        raise ModuleNotFoundError(f"{descriptor_name} don't exist.")

    if variant_type is None and variant is None:
        # `temp` is a temporary dictionary that holds the object instance
        # Attention: `exec()` function has been used to dynamically create the desired
        #   object instance.
        # `exec_string` is the expression that is being executed by the `exec()` function.

        temp = {}
        exec_string = "temp['instance'] = {0}.create()".format(str(select_descriptor(descriptor_name)).split('\'')[1])
        exec(exec_string)

        return temp['instance']

    if bool(variant_type) ^ bool(variant):
        raise ValueError("Either one of the parameters variant_type of variant is None")

    if variant_type not in get_all_variants().get(descriptor_name):
        raise ValueError(f"The variant type {variant_type} for class {descriptor_name} doesn't exist.")

    # if type(variant) is not int:
    #     raise TypeError(f"variant value should be a int type")

    if variant not in get_variants(descriptor_name, variant_type).values():
        raise ValueError(f"The variant {variant} of type {variant_type} for class {descriptor_name} doesn't exist.")

    # `temp` is a temporary dictionary that holds the object instance
    # Attention: `exec()` function has been used to dynamically create the desired
    #   object instance.
    # `exec_string` is the expression that is being executed by the `exec()` function.
    temp = {}
    exec_string = "temp['instance'] = {0}.create({1}={2})".format(str(select_descriptor(descriptor_name)).split('\'')[1],
                                                                          variant_type, variant)
    exec(exec_string)

    return temp['instance']


def available_attributes(var):
    """
    Returns the attributes of a `class` or `object`.

    Important:
        This function checks class/object properties to find `get*` methods and assumes methods starting with `get`
        would return an attribute value of a class/object.

    Args:
        var (`str` or `object`): An object or name of the class.

    Returns:
        (`list`): A list of strings containing the attribute (supposed) names.

    Examples:
        .. attention::
            If `var` is a class name e.g. 'FAST'

            .. code-block:: python

                In[1]: from src.detector_descriptor import *
                In[2]: available_attributes('FAST')
                Out[2]: ['DefaultName', 'NonmaxSuppression', 'Threshold', 'Type']

            If `var` is an object instance.

            .. code-block:: python

                In[1]: from src.detector_descriptor import *
                In[2]: obj = initialize_descriptor('AKAZE')
                In[3]: available_attributes(obj)
                Out[4]:
                ['DefaultName',
                 'DescriptorChannels',
                 'DescriptorSize',
                 'DescriptorType',
                 'Diffusivity',
                 'NOctaveLayers',
                 'NOctaves',
                 'Threshold']

    Raises:
        ModuleNotFoundError: If the class (`var`) doesn't exist.
    """
    if type(var) is str:
        if var in get_all_detectors():
            class_ = select_detector(var)
        elif var in get_all_descriptors():
            class_ = select_descriptor(var)
        else:
            raise ModuleNotFoundError(f"The class: {var} doesn't exist")

        # Check class (class_) properties to find `get*` methods and return them as a list of string.
        # Assuming methods starting with `get` would return an attribute of a class.
        attributes = [value[3:] for value in dir(class_) if 'get' in value and '__' not in value]
    else:
        attributes = [value[3:] for value in dir(var) if 'get' in value and '__' not in value]

    return attributes


def get_attribute(obj, attribute_name):
    """
    Returns the value of the attribute of an object.

    Args:
        obj (:obj:`cv2`):
        attribute_name (`str`): The name of the attribute. *`attribute_name` should have a **capital case** value*

    .. attention::
        All the attributes can be found using :func:`~src.detector_descriptor.available_attributes`

    Returns:
        The value of the attribute.

    .. note::
        This method uses `exec`_

    Examples:
        .. code-block:: python

            In[1]: from src.detector_descriptor import *
            In[2]: obj = initialize_detector('FAST', 'type', 1)
            In[3]: get_attribute(obj, 'Type')
            Out[9]: 1

    Raises:
        AttributeError: If the attribute doesn't exist for the specified object `obj`

    See Also:
        :func:`~src.detector_descriptor.available_attributes`
        :func:`~src.detector_descriptor.set_attribute`

    .. _exec:
        https://docs.python.org/3/library/functions.html#exec
    """
    if attribute_name not in available_attributes(obj):
        raise AttributeError(f"{obj} has no attribute {attribute_name}.\n Make sure the attribute_name is capital case.")

    temp = {}
    exec_string = f"temp['attribute'] = obj.get{attribute_name}()"
    # `exec` is used to dynamically execute python code
    exec(exec_string)
    return temp['attribute']


def set_attribute(obj, attribute_name, val):
    """
    Assigns a value to the attribute of an object.

    Args:
        obj (:obj:`cv2`): A `cv2` object instance.
        attribute_name (`str`): The name of the attribute. `attribute_name` should have a **capital case** value.
        val (`int`): The value that would be assigned. `val` is usually of type `int` but it can be of other types as well.

    .. attention::
        All the attributes can be found using :func:`~src.detector_descriptor.available_attributes`

    Returns:
        The value of the attribute.

    .. note::
        This method uses `exec`_

    Examples:
        .. code-block:: python

            In[1]: from src.detector_descriptor import *
            In[2]: obj = initialize_detector('FAST', 'type', 1)
            In[3]: set_attribute(obj, 'Type', 2)
            Out[9]: 1

    Raises:
        AttributeError: If the attribute doesn't exist for the specified object `obj`

    See Also:
        :func:`~src.detector_descriptor.available_attributes`
        :func:`~src.detector_descriptor.get_attribute`

    .. _exec:
        https://docs.python.org/3/library/functions.html#exec
    """
    if attribute_name not in available_attributes(obj):
        raise AttributeError(f"{obj} has no attribute {attribute_name}.\n Make sure the attribute_name is capital case.")

    exec_string = f"obj.set{attribute_name}({val})"
    # `exec` is used to dynamically execute python code
    exec(exec_string)


all_detectors = list(get_all_detectors().keys())
all_descriptors = list(get_all_descriptors().keys())
# Todo: Check better testing techniques
# Todo: change eval to ast.literal_eval