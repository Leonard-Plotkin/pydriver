# -*- coding: utf-8 -*-
"""
Coordinate system
----------------------------------------
The coordinate system is right-handed with positive x-axis pointing right, y-axis pointing down and z-axis pointing away from the viewer.

NumPy array types
----------------------------------------
.. data:: FLOAT_dtype
    Alias for the NumPy float type used in the library

    Use this type for your data to reduce conversions between float data types.

.. data:: Position_dtype
    Structure for object and keypoint positions

    x: :const:`FLOAT_dtype`
        x-coordinate
    y: :const:`FLOAT_dtype`
        y-coordinate
    z: :const:`FLOAT_dtype`
        z-coordinate
    rotation_y: :const:`FLOAT_dtype`
        Rotation around y-axis in radian, 0 points to the positive x-direction

.. data:: Detection_dtype
    Structure for detections

    category: str
        Object category, *'negative'* for negative detections
    position: :const:`Position_dtype`
        Object position
    height: :const:`FLOAT_dtype`
        Object height
    width: :const:`FLOAT_dtype`
        Object width
    length: :const:`FLOAT_dtype`
        Object length
    weight: :const:`FLOAT_dtype`
        Detection weight
"""
from __future__ import absolute_import, division


from .version import __version_info__, __version__

from .common.constants import FLOAT_dtype, Position_dtype, Detection_dtype

from . import datasets
from . import detectors
from . import evaluation
from . import features
from . import geometry
from . import keypoints
from . import pcl
from . import preprocessing
from . import stereo

__all__ = [
    'FLOAT_dtype', 'Position_dtype', 'Detection_dtype',
    'datasets', 'detectors', 'evaluation', 'features', 'geometry', 'keypoints', 'pcl', 'preprocessing', 'stereo',
    '__version_info__', '__version__',
]
