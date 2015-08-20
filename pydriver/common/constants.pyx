# -*- coding: utf-8 -*-
"""
Python constants, changing does not require full recompilation
"""
from __future__ import absolute_import, division

import multiprocessing

import numpy as np

from .cconstants cimport C_PI, C_CATEGORY_LENGTH


# === proxies for accessing compile-time C constants from Python code ===

PI = C_PI
CATEGORY_LENGTH = C_CATEGORY_LENGTH


# === Python constants ===

# recommended number of threads (where applicable)
NUMBER_OF_THREADS = multiprocessing.cpu_count()


# === run-time type definitions ===

# type definition for float values (where they are not required to be of some specific float type)
# must match FLOAT_t (in structs.h and structs.pxd)
FLOAT_dtype = np.float32

# NumPy array type for object and keypoint positions
# it has to define an identical structure as Position (in structs.h and structs.pxd)
Position_dtype = np.dtype([
    ('x', FLOAT_dtype),
    ('y', FLOAT_dtype),
    ('z', FLOAT_dtype),
    # rotation around Y-axis (also used as local reference frame of keypoints/features)
    ('rotation_y', FLOAT_dtype),
])

# NumPy array type for detections
# it has to define an identical structure as Detection (in structs.h and structs.pxd)
Detection_dtype = np.dtype([
    # category of the object, 'negative' for negative detections (string with fixed number of characters)
    ('category', 'S%d' % CATEGORY_LENGTH),
    # position of the object
    ('position', Position_dtype),
    # dimensions of the object
    ('height', FLOAT_dtype),
    ('width', FLOAT_dtype),
    ('length', FLOAT_dtype),
    # weight of Detection
    ('weight', FLOAT_dtype),
])

# NumPy array type containing tetragonal prism data
# it has to define an identical structure as TetragonalPrism (in structs.h and structs.pxd)
TetragonalPrism_dtype = np.dtype([
    ('xyz', FLOAT_dtype, (4, 3)),
    ('height_min', FLOAT_dtype),
    ('height_max', FLOAT_dtype),
])
