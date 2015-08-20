# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

cimport numpy as cnp

from .cconstants cimport C_CATEGORY_LENGTH


cdef extern from "structs.h":
    # type definition for float values (where they are not required to be of some specific float type)
    # the type "cnp.float_t" doesn't matter here except for Cython choice of conversion function between C and Python values
    # ... so it can have more or less precision than the actual type defined in structs.h
    ctypedef cnp.float_t FLOAT_t

    cdef struct Position:
        FLOAT_t x
        FLOAT_t y
        FLOAT_t z
        FLOAT_t rotation_y

    cdef struct Detection:
        char[C_CATEGORY_LENGTH] category
        Position position
        FLOAT_t height
        FLOAT_t width
        FLOAT_t length
        FLOAT_t weight

    cdef struct TetragonalPrism:
        FLOAT_t xyz[4][3]
        FLOAT_t height_min
        FLOAT_t height_max

    cdef struct SHOTFeature:
        FLOAT_t descriptor[352]
        FLOAT_t rf[9]
    cdef struct SHOTColorFeature:
        FLOAT_t descriptor[1344]
        FLOAT_t rf[9]
