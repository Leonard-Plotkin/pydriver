# -*- coding: utf-8 -*-
"""Common functions

Use functions_include.pyx directly for significant speedup for large numbers of calls.
"""
from __future__ import absolute_import, division

from .structs cimport FLOAT_t


# fast C functions
include "../common/functions_include.pyx"


# slow Python proxies to some of them

cpdef pyRotate2DXY(FLOAT_t x, FLOAT_t y, FLOAT_t angle):
    cdef:
        FLOAT_t cx = x, cy = y
    rotate2DXY(&cx, &cy, angle)
    return (cx, cy)

cpdef pyRotate2DXZ(FLOAT_t x, FLOAT_t z, FLOAT_t angle):
    cdef:
        FLOAT_t cx = x, cz = z
    rotate2DXZ(&cx, &cz, angle)
    return (cx, cz)

cpdef FLOAT_t pyNormalizeAngle(FLOAT_t angle):
    return normalizeAngle(angle)
