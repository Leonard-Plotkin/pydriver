# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

from .structs cimport FLOAT_t


# --- rotation ---
cdef void rotate2DXY(FLOAT_t *x, FLOAT_t *y, FLOAT_t angle) nogil
cdef void rotate2DXZ(FLOAT_t *x, FLOAT_t *z, FLOAT_t angle) nogil
cdef void rotate2DXYSinCos(FLOAT_t *x, FLOAT_t *y, FLOAT_t angle_sin, FLOAT_t angle_cos) nogil
cdef void rotate2DXZSinCos(FLOAT_t *x, FLOAT_t *z, FLOAT_t angle_sin, FLOAT_t angle_cos) nogil

# --- angle normalization ---
cdef FLOAT_t normalizeAngle(FLOAT_t angle) nogil
