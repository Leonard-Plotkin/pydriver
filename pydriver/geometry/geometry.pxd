# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

cimport numpy as cnp

from ..common.structs cimport FLOAT_t


cdef extern from "stdlib.h":
    void *memcpy(void *dest, void *src, size_t n) nogil

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(cnp.ndarray arr, int flags)

cpdef cnp.ndarray[FLOAT_t, ndim=2] homogenuous2cartesian(cnp.ndarray[FLOAT_t, ndim=2] homCoords, bint inplace = *)
cpdef cnp.ndarray[FLOAT_t, ndim=2] cartesian2homogenuous(cnp.ndarray[FLOAT_t, ndim=2] carCoords)
cpdef cnp.ndarray[FLOAT_t, ndim=2] affineTransform(cnp.ndarray[FLOAT_t, ndim=2] carCoords, cnp.ndarray[FLOAT_t, ndim=2] transformation)

cpdef cnp.ndarray[FLOAT_t, ndim=2] image2space(cnp.ndarray disparityMap, cnp.ndarray[FLOAT_t, ndim=2] reprojection)

cpdef dict transform3DBox(dict box3D, cnp.ndarray[FLOAT_t, ndim=2] transformation)
cpdef dict project3DBox(dict box3D, cnp.ndarray[FLOAT_t, ndim=2] projection)
cpdef cnp.ndarray[FLOAT_t, ndim=2] getNormalizationTransformation(dict box3D)
cpdef list extractNormalizedOrientedBoxes(FLOAT_t[:, :] xyz, cnp.uint8_t[:, :] rgb, list boxes)
cpdef cnp.ndarray[FLOAT_t, ndim=2] get3DBoxVertices(dict box3D)
