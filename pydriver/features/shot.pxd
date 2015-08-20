# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

from pydriver.pcl.pcl cimport PCLHelper  # a relative import would fail at runtime

from .base cimport FeatureExtractor


cdef class SHOTExtractor(FeatureExtractor):
    cpdef tuple getFeatures(SHOTExtractor self, dict scene, PCLHelper keypointCloud)
    cdef tuple _computeSHOT(SHOTExtractor self, PCLHelper cloud, PCLHelper keypointCloud)

cdef class SHOTColorExtractor(SHOTExtractor):
    cdef tuple _computeSHOT(SHOTColorExtractor self, PCLHelper cloud, PCLHelper keypointCloud)
