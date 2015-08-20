# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

from pydriver.pcl.pcl cimport PCLHelper  # a relative import would fail at runtime


cdef class FeatureExtractor:
    cdef:
        readonly:
            dict params
    cpdef tuple getFeatures(FeatureExtractor self, dict scene, PCLHelper keypointCloud)
    cpdef dict _getDefaults(FeatureExtractor self)
