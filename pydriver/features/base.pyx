# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

import warnings

from pydriver.pcl.pcl cimport PCLHelper  # a relative import would fail at runtime


cdef class FeatureExtractor(object):
    """Base class for feature extractors

    :Parameters:
        fixedY: float, optional
            Ignore y-coordinate of keypoint and use this value, disabled by default
        includeY: bool, optional
            Include y-coordinate of keypoint as first feature dimension, *False* by default
    """
    def __init__(FeatureExtractor self, **kwargs):
        """Initialize feature extractor"""
        # store given arguments
        self.params = kwargs
        # set defaults if not given
        for key, value in self._getDefaults().items():
            if key not in self.params:
                self.params[key] = value

        if self.params['fixedY'] is not None and self.params['includeY']:
            warnings.warn(UserWarning("Using 'fixedY' together with 'includeY' produces a constant value for the first dimension."))

    cpdef dict _getDefaults(FeatureExtractor self):
        """Get dictionary with default parameters"""
        return {'fixedY': None, 'includeY': False}

    @property
    def dims(FeatureExtractor self):
        """Get number of dimensions of the returned feature"""
        raise NotImplementedError("FeatureExtractor is an abstract class, use subclasses.")

    cpdef tuple getFeatures(FeatureExtractor self, dict scene, PCLHelper keypointCloud):
        """Extract features from scene at given keypoints

        Returned keypoints may be identical to *keypointCloud* but can also differ (e.g. if the
        feature ignores the original y-coordinate of keypoints). The number of returned keypoints is
        equal or less than the number of given keypoints (some keypoints may produce invalid features).

        Features is a :const:`~pydriver.FLOAT_dtype` array of shape *(number of keypoints, number of feature dimensions)*.

        :Parameters:
            scene: dict
                Scene information as described in :class:`~pydriver.preprocessing.preprocessing.Preprocessor` documentation
            keypointCloud: :class:`~pydriver.pcl.pcl.PCLHelper`
                Cloud with keypoints

        :Returns: tuple with keypoints and features
        :Returntype: tuple(np.ndarray[:const:`~pydriver.Position_dtype`], np.ndarray[FLOAT_dtype, ndim=2])
        """
        raise NotImplementedError("FeatureExtractor is an abstract class, use subclasses.")
