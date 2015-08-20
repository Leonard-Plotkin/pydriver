# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

cimport numpy as cnp
import numpy as np

from ..common.structs cimport FLOAT_t
from pydriver.pcl.pcl cimport PCLHelper  # a relative import would fail at runtime
from ..common.constants import FLOAT_dtype, Position_dtype
from .. import keypoints

from .base cimport FeatureExtractor


cdef class SHOTExtractor(FeatureExtractor):
    """Compute SHOT features

    Feature has 352 dimensions (or 353 including keypoint y-coordinate).

    :Parameters:
        shotRadius: float
            Radius of SHOT feature

    For other parameters, see :class:`~.base.FeatureExtractor`.
    """
    @property
    def dims(SHOTExtractor self):
        return 352 + (1 if self.params['includeY'] else 0)

    cpdef tuple getFeatures(SHOTExtractor self, dict scene, PCLHelper keypointCloud):
        cdef:
            cnp.ndarray rawFeature, validityMask
            cnp.ndarray[FLOAT_t, ndim=2] rawKeypoints, normals
            cnp.ndarray[FLOAT_t] lrfs

            # returned results
            cnp.ndarray[FLOAT_t, ndim=2] feature
            cnp.ndarray keypointPositions   # dtype = Position_dtype
        # get input keypoints as array
        rawKeypoints = keypointCloud.toArray(flat = True, extractRGB = False)

        if rawKeypoints.shape[0] == 0:
            # nothing to do, return empty arrays
            return (np.empty(0, dtype = Position_dtype), np.empty((0, self.dims), dtype = FLOAT_dtype))

        if self.params['fixedY'] is not None:
            # replace y-coordinate of keypoints
            rawKeypoints[:, 1] = self.params['fixedY']
            # construct new keypoint cloud with changed y-coordinate
            keypointCloud = PCLHelper(rawKeypoints)

        # compute features
        rawFeature, validityMask = self._computeSHOT(scene['cloud'], keypointCloud)
        if validityMask.any():
            # there are invalid values
            invertedValidityMask = np.invert(validityMask)
            rawFeature = rawFeature[invertedValidityMask]
            rawKeypoints = rawKeypoints[invertedValidityMask]
        # keypointCloud may be out of sync now, don't reconstruct it since we don't need it anymore
        del keypointCloud

        # extract first SHOT LRF axis and use it as normal (we only need repeatability)
        normals = rawFeature['rf'][:, 0]
        lrfs = keypoints.normals2lrfs(rawKeypoints, normals)

        # process raw SHOT data to final feature
        feature = np.empty((rawFeature.shape[0], self.dims), dtype = FLOAT_dtype)
        if self.params['includeY']:
            # store y-coordinate in first dimension
            feature[:, 0] = rawKeypoints[:, 1]
            # copy SHOT feature to other dimensions
            feature[:, 1:] = rawFeature['descriptor']
        else:
            # get SHOT feature
            feature = rawFeature['descriptor']

        # process raw keypoints to final keypoints (x, y, z, rotation_y)
        keypointPositions = np.empty(rawKeypoints.shape[0], dtype = Position_dtype)
        # copy x-, y-, and z-coordinates
        keypointPositions.view(FLOAT_dtype).reshape(keypointPositions.shape[0], 4)[:, :3] = rawKeypoints
        # copy LRFs
        keypointPositions['rotation_y'] = lrfs

        return (keypointPositions, feature)

    cdef tuple _computeSHOT(SHOTExtractor self, PCLHelper cloud, PCLHelper keypointCloud):
        """Helper function to compute the raw feature so SHOTColorExtractor can overwrite it"""
        return cloud.computeSHOT(self.params['shotRadius'], keypointCloud)


cdef class SHOTColorExtractor(SHOTExtractor):
    """Compute SHOT+Color features

    Feature has 1344 dimensions (or 1345 including keypoint y-coordinate).

    Parameters are identical to those of :class:`SHOTExtractor`.
    """
    @property
    def dims(SHOTColorExtractor self):
        return 1344 + (1 if self.params['includeY'] else 0)
    cdef tuple _computeSHOT(SHOTColorExtractor self, PCLHelper cloud, PCLHelper keypointCloud):
        return cloud.computeSHOTColor(self.params['shotRadius'], keypointCloud)
