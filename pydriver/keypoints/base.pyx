# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

cimport libc.math

cimport numpy as cnp
import numpy as np

from ..common.structs cimport FLOAT_t
from ..common.constants import FLOAT_dtype


class KeypointExtractor(object):
    """Base class for keypoint extractors"""
    def __init__(self, **kwargs):
        """Initialize keypoint extractor"""
        # store given arguments
        self.params = kwargs
        # set defaults if not given
        for key, value in self._getDefaults().items():
            if key not in self.params:
                self.params[key] = value
    def _getDefaults(self):
        """Get dictionary with default parameters"""
        return {}
    def getKeypointCloud(self, dict scene):
        """Extract keypoints from scene

        :Parameters:
            scene: dict
                Scene information as described in :class:`~pydriver.preprocessing.preprocessing.Preprocessor` documentation

        :Returns: cloud with keypoints
        :Returntype: :class:`~pydriver.pcl.pcl.PCLHelper`
        """
        raise NotImplementedError("KeypointExtractor is an abstract class, use subclasses.")


cpdef cnp.ndarray[FLOAT_t] normals2lrfs(cnp.ndarray[FLOAT_t, ndim=2] keypoints, cnp.ndarray[FLOAT_t, ndim=2] normals):
    """Compute local reference frames for keypoints and their normals

    The LRF will be computed to point to (0,0,0) to resolve ambiguity. For normals pointing along the y-axis
    the resulting LRF will be 0.

    Keypoints and normals must have the shape *(number of keypoints, 3)*.

    :Parameters:
        keypoints: np.ndarray[FLOAT_dtype, ndim=2]
            Array of keypoint coordinates
        normals: np.ndarray[FLOAT_dtype, ndim=2]
            Array of keypoint normals

    :Returns: rotation angles around y-axis of normals projected to xz plane
    :Returntype: np.ndarray[FLOAT_dtype]
    """
    cdef:
        size_t nKeypoints, i
        FLOAT_t nx, nz
        cnp.ndarray[FLOAT_t] lrfs
    # sanity checks
    assert keypoints.shape[0] == normals.shape[0], "Keypoints and normals arrays must contain the same number of entries"
    assert keypoints.shape[1] == 3, "Keypoints array must be of shape (n, 3) for n keypoints"
    assert normals.shape[1] == 3, "Normals array must be of shape (n, 3) for n keypoints"

    # initialization
    nKeypoints = keypoints.shape[0]
    lrfs = np.empty(nKeypoints, dtype = FLOAT_dtype)

    for i in range(nKeypoints):
        # we need only the X/Z projection of the normal
        nx = normals[i, 0]
        nz = normals[i, 2]

        if normals[i].dot(keypoints[i]) > 0.0:
            # invert normal if it points in same direction as viewing direction to get a unique and unambiguous LRF
            nx *= -1
            nz *= -1

        lrfs[i] = <FLOAT_t>-libc.math.atan2(nz, nx)     # -atan2 because of the right-handed 3D coordinate system
    return lrfs
