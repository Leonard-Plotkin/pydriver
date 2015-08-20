# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

from .base import KeypointExtractor


class ISSExtractor(KeypointExtractor):
    """ISS keypoint extractor

    If *leafSize* is given the cloud will be downsampled before keypoint extraction.

    See :meth:`~pydriver.pcl.pcl.PCLHelper.getISSPoints()` for parameter details.

    :Parameters:
        salientRadius: float
            ISS salient radius
        nonMaxRadius: float
            Non maxima suppression radius
        minNeighbors: int, optional
            Minimum neighbors for ISS keypoint
        threshold21: float, optional
            Upper bound on the ratio between the second and the first eigenvalue
        threshold32: float, optional
            Upper bound on the ratio between the third and the second eigenvalue
        angleThreshold: float, optional
            Angle threshold that marks points as boundary or regular, only for boundary detection version
        normalRadius: float, optional
            Radius for surface normal estimation, only for boundary detection version
        borderRadius: float, optional
            Radius for boundary estimation, only for boundary detection version
        leafSize: float, optional
            Voxel grid leaf size, disabled by default

    See :class:`~.base.KeypointExtractor` for further information.
    """
    def _getDefaults(self):
        return {'leafSize': None}
    def getKeypointCloud(self, dict scene):
        # get downsampled cloud (or the original one if downsampling is deactivated)
        if self.params['leafSize']:
            cloud_ds = scene['cloud'].downsampleVoxelGrid(self.params['leafSize'])
        else:
            cloud_ds = scene['cloud']

        # construct ISS parameters
        validParams = ('salientRadius', 'nonMaxRadius', 'minNeighbors', 'threshold21', 'threshold32', 'angleThreshold', 'normalRadius', 'borderRadius')
        params = {}
        for param in validParams:
            if param in self.params:
                params[param] = self.params[param]

        keypointCloud = cloud_ds.getISSPoints(**params)
        return keypointCloud
