# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

from .base import KeypointExtractor


class HarrisExtractor(KeypointExtractor):
    """Harris keypoint extractor

    If *leafSize* is given the cloud will be downsampled before keypoint extraction.

    See :meth:`~pydriver.pcl.pcl.PCLHelper.getHarrisPoints()` for parameter details.

    :Parameters:
        radius: float
            Radius for normal estimation and non maxima suppression
        refine: bool, optional
            Flag whether to refine keypoints
        threshold: FLOAT_dtype, optional
            Threshold value for detecting corners
        method: int, optional
            Method to use (1-5)
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

        # construct Harris parameters
        validParams = ('radius', 'refine', 'threshold', 'method')
        params = {}
        for param in validParams:
            if param in self.params:
                params[param] = self.params[param]

        keypointCloud = cloud_ds.getHarrisPoints(**params)
        return keypointCloud
