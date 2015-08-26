# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

import os, pickle, warnings

cimport numpy as cnp
import numpy as np

from ..common.constants import FLOAT_dtype
from .. import geometry, pcl, stereo


class Preprocessor(object):
    """Frame preprocessing pipeline"""

    def __init__(self, reconstructor = 'default', processors = None, dirCache = None):
        """Initialize preprocessor

        If cache directory is given and valid, cached frames will be loaded from cache and uncached frames will be cached.

        In case you don't want to utilize a reconstructor, pass a None value explicitly (explicit value for default reconstructor
        is the string 'default'). For disabling processors pass an empty tuple.

        .. warning:: There is no verification whether the current preprocessing pipeline is identical to the one used to cache
            the frame, so pay attention to not to use the cache directory of a different configuration or you will get misleading results.

        :Parameters:
            reconstructor: object, optional
                Reconstructor object instance which provides a *process(scene)* function returning a dictionary to include in scene information, :class:`StereoReconstructor() <pydriver.stereo.stereo.StereoReconstructor>` by default
            processors: tuple of :class:`cloud processors <CloudProcessor>`, optional
                Cloud processors to apply after initial cloud reconstruction, (:class:`GroundPlaneProcessor`, :class:`RemoveNaNProcessor`) by default
            dirCache: str, optional
                Cache directory, disabled by default
        """
        # 3D reconstruction
        if reconstructor == 'default':
            self.reconstructor = stereo.StereoReconstructor()
        else:
            self.reconstructor = reconstructor

        # list with cloud processors
        self.processors = processors if processors is not None else (
            # default processors
            GroundPlaneProcessor(),     # detect, remove and adjust ground plane
            RemoveNaNProcessor(),       # remove NaN values from cloud
        )

        # set and check cache directory
        self.dirCache = dirCache
        if self.dirCache and not os.path.isdir(self.dirCache):
            warnings.warn(RuntimeWarning("Cache directory '%s' not found, cache disabled" % self.dirCache))
            self.dirCache = None

    def process(self, frame):
        """Process frame and get the processed scene information

        See :class:`~pydriver.datasets.base.BaseReader` for frame format description. The keys 'dataset' and 'frameId' are only
        required when using cache. Other requirements depend on utilized reconstructor and processors.

        Extracted scene information will be stored in a dictionary. It contains the frame information as well as any keys and
        values produced by the reconstructor and processors. The transformation applied to the cloud is stored as a 4x4 matrix
        in key 'transformation' which is an identity matrix by default.

        :Parameters:
            frame: dict
                Dictionary with frame information

        :Returns: scene
        :Returntype: dict
        """
        # get path to cache file of this frame, None if cache can't be used
        if self.dirCache:
            if 'dataset' in frame and 'frameId' in frame:
                fileCache = os.path.join(self.dirCache, '%s_%s.bin' % (frame['dataset'], frame['frameId']))
            else:
                warnings.warn(RuntimeWarning("Frames must contain the keys 'dataset' and 'frameId' in order to use the cache."))
                fileCache = None
        else:
            fileCache = None
        # load scene from cache if configured and scene is already cached
        if fileCache and os.path.isfile(fileCache):
            with open(fileCache, 'rb') as f:
                try:
                    scene = pickle.load(f)
                except Exception as e:
                    warnings.warn(RuntimeWarning("Could not load from cache. Exception: %s" % e))
                    scene = None
            if scene is not None:
                return scene

        # initialize scene information
        scene = {}
        # copy frame information to scene
        for k, v in frame.items():
            scene[k] = v
        # initial transformation is an identity
        scene['transformation'] = np.matrix(np.identity(4, dtype = FLOAT_dtype), copy = False)

        if self.reconstructor:
            # copy reconstruction information to scene
            for k, v in self.reconstructor.process(scene).items():
                scene[k] = v

        # apply cloud processors
        for processor in self.processors:
            processor.process(scene)

        # cache scene if cache is configured
        if fileCache:
            try:
                with open(fileCache, 'wb') as f:
                    pickle.dump(scene, f, protocol = pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                warnings.warn(RuntimeWarning("Could not save to cache. Exception: %s" % e))
        return scene


# -- lidar reconstructor

class LidarReconstructor(object):
    """Class for reconstructing point clouds with lidar and optionally image data

    Required scene information: lidar['XYZ'].

    Optional scene information: lidar['RGB'], img_left, img_right, calibration['projection_left'], calibration['projection_right'].

    Provided scene information: cloud.
    """

    def __init__(self, useImageColor = False, removeInvisible = False):
        """Initialize reconstructor

        The reconstructor will use scene['lidar']['XYZ'] for point coordinates and scene['lidar']['RGB'] as coloring
        information if available. If *useImageColor* is True the color of points visible in the images will be overwritten
        by image color using the 'projection_left' and 'projection_right' matrices in scene['calibration'].

        :Parameters:
            useImageColor: bool, optional
                Flag whether to use color images for color reconstruction, *False* by default
            removeInvisible: bool, optional
                Flag whether to remove points which are not visible in images, *False* by default
        """
        self._useImageColor = useImageColor
        self._removeInvisible = removeInvisible

    def process(self, scene):
        cdef:
            cnp.ndarray XYZ, RGB
            cnp.ndarray img_left, img_right, img
            cnp.ndarray XY_left, XY_right, XY
            cnp.ndarray mask
            int i, x, y

        XYZ = scene['lidar']['XYZ']
        if 'RGB' in scene['lidar']:
            # use available lidar color information
            RGB = scene['lidar']['RGB']
        else:
            # initialize color to gray
            RGB = np.zeros((XYZ.shape[0], 3), dtype = np.uint8)
            RGB[:, :] = 127

        if self._useImageColor or self._removeInvisible:
            if 'calibration' not in scene:
                raise ValueError("Scene must provide 'calibration' key for using images.")
            if 'img_left' in scene and 'projection_left' in scene['calibration']:
                img_left = scene['img_left']
                XY_left = np.round(geometry.affineTransform(XYZ, scene['calibration']['projection_left'])).astype(np.int32)
            else:
                img_left = None
                XY_left = None
            if 'img_right' in scene and 'projection_right' in scene['calibration']:
                img_right = scene['img_right']
                XY_right = np.round(geometry.affineTransform(XYZ, scene['calibration']['projection_right'])).astype(np.int32)
            else:
                img_right = None
                XY_right = None
            if XY_left is None and XY_right is None:
                raise ValueError("Scene must provide at least one image with calibration for using images.")
            if self._removeInvisible:
                # used for masking invisible points, mask everything initially
                mask = np.ones(XYZ.shape[0], dtype = np.bool)

            # initialize tuples with available images and projected coordinates
            data = [(img, XY) for img, XY in [(img_left, XY_left), (img_right, XY_right)] if img is not None and XY is not None]
            for i in range(XYZ.shape[0]):
                for img, XY in data:
                    x = XY[i, 0]
                    y = XY[i, 1]
                    if XYZ[i, 2]>0 and x>=0 and y>=0 and y<img.shape[0] and x<img.shape[1]:
                        # point visible
                        if self._useImageColor:
                            # set color
                            RGB[i, 0] = img[y, x, 0]
                            RGB[i, 1] = img[y, x, 1]
                            RGB[i, 2] = img[y, x, 2]
                        if self._removeInvisible:
                            # unmask
                            mask[i] = False
                        # no need to look at other images
                        break
            if self._removeInvisible:
                # extract remaining visible points
                invertedMask = np.invert(mask)
                XYZ = XYZ[invertedMask]
                RGB = RGB[invertedMask]

        result = {'cloud': pcl.PCLHelper(XYZ, RGB)}
        return result


# --- cloud processors ---

class CloudProcessor(object):
    """Base class for cloud processors"""
    def __init__(self, **kwargs):
        """Initialize cloud processor"""
        # store given arguments
        self.params = kwargs
        # set defaults if not given
        for key, value in self._getDefaults().items():
            if key not in self.params:
                self.params[key] = value
    def _getDefaults(self):
        """Get dictionary with default parameters"""
        return {}
    def process(self, scene):
        """Process scene inplace

        :Parameters:
            scene: dict
                Scene information as described in :class:`Preprocessor` documentation
        """
        # don't do anything
        pass

class RemoveNaNProcessor(CloudProcessor):
    """Remove *NaN* values

    The resulting cloud will be **not** organized. Useful for decreasing cloud size and improving performance if organized clouds are not required.

    Required scene information: cloud.

    Modified scene information: cloud.
    """
    def process(self, scene):
        scene['cloud'].removeNaN()

class ViewportProcessor(CloudProcessor):
    """Restrict viewport

    Required scene information: cloud.

    Modified scene information: cloud.

    :Parameters:
        viewport: tuple, optional
            Tuple of 3 tuples with coordinate ranges for x-, y-, and z-coordinates, *((-50,50), (-4,0), (0,100))* by default
    """
    def _getDefaults(self):
        return {
            'viewport': (
                (-50, 50),  # X range
                (-4, 0),    # Y range (extract points between 0 and 4 meters above the ground, ground plane expected to be adjusted)
                (0, 100),   # Z range
            ),
        }
    def process(self, scene):
        scene['cloud'].restrictViewport(self.params['viewport'])

class DownsampleProcessor(CloudProcessor):
    """Downsample cloud

    The resulting cloud will be **not** organized.

    Required scene information: cloud.

    Modified scene information: cloud.

    :Parameters:
        leafSize: float, optional
            Voxel grid leaf size, *0.1* by default
    """
    def _getDefaults(self):
        return {'leafSize': 0.1}
    def process(self, scene):
        scene['cloud'] = scene['cloud'].downsampleVoxelGrid(self.params['leafSize'])

class GroundPlaneProcessor(CloudProcessor):
    """Detect and remove ground plane, adjust cloud to have ground plane at y=0

    The given viewport will be used to extract a smaller portion of the cloud for ground plane detection. This speeds up
    ground detection and decreases wrong detections. The ground is expected anywhere between 0 and 4 meters below the
    camera by default, restrict this range according to your scenario to get faster and more reliable detections.

    Required scene information: cloud, transformation.

    Modified scene information: cloud, transformation.

    :Parameters:
        viewport: tuple, optional
            Tuple of 3 tuples with coordinate ranges for x-, y-, and z-coordinates, *None* to disable, *((-10,10), (0,4), (5,35))* by default
        leafSize: float, optional
            Downsample cloud for faster ground plane detection, *0* to disable, *0.3* by default
        maxAngle: float, optional
            Maximal angle (in radian) between detected ground plane and xz plane, *Pi/4* by default
        maxDistance: float, optional
            Maximal distance to the plane for a point to be considered an inlier (RANSAC parameter), *0.1* by default
        remove: bool, optional
            Remove ground plane after detection, *True* by default
        adjust: bool, optional
            Adjust cloud to have ground plane at y=0, *True* by default
    """
    def _getDefaults(self):
        return {
            'viewport': (
                (-10, 10),  # X range
                (0, 4),     # Y range (ground plane expected to be 0 to 4 meters below the camera)
                (5, 35),    # Z range (don't consider points too close to the vehicle, their estimation can be flawed
            ),
            'leafSize': 0.3,
            'maxAngle': np.deg2rad(45),
            'maxDistance': 0.1,
            'remove': True,
            'adjust': True,
        }
    def process(self, scene):
        if not (self.params['adjust'] or self.params['remove']):
            # nothing to do
            warnings.warn(RuntimeWarning("GroundPlaneProcessor has nothing to do, neither adjustment nor removal requested."))
            return

        # copy original cloud
        cloudDetection = scene['cloud'].copy()

        if self.params['viewport'] is not None:
            # restrict reconstruction data to ground viewport where ground plane is expected
            cloudDetection.restrictViewport(self.params['viewport'])

        # remove NaN values in the cloud used for detection, we don't need an organized cloud here
        cloudDetection.removeNaN()

        # downsample cloud for faster ground plane detection
        if self.params['leafSize'] > 0:
            cloudDetection = cloudDetection.downsampleVoxelGrid(self.params['leafSize'])

        # sanity check
        if cloudDetection.getCloudSize()[0] < 10:
            # invalid input, e.g. empty cloud
            raise ValueError("Ground plane could not be processed, is the cloud empty in the expected ground viewport?")

        # detect ground plane
        groundPlane, gpTransformation = cloudDetection.detectGroundPlane(self.params['maxAngle'], self.params['maxDistance'])

        if self.params['remove']:
            # remove ground plane
            scene['cloud'].removeGroundPlane(self.params['maxDistance'], groundPlane)

        if self.params['adjust']:
            # adjust ground plane
            scene['cloud'].transform(gpTransformation)
            # update transformation
            scene['transformation'] = gpTransformation * scene['transformation']
