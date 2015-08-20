# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

cimport numpy as cnp
import numpy as np

import skimage.exposure, skimage.util

from ..common.structs cimport FLOAT_t
from ..common.constants import FLOAT_dtype
from .. import geometry, pcl


# --- utility functions ---

def disparity2depth(cnp.ndarray[FLOAT_t, ndim=2] disparityMap, cnp.ndarray[FLOAT_t, ndim=2] reprojection):
    """Compute depth map from disparity map and reprojection matrix"""
    # depth = fT / disparity = (1 / disparity) * (f / (1/T))
    depthMap = (1 / disparityMap) * (reprojection[2, 3] / abs(reprojection[3, 2]))
    return depthMap.astype(FLOAT_dtype, copy = False)

def depth2disparity(cnp.ndarray[FLOAT_t, ndim=2] depthMap, cnp.ndarray[FLOAT_t, ndim=2] reprojection):
    """Compute disparity map from depth map and reprojection matrix"""
    # depth = fT / disparity
    # => disparity = 1 / (depth / fT) = 1 / (depth / (f / (1/T)))
    disparityMap = 1 / (depthMap / (reprojection[2, 3] / abs(reprojection[3, 2])))
    return disparityMap.astype(FLOAT_dtype, copy = False)


# --- Stereo reconstructor class ---

class StereoReconstructor(object):
    """Pipeline for reconstructing 3D point clouds from stereo images

    The class preprocesses the input images and then uses matchers (e.g. :class:`OpenCVMatcher`) in the specified order
    to produce the final disparity map out of their results.

    The function :meth:`computeDisparity()` is implemented so a :class:`StereoReconstructor` can also be used as a matcher in another :class:`StereoReconstructor`.

    Required scene information: img_left, img_right, calibration['reprojection'].

    Provided scene information: cloud, disparityMap_left, depthMap_left.
    """

    def __init__(self, tuple matchers = None):
        """Initialize *StereoReconstructor* instance

        Given matchers must implement the *computeDisparity(img_left, img_right)* function which returns the disparity map. Invalid
        disparity values must be negative (**NOT** *NaN*).

        If *matchers* is not given, :class:`OpenCVMatcher() <OpenCVMatcher>` will be used.

        :Parameters:
            matchers: tuple, optional
                Instances of matchers to use, their results will be fused
        """
        if matchers is None:
            matcher = OpenCVMatcher()
            self.matchers = (matcher,)
        else:
            self.matchers = matchers

    def process(self, scene):
        """Process scene and return 3D information

        The returned perspective is that of the left image.

        :Parameters:
            scene: dict
                Scene information

        :Returns: dictionary with cloud, disparity map and depth map
        :Returntype: dict{'cloud': :class:`~pydriver.pcl.pcl.PCLHelper`, 'disparityMap_left': np.ndarray[FLOAT_dtype, ndim=2], 'depthMap_left': np.ndarray[FLOAT_dtype, ndim=2]}
        """
        cdef:
            cnp.ndarray[FLOAT_t, ndim=2] disparityMap_left
            cnp.ndarray[FLOAT_t, ndim=3] reconstruction_left
        # 3D reconstruction
        disparityMap_left = self.computeDisparity(scene['img_left'], scene['img_right'])
        # invalidate invalid (-1) points and those being very far away (not reliable)
        disparityMap_left[np.where(disparityMap_left < 1.0)] = np.nan
        # reconstruct XYZ values
        reconstruction_left = geometry.image2space(disparityMap_left, scene['calibration']['reprojection'])

        # construct cloud
        cloud_left = pcl.PCLHelper(reconstruction_left, scene['img_left'])

        result = {'cloud': cloud_left,
                  'disparityMap_left': disparityMap_left,
                  'depthMap_left': disparity2depth(disparityMap_left, scene['calibration']['reprojection']),
                  }
        return result

    def computeDisparity(self, cnp.ndarray img_left, cnp.ndarray img_right):
        """Compute left disparity map for a pair of left and right images

        Image data type and number of channels must be accepted by all matchers.

        This function allows a :class:`StereoReconstructor` instance to be used as a matcher in another :class:`StereoReconstructor` instance.

        :Parameters:
            img_left: np.ndarray[ndim=2 or 3]
                Array with left image data
            img_right: np.ndarray[ndim=2 or 3]
                Array with right image data

        :Returns: Array with disparity map (shape equals to height and width of input images)
        :Returntype: np.ndarray[FLOAT_dtype, ndim=2]
        """
        cdef:
            cnp.ndarray[FLOAT_t, ndim=2] disp_left, cur_disp_left

        # initialize disparity map to None for faster computation with first window_size
        disp_left = None

        # fuse disparity maps with these disparity settings
        for matcher in self.matchers:
            # compute current disparity map
            cur_disp_left = matcher.computeDisparity(img_left, img_right)
            if disp_left is None:
                # no disparity map yet, replace it with current disparity map
                disp_left = cur_disp_left
            else:
                # set disparity values where they are currently missing (-1 values) or very far away (not reliable)
                missing_ids = np.where(disp_left < 1.0)
                disp_left[missing_ids] = cur_disp_left[missing_ids]
        return disp_left


class OpenCVMatcher(object):
    """Disparity computation with `OpenCV <http://opencv.org/>`_ *StereoSGBM* class

    Initialization raises an *ImportError* if *cv2* can not be imported.
    """

    def __init__(self, **kwargs):
        """Initialize OpenCVMatcher instance

        Keyword arguments are the non-default parameters to pass to *cv2.StereoSGBM()*.

        The class uses its own default parameters which do not match OpenCV defaults.
        """
        # keyword arguments for cv2.StereoSGBM()
        SADWindowSize = kwargs.get('SADWindowSize', 1)
        self.params = {
            'minDisparity': kwargs.get('minDisparity', 0),
            'numDisparities': kwargs.get('numDisparities', 128),
            'SADWindowSize': SADWindowSize,
            'P1': kwargs.get('P1', 16 * 3 * min(SADWindowSize, 7)**2),  # X * 3[channels] * SADWindowSize**2
            'P2': kwargs.get('P2', 128 * 3 * min(SADWindowSize, 7)**2), # X * 3[channels] * SADWindowSize**2
            'disp12MaxDiff': kwargs.get('disp12MaxDiff', 1),
            'preFilterCap': kwargs.get('preFilterCap', 0),
            'uniquenessRatio': kwargs.get('uniquenessRatio', 10),
            'speckleWindowSize': kwargs.get('speckleWindowSize', 200),
            'speckleRange': kwargs.get('speckleRange', 16),
            'fullDP': True,                                             # True uses a lot of memory
        }
        # import OpenCV here so we can dynamically decide whether to use this class
        import cv2
        # initialize StereoSGBM() instance
        self._StereoSGBM = cv2.StereoSGBM(**self.params)

    def computeDisparity(self, cnp.ndarray img_left, cnp.ndarray img_right):
        """Compute left disparity map for a pair of left and right images

        Input images can have any type. They must be of same size and have 2 or 3 dimensions (3 channels if 3-dimensional).

        :Parameters:
            img_left: np.ndarray[ndim=2 or 3]
                Array with left image data
            img_right: np.ndarray[ndim=2 or 3]
                Array with right image data

        :Returns: Array with disparity map (shape equals to height and width of input images)
        :Returntype: np.ndarray[FLOAT_dtype, ndim=2]
        """
        cdef:
            cnp.ndarray[FLOAT_t, ndim=2] disp_left
        # make sure we have uint8 (copy data only if required)
        img_left = skimage.util.img_as_ubyte(img_left)
        img_right = skimage.util.img_as_ubyte(img_right)

        disp_left = self._StereoSGBM.compute(img_left, img_right).astype(FLOAT_dtype, copy = False)
        disp_left /= 16.0   # required by cv2 algorithm to get the true disparity
        return disp_left
