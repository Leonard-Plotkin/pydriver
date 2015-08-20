# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

import functools, os

import numpy as np
import scipy.misc, skimage.util


def loadImage(filename, flatten = False):
    """Load image file into NumPy array

    The shape of the resulting array will be (height, width) and possibly one
    further dimension with 3 (RGB) or 4 (RGBA) values for channel information.

    See :func:`scipy.misc.imread()` for further information.

    :Parameters:
        filename: string
            File name of image to load
        flatten: boolean, optional
            Convert output to grayscale, *False* by default

    :Returns: NumPy array with image data
    :Returntype: np.ndarray[np.uint8, ndim=2 or 3]
    """
    return skimage.util.img_as_ubyte(scipy.misc.imread(filename, flatten = flatten))


def _union(*args):
    """Return a set containing the union of given sequence of iterators"""
    if not args:
        return set()    # return an empty set if no iterators given
    else:
        return functools.reduce(lambda a,b: set(a) | set(b), args)


class BaseReader(object):
    """Base class for data extractor interfaces

    Frame IDs should be implemented as integers, dataset IDs as strings (or *None* if not applicable).

    See :meth:`getFrameInfo()` for frame format description.
    """

    def __init__(self, directory):
        """Initialize data extractor

        :Parameters:
            directory: string
                Data directory
        """
        if not os.path.exists(directory):
            raise IOError("Could not find directory '%s'" % directory)
        self._dir = directory

    def getFramesInfo(self, frameStart = 0, frameStop = None, frameStep = 1, datasets = None):
        """Lazy generator function to get information about frames

        The function returns the same information as :func:`getFrameInfo`. A frame will not be processed until the generator reaches it.

        The parameters *frameStart*, *frameStop* and *frameStep* will be applied to each requested dataset separately.

        :Parameters:
            frameStart: int, optional
                First frame, 0 by default
            frameStop: int, optional
                Frame at which to stop (will not be yielded), no end by default
            frameStep: int, optional
                Yield every n-th frame of a dataset, 1 by default
            datasets: list of str, optional
                Restrict generator to specified datasets, all datasets by default
        """
        if datasets is None:
            datasets = self.getDatasets()
        for current_dataset in datasets:
            for frameId in self.getFrameIds(current_dataset)[frameStart : frameStop : frameStep]:
                yield self.getFrameInfo(frameId, current_dataset)

    def getDatasets(self):
        """Get sorted list of datasets"""
        raise NotImplementedError("getDatasets() is not implemented in BaseReader")

    def getFrameIds(self, dataset = None):
        """Get sorted list of frame IDs of the optionally specified dataset (if reader supports multiple datasets)"""
        raise NotImplementedError("getFrameIds() is not implemented in BaseReader")

    def getFrameInfo(self, frameId, dataset = None):
        """Get information about a frame in the specified dataset

        :Parameters:
            frameId: int
                Frame ID of the requested frame
            dataset: str or None, optional
                Dataset with the requested frame, only required by readers with multiple datasets

        :Returns: Dictionary with information about the frame:

            'dataset': str or None
                Dataset of the frame, *None* for readers without multiple datasets

            'frameId': int
                Frame ID

            'img_left': NumPy array
                Left image, can be None

            'img_right': NumPy array
                Right image, can be None

            'calibration': dict
                Calibration matrices
                    'reprojection': np.matrix[FLOAT_dtype]
                        3D reconstruction out of disparity, shape: (4, 4)
                    'projection_left': np.matrix[FLOAT_dtype]
                        Projection of camera coordinates to left camera image, shape: (3, 4)
                    'projection_right': np.matrix[FLOAT_dtype]
                        Projection of camera coordinates to right camera image, shape: (3, 4)

            'labels': list of dictionaries
                List with labels in this frame. Each label contains the following keys:
                    'category': string
                        Possible values depend on the Reader subclass
                    'box2D': dict
                        Bounding box in the left image, keys: *'left'*, *'top'*, *'right'*, *'bottom'*
                    'box3D': dict
                        'location': dict
                            Center of the 3D box, keys: *'x'*, *'y'*, *'z'*
                        'dimensions': dict
                            Size of the 3D box, keys: *'height'*, *'width'*, *'length'*
                        'rotation_y': float
                            Object rotation around Y-axis in camera coordinates [-pi...pi], 0 = facing along X-axis
                    'info': dict
                        Information depending on the Reader subclass
        """
        raise NotImplementedError("getFrameInfo() is not implemented in BaseReader")
