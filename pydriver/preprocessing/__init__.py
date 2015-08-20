# -*- coding: utf-8 -*-
"""
Module for preprocessing frames returned by :mod:`~pydriver.datasets` module and extracting (3D) scene information on which the detection algorithms (:mod:`~pydriver.keypoints`, :mod:`~pydriver.features`, etc.) will operate.

The class :class:`~.preprocessing.Preprocessor` implements a preprocessing pipeline whereas :class:`~.preprocessing.CloudProcessor` and its
derived classes offer various cloud modifications which can be combined in the :class:`~.preprocessing.Preprocessor`.

.. automodule:: pydriver.preprocessing.preprocessing
"""
from __future__ import absolute_import, division

# main pipeline
from .preprocessing import Preprocessor

# lidar reconstructor
from .preprocessing import LidarReconstructor

# cloud processors
from .preprocessing import CloudProcessor   # base class
from .preprocessing import RemoveNaNProcessor, ViewportProcessor, DownsampleProcessor, GroundPlaneProcessor
