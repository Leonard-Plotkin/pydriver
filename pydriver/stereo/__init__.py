# -*- coding: utf-8 -*-
"""
Module for extracting 3D information from stereo image pairs.

The class :class:`~.stereo.StereoReconstructor` implements the reconstructing pipeline whereas :class:`~.stereo.OpenCVMatcher`
is an implementation of a specific disparity computation method.

Stereo
----------------------------------------
.. automodule:: pydriver.stereo.stereo
"""
from __future__ import absolute_import, division

# stereo reconstruction pipeline
from .stereo import StereoReconstructor

# utility functions
from .stereo import disparity2depth, depth2disparity

# classes for computing "simple" disparity maps
from .stereo import OpenCVMatcher
