# -*- coding: utf-8 -*-
"""
Geometry module for coordinate transformations and segmentation.

.. automodule:: pydriver.geometry.geometry
"""
from __future__ import absolute_import, division

from .geometry import homogenuous2cartesian, cartesian2homogenuous, affineTransform
from .geometry import image2space

from .geometry import transform3DBox, project3DBox
from .geometry import getNormalizationTransformation, extractNormalizedOrientedBoxes
from .geometry import get3DBoxVertices
