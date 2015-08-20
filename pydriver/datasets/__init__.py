# -*- coding: utf-8 -*-
"""
Module for extracting data from datasets. Every dataset submodule should implement a class derived from :class:`~.base.BaseReader`.

Base
----------------------------------------
.. automodule:: pydriver.datasets.base

KITTI
----------------------------------------
.. automodule:: pydriver.datasets.kitti

Utils
----------------------------------------
.. automodule:: pydriver.datasets.utils
"""
from __future__ import absolute_import, division

from .utils import labels2detections, detections2labels
from . import kitti
