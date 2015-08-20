# -*- coding: utf-8 -*-
"""
Feature extraction module

Base
----------------------------------------
.. automodule:: pydriver.features.base

SHOT
----------------------------------------
.. automodule:: pydriver.features.shot
"""
from __future__ import absolute_import, division

from .base import FeatureExtractor  # base class
from .shot import SHOTExtractor, SHOTColorExtractor
