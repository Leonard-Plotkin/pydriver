# -*- coding: utf-8 -*-
"""
`Point Cloud Library <http://pointclouds.org/>`_ interface.

.. automodule:: pydriver.pcl.pcl
"""
from __future__ import absolute_import, division

import os, platform

if platform.system() == 'Windows':
    old_PATH = os.environ['PATH']   # store old PATH value

    # add path to pcl_helper library (DLL on windows) to PATH environment variable
    # (Note: adding to sys.path is not sufficient)
    os.environ['PATH'] += os.pathsep + os.path.join(os.path.dirname(__file__), 'pcl_helper', 'lib')

from .pcl import PCLHelper

if platform.system() == 'Windows':
    os.environ['PATH'] = old_PATH   # restore old PATH value
