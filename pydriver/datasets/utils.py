# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

import numpy as np

import copy

from ..common.constants import Detection_dtype
from .. import geometry


def labels2detections(labels, transformation = None):
    """Convert list of dictionaries with labels to NumPy array of detections

    If transformation is given, the result will be transformed accordingly.

    See :class:`~.base.BaseReader` for label format description.

    :Returns: NumPy array with detections
    :Returntype: np.ndarray[Detection_dtype]
    """
    # initialize list with Detection tuples
    detections = []
    for label in labels:
        box3D = label['box3D']
        if transformation is not None:
            box3D = geometry.transform3DBox(box3D, transformation)

        Detection = (
                label['category'].encode('ascii'),  # category
                (
                 box3D['location']['x'], box3D['location']['y'], box3D['location']['z'], box3D['rotation_y'],   # position tuple
                 ),
                box3D['dimensions']['height'],  # dimensions...
                box3D['dimensions']['width'],
                box3D['dimensions']['length'],
                1.0,    # weight
                )
        detections.append(Detection)
    # convert list to np.array and return it
    return np.array(detections, dtype = Detection_dtype)

def detections2labels(detections, transformation = None, projection = None, imgShape = None):
    """Convert NumPy array of detections to a list of dictionaries with labels

    The key *'info'* will contain a dictionary with the 'weight' key.

    If transformation is given, the result will be transformed accordingly (before projection to 2D box).
    If projection matrix is given, the result will contain the 'box2Duntruncated' key. If imgShape (height, width) is also given, 2D boxes
    will be truncated to be within image boundaries and saved as 'box2D', the value ['info']['truncated'] will be set as well.

    See :class:`~.base.BaseReader` for label format description.

    :Returns: list of dictionaries with labels
    :Returntype: list of dictionaries
    """
    # initialize list with labels
    labels = []
    for i in range(detections.shape[0]):
        Detection = detections[i]
        label = {
                'category': Detection['category'].decode('ascii'),
                'box3D': {
                          'location': {'x': Detection['position']['x'], 'y': Detection['position']['y'], 'z': Detection['position']['z']},
                          'dimensions': {'height': Detection['height'], 'width': Detection['width'], 'length': Detection['length']},
                          'rotation_y': Detection['position']['rotation_y'],
                          },
                'info': {'weight': Detection['weight']},
                }
        if transformation is not None:
            label['box3D'] = geometry.transform3DBox(label['box3D'], transformation)
        if projection is not None:
            label['box2Duntruncated'] = geometry.project3DBox(label['box3D'], projection)
            if imgShape is not None:
                box2Dtruncated = copy.deepcopy(label['box2Duntruncated'])
                box2Dtruncated['left'] = max(0, box2Dtruncated['left'])
                box2Dtruncated['top'] = max(0, box2Dtruncated['top'])
                box2Dtruncated['right'] = min(imgShape[1], box2Dtruncated['right'])
                box2Dtruncated['bottom'] = min(imgShape[0], box2Dtruncated['bottom'])
                label['box2D'] = box2Dtruncated

                # calculate truncation
                org_size = (label['box2Duntruncated']['right']-label['box2Duntruncated']['left']) * (label['box2Duntruncated']['bottom']-label['box2Duntruncated']['top'])
                trunc_size = (label['box2D']['right']-label['box2D']['left']) * (label['box2D']['bottom']-label['box2D']['top'])
                if org_size > 0:
                    label['info']['truncated'] = 1.0 - (trunc_size / org_size)
                else:
                    label['info']['truncated'] = 0.0
        labels.append(label)
    return labels
