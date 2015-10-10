# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

from collections import namedtuple
import copy
import math
import os
import re
import warnings

import numpy as np

from ..common.constants import FLOAT_dtype
from ..common.functions import pyNormalizeAngle
from ..geometry import affineTransform

from .base import _union, loadImage, BaseReader


# utility functions

def getKITTIGroundTruth(labels, categories, categoriesOpt, mode = 'moderate'):
    """Get mandatory and optional ground truth out of a list of labels

    The KITTI dataset defines criteria which labels have to be detected in order to avoid a false negative. There are three evaluation
    modes 'easy', 'moderate' and 'hard'. To avoid penalizing good detection algorithms labels which are hard to detect become optional
    ground truth so their detection will not result in a false positive. The harder the mode the more labels become mandatory ground truth.

    The category 'dontcare' is always included in optional ground truth.

    :Parameters:
        labels: list of dicts
            List with labels provided by dataset reader
        categories: list of strings
            Mandatory categories
        categoriesOpt: list of strings
            Optional categories
        mode: string, optional
            Evaluation mode, 'moderate' by default

    :Returns: mandatory and optional ground truth
    :Returntype: tuple(groundTruth: list, groundTruthOpt: list)
    """
    # evaluation modes according to KITTI object evaluation rules
    MODES = {
             'easy': {'minHeight': 40, 'maxOcclusion': 0, 'maxTruncation': 0.15},
             'moderate': {'minHeight': 25, 'maxOcclusion': 1, 'maxTruncation': 0.30},
             'hard': {'minHeight': 25, 'maxOcclusion': 2, 'maxTruncation': 0.50},
             }
    mode = MODES[mode]      # get mode dictionary
    groundTruth = []        # ground truth (missing detection is a false negative)
    groundTruthOpt = []     # optional ground truth (missing detection is ok)
    for label in labels:
        if label['category'] in categoriesOpt or label['category'] == 'dontcare':
            groundTruthOpt.append(label)
        elif label['category'] in categories:
            if label['info']['occluded'] > mode['maxOcclusion']:
                groundTruthOpt.append(label)
            elif label['info']['truncated'] > mode['maxTruncation']:
                groundTruthOpt.append(label)
            elif label['box2D']['bottom'] - label['box2D']['top'] < mode['minHeight']:
                groundTruthOpt.append(label)
            else:
                # label not optional
                groundTruth.append(label)
    return (groundTruth, groundTruthOpt)

def correctKITTILabelForStereo(label):
    """Roughly adjust for an empirically estimated labeling error w.r.t. stereo reconstruction in KITTI dataset"""
    # TODO: check extensively
    base = 15.0
    scale = 1.07
    new_label = copy.deepcopy(label)
    if new_label['box3D']['location']['z'] > base:
        new_label['box3D']['location']['z'] = base + (new_label['box3D']['location']['z']-base)*scale
    return new_label

def writeLabels(labels, filepath, includeAlpha = True):
    """Write labels to file

    The produced file can be used for evaluation on KITTI servers. Labels are expected to use the camera coordinate system.

    :Parameters:
        labels: list
            List with object labels
        filepath: string
            Path to the file to use
        includeAlpha: bool, optional
            Write alpha values (observation angle) to file, *True* by default
    """
    def label2line(label):
        # convert lower case category to KITTI categories
        category = label['category']
        if category == 'dontcare':
            category = 'DontCare'
        else:
            category = category[0].upper() + category[1:]

        # compute alpha if required
        if includeAlpha:
            # set to object orientation
            alpha = label['box3D']['rotation_y']
            # adjust to X/Z observation angle of object center
            alpha -= -math.atan2(label['box3D']['location']['z'], label['box3D']['location']['x']) - 1.5*math.pi
            # wrap to +/-Pi
            alpha = pyNormalizeAngle(alpha)
            # convert to string
            alpha = '%.2f' % alpha
        else:
            # set to KITTI default (invalid) value
            alpha = '-10'

        label_line = '%(category)s %(truncated).2f %(occluded)d %(alpha)s %(left).2f %(top).2f %(right).2f %(bottom).2f %(height).2f %(width).2f %(length).2f %(x).2f %(y).2f %(z).2f %(rotation_y).2f %(score).2f\n' % {
            'category': category,
            'truncated': label['info']['truncated'],
            'occluded': -1,     # invalid value to be ignored by KITTI evaluation
            'alpha': alpha,
            'left': label['box2D']['left'],
            'top': label['box2D']['top'],
            'right': label['box2D']['right'],
            'bottom': label['box2D']['bottom'],
            'height': label['box3D']['dimensions']['height'],
            'width': label['box3D']['dimensions']['width'],
            'length': label['box3D']['dimensions']['length'],
            'x': label['box3D']['location']['x'],
            'y': label['box3D']['location']['y'] + label['box3D']['dimensions']['height'] / 2.0,
            'z': label['box3D']['location']['z'],
            'rotation_y': label['box3D']['rotation_y'],
            'score': label['info']['weight']*100,   # multiply by 100 to avoid precision loss
        }
        return label_line

    with open(filepath, mode='w') as f:
        for label in labels:
            f.write(label2line(label))


# named tuples definition

_NavigationInfo = namedtuple('_NavigationInfo', [
    'lat',      # latitude of the oxts-unit (deg)
    'lon',      # longitude of the oxts-unit (deg)
    'alt',      # altitude of the oxts-unit (m)
    'roll',     # roll angle (rad),  0 = level, positive = left side up (-pi..pi)
    'pitch',    # pitch angle (rad), 0 = level, positive = front down (-pi/2..pi/2)
    'yaw',      # heading (rad),     0 = east,  positive = counter clockwise (-pi..pi)
    'vn',       # velocity towards north (m/s)
    've',       # velocity towards east (m/s)
    'vf',       # forward velocity, i.e. parallel to earth-surface (m/s)
    'vl',       # leftward velocity, i.e. parallel to earth-surface (m/s)
    'vu',       # upward velocity, i.e. perpendicular to earth-surface (m/s)
    'ax',       # acceleration in x, i.e. in direction of vehicle front (m/s^2)
    'ay',       # acceleration in y, i.e. in direction of vehicle left (m/s^2)
    'az',       # acceleration in z, i.e. in direction of vehicle top (m/s^2)
    'af',       # forward acceleration (m/s^2)
    'al',       # leftward acceleration (m/s^2)
    'au',       # upward acceleration (m/s^2)
    'wx',       # angular rate around x (rad/s)
    'wy',       # angular rate around y (rad/s)
    'wz',       # angular rate around z (rad/s)
    'wf',       # angular rate around forward axis (rad/s)
    'wl',       # angular rate around leftward axis (rad/s)
    'wu',       # angular rate around upward axis (rad/s)
    'posacc',   # velocity accuracy (north/east in m)
    'velacc',   # velocity accuracy (north/east in m/s)
    'navstat',  # navigation status
    'numsats',  # number of satellites tracked by primary GPS receiver
    'posmode',  # position mode of primary GPS receiver
    'velmode',  # velocity mode of primary GPS receiver
    'orimode',  # orientation mode of primary GPS receiver
    ])


# dataset readers implementation

class KITTIReader(BaseReader):
    """Abstract data extractor for KITTI_ datasets

    This class relies on presence of at least one image for every frame to detect available frames. Lidar data is optional.

    See :class:`~.base.BaseReader` for more information.

    .. _KITTI: http://www.cvlibs.net/datasets/kitti/
    """

    def getDatasets(self):
        raise NotImplementedError("getDatasets() is not implemented in KITTIReader")

    def getFrameIds(self, dataset = None):
        def _filesToFrames(filenames):
            def _getFrameId(filename):
                match = re.match("(\d{6}).png", filename)
                if match:
                    return int(match.groups()[0])
                else:
                    return None
            return [frameId for frameId in map(_getFrameId, filenames) if frameId is not None]
        image_lists = [os.listdir(image_dir) for image_dir in self._getImageDirs(dataset) if os.path.isdir(image_dir)]
        return sorted(list(_union(*map(_filesToFrames, image_lists))))

    def getFrameInfo(self, frameId, dataset = None):
        """Get information about a frame in the specified dataset

        :Parameters:
            frameId: int
                Frame ID of the requested frame
            dataset: str or None, optional
                Dataset with the requested frame, only required by :class:`KITTITrackletsReader`

        :Returns: Dictionary with information about the frame:

            'dataset': str or None
                Dataset of the frame, *None* for :class:`KITTIObjectsReader`

            'frameId': int
                Frame ID

            'img_left': NumPy array
                Left image, can be None

            'img_right': NumPy array
                Right image, can be None

            'lidar': dict{'XYZ': NumPy array, 'RGB': NumPy array}
                XYZ (transformed to camera coordinates, rectification matrix applied) and RGB (reflectance, gray) data from lidar sensor, can be None

            'calibration': dict
                Calibration matrices
                    'reprojection': np.matrix[FLOAT_dtype]
                        3D reconstruction out of disparity, shape: (4, 4)
                    'projection_left': np.matrix[FLOAT_dtype]
                        Projection of camera coordinates to left camera image, shape: (3, 4)
                    'projection_right': np.matrix[FLOAT_dtype]
                        Projection of camera coordinates to right camera image, shape: (3, 4)
                    'rect': np.matrix[FLOAT_dtype]
                        Rectification matrix, shape: (4, 4)
                    'lidar2cam': np.matrix[FLOAT_dtype]
                        Transformation from lidar to camera coordinates, shape: (4, 4)

            'labels': list of dictionaries
                List with labels in this frame. Each label contains the following keys:
                    'category': string
                        Possible values: 'car', 'van', 'truck', 'pedestrian', 'person_sitting', 'cyclist', 'tram', 'misc', 'dontcare'
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
                        'truncated': float
                            Float from 0 (non-truncated) to 1 (truncated), where *truncated* refers to the object leaving image boundaries
                        'occluded': int
                            Occlusion status (0 = fully visible, 1 = partly occluded, 2 = largely occluded, 3 = unknown)
                        'trackId': int, optional
                            Unique tracking id of this object within this sequence, supplied only by `KITTITrackletsReader`
        """
        img_dir_left, img_dir_right = self._getImageDirs(dataset)
        img_file_left = os.path.join(img_dir_left, "%06d.png" % frameId)
        img_file_right = os.path.join(img_dir_right, "%06d.png" % frameId)
        calibration = self._getCamCalibration(frameId, dataset)

        return {
                'dataset': dataset,
                'frameId': frameId,
                'img_left': loadImage(img_file_left) if os.path.isfile(img_file_left) else None,
                'img_right': loadImage(img_file_right) if os.path.isfile(img_file_right) else None,
                'calibration': calibration,
                'lidar': self._getLidarPoints(calibration, frameId, dataset),
                'labels': self._getFrameLabels(frameId, dataset),
                }


    # -- directory functions ---

    def _getImageDirs(self, dataset = None):
        raise NotImplementedError("_getImageDirs() is not implemented in KITTIReader")

    def _getLidarDir(self, dataset = None):
        raise NotImplementedError("_getLidarDir() is not implemented in KITTIReader")

    def _getCalibrationDir(self):
        return os.path.join(self._dir, "calib")

    def _getLabelsDir(self):
        raise NotImplementedError("_getLabelsDir() is not implemented in KITTIReader")


    # --- internal functions ---

    def _getLabelData(self, values):
        # function expects the first value in values list being the category
        labelData = {
                     # see KITTI's devkit/readme.txt
                     'type': values[0],                     # category of object: 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc' or 'DontCare'
                     'truncated': float(values[1]),         # float from 0 (non-truncated) to 1 (truncated), where truncated refers to the object leaving image boundaries
                     'occluded': int(values[2]),            # integer (0,1,2,3) indicating occlusion state: 0 = fully visible, 1 = partly occluded, 2 = largely occluded, 3 = unknown
                     'alpha': float(values[3]),             # observation angle of object, ranging [-pi..pi]
                     'bbox': {                              # 2D bounding box of object in the image
                              'left': float(values[4]),
                              'top': float(values[5]),
                              'right': float(values[6]),
                              'bottom': float(values[7]),
                              },
                     'dimensions': {                        # 3D object dimensions
                                    'height': float(values[8]),
                                    'width': float(values[9]),
                                    'length': float(values[10]),
                                    },
                     'location': {                          # location of front-center-bottom point of the 3D bounding box
                                  'x': float(values[11]),
                                  'y': float(values[12]),
                                  'z': float(values[13]),
                                  },
                     'rotation_y': float(values[14]),       # rotation ry around Y-axis in camera coordinates [-pi..pi], 0 = facing along X-axis
                     }
        return labelData

    def _processLabel(self, kitti_label):
        """Transform KITTI label to universal format

        See :class:`~.base.BaseReader` for label format description.
        """
        label = {
                 'category': kitti_label['type'].lower(),
                 'box2D': kitti_label['bbox'].copy(),
                 'box3D': {
                           'location': {
                                        'x': kitti_label['location']['x'],
                                        'y': kitti_label['location']['y'] - kitti_label['dimensions']['height'] / 2.0,  # move to center
                                        'z': kitti_label['location']['z'],
                                        },
                           'dimensions': kitti_label['dimensions'].copy(),
                           'rotation_y': kitti_label['rotation_y'],
                           },
                 'info': {
                          'truncated': kitti_label['truncated'],
                          'occluded': kitti_label['occluded'],
                          },
                 }
        if 'trackId' in kitti_label:
            # set trackId if given
            label['info']['trackId'] = kitti_label['trackId']
        return label

    def _getFrameLabels(self, frameId, dataset = None):
        raise NotImplementedError("_getFrameLabels() is not implemented in KITTIReader")

    def _getCamCalibration(self, frameId, dataset = None):
        raise NotImplementedError("_getCamCalibration() is not implemented in KITTIReader")

    def _readCamCalibration(self, filename):
        def line2values(line):
            return [float(v) for v in line.strip().split(" ")[1:]]
        def getMatrix(values, shape):
            return np.matrix(values, dtype = FLOAT_dtype).reshape(shape)
        def padMatrix(matrix_raw):
            matrix = np.matrix(np.zeros((4,4), dtype = FLOAT_dtype), copy = False)
            matrix[:matrix_raw.shape[0], :matrix_raw.shape[1]] = matrix_raw
            matrix[3, 3] = 1
            return matrix

        with open(filename, 'r') as f:
            data = f.read().split("\n")

        #P0 = getMatrix(line2values(data[0]), (3, 4))
        #P1 = getMatrix(line2values(data[1]), (3, 4))
        P2 = getMatrix(line2values(data[2]), (3, 4))
        P3 = getMatrix(line2values(data[3]), (3, 4))

        Rect = padMatrix(getMatrix(line2values(data[4]), (3, 3)))
        Velo2Cam = padMatrix(getMatrix(line2values(data[5]), (3, 4)))
        #Imu2Velo = padMatrix(getMatrix(line2values(data[6]), (3, 4)))

        P_left = P2
        P_right = P3

        # see for example http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#stereorectify
        f = P_left[0, 0]
        Tx = (P_right[0, 3] - P_left[0, 3]) / f
        cx_left = P_left[0, 2]
        cx_right = P_right[0, 2]
        cy = P_left[1, 2]
        # Depth = f*(T/disparity), see for example http://www.gergltd.com/cse486/project4/
        # see http://stackoverflow.com/questions/11406849/using-opencv-to-generate-3d-points-assuming-frontal-parallel-configuration

        reprojection = np.matrix([
                                  [1, 0, 0, -cx_left],
                                  [0, 1, 0,  -cy],
                                  [0, 0, 0,    f],
                                  [0, 0, -1/Tx, (cx_left - cx_right) / Tx],
                                  ], dtype = FLOAT_dtype)
        result = {
            'projection_left': P_left,
            'projection_right': P_right,
            'rect': Rect,
            'velo2cam': Velo2Cam,
            'reprojection': reprojection,
        }
        return result

    def _getLidarPoints(self, calibration, frameId, dataset = None):
        filename = os.path.join(self._getLidarDir(dataset), '%06d.bin' % frameId)
        if not os.path.isfile(filename):
            # no data
            return None
        # read matrix
        data = np.fromfile(filename, np.float32)
        # reshape to points with 4 coordinates each
        data = data.reshape(data.shape[0] // 4, 4)
        # XYZ coordinates transformed to camera coordinates
        XYZ = affineTransform(data[:, :3], calibration['rect']*calibration['velo2cam'])
        # reflectance
        R = (256 * data[:, 3]).astype(np.uint8)
        # convert reflectance to RGB
        RGB = np.ndarray((XYZ.shape[0], 3), dtype = np.uint8)
        RGB[:, 0] = R
        RGB[:, 1] = R
        RGB[:, 2] = R
        return {'XYZ': XYZ, 'RGB': RGB}


class KITTIObjectsReader(KITTIReader):
    """Data extractor for KITTI objects dataset

    The data directory must contain the directories *'calib'*, *'image_2'* and/or *'image_3'* and optionally *'label_2'*.

    See :class:`KITTIReader` for more information.
    """

    def getDatasets(self):
        return ['None']     # return the one and only dataset

    # -- directory functions ---

    def _getImageDirs(self, dataset = None):
        # we have no datasets, ignore the dataset parameter
        return (os.path.join(self._dir, "image_2"), os.path.join(self._dir, "image_3"))

    def _getLidarDir(self, dataset = None):
        # we have no datasets, ignore the dataset parameter
        return os.path.join(self._dir, "velodyne")

    def _getLabelsDir(self):
        dir = os.path.join(self._dir, "label_2")
        if os.path.exists(dir):
            return dir
        else:
            # no labels given
            return None


    # --- internal functions ---

    def _getFrameLabels(self, frameId, dataset = None):
        if self._getLabelsDir() is None:
            # no labels given
            return []
        else:
            with open(os.path.join(self._getLabelsDir(), "%06d.txt" % frameId), 'r') as f:
                text_data = [[value for value in line.split(" ")] for line in f.read().split("\n") if line]
            labels = []
            for line in text_data:
                # get label data (starting with the category)
                labelData = self._getLabelData(line)
                # transform to universal label format and append to labels
                labels.append(self._processLabel(labelData))
            return labels

    def _getCamCalibration(self, frameId, dataset = None):
        return self._readCamCalibration(os.path.join(self._getCalibrationDir(), "%06d.txt" % frameId))


class KITTITrackletsReader(KITTIReader):
    """Data extractor for KITTI tracklets dataset

    The data directory must contain the directories *'calib'*, *'image_02'* and/or *'image_03'*.
    Optional directories are *'label_02'* and *'oxts'*.

    See :class:`KITTIReader` for more information.
    """

    def __init__(self, directory):
        super(KITTITrackletsReader, self).__init__(directory)
        # initialize cache
        self._cache = {
                       'calibration': {},
                       'labels': {},
                       'oxts': {'files': {}, 'data': {}},
                       }

    def getDatasets(self):
        # use a dummy dataset, we only need the directory above it
        return sorted(list(_union(*[os.listdir(os.path.dirname(image_dir)) for image_dir in self._getImageDirs('0')])))

    def getFrameInfo(self, frameId, dataset):
        """See :func:`~pydriver.datasets.base.BaseReader.getFrameInfo` for general description

        If navigation data is available (in the *'oxts'* directory), the result will contain an additional key:

        'navigation': OrderedDict
            'lat': float
                latitude of the oxts-unit (deg)
            'lon': float
                longitude of the oxts-unit (deg)
            'alt': float
                altitude of the oxts-unit (m)
            'roll': float
                roll angle (rad),  0 = level, positive = left side up (-pi..pi)
            'pitch': float
                pitch angle (rad), 0 = level, positive = front down (-pi/2..pi/2)
            'yaw': float
                heading (rad),     0 = east,  positive = counter clockwise (-pi..pi)
            'vn': float
                velocity towards north (m/s)
            've': float
                velocity towards east (m/s)
            'vf': float
                forward velocity, i.e. parallel to earth-surface (m/s)
            'vl': float
                leftward velocity, i.e. parallel to earth-surface (m/s)
            'vu': float
                upward velocity, i.e. perpendicular to earth-surface (m/s)
            'ax': float
                acceleration in x, i.e. in direction of vehicle front (m/s^2)
            'ay': float
                acceleration in y, i.e. in direction of vehicle left (m/s^2)
            'az': float
                acceleration in z, i.e. in direction of vehicle top (m/s^2)
            'af': float
                forward acceleration (m/s^2)
            'al': float
                leftward acceleration (m/s^2)
            'au': float
                upward acceleration (m/s^2)
            'wx': float
                angular rate around x (rad/s)
            'wy': float
                angular rate around y (rad/s)
            'wz': float
                angular rate around z (rad/s)
            'wf': float
                angular rate around forward axis (rad/s)
            'wl': float
                angular rate around leftward axis (rad/s)
            'wu': float
                angular rate around upward axis (rad/s)
            'posacc': float
                velocity accuracy (north/east in m)
            'velacc': float
                velocity accuracy (north/east in m/s)
            'navstat': int
                navigation status
            'numsats': int
                number of satellites tracked by primary GPS receiver
            'posmode': int
                position mode of primary GPS receiver
            'velmode': int
                velocity mode of primary GPS receiver
            'orimode': int
                orientation mode of primary GPS receiver
        """
        info = super(KITTITrackletsReader, self).getFrameInfo(frameId, dataset)
        # add navigation information if available
        oxts = self._getOxtsInfo(frameId, dataset)
        if oxts is not None:
            info['navigation'] = oxts
        return info


    # -- directory functions ---

    def _getImageDirs(self, dataset):
        return (os.path.join(self._dir, "image_02", dataset), os.path.join(self._dir, "image_03", dataset))

    def _getLidarDir(self, dataset = None):
        return os.path.join(self._dir, "velodyne", dataset)

    def _getLabelsDir(self):
        dir = os.path.join(self._dir, "label_02")
        if os.path.exists(dir):
            return dir
        else:
            # no labels given
            return None

    def _getOxtsDir(self):
        oxtsDir = os.path.join(self._dir, "oxts")
        if os.path.exists(oxtsDir):
            return oxtsDir
        else:
            # no oxts data given
            warnings.warn(UserWarning('"{}" directory not found, navigation data not available.'.format(oxtsDir)))
            return None
    def _getOxtsFile(self, dataset):
        oxtsDir = self._getOxtsDir()
        if oxtsDir is None:
            return None
        oxtsFile = os.path.join(oxtsDir, '{}.txt'.format(dataset))
        if os.path.exists(oxtsFile):
            return oxtsFile
        else:
            # no oxts data given
            warnings.warn(UserWarning('"{}" file not found, navigation data for dataset {} not available.'.format(oxtsFile, dataset)))
            return None

    # --- internal functions ---

    def _getFrameLabels(self, frameId, dataset):
        if self._getLabelsDir() is None:
            # no labels given
            return []
        else:
            return self._getDatasetLabels(dataset).get(frameId, [])

    def _getDatasetLabels(self, dataset):
        if dataset not in self._cache['labels']:
            with open(os.path.join(self._getLabelsDir(), "%s.txt" % dataset), 'r') as f:
                text_data = [[value for value in line.split(" ")] for line in f.read().split("\n") if line]
            labels = {}
            for line in text_data:
                frameId = int(line[0])
                if frameId not in labels:
                    labels[frameId] = []

                # get label data (starting with the category)
                labelData = self._getLabelData(line[2:])
                # unique tracking id of this object within this sequence (specific to tracklets dataset)
                labelData['trackId'] = int(line[1])
                # transform to universal label format
                label = self._processLabel(labelData)
                labels[frameId].append(label)
            self._cache['labels'][dataset] = labels
        return self._cache['labels'][dataset]

    def _getCamCalibration(self, frameId, dataset):
        if dataset not in self._cache['calibration']:
            self._cache['calibration'][dataset] = self._readCamCalibration(os.path.join(self._getCalibrationDir(), "%s.txt" % dataset))
        return self._cache['calibration'][dataset]

    def _getOxtsInfo(self, frameId, dataset):
        """Extract oxts navigation data"""
        if dataset not in self._cache['oxts']['files']:
            # dataset file not in cache
            oxtsFile = self._getOxtsFile(dataset)
            if oxtsFile is None:
                # no oxts file
                self._cache['oxts']['files'][dataset] = None
            else:
                # read file lines
                with open(oxtsFile, 'r') as f:
                    lines = [line for line in f.read().strip().split('\n') if line]
                self._cache['oxts']['files'][dataset] = lines
            # initialize dataset dictionary, key: frameId, value: _NavigationInfo-like OrderedDict or None
            self._cache['oxts']['data'][dataset] = {}
        # assertion: self._cache['oxts']['files'][dataset] exists (list of strings or None)
        # assertion: self._cache['oxts']['data'][dataset] dict exists (can be empty)
        if frameId not in self._cache['oxts']['data'][dataset]:
            # get text file lines for this dataset
            lines = self._cache['oxts']['files'][dataset]
            if lines is None:
                # no information available
                self._cache['oxts']['data'][dataset][frameId] = None
            else:
                if frameId >= len(lines):
                    raise ValueError('Navigation information for frame {} in dataset {} not found.'.format(frameId, dataset))
                # list with values (as strings) in text file line for this frame
                values_str = lines[frameId].strip().split(' ')
                # process and cache frame data
                values_float = [float(v) for v in values_str[:25]]
                values_int = [int(v) for v in values_str[25:]]
                self._cache['oxts']['data'][dataset][frameId] = _NavigationInfo(*tuple(values_float + values_int))._asdict()
        # assertion: self._cache['oxts']['data'][dataset][frameId] exists (_NavigationInfo-like OrderedDict or None)
        return self._cache['oxts']['data'][dataset][frameId]
