# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

include "../common/functions_include.pyx"

import os, pickle, tempfile, warnings, zipfile

cimport libc.math

cimport cython

cimport numpy as cnp
import numpy as np

import sklearn.cluster

from ..common.structs cimport FLOAT_t, Position
from ..common.constants import FLOAT_dtype, Detection_dtype
from ..datasets.utils import detections2labels
from ..geometry import get3DBoxVertices

from . import vocabularies

try:
    import shapely.geometry
except ImportError:
    # shapely not available, deactivate some functionality
    USE_SHAPELY = False
else:
    # shapely available
    USE_SHAPELY = True


class Detector(object):
    """Class for learning and recognition

    This class implements learning features of different types and storing them in their respective :class:`vocabularies <.vocabularies.Vocabulary>` which
    can then be used to perform detection and recognition.
    """

    def __init__(self, featureTypes, vocabularyGenerator = None):
        """Initialize Detector instance

        :Parameters:
            featureTypes: list of tuples(str, int)
                Each tuple has to contain the unique name of the feature type and the number of its dimensions
            vocabularyGenerator: callable, optional
                Function to use for creating new vocabulary objects, must accept dimensions as first positional argument and the feature name as the second, :class:`Vocabulary(dimensions) <.vocabularies.Vocabulary>` by default
        """
        self._featureTypes = featureTypes

        # set vocabulary generator
        if vocabularyGenerator is None:
            self._vocabularyGenerator = _default_vocabularyGenerator
        else:
            self._vocabularyGenerator = vocabularyGenerator

        # vocabularies for different feature types
        self._vocabularies = {}
        # initialize vocabularies
        for f_name, f_dims in featureTypes:
            self._vocabularies[f_name] = self.vocabularyGenerator(f_dims, f_name)

    @property
    def featureTypes(self):
        return self._featureTypes
    @property
    def vocabularyGenerator(self):
        return self._vocabularyGenerator

    def save(self, file):
        """Save model to file or file-like object

        No further compression is applied, but sub-objects (like :class:`vocabularies <.vocabularies.Vocabulary>`) can compress their data.

        file: str or file-like object
            If file is a string, the file will be created
        """
        params = self._getParams()
        with zipfile.ZipFile(file, 'w') as zf:
            zf.writestr('params.dict', pickle.dumps(params, protocol = pickle.HIGHEST_PROTOCOL))
            for f_name, f_dims in self.featureTypes:
                # write vocabulary to temporary file to avoid memory overflows
                with tempfile.NamedTemporaryFile(delete = False) as tf:
                    self._vocabularies[f_name].save(tf)
                    tf_name = tf.name
                # temporary file has to be closed on Windows plattforms to be opened again (by ZipFile)
                # write vocabulary data to final zip file or file-like object
                zf.write(tf_name, '%s.voc' % f_name)
                # delete temporary file
                os.remove(tf_name)

    @classmethod
    def load(cls, file):
        """Load model from file or file-like object and return the new instance

        file: str or file-like object
            If file is a string, the file will be opened and read
        """
        with zipfile.ZipFile(file, 'r') as zf:
            params = pickle.load(zf.open('params.dict', 'r'))
            result = cls._fromParams(params)

            for f_name, f_dims in result.featureTypes:
                # read vocabulary file into temporary file since ZipFile has problems reading from ZipFile.open() directly
                with tempfile.NamedTemporaryFile(delete = False) as tf:
                    tf.write(zf.open('%s.voc' % f_name, 'r').read())
                    tf_name = tf.name
                # load vocabulary
                # create a dummy vocabulary with 1 dimension using the generator and use it to load the saved vocabulary
                result._vocabularies[f_name] = result.vocabularyGenerator(1, f_name).load(tf_name)
                # remove temporary file
                os.remove(tf_name)
        return result

    def addWords(self, str category, str featureType, cnp.ndarray[FLOAT_t, ndim=2] features, cnp.ndarray keypoints = None, dict box3D = None):
        """Add new visual words

        The function adds features and their associated "meaning" (positive/negative example, object configuration if positive) to
        appropriate internal :class:`vocabularies <.vocabularies.Vocabulary>` which will be used for learning and recognition.
        Only features which are fully defined (no *NaN* values) are learned.

        category: str
            Category of object (or *'negative'*)
        featureType: str
            String indicating the feature type
        features: np.ndarray[FLOAT_dtype, ndim=2]
            Array with feature vectors
        keypoints: np.ndarray[:const:`~pydriver.Position_dtype`], optional
            Array with keypoints, only for positive examples
        box3D: dict, optional
            3D box of object, only for positive examples, see :class:`~pydriver.datasets.base.BaseReader` for box format description
        """
        if category != 'negative':
            # positive example
            detections = _computeDetections(keypoints, category, box3D)
            self._vocabularies[featureType].addWords(features, detections)
        else:
            # negative example
            self._vocabularies[featureType].addWords(features)

    def learn(self, nVocMaxRandomSamples=0, nStorageMaxRandomSamples=0, keepData=False, verbose=False):
        """Perform learning on stored features

        The detector can be used for recognition afterwards.
        """
        for f_name, voc in self._vocabularies.items():
            voc.prepareForRecognition(nMaxRandomSamples=nVocMaxRandomSamples, nStorageMaxRandomSamples=nStorageMaxRandomSamples, keepData=keepData, verbose=verbose)

    def getDetections(self, dict featuresData, suppressNegatives = False):
        """Get detections for given features and their corresponding keypoint positions

        featuresData containing no data is a valid case.

        :Parameters:
            featuresData: dict{str featureType: tuple(np.ndarray[Position_dtype] keypoints, np.ndarray[FLOAT_dtype, ndim=2] features)}
                Dictionary with feature types containing their keypoints and features
            suppressNegatives: bool, optional
                Flag whether to produce only positive detections, *False* by default

        :Returns: Array with detections
        :Returntype: np.ndarray[:const:`~pydriver.Detection_dtype`]
        """
        cdef:
            int i
        # initialize list with detections arrays
        detections_list = []
        for f_name, f_dims in self.featureTypes:
            # get data for current feature type
            featureData = featuresData.get(f_name, None)
            if featureData is not None:
                if self._vocabularies[f_name].isPreparedForRecognition:
                    # extract tuple
                    cur_keypoints, cur_features = featureData
                    # sanity check
                    assert cur_features.shape[0] == cur_keypoints.shape[0], "Features and keypoints must match (feature type name: %s, features: %d, keypoints: %d)" % (f_name, cur_features.shape[0], cur_keypoints.shape[0])
                    # recognize features, get list of Detection arrays relative to their keypoint
                    cur_detections_list = self._vocabularies[f_name].recognizeFeatures(cur_features, suppressNegatives)
                    # sanity check
                    assert cur_keypoints.shape[0] == len(cur_detections_list), "Keypoints and detections list must match (feature type name: %s, keypoints: %d, detection lists: %d)" % (f_name, cur_keypoints.shape[0], len(cur_detections_list))
                    for i in range(len(cur_detections_list)):
                        # account for keypoint position
                        _adjustDetections(cur_keypoints[i], cur_detections_list[i]['position'])
                    # store detections for later use
                    detections_list += cur_detections_list
                else:
                    warnings.warn(RuntimeWarning("Vocabulary for feature type %s is not prepared for recognition, skipping." % f_name))
        # concatenate detections for all feature types
        if len(detections_list) > 0:
            detections = np.concatenate(detections_list)
        else:
            detections = np.empty(0, dtype = Detection_dtype)
        return detections

    def recognize(self, dict featureData, suppressNegatives = False):
        """Get detections aggregated using non-maximum suppression

        See :meth:`getDetections()` for more information about parameters.

        :Returns: Array with detections
        :Returntype: np.ndarray[:const:`~pydriver.Detection_dtype`]
        """
        cdef:
            cnp.ndarray detections_raw
            cnp.ndarray[FLOAT_t] weights
            FLOAT_t[:] weights_view     # fast Cython view
            cnp.ndarray mask
            int iHyp, iOtherHyp
        # get raw hypotheses
        detections_raw = self.getDetections(featureData, suppressNegatives)
        if not USE_SHAPELY:
            # shapely module not available, no non-maximum suppression possible
            warnings.warn(UserWarning('The module "shapely" could not be imported, non-maximum suppression deactivated.'))
            return detections_raw

        # copy weights since we will modify them to exclude processed hypotheses
        weights = detections_raw['weight'].copy()
        weights_view = weights
        # initialize mask, mask everything by default
        mask = np.ones(detections_raw.shape[0], dtype = np.bool)

        # get list with vertex arrays
        vertices_list = [get3DBoxVertices(label['box3D']) for label in detections2labels(detections_raw)]
        # get array with widths and lengths (for speedup)
        detections_sizes = detections_raw[['width', 'length']]
        # create flat shapes representing hypotheses
        shapes = []
        for i in range(detections_raw.shape[0]):
            if np.any(detections_sizes[i] == 0):
                # create point representing the center (since the polygon would have zero area)
                shapes.append(shapely.geometry.Point(detections_raw['x'][i], detections_raw['z'][i]))
            else:
                # create polygon (with clockwise X/Z coordinates)
                shapes.append(shapely.geometry.Polygon([(vertices_list[i][iVertex,0], vertices_list[i][iVertex,2]) for iVertex in (0, 1, 3, 2)]))

        # loop until everything is processed
        while True:
            # get index of next (not yet processed) hypothesis with maximum weight
            iHyp = np.argmax(weights)
            if weights_view[iHyp] < 0:
                # all hypotheses processed, exit loop
                break
            # unmask best hypothesis
            mask[iHyp] = False
            # mark best hypothesis as processed
            weights_view[iHyp] = -1

            # loop through other hypotheses
            this_shape = shapes[iHyp]
            for iOtherHyp in range(detections_raw.shape[0]):
                # check only if not already processed/excluded
                if weights_view[iOtherHyp] >= 0:
                    other_shape = shapes[iOtherHyp]
                    if this_shape.overlaps(other_shape):
                        # overlap exists, exclude other hypothesis (mark as processed)
                        weights_view[iOtherHyp] = -1

        # get unmasked detections and return them
        return detections_raw[np.invert(mask)]

    def _getParams(self):
        """Get parameters dictionary to save"""
        params = {
                  'featureTypes': self.featureTypes,
                  'vocabularyGenerator': self.vocabularyGenerator,
                  }
        return params

    @classmethod
    def _fromParams(cls, params):
        """Create instance from loaded parameters dictionary

        The vocabularies of the result remain uninitialized.
        """
        result = cls(featureTypes = params['featureTypes'], vocabularyGenerator = params['vocabularyGenerator'])
        return result


# --- internal functions ---

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _adjustDetections(Position keypoint, Position[:] detectionsPos):
    """Translate and rotate an array of relative detections according to position and orientation of their keypoint"""
    cdef:
        size_t i
        FLOAT_t rsin, rcos
    # precompute sin and cos for faster rotation
    rsin = libc.math.sin(keypoint.rotation_y)
    rcos = libc.math.cos(keypoint.rotation_y)
    for i in range(<size_t>detectionsPos.shape[0]):
        # account for keypoint rotation
        rotate2DXZSinCos(&detectionsPos[i].x, &detectionsPos[i].z, rsin, rcos)                              # rotate relative detection position
        detectionsPos[i].rotation_y = normalizeAngle(detectionsPos[i].rotation_y + keypoint.rotation_y)     # rotate and normalize detection orientation
        # account for keypoint translation
        detectionsPos[i].x += keypoint.x
        detectionsPos[i].y += keypoint.y
        detectionsPos[i].z += keypoint.z

cpdef cnp.ndarray _computeDetections(cnp.ndarray keypoints, str category, dict box3D):
    """Get positive detections relative to keypoints

    :Parameters:
        keypoints: np.ndarray[Position_dtype]
            Keypoint coordinates
        category: str
            Category of object
        box3D: dict
            3D box of object

    :Returns: NumPy array with detections
    :Returntype: np.ndarray[Detection_dtype]
    """
    cdef:
        size_t i
        FLOAT_t dx, dy, dz  # vector to object center
        cnp.ndarray detections
    # initialize resulting detections array
    detections = np.empty(keypoints.shape[0], dtype = Detection_dtype)

    for i in range(<size_t>keypoints.shape[0]):
        # compute XYZ vector to object center which is at (0,0,0)
        dx = box3D['location']['x'] - keypoints[i]['x']
        dy = box3D['location']['y'] - keypoints[i]['y']
        dz = box3D['location']['z'] - keypoints[i]['z']
        # rotate vector according to LRF
        rotate2DXZ(&dx, &dz, -keypoints[i]['rotation_y'])

        detections[i] = (
                    category,   # will be automatically cut off if too long
                    (
                        dx,
                        dy,
                        dz,
                        normalizeAngle(box3D['rotation_y'] - keypoints[i]['rotation_y']),
                    ),
                    box3D['dimensions']['height'],
                    box3D['dimensions']['width'],
                    box3D['dimensions']['length'],
                    1.0,    # Detection weight
                    )
    return detections

def _default_vocabularyGenerator(dimensions, featureName):
    """Default vocabulary generator function"""
    return vocabularies.Vocabulary(dimensions)
