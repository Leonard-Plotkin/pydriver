# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

import math, operator

import numpy as np

from ..common.constants import FLOAT_dtype


class Evaluator(object):
    """Evaluation curve representation

    Average precision and orientation similarity measures will compute the area under the monotonically
    decreasing function of maximum performance w.r.t. increasing minimum recall values.

    The negative category of hypotheses is processed correctly in the sense that it will be ignored during
    matching of hypotheses to ground truth.

    Evaluation is performed under the assumption that the exact positive category does not matter. Create
    multiple instances for different categories and only supply ground truth / hypotheses of desired category
    if you want to evaluate a multi-category recognition scenario.
    """

    def __init__(self, minOverlap = 0.5, nPoints = 0, nRecallIntervals = 0):
        """Initialize Evaluator

        Overlapping is computed as *intersection(ground truth, hypothesis) / union(ground truth, hypothesis)* considering
        their 2D bounding boxes. Minimum overlap is the criterion for matching hypotheses to ground truth.

        The number of individual points will correspond to the number of (different) weights of supplied hypotheses for
        nPoints<=0 or their minimum weights will be linearly spaced between minimum and maximum weights of those
        hypotheses for nPoints>0.

        Minimum recall values for averaging will be drawn from all available points for nRecallIntervals<=0 or will be linearly spaced
        between 0.0 and 1.0 (e.g. 10 intervals for nRecallIntervals=10: minimum recall = 0.0, 0.1, ..., 0.9, 1.0) for nRecallIntervals>0.

        :Parameters:
            minOverlap: float, optional
                Minimum overlap between ground truth and hypothesis to match them to each other, *0.5* by default
            nPoints: int, optional
                Number of individual points to evaluate, *0* by default
            nRecallIntervals: int, optional
                Number of intervals for average measures, *0* by default
        """
        # minimal overlap for matching
        self._minOverlap = minOverlap
        # number of points
        self._nPoints = nPoints
        # number of recall intervals
        self._nRecallIntervals = nRecallIntervals
        # undocumented flag for emulating KITTI evaluation, use with caution
        self._emulateKITTI = False
        # added frames
        self._frames = []
        # initialize cache
        self._cache = {}
        self._resetCache()

    def addFrame(self, groundTruth, groundTruthOpt, hypotheses):
        """Add frame for evaluation

        Function expects everything as list of labels (see :class:`~pydriver.datasets.base.BaseReader` for format description).

        The cached points will be updated.

        :Parameters:
            groundTruth : list
                Labels with mandatory ground truth
            groundTruthOpt : list
                Labels with optional ground truth
            hypotheses : list
                Labels with hypotheses, must contain 'info' dictionary with 'weight' key
        """
        frame = {
            'gt': groundTruth,
            'gtOpt': groundTruthOpt,
            'hyp': [h for h in hypotheses if h['category'] != 'negative'],  # ignore negative hypotheses
        }
        self._frames.append(frame)
        # update existing points
        for evPoint in self._cache['points'].values():
            evPoint.addFrame(frame['gt'], frame['gtOpt'], frame['hyp'])
        # update stored weights (set of unique values)
        for hyp in frame['hyp']:
            self._cache['weights'].add(hyp['info']['weight'])
        # reset all other cached values
        self._resetCache()

    def getPoint(self, minWeight):
        """Get :class:`EvaluatorPoint` corresponding to all hypotheses above given minimum weight

        The result will be cached.
        """
        # get all hypotheses weights as sorted NumPy array
        weights = self._getWeights(nPoints = 0)
        # find the minimal existing hypothesis weight >= minWeight
        realMinWeightIndex = np.searchsorted(weights, minWeight)
        if realMinWeightIndex == weights.shape[0]:
            # no hypotheses >= minWeight
            # map all those requests to a point above any possible weight
            realMinWeight = np.inf
        else:
            realMinWeight = weights[realMinWeightIndex]
        # replace minWeight with realMinWeight
        # the point: avoid constructing identical points between two hypotheses again and again
        minWeight = realMinWeight

        if minWeight not in self._cache['points']:
            evPoint = EvaluatorPoint(minOverlap = self._minOverlap, minWeight = minWeight)
            for frame in self._frames:
                evPoint.addFrame(frame['gt'], frame['gtOpt'], frame['hyp'])
            self._cache['points'][minWeight] = evPoint
        return self._cache['points'][minWeight]

    def getPoints(self):
        """Get all points according to self._nPoints as list of :class:`EvaluatorPoint`"""
        if 'getPoints' not in self._cache:
            # get all desired points (possibly multiple pointers to same instance)
            points = [self.getPoint(minWeight) for minWeight in self._getWeights(self._nPoints)]
            # get their sorted unique minimum weights
            minWeights = sorted(set([point.minWeight for point in points]))
            # get only unique instances
            points = [self.getPoint(minWeight) for minWeight in minWeights]
            # cache the result
            self._cache['getPoints'] = points
        return self._cache['getPoints']

    def getValues(self):
        """Get recall, precision and OS values suitable for plotting

        The function returns a dictionary with keys 'recall', 'precision' and 'OS'. Each of them contains
        a list with respective values sorted by recall. The produced recall/measure curves are convex.
        """
        valuesRecall = [0] + [recall for recall, length in self._getIntervals(self._nRecallIntervals)]
        valuesPrecision = [self._getMaxPrecision(recall) for recall in valuesRecall]
        valuesOS = [self._getMaxOS(recall) for recall in valuesRecall]
        return {'recall': valuesRecall, 'precision': valuesPrecision, 'OS': valuesOS}

    @property
    def aprecision(self):
        """Average precision"""
        return self._avalue(self._getMaxPrecision, self._nRecallIntervals)

    @property
    def aos(self):
        """Average orientation similarity"""
        return self._avalue(self._getMaxOS, self._nRecallIntervals)

    # --- internal functions ---
    def _resetCache(self):
        result = {}
        # points: cache for EvaluatorPoint objects (minWeight: EvaluatorPoint)
        if 'points' not in self._cache:
            result['points'] = {}                       # create empty points dictionary
        else:
            result['points'] = self._cache['points']    # preserve existing points
        # weights: cache for unique hypotheses weights (set of unique values)
        if 'weights' not in self._cache:
            result['weights'] = set()                   # create empty set of weights
        else:
            result['weights'] = self._cache['weights']  # preserve existing weights
        self._cache = result

    def _avalue(self, valuefunc, nRecallIntervals):
        """Average measure"""
        return sum([valuefunc(minRecall)*length for minRecall,length in self._getIntervals(nRecallIntervals)])

    def _getIntervals(self, nRecallIntervals):
        """Get recall values and the lengths they are covering

        A value <=0 for nRecallIntervals will use all available recall values from assigned evaluators.
        """
        if self._emulateKITTI:
            # emulate KITTI evaluation, don't pass nRecallIntervalls since KITTI uses exactly 11
            return self._getIntervalsKITTI()

        if nRecallIntervals > 0:
            # evenly spaced intervals
            minRecalls = np.linspace(0, 1, nRecallIntervals+1)
        else:
            # all unique recall values
            minRecalls = sorted(set([0] + [evPoint.recall for evPoint in self.getPoints()]))
        # compute differences between n-th and (n+1)-th recall value
        lengths = np.diff(minRecalls)
        # return tuples of recall values and their corresponding lengths (without the first value which covers zero length)
        return [(minRecalls[i+1], lengths[i]) for i in range(lengths.shape[0])]
    def _getIntervalsKITTI(self, nRecallIntervals = 11):
        """Test function for emulating KITTI averaging"""
        if nRecallIntervals < 1:
            nRecallIntervals = 11
        minRecalls = np.linspace(0, 1, nRecallIntervals)
        lengths = [1.0 / nRecallIntervals] * nRecallIntervals
        return [(minRecalls[i], lengths[i]) for i in range(nRecallIntervals)]

    def _getWeights(self, nPoints = 0):
        """Get mininum weights for points

        Get all unique weight values of supplied hypotheses for nPoints<=0 or use linear spacing between minimum and
        maximum weight otherwise. The result is a NumPy array.
        """
        if 'getWeightsExact' not in self._cache:
            # convert stored set to sorted NumPy array
            self._cache['getWeightsExact'] = np.array(sorted(self._cache['weights']), dtype = np.float)
        weights = self._cache['getWeightsExact']

        if nPoints > 0:
            # create linear spacing
            if weights.shape[0] < 2:
                weights = np.linspace(0, 1, nPoints)
            else:
                weights = np.linspace(weights.min(), weights.max(), nPoints)
        return weights

    def _getMaxPrecision(self, minRecall):
        """Get maximal precision of evaluators with specified minimum recall"""
        return max([0] + [evPoint.precision for evPoint in self._getMinRecallEvaluators(minRecall)])
    def _getMaxOS(self, minRecall):
        """Get maximal orientation similarity of evaluators with specified minimum recall"""
        return max([0] + [evPoint.os for evPoint in self._getMinRecallEvaluators(minRecall)])
    def _getMinRecallEvaluators(self, minRecall):
        """Get evaluators with specified minimum recall"""
        if 'getMinRecallEvaluators' not in self._cache:
            self._cache['getMinRecallEvaluators'] = {}
        if minRecall not in self._cache['getMinRecallEvaluators']:
            self._cache['getMinRecallEvaluators'][minRecall] = [evPoint for evPoint in self.getPoints() if evPoint.recall >= minRecall]
        return self._cache['getMinRecallEvaluators'][minRecall]


class EvaluatorPoint(object):
    """Evaluation point representation"""
    def __init__(self, minOverlap = 0.5, minWeight = 0.0):
        """Initialize Evaluator instance

        Overlapping is computed as *intersection(ground truth, hypothesis) / union(ground truth, hypothesis)* considering
        their 2D bounding boxes. Minimum overlap is the criterion for matching hypotheses to ground truth.

        :Parameters:
            minOverlap: float, optional
                Minimum overlap between ground truth and hypothesis to count the latter as true positive, *0.5* by default
            minWeight: float, optional
                Dismiss hypotheses with lesser weight, *0.0* by default
        """
        self._minOverlap = minOverlap
        self._minWeight = minWeight

        self.TP = 0         # true positives
        self.FN = 0         # false negatives
        self.FP = 0         # false positives

        self._os_sum = 0    # non-normalized orientation similarity

    @property
    def minOverlap(self):
        return self._minOverlap
    @property
    def minWeight(self):
        return self._minWeight

    @property
    def objects(self):
        """Number of ground truth objects (detected and missed)"""
        return self.TP + self.FN
    @property
    def detections(self):
        """Number of detections (true and false)"""
        return self.TP + self.FP
    @property
    def recall(self):
        """Recall"""
        if self.objects > 0:
            return self.TP / self.objects
        else:
            return 0.0
    @property
    def precision(self):
        """Precision"""
        if self.detections > 0:
            return self.TP / self.detections
        else:
            return 0.0
    @property
    def os(self):
        """Normalized orientation similarity"""
        if self.detections > 0:
            return self._os_sum / self.detections
        else:
            return 0.0

    def addFrame(self, groundTruth, groundTruthOpt, hypotheses):
        """Add frame for evaluation

        Function expects everything as list of labels (see :class:`~pydriver.datasets.base.BaseReader` for format description).

        :Parameters:
            groundTruth : list
                Labels with mandatory ground truth
            groundTruthOpt : list
                Labels with optional ground truth
            hypotheses : list
                Labels with hypotheses, must contain 'info' dictionary with 'weight' key
        """
        # get positive hypotheses with specified minimum confidence
        hypotheses_pos = [h for h in hypotheses if h['category'] != 'negative' and h['info']['weight'] >= self._minWeight]

        gt_matches, gtOpt_matches, hypotheses_FP = _get2DMatches(groundTruth, groundTruthOpt, hypotheses_pos, self._minOverlap)

        # add number of false positives
        self.FP += len(hypotheses_FP)

        # process obligatory ground truth
        # (matches with optional ground truth do not affect evaluation results except that they don't count as false positives)
        for iGT in range(len(groundTruth)):
            if gt_matches[iGT] == -1:
                # no match, add false negative
                self.FN += 1
            else:
                # match, add true positive
                self.TP += 1
                # add orientation similarity
                self._os_sum += _getOS(groundTruth[iGT]['box3D'], hypotheses_pos[gt_matches[iGT]]['box3D'])


# --- module-wide internal functions ---

def _getOS(box1, box2):
    """Compute orientation similarity between two 3D boxes"""
    angle_diff = box1['rotation_y'] - box2['rotation_y']
    return (1 + math.cos(angle_diff)) / 2

def _get2DMatches(groundTruth, groundTruthOpt, hypotheses, minOverlap):
    """Match ground truth and optional ground truth to hypotheses

    Function expects everything as label (not as Detection) and positive hypotheses only.

    This is a greedy algorithm which looks for the best match in each iteration. All mandatory ground truth
    labels are processed first before the function looks for matches to optional ground truth.
    """
    # initialize list of indices of unmatched hypotheses
    unmatchedHypotheses = list(range(len(hypotheses)))
    # mandatory ground truth
    gt_matches = np.empty(len(groundTruth), dtype = np.int)         # indexes of hypotheses which match ground truth labels, -1 is "unmatched"
    gt_matches[:] = -1                                              # defaults to "unmatched"
    # optional ground truth
    gtOpt_matches = np.empty(len(groundTruthOpt), dtype = np.int)   # indexes of hypotheses which match optional ground truth labels, -1 is "unmatched"
    gtOpt_matches[:] = -1                                           # defaults to "unmatched"

    # match to mandatory ground truth
    overlap = _get2DOverlapMatrix(groundTruth, hypotheses)          # initialize matrix with overlap values between ground truth and hypotheses
    while np.any(overlap >= minOverlap):
        # find best match between label and hypothesis
        iLabel, iHypothesis = np.unravel_index(overlap.argmax(), overlap.shape)
        # assign hypothesis to the label
        gt_matches[iLabel] = iHypothesis
        # remove hypothesis from list of unmatched hypotheses
        unmatchedHypotheses.remove(iHypothesis)
        # reset overlap value for this hypothesis and label
        overlap[iLabel, :] = -1
        overlap[:, iHypothesis] = -1

    # match to optional ground truth
    overlap = _get2DOverlapMatrix(groundTruthOpt, hypotheses)       # initialize matrix with overlap values between optional ground truth and hypotheses
    # reset overlap values for hypotheses which were already matched to ground truth
    for iHypothesis in gt_matches:
        if iHypothesis >= 0:
            overlap[:, iHypothesis] = -1
    while np.any(overlap >= minOverlap):
        # find best match between label and hypothesis
        iLabel, iHypothesis = np.unravel_index(overlap.argmax(), overlap.shape)
        # assign hypothesis to the label
        gtOpt_matches[iLabel] = iHypothesis
        # remove hypothesis from list of unmatched hypotheses
        unmatchedHypotheses.remove(iHypothesis)
        # reset overlap value for this hypothesis and label
        overlap[iLabel, :] = -1
        overlap[:, iHypothesis] = -1

    return gt_matches, gtOpt_matches, unmatchedHypotheses

def _get2DOverlapMatrix(truth, hypotheses):
    overlap = np.zeros((len(truth), len(hypotheses)), dtype = FLOAT_dtype)
    for j in range(overlap.shape[1]):
        for i in range(overlap.shape[0]):
                overlap[i, j] = _get2DOverlap(truth[i]['box2D'], hypotheses[j]['box2D'])
    return overlap

def _get2DOverlap(box1, box2):
    """Get box overlap between 0 and 1"""
    areaOverlap = _get2DArea(_get2DOverlapBox(box1, box2))
    return areaOverlap / (_get2DArea(box1) + _get2DArea(box2) - areaOverlap)

def _get2DOverlapBox(box1, box2):
    """Get 2D box where the two boxes overlap"""
    result = {
              'left': max(box1['left'], box2['left']),
              'top': max(box1['top'], box2['top']),
              'right': min(box1['right'], box2['right']),
              'bottom': min(box1['bottom'], box2['bottom']),
              }
    # ensure that right>=left and bottom>=top
    result['right'] = max(result['left'], result['right'])
    result['bottom'] = max(result['top'], result['bottom'])
    return result

def _get2DArea(box):
    """Get area of a 2D box"""
    return (box['right']-box['left']) * (box['bottom']-box['top'])
