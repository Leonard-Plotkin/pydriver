# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

from cpython cimport bool as py_bool

cimport numpy as cnp

from ..common.structs cimport FLOAT_t, Detection, Position


cdef class PCLHelper:
    cdef:
        readonly:
            tuple bgColor       # background color for visualization (3 values)
            tuple camPosition   # camera position for visualization (3 tuples with 3 values each)
            FLOAT_t kpSize, kpNormalLength  # keypoint size, keypoint normal length
            tuple kpColor                   # keypoint color
            FLOAT_t detNegSize  # edge length of negative detections cubes
            # colors of detections visualization: negative, positive, ground truth, optional ground truth
            tuple detNegColor, detPosColor, detGTColor, detGTOptColor
            # normals estimation parameters
            int normals_k
            FLOAT_t normals_radius
        void* me

    # Python/C functions (cpdef)
    cpdef setNormalsKSearch(PCLHelper self, int k_search)
    cpdef setNormalsRadius(PCLHelper self, FLOAT_t radius)

    cpdef tuple getCloudSize(PCLHelper self)
    cpdef py_bool isOrganized(PCLHelper self)
    cpdef fromArray(PCLHelper self, cnp.ndarray XYZ, cnp.ndarray RGB = *)
    cpdef toArray(PCLHelper self, py_bool flat, py_bool extractRGB)
    cpdef PCLHelper copy(PCLHelper self)
    cpdef addCloud(PCLHelper self, PCLHelper addend)
    cpdef removeNaN(PCLHelper self)

    cpdef setBGColor(PCLHelper self, tuple color)
    cpdef setCameraPosition(PCLHelper self, tuple position)
    cpdef setKeypointsVisualization(PCLHelper self, FLOAT_t size = *, FLOAT_t normalLength = *, tuple color = *)
    cpdef setDetectionsVisualization(PCLHelper self, FLOAT_t negSize = *, tuple negColor = *, tuple posColor = *, tuple GTColor = *, tuple GTOptColor = *)
    cpdef visualize(PCLHelper self, title = *, py_bool fullscreen = *)
    cpdef visualizeKeypoints(PCLHelper self, PCLHelper keypoints, PCLHelper normals = *, title = *, py_bool fullscreen = *)
    cpdef visualizeDetections(PCLHelper self, Detection[:] detections = *, Detection[:] groundTruth = *, Detection[:] groundTruthOpt = *, title = *, py_bool fullscreen = *)
    cpdef save(PCLHelper self, filename)

    cpdef PCLHelper downsampleVoxelGrid(PCLHelper self, FLOAT_t leafSize)
    cpdef restrictViewport(PCLHelper self, tuple viewport)

    cpdef tuple detectGroundPlane(PCLHelper self, FLOAT_t maxPlaneAngle, FLOAT_t distanceThreshold)
    cpdef removeGroundPlane(PCLHelper self, FLOAT_t distanceThreshold, cnp.ndarray[FLOAT_t] coefficients)
    cpdef transform(PCLHelper self, cnp.ndarray[FLOAT_t, ndim=2] transformation)

    cpdef list getConnectedComponents(PCLHelper self, FLOAT_t distanceThreshold)

    cpdef PCLHelper extractOrientedBoxes(PCLHelper self, list boxes3D, FLOAT_t margin = *, py_bool invert = *)
    cpdef PCLHelper _extractTetragonalPrisms(PCLHelper self, cnp.ndarray prisms, py_bool invert = *)

    cpdef tuple getNormalsOfCloud(PCLHelper self, PCLHelper pointCloud, FLOAT_t radius)

    cpdef PCLHelper getHarrisPoints(PCLHelper self, FLOAT_t radius, py_bool refine = *, FLOAT_t threshold = *, int method = *)
    cpdef PCLHelper getISSPoints(PCLHelper self, FLOAT_t salientRadius, FLOAT_t nonMaxRadius, int minNeighbors = *, FLOAT_t threshold21 = *, FLOAT_t threshold32 = *, FLOAT_t angleThreshold = *, FLOAT_t normalRadius = *, FLOAT_t borderRadius = *)

    cpdef tuple computeSHOT(PCLHelper self, FLOAT_t radius, PCLHelper keypoints)
    cpdef tuple computeSHOTColor(PCLHelper self, FLOAT_t radius, PCLHelper keypoints)

    # C functions (cdef)
    # currently no declarations
