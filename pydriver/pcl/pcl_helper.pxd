# -*- coding: utf-8 -*-
"""Cython definition file for pcl_helper library"""
from __future__ import absolute_import, division

from libcpp cimport bool as c_bool

from ..common.structs cimport FLOAT_t, Detection, Position, TetragonalPrism, SHOTFeature, SHOTColorFeature


# make clear that the void pointer used in imported library functions refers to a CPCLHelper instance
ctypedef void* CPCLHelperPtr
cdef extern from "pcl_helper_exports.h":
    # factory functions that create and release instances of CPCLHelper
    CPCLHelperPtr __stdcall getPCLHelper()
    void __stdcall freePCLHelper(CPCLHelperPtr h)

    # implementation functions
    void setNormalsParams(CPCLHelperPtr h, int k_search, FLOAT_t radius)

    size_t getCloudSize(CPCLHelperPtr h, size_t *width, size_t *height)
    void fromArray(CPCLHelperPtr h, size_t width, size_t height, FLOAT_t *pPoints, char *pRGB)
    void toArray(CPCLHelperPtr h, FLOAT_t *pXYZ, char *pRGB)
    void copyCloud(CPCLHelperPtr h, CPCLHelperPtr out)
    void addCloud(CPCLHelperPtr h, CPCLHelperPtr addendHelper)
    void removeNaN(CPCLHelperPtr h, )

    void setBGColor(CPCLHelperPtr h, FLOAT_t r, FLOAT_t g, FLOAT_t b)
    void setCameraPosition(CPCLHelperPtr h, FLOAT_t camPosX, FLOAT_t camPosY, FLOAT_t camPosZ, FLOAT_t camViewX, FLOAT_t camViewY, FLOAT_t camViewZ, FLOAT_t camUpX, FLOAT_t camUpY, FLOAT_t camUpZ)
    void setKeypointsVisualization(CPCLHelperPtr h, FLOAT_t size, FLOAT_t normalLength, FLOAT_t r, FLOAT_t g, FLOAT_t b)
    void setDetectionsVisualization(CPCLHelperPtr h, FLOAT_t negSize, FLOAT_t negR, FLOAT_t negG, FLOAT_t negB, FLOAT_t posR, FLOAT_t posG, FLOAT_t posB, FLOAT_t GTR, FLOAT_t GTG, FLOAT_t GTB, FLOAT_t GTOptR, FLOAT_t GTOptG, FLOAT_t GTOptB)
    void visualize(CPCLHelperPtr h, char *title, c_bool fullscreen)
    void visualizeKeypoints(CPCLHelperPtr h, char *title, CPCLHelperPtr keypointsHelper, CPCLHelperPtr normalsHelper, c_bool fullscreen)
    void visualizeDetections(CPCLHelperPtr h, char *title, size_t nDetections, Detection *detections, size_t nGroundTruth, Detection *groundTruth, size_t nGroundTruthOpt, Detection *groundTruthOpt, c_bool fullscreen)
    void saveCloud(CPCLHelperPtr h, char *filename)

    void downsampleVoxelGrid(CPCLHelperPtr h, FLOAT_t leafSize, CPCLHelperPtr out)
    void restrictViewport(CPCLHelperPtr h, FLOAT_t x_min, FLOAT_t x_max, FLOAT_t y_min, FLOAT_t y_max, FLOAT_t z_min, FLOAT_t z_max)

    void detectGroundPlane(CPCLHelperPtr h, FLOAT_t maxPlaneAngle, FLOAT_t distanceThreshold, FLOAT_t *coefficients, FLOAT_t *transformation)
    void removeGroundPlane(CPCLHelperPtr h, FLOAT_t distanceThreshold, const FLOAT_t *coefficients)
    void transform(CPCLHelperPtr h, const FLOAT_t *transformation)

    size_t getConnectedComponents(CPCLHelperPtr h, FLOAT_t distanceThreshold, void **pLabelsIndices)
    void extractConnectedComponents(CPCLHelperPtr h, void *pLabelsIndices, CPCLHelperPtr *components)

    void extractTetragonalPrisms(CPCLHelperPtr h, size_t nPrisms, TetragonalPrism *prisms, CPCLHelperPtr out, c_bool invert)

    void getNormalsOfCloud(CPCLHelperPtr h, CPCLHelperPtr inputHelper, FLOAT_t radius, FLOAT_t *pNormals)

    int getHarrisPoints(CPCLHelperPtr h, FLOAT_t radius, CPCLHelperPtr out, c_bool refine, FLOAT_t threshold, int method)
    int getISSPoints(CPCLHelperPtr h, FLOAT_t salientRadius, FLOAT_t nonMaxRadius, int numberOfThreads, CPCLHelperPtr out, int minNeighbors, FLOAT_t threshold21, FLOAT_t threshold32, FLOAT_t angleThreshold, FLOAT_t normalRadius, FLOAT_t borderRadius)

    size_t computeSHOT(CPCLHelperPtr h, CPCLHelperPtr keypointHelper, FLOAT_t radius, SHOTFeature *out)
    size_t computeSHOTColor(CPCLHelperPtr h, CPCLHelperPtr keypointHelper, FLOAT_t radius, SHOTColorFeature *out)
