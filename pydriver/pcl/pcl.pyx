# -*- coding: utf-8 -*-
"""
.. data:: SHOTFeature_dtype
    Structure for SHOT features extracted by PCL

    descriptor: :const:`~pydriver.FLOAT_dtype` [352]
        Feature descriptor
    rf: :const:`~pydriver.FLOAT_dtype` [3,3]
        Reference frame

.. data:: SHOTColorFeature_dtype
    Structure for SHOT+Color features extracted by PCL

    descriptor: :const:`~pydriver.FLOAT_dtype` [1344]
        Feature descriptor
    rf: :const:`~pydriver.FLOAT_dtype` [3,3]
        Reference frame
"""
from __future__ import absolute_import, division

include "../common/functions_include.pyx"

from cython.view cimport array as cvarray

from libc.stdlib cimport malloc, free
from libcpp cimport bool as c_bool
from cpython cimport bool as py_bool
cimport libc.math
from libcpp.vector cimport vector

cimport numpy as cnp
import numpy as np

from ..common.structs cimport FLOAT_t, Detection, TetragonalPrism, SHOTFeature, SHOTColorFeature
from ..common.constants import FLOAT_dtype, Detection_dtype, TetragonalPrism_dtype, NUMBER_OF_THREADS, PI

from . cimport pcl_helper


# === run-time type definitions ===

# dtype for NumPy arrays containing (raw) SHOTFeature
# must define an identical structure as SHOTFeature (in common/structs.h)
SHOTFeature_dtype = np.dtype([
                              ('descriptor', FLOAT_dtype, 352),
                              ('rf', FLOAT_dtype, (3,3)),
                              ])
# dtype for NumPy arrays containing (raw) SHOTColorFeature
# must define an identical structure as SHOTColorFeature (in common/structs.h)
SHOTColorFeature_dtype = np.dtype([
                                   ('descriptor', FLOAT_dtype, 1344),
                                   ('rf', FLOAT_dtype, (3,3)),
                                   ])


cdef class PCLHelper:
    """Point Cloud Library helper

    An instance of this class represents a single XYZRGB point cloud.
    """
    def __cinit__(self):
        self.me = pcl_helper.getPCLHelper()
    def __dealloc__(self):
        pcl_helper.freePCLHelper(self.me)

    def __init__(self, XYZ = None, RGB = None, int normals_k = 30, FLOAT_t normals_radius = 0.0, tuple bgColor = None, tuple camPosition = None, dict kpVisParams = None, dict detVisParams = None):
        """Initialize a new PCLHelper instance

        The default background is dark green so black and missing regions can be easily distinguished.

        This object can be pickled with the :mod:`pickle` Python module.

        :Parameters:
            XYZ: np.ndarray[FLOAT_dtype, ndim=2 or 3], optional
                X, Y and Z coordinates of points, see :meth:`fromArray`
            RGB: np.ndarray[np.uint8, ndim=2 or 3], optional
                R, G and B colors of points, see :meth:`fromArray`
            normals_k: int, optional
                See :meth:`setNormalsKSearch()`, 30 by default
            normals_radius: FLOAT_dtype, optional
                See :meth:`setNormalsRadius()`, disabled by default
            bgColor: tuple, optional
                Background color for visualization, see :meth:`setBGColor()`
            camPosition: tuple, optional
                Camera position for visualization, see :meth:`setCameraPosition()`
            kpVisParams: dict, optional
                Dictionary with keypoints visualization parameters, keys: *'size'*, *'normalLength'*, *'color'*, see :meth:`setKeypointsVisualization()`
            detVisParams: dict, optional
                Dictionary with detection visualization parameters, keys: *'negSize'*, *'negColor'*, *'posColor'*, *'GTColor'*, *'GTOptColor'*, see :meth:`setDetectionsVisualization()`
        """
        if normals_radius > 0:
            self.setNormalsRadius(normals_radius)
        else:
            self.setNormalsKSearch(normals_k)

        if bgColor is not None:
            self.setBGColor(bgColor)
        else:
            self.setBGColor((0.0, 0.1, 0.0))    # dark green
        if camPosition is not None:
            self.setCameraPosition(camPosition)
        else:
            # default to a comfortable macro scene viewing and rotation position
            self.setCameraPosition((
                                   (0, 0, -5),  # position
                                   (0, 0, 10),  # view
                                   (0, -1, 0),  # up
                                   ))
        if kpVisParams is None:
            kpVisParams = {}    # allows us to call kpVisParams.get()
        self.setKeypointsVisualization(kpVisParams.get('size', 5),
                                       kpVisParams.get('normalLength', 0.5),
                                       kpVisParams.get('color', (1.0, 0.0, 0.0)),           # red
                                       )
        if detVisParams is None:
            detVisParams = {}   # allows us to call detVisParams.get()
        self.setDetectionsVisualization(detVisParams.get('negSize', 1),
                                        detVisParams.get('negColor', (1.0, 0.0, 0.0)),      # red
                                        detVisParams.get('posColor', (1.0, 1.0, 1.0)),      # white
                                        detVisParams.get('GTColor', (0.0, 1.0, 0.0)),       # green
                                        detVisParams.get('GTOptColor', (1.0, 1.0, 0.0)),    # yellow
                                        )

        if XYZ is not None:
            self.fromArray(XYZ, RGB)    # RGB = None is a valid value

    def __reduce__(self):
        """Pickling implementation"""
        return (PCLHelper, self.toArray(flat = False, extractRGB = True) + (
            self.normals_k,
            self.normals_radius,
            self.bgColor,
            self.camPosition,
            {'size': self.kpSize, 'normalLength': self.kpNormalLength, 'color': self.kpColor},
            {'negSize': self.detNegSize, 'negColor': self.detNegColor, 'posColor': self.detPosColor, 'GTColor': self.detGTColor, 'GTOptColor': self.detGTOptColor},
            ))

    cpdef setNormalsKSearch(PCLHelper self, int k):
        """Set normals estimation method to kNN"""
        self.normals_k = k
        self.normals_radius = 0
        pcl_helper.setNormalsParams(self.me, k, 0)

    cpdef setNormalsRadius(PCLHelper self, FLOAT_t radius):
        """Set normals estimation method to radius search"""
        self.normals_k = 0
        self.normals_radius = radius
        pcl_helper.setNormalsParams(self.me, 0, radius)

    cpdef tuple getCloudSize(PCLHelper self):
        """Get cloud size

        :Returns: Tuple with total points number and shape
        :Returntype: tuple(nPoints, (width, height))
        """
        cdef:
            size_t nPoints, width, height
        nPoints = pcl_helper.getCloudSize(self.me, &width, &height)
        return (nPoints, (width, height))

    cpdef py_bool isOrganized(PCLHelper self):
        """Check whether the cloud is organized

        Organized clouds have meaningful width and height, invalid points must be set to *NaN*. Some operations require such organized clouds and some
        operations (such as :meth:`removeNaN()`) make organized clouds unorganized.
        """
        # check whether height > 1
        return (self.getCloudSize()[1][1] > 1)

    cpdef fromArray(PCLHelper self, cnp.ndarray XYZ, cnp.ndarray RGB = None):
        """Construct point cloud from array(s)

        3-dimensional arrays result in organized clouds, see :meth:`isOrganized()`.

        See :meth:`toArray`.

        :Parameters:
            XYZ: np.ndarray[FLOAT_dtype, ndim=2 or 3]
                *(nPoints, 3)*- or *(height, width, 3)*-shaped array with XYZ coordinates
            RGB: np.ndarray[np.uint8, ndim=2 or 3], optional
                *(nPoints, 3)*- or *(height, width, 3)*-shaped array with RGB values
        """
        cdef:
            cnp.ndarray[FLOAT_t, ndim=3] XYZ_3D
            cnp.ndarray[cnp.uint8_t, ndim=3] RGB_3D
        if XYZ.shape[0] == 0:
            # nothing to do
            return

        # get 3D array with XYZ data
        # Note: len(XYZ.shape) would produce a Cython error due to typed declaration of XYZ
        if len((<object>XYZ).shape) == 2:
            # array is 2-dimensional, XYZ[i] = (x, y, z)
            XYZ_3D = XYZ.reshape((1, XYZ.shape[0], XYZ.shape[1]))
        else:
            # array is already 3-dimensional, XYZ[cam_y, cam_x] = (x, y, z)
            XYZ_3D = XYZ
        XYZ_3D = np.ascontiguousarray(XYZ_3D)

        if RGB is None:
            pcl_helper.fromArray(self.me, XYZ_3D.shape[1], XYZ_3D.shape[0], &XYZ_3D[0, 0, 0], <char*>0)
        else:
            # get 3D array with RGB data
            # Note: len(RGB.shape) would produce a Cython error due to typed declaration of RGB
            if len((<object>RGB).shape) == 2:
                # array is 2-dimensional, RGB[i] = (r, g, b)
                RGB_3D = RGB.reshape((1, RGB.shape[0], RGB.shape[1]))
            else:
                # array is already 3-dimensional, RGB[cam_y cam_x] = (r, g, b)
                RGB_3D = RGB
            RGB_3D = np.ascontiguousarray(RGB_3D)
            pcl_helper.fromArray(self.me, XYZ_3D.shape[1], XYZ_3D.shape[0], &XYZ_3D[0, 0, 0], <char*>&RGB_3D[0, 0, 0])

    cpdef toArray(PCLHelper self, py_bool flat, py_bool extractRGB):
        """Get *(nPoints, 3)*- or *(height, width, 3)*-shaped array(s) with cloud points' data

        The type of XYZ array is FLOAT_dtype, the type of RGB array is np.uint8.

        If extractRGB is True the result will be a tuple *(XYZ, RGB)* with coordinates and color of cloud points. Otherwise
        the result will be the XYZ data array.

        See :meth:`fromArray`.

        :Parameters:
            flat: bool
                Indicate whether to extract a flat *(nPoints, 3)*-shaped array instead of a *(height, width, 3)*-shaped one
            extractRGB: bool
                Indicate whether to extract RGB data in addition to XYZ data

        :Returns: XYZ data or tuple(XYZ, RGB)
        :Returntype: np.ndarray[FLOAT_dtype, ndim=2 or 3] or tuple(np.ndarray[FLOAT_dtype, ndim=2 or 3], np.ndarray[np.uint8, ndim=2 or 3])
        """
        cdef:
            cnp.ndarray XYZ, RGB
            size_t nPoints, width, height
        nPoints, (width, height) = self.getCloudSize()

        if flat:
            XYZ = np.empty((nPoints, 3), dtype = FLOAT_dtype)
        else:
            XYZ = np.empty((height, width, 3), dtype = FLOAT_dtype)

        if extractRGB:
            if flat:
                RGB = np.empty((nPoints, 3), dtype = np.uint8)
            else:
                RGB = np.empty((height, width, 3), dtype = np.uint8)

            if nPoints > 0:
                pcl_helper.toArray(self.me, <FLOAT_t*>XYZ.data, <char*>RGB.data)
            return (XYZ, RGB)
        else:
            if nPoints > 0:
                pcl_helper.toArray(self.me, <FLOAT_t*>XYZ.data, <char*>0)
            return XYZ

    cpdef PCLHelper copy(PCLHelper self):
        """Create a copy of this cloud

        Only the points are copied, no parameters will be set.

        :Returns: copied cloud
        :Returntype: PCLHelper
        """
        cdef:
            PCLHelper result
        result = PCLHelper()
        pcl_helper.copyCloud(self.me, result.me)
        return result

    cpdef addCloud(PCLHelper self, PCLHelper addend):
        """Merge another cloud into this one

        Destroys point cloud organization.
        """
        pcl_helper.addCloud(self.me, addend.me)

    cpdef removeNaN(PCLHelper self):
        """Remove *NaN* values from point cloud

        Destroys point cloud organization.
        """
        pcl_helper.removeNaN(self.me)

    cpdef setBGColor(PCLHelper self, tuple color):
        """Set background color for visualization

        :Parameters:
            color: tuple
                Contains 3 float values (RGB) between 0 and 1
        """
        self.bgColor = color
        pcl_helper.setBGColor(self.me, self.bgColor[0], self.bgColor[1], self.bgColor[2])

    cpdef setCameraPosition(PCLHelper self, tuple camPosition):
        """Set camera position for visualization

        :Parameters:
            camPosition: tuple
                Contains 3 tuples (position, view, up) with 3 float values (x, y, z) each
        """
        self.camPosition = camPosition
        pcl_helper.setCameraPosition(self.me,
            self.camPosition[0][0], self.camPosition[0][1], self.camPosition[0][2],     # position
            self.camPosition[1][0], self.camPosition[1][1], self.camPosition[1][2],     # view
            self.camPosition[2][0], self.camPosition[2][1], self.camPosition[2][2],     # up
        )

    cpdef setKeypointsVisualization(PCLHelper self, FLOAT_t size = -1, FLOAT_t normalLength = -1, tuple color = None):
        """Set parameters for keypoints visualization

        Color must be given as tuple with three RGB float values between 0.0 and 1.0. Parameters which are not given will not be changed.

        :Parameters:
            size: FLOAT_dtype, optional
                Size of keypoints (in pixels)
            normalLength: FLOAT_dtype, optional
                Length of normals
            color: tuple, optional
                Color of keypoints and their normals, contains 3 float values (RGB) between 0 and 1
        """
        if size > 0:
            self.kpSize = size
        if normalLength > 0:
            self.kpNormalLength = normalLength
        self.kpColor = color or self.kpColor
        pcl_helper.setKeypointsVisualization(self.me, self.kpSize, self.kpNormalLength, self.kpColor[0], self.kpColor[1], self.kpColor[2])

    cpdef setDetectionsVisualization(PCLHelper self, FLOAT_t negSize = -1, tuple negColor = None, tuple posColor = None, tuple GTColor = None, tuple GTOptColor = None):
        """Set parameters for detections visualization

        Colors must be given as tuples with three RGB float values between 0.0 and 1.0. Parameters which are not given will not be changed.

        :Parameters:
            negSize: FLOAT_dtype, optional
                Edge length of cubes visualizing negative detections
            negColor: tuple, optional
                Color of negative detections
            posColor: tuple, optional
                Color of positive detections
            GTColor: tuple, optional
                Color of ground truth
            GTOptColor: tuple, optional
                Color of optional ground truth
        """
        if negSize > 0:
            self.detNegSize = negSize
        self.detNegColor = negColor or self.detNegColor
        self.detPosColor = posColor or self.detPosColor
        self.detGTColor = GTColor or self.detGTColor
        self.detGTOptColor = GTOptColor or self.detGTOptColor
        pcl_helper.setDetectionsVisualization(self.me,
            self.detNegSize,
            self.detNegColor[0], self.detNegColor[1], self.detNegColor[2],
            self.detPosColor[0], self.detPosColor[1], self.detPosColor[2],
            self.detGTColor[0], self.detGTColor[1], self.detGTColor[2],
            self.detGTOptColor[0], self.detGTOptColor[1], self.detGTOptColor[2],
        )

    cpdef visualize(PCLHelper self, title = "Point Cloud Visualization", py_bool fullscreen = False):
        """Interactively visualize the point cloud

        :Parameters:
            title: string, optional
                Window title, *"Point Cloud Visualization"* by default
            fullscreen: bool, optional
                Use full screen instead of window, *False* by default
         """
        pcl_helper.visualize(self.me, title.encode('ascii'), fullscreen)

    cpdef visualizeKeypoints(PCLHelper self, PCLHelper keypoints, PCLHelper normals = None, title = "Keypoints Visualization", py_bool fullscreen = False):
        """Interactively visualize the point cloud with given keypoints and optionally their normals

        Use :meth:`setKeypointsVisualization()` to change how the keypoints will be visualized.

        :Parameters:
            keypoints: PCLHelper
                PCLHelper instance containing the keypoints
            normals: PCLHelper, optional
                PCLHelper instance where XYZ values represent the normal vector for each keypoint in *keypoints*
            title: string, optional
                Window title, *"Keypoints Visualization"* by default
            fullscreen: bool, optional
                Use full screen instead of window, *False* by default
        """
        if normals is not None:
            assert keypoints.getCloudSize() == normals.getCloudSize(), "keypoints and normals must have exactly the same cloud size and shape if both are given"
        pcl_helper.visualizeKeypoints(self.me, title.encode('ascii'), keypoints.me, <pcl_helper.CPCLHelperPtr>0 if normals is None else normals.me, fullscreen)

    cpdef visualizeDetections(PCLHelper self, Detection[:] detections = None, Detection[:] groundTruth = None, Detection[:] groundTruthOpt = None, title = "Detections Visualization", py_bool fullscreen = False):
        """Interactively visualize the point cloud with optionally specified ground truth and detection hypotheses

        Use :meth:`setDetectionsVisualization()` to change how the detections will be visualized.

        :Parameters:
            detections: np.ndarray[:const:`~pydriver.Detection_dtype`], optional
                Detection hypotheses
            groundTruth: np.ndarray[:const:`~pydriver.Detection_dtype`], optional
                Ground truth for this cloud
            groundTruthOpt: np.ndarray[:const:`~pydriver.Detection_dtype`], optional
                Optional ground truth for this cloud
            title: string, optional
                Window title, *"Detections Visualization"* by default
            fullscreen: bool, optional
                Use full screen instead of window, *False* by default
        """
        cdef:
            Detection[:] cDetections, cGroundTruth, cGroundTruthOpt
        if detections is None:
            cDetections = np.empty(0, dtype = Detection_dtype)
        else:
            cDetections = np.ascontiguousarray(detections)
        if groundTruth is None:
            cGroundTruth = np.empty(0, dtype = Detection_dtype)
        else:
            cGroundTruth = np.ascontiguousarray(groundTruth)
        if groundTruthOpt is None:
            cGroundTruthOpt = np.empty(0, dtype = Detection_dtype)
        else:
            cGroundTruthOpt = np.ascontiguousarray(groundTruthOpt)

        pcl_helper.visualizeDetections(self.me, title.encode('ascii'),
            cDetections.shape[0], <Detection*>(&cDetections[0] if cDetections.shape[0]>0 else <Detection*>0),
            cGroundTruth.shape[0], <Detection*>(&cGroundTruth[0] if cGroundTruth.shape[0]>0 else <Detection*>0),
            cGroundTruthOpt.shape[0], <Detection*>(&cGroundTruthOpt[0] if cGroundTruthOpt.shape[0]>0 else <Detection*>0),
            fullscreen,
        )

    cpdef save(PCLHelper self, filename):
        """Save point cloud to file in PCD format"""
        pcl_helper.saveCloud(self.me, filename.encode('ascii'))

    cpdef PCLHelper downsampleVoxelGrid(PCLHelper self, FLOAT_t leafSize):
        """Get PCLHelper containing the cloud downsampled with PCL VoxelGrid filter

        The resulting cloud is not organized.

        :Parameters:
            leafSize: float
                Voxel grid leaf size

        :Returns: downsampled cloud
        :Returntype: PCLHelper
        """
        cdef:
            PCLHelper downsampledCloud = PCLHelper()
        pcl_helper.downsampleVoxelGrid(self.me, leafSize, downsampledCloud.me)
        return downsampledCloud

    cpdef restrictViewport(PCLHelper self, tuple viewport):
        """Restrict viewport to given coordinate ranges

        :Parameters:
            viewport: tuple
                3-tuple of 2-tuples defining x-, y- and z-ranges
        """
        pcl_helper.restrictViewport(self.me, viewport[0][0], viewport[0][1], viewport[1][0], viewport[1][1], viewport[2][0], viewport[2][1])

    cpdef tuple detectGroundPlane(PCLHelper self, FLOAT_t maxPlaneAngle, FLOAT_t distanceThreshold):
        """Detect ground plane, return ground plane and correspondent transformation

        Ground plance is returned as a NumPy array of 4 elements containing a 3D normal vector pointing to positive y-direction
        and signed distance of origin to ground plane (Hesse normal form).

        Transformation is returned as NumPy matrix of shape (4, 4). The point under the camera with respect to ground
        plane normal will become the point (0, 0, 0).

        :Parameters:
            maxPlaneAngle: FLOAT_dtype
                Maximal angle between xz plane and detected ground plane
            distanceThreshold: FLOAT_dtype
                Maximal distance to the plane for a point to be considered an inlier

        :Returns: ground plane coefficients and correspondent transformation
        :Returntype: tuple(np.ndarray[FLOAT_dtype], np.matrix[FLOAT_dtype, ndim=2])
        """
        cdef:
            cnp.ndarray[FLOAT_t] coefficients, transformation
        coefficients = np.empty(4, dtype = FLOAT_dtype)
        transformation = np.empty(16, dtype = FLOAT_dtype)
        pcl_helper.detectGroundPlane(self.me, maxPlaneAngle, distanceThreshold, &coefficients[0], &transformation[0])
        # convert to matrix
        mTransformation = np.matrix(transformation, copy=False).reshape(4, 4)
        return coefficients, mTransformation

    cpdef removeGroundPlane(PCLHelper self, FLOAT_t distanceThreshold, cnp.ndarray[FLOAT_t] coefficients):
        """Remove ground plane

        :Parameters:
            distanceThreshold: FLOAT_dtype
                Maximal distance to the plane for a point to be considered an inlier
            coefficients: np.ndarray[FLOAT_dtype]
                Array of shape (4, 1) with ground plane information as returned by :meth:`detectGroundPlane()`
        """
        pcl_helper.removeGroundPlane(self.me, distanceThreshold, &coefficients[0])

    cpdef transform(PCLHelper self, cnp.ndarray[FLOAT_t, ndim=2] transformation):
        """Rotate and translate point cloud according to given affine transformation

        :Parameters:
            transformation: np.ndarray[FLOAT_dtype, ndim=2]
                NumPy array of shape (4, 4) with affine transformation
        """
        cdef:
            cnp.ndarray[FLOAT_t, ndim=2] cTransformation
        cTransformation = np.ascontiguousarray(transformation)
        pcl_helper.transform(self.me, &cTransformation[0, 0])

    cpdef list getConnectedComponents(PCLHelper self, FLOAT_t distanceThreshold):
        """Get PCLHelper list with segmented components

        The cloud needs to be organized.

        :Parameters:
            distanceThreshold: FLOAT_dtype
                Threshold to use for segmentation

        :Returns: List with PCLHelpers containing the segments
        :Returntype: list(PCLHelper)
        """
        cdef:
            size_t i
            size_t nComponents              # number of components
            void *pLabelsIndices = <void*>0 # pointer to vector<pcl::PointIndices>, initialize to avoid compiler warning
            list components                 # Python list with PCLHelpers containing components, will be returned
            PCLHelper py_helper             # current PCLHelper object
            void **c_helpers = <void**>0    # pointer to array of CPCLHelper pointers, initialize to avoid compiler warning
        assert self.isOrganized(), "Cloud needs to be organized in order to extract connected components."

        # initialize list which will be returned later
        components = []

        # get components and their number
        nComponents = pcl_helper.getConnectedComponents(self.me, distanceThreshold, &pLabelsIndices)

        if nComponents > 0:
            # reserve space for CPCLHelper pointer array
            c_helpers = <void**>malloc(nComponents * sizeof(void*))

            # create PCLHelper objects
            for i in range(nComponents):
                # create new PCLHelper object
                py_helper = PCLHelper()
                # add it to list
                components.append(py_helper)
                # add its CPCLHelper pointer to c_helpers array
                c_helpers[i] = <void*>(py_helper.me)

            # extract components and free memory reserved by pLabelsIndices
            pcl_helper.extractConnectedComponents(self.me, pLabelsIndices, <pcl_helper.CPCLHelperPtr*>c_helpers)

            # free CPCLHelper pointer array
            free(c_helpers)
        else:
            # call even for 0 components to free memory reserved by pLabelsIndices
            pcl_helper.extractConnectedComponents(self.me, pLabelsIndices, <pcl_helper.CPCLHelperPtr*>0)

        return components

    cpdef PCLHelper extractOrientedBoxes(PCLHelper self, list boxes3D, FLOAT_t margin = 0, py_bool invert = False):
        """Get new PCLHelper with points inside or outside the given oriented boxes

        The function returns all extracted points as one cloud and does not apply any transformations.

        The resulting cloud is not organized.

        :Parameters:
            boxes3D: list of dicts
                List of dictionaries with 3D box information, see :class:`~pydriver.datasets.base.BaseReader` for box format description
            margin: FLOAT_dtype, optional
                Safety margin to be added at each side, 0 by default
            invert: bool, optional
                Get points inside the boxes if *False* or outside if *True*, *False* by default
        """
        cdef:
            FLOAT_t x, y, z, width, height, length, rotation_y
            cnp.ndarray prisms
            cnp.ndarray[FLOAT_t, ndim=2] hull
            int i
        prisms = np.empty(len(boxes3D), dtype = TetragonalPrism_dtype)
        for iBox in range(len(boxes3D)):
            box3D = boxes3D[iBox]
            x = box3D['location']['x']
            y = box3D['location']['y']
            z = box3D['location']['z']
            width = box3D['dimensions']['width']
            height = box3D['dimensions']['height']
            length = box3D['dimensions']['length']
            rotation_y = box3D['rotation_y']

            hull = prisms[iBox]['xyz']

            # non-oriented rectangular planar hull coordinates with center at (0, 0, 0) facing along the X-axis
            hull[:, 1] = 0  # set y coordinate to 0 for all points

            hull[0, 0] = <FLOAT_t>(+0.5*length + margin)
            hull[0, 2] = <FLOAT_t>(-0.5*width - margin)
            hull[1, 0] = <FLOAT_t>(-0.5*length - margin)
            hull[1, 2] = <FLOAT_t>(-0.5*width - margin)
            hull[2, 0] = <FLOAT_t>(-0.5*length - margin)
            hull[2, 2] = <FLOAT_t>(+0.5*width + margin)
            hull[3, 0] = <FLOAT_t>(+0.5*length + margin)
            hull[3, 2] = <FLOAT_t>(+0.5*width + margin)

            # set oriantation, i.e. rotate hull vertices
            for i in range(hull.shape[0]):
                rotate2DXZ(&hull[i, 0], &hull[i, 2], rotation_y)

            # move hull to origin
            hull[:, 0] += x
            hull[:, 1] += y
            hull[:, 2] += z

            # set min/max height
            prisms[iBox]['height_min'] = <FLOAT_t>(-0.5*height - margin)
            prisms[iBox]['height_max'] = <FLOAT_t>(+0.5*height + margin)

        return self._extractTetragonalPrisms(prisms, invert = invert)

    cpdef PCLHelper _extractTetragonalPrisms(PCLHelper self, cnp.ndarray prisms, py_bool invert = False):
        """Filter point cloud by prisms each defined by planar hulls with 4 points and height limits

        Lower and upper height limits are defined relative to hull surface along the hull surface normal.

        The resulting cloud is not organized.

        :Parameters:
            prisms: np.ndarray[:const:`pydriver.common.constants.TetragonalPrism_dtype`]
                Array with prisms
            invert: bool, optional
                Get points inside the prisms if *False* or outside if *True*, *False* by default

        :Returns: filtered cloud
        :Returntype: PCLHelper
        """
        cdef:
            PCLHelper result = PCLHelper()
            c_bool c_invert = False
        prisms = np.ascontiguousarray(prisms)

        if invert:
            c_invert = True     # avoid compiler warning which would be produced by "c_invert = invert"

        # extract points into the new PCLHelper
        pcl_helper.extractTetragonalPrisms(self.me, prisms.shape[0], <TetragonalPrism*>prisms.data, result.me, c_invert)
        return result

    cpdef tuple getNormalsOfCloud(PCLHelper self, PCLHelper pointCloud, FLOAT_t radius):
        """Get surface normals at positions given by *pointCloud*

        Normals are computed based on points of this PCLHelper within the given radius. The resulting normals array has the shape *(nPoints, 3)*.

        :Returns: tuple containing normals and a mask specifying which normals are valid
        :Returntype: tuple(np.ndarray[FLOAT_dtype, ndim=2] normals, np.ndarray[bool] validityMask)
        """
        cdef:
            cnp.ndarray validityMask
            cnp.ndarray[FLOAT_t, ndim=2] normals
            size_t nPoints
        nPoints = pointCloud.getCloudSize()[0]

        normals = np.empty(shape = (nPoints, 3), dtype = FLOAT_dtype)
        if nPoints > 0:
            pcl_helper.getNormalsOfCloud(self.me, pointCloud.me, radius, &normals[0, 0])

        # check validity
        validityMask = np.isnan(np.sum(normals, axis = 1))  # True for invalid values

        return normals, validityMask

    cpdef PCLHelper getHarrisPoints(PCLHelper self, FLOAT_t radius, py_bool refine = True, FLOAT_t threshold = 0, int method = 1):
        """Get PCLHelper containing the cloud with Harris keypoints

        See `pcl::HarrisKeypoint3D <http://docs.pointclouds.org/1.7.1/classpcl_1_1_harris_keypoint3_d.html>`_ and
        `available methods <http://docs.pointclouds.org/1.7.1/classpcl_1_1_harris_keypoint3_d.html#a372b4e5dd47b17cb70dd3a8f6e9e4187>`_
        for further information.

        :Parameters:
            radius: FLOAT_dtype
                Radius for normal estimation and non maxima suppression
            refine: bool, optional
                Flag whether to refine keypoints, *True* by default
            threshold: FLOAT_dtype, optional
                Threshold value for detecting corners, *0* by default
            method: int, optional
                Method to use (1-5), *1* by default
        """
        cdef:
            PCLHelper harrisCloud = PCLHelper()
        pcl_helper.getHarrisPoints(self.me, radius, harrisCloud.me, refine, threshold, method)
        return harrisCloud

    cpdef PCLHelper getISSPoints(PCLHelper self, FLOAT_t salientRadius, FLOAT_t nonMaxRadius, int minNeighbors = 5, FLOAT_t threshold21 = 0.975, FLOAT_t threshold32 = 0.975, FLOAT_t angleThreshold = PI / 2.0, FLOAT_t normalRadius = 0, FLOAT_t borderRadius = 0):
        """Get PCLHelper containing the cloud with ISS keypoints

        See `pcl::ISSKeypoint3D <http://docs.pointclouds.org/1.7.1/classpcl_1_1_i_s_s_keypoint3_d.html>`_ for further information.
        """
        cdef:
            PCLHelper issCloud = PCLHelper()
        pcl_helper.getISSPoints(self.me, salientRadius, nonMaxRadius, NUMBER_OF_THREADS, issCloud.me, minNeighbors, threshold21, threshold32, angleThreshold, normalRadius, borderRadius)
        return issCloud

    cpdef tuple computeSHOT(PCLHelper self, FLOAT_t radius, PCLHelper keypoints):
        """Compute the SHOT feature (descriptor and 3D local reference frame) with given radius for specified keypoint cloud

        :Returns: tuple containing SHOT features and a mask specifying which features are valid
        :Returntype: tuple(np.ndarray[:const:`SHOTFeature_dtype`] SHOTFeatures, np.ndarray[bool] validityMask)
        """
        cdef:
            cnp.ndarray shotFeatures, validityMask
            size_t nKeypoints, nFeatures
        # get number of keypoints
        nKeypoints = keypoints.getCloudSize()[0]
        # reserve space for features
        shotFeatures = np.empty(nKeypoints, dtype = SHOTFeature_dtype)
        if nKeypoints > 0:
            # compute SHOT features if any keypoints are given
            nFeatures = pcl_helper.computeSHOT(self.me, keypoints.me, radius, <SHOTFeature*>shotFeatures.data)
            # sanity check
            if nFeatures <> nKeypoints:
                raise ValueError("Features and Keypoints diverge! %s keypoints, %s features" % (nKeypoints, nFeatures))
        # check feature validity (works for 0 keypoints, too)
        validityMask = np.isnan(np.sum(shotFeatures['rf'], axis = (1, 2)))  # True for invalid values

        return shotFeatures, validityMask

    cpdef tuple computeSHOTColor(PCLHelper self, FLOAT_t radius, PCLHelper keypoints):
        """Compute the SHOT+Color feature (descriptor and 3D local reference frame) with given radius for specified keypoint cloud

        :Returns: tuple containing SHOT+Color features and a mask specifying which features are valid
        :Returntype: tuple(np.ndarray[:const:`SHOTColorFeature_dtype`] SHOTColorFeatures, np.ndarray[bool] validityMask)
        """
        cdef:
            cnp.ndarray shotFeatures, validityMask
            size_t nKeypoints, nFeatures
        # get number of keypoints
        nKeypoints = keypoints.getCloudSize()[0]
        # reserve space for features
        shotFeatures = np.empty(nKeypoints, dtype = SHOTColorFeature_dtype)
        if nKeypoints > 0:
            # compute SHOT features if any keypoints are given
            nFeatures = pcl_helper.computeSHOTColor(self.me, keypoints.me, radius, <SHOTColorFeature*>shotFeatures.data)
            # sanity check
            if nFeatures <> nKeypoints:
                raise ValueError("Features and Keypoints diverge! %s keypoints, %s features" % (nKeypoints, nFeatures))
        # check feature validity (works for 0 keypoints, too)
        validityMask = np.isnan(np.sum(shotFeatures['rf'], axis = (1, 2)))  # True for invalid values

        return shotFeatures, validityMask
