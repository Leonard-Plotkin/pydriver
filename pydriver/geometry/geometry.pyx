# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

import copy, warnings

cimport libc.math
from libc.stdlib cimport malloc, free

cimport cython
from cython.parallel cimport prange

cimport numpy as cnp
import numpy as np

from ..common.structs cimport FLOAT_t
from ..common.constants import FLOAT_dtype
from ..common import functions, opencl_config


# check OpenCV availability
try:
    import cv2
except ImportError:
    USE_OPENCV = False
else:
    USE_OPENCV = True


# --- coordinate transformations ---

cpdef cnp.ndarray[FLOAT_t, ndim=2] homogenuous2cartesian(cnp.ndarray[FLOAT_t, ndim=2] homCoords, bint inplace = False):
    """Transform homogenuous coordinates to cartesian coordinates

    If *inplace* is *True* the data will not be copied and the input array will be modified.
    """
    cdef:
        cnp.ndarray[FLOAT_t, ndim=2] carCoords
        int i
    if inplace:
        carCoords = homCoords[:, :-1]
    else:
        carCoords = homCoords[:, :-1].copy()
    for i in range(carCoords.shape[1]):
        carCoords[:, i] /= homCoords[:, -1]
    return carCoords

cpdef cnp.ndarray[FLOAT_t, ndim=2] cartesian2homogenuous(cnp.ndarray[FLOAT_t, ndim=2] carCoords):
    """Transform cartesian coordinates to homogenuous coordinates"""
    cdef:
        cnp.ndarray[FLOAT_t, ndim=2] homCoords
    homCoords = np.ones((carCoords.shape[0], carCoords.shape[1]+1), dtype = FLOAT_dtype)
    homCoords[:, :-1] = carCoords
    return homCoords

cpdef cnp.ndarray[FLOAT_t, ndim=2] affineTransform(cnp.ndarray[FLOAT_t, ndim=2] carCoords, cnp.ndarray[FLOAT_t, ndim=2] transformation):
    """Apply affine transformation to cartesian coordinates

    :Parameters:
        carCoords: np.ndarray[FLOAT_dtype, ndim=2]
            Array with cartesian coordinates
        transformation: np.ndarray[FLOAT_dtype, ndim=2]
            Transformation matrix

    :Returns: Array with transformed coordinates
    :Returntype: np.ndarray[FLOAT_dtype, ndim=2]
    """
    return homogenuous2cartesian((_asMatrix(transformation) * cartesian2homogenuous(carCoords).T).T.A, inplace = True)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef cnp.ndarray[FLOAT_t, ndim=2] image2space(cnp.ndarray disparityMap, cnp.ndarray[FLOAT_t, ndim=2] reprojection):
    """Reproject a full image to 3D space

    Function tries to use following implementations in this order:

    1. own OpenCL implementation (uses the best available device such as GPU or CPU)
    2. OpenCV
    3. own CPU implementation

    :Parameters:
        disparityMap: np.ndarray[ndim=2]
            Disparity map of any float type
        reprojection: np.ndarray[FLOAT_dtype, ndim=2]
            Matrix used for reprojection

    :Returns: Array with reprojected x, y and z coordinates
    :Returntype: np.ndarray[FLOAT_dtype, ndim=2]
    """
    cdef:
        # resulting coordinates
        cnp.ndarray[FLOAT_t, ndim=3] spaceCoords
        # views for faster access
        FLOAT_t[:,:,:] spaceCoords_view
        FLOAT_t[:,:] disparityMap_view

        int width, height, x, y
        FLOAT_t fx, fy, fz, fw
        FLOAT_t q03, q13, q32, q33
    if opencl_config.USE_OPENCL:
        # use OpenCL implementation (CPU/GPU)
        # OpenCL only accepts np.float32
        return _image2space_cl(disparityMap.astype(dtype = np.float32, copy = False), reprojection).astype(FLOAT_dtype, copy = False)
    elif USE_OPENCV:
        warnings.warn(RuntimeWarning("OpenCL is not available, falling back to OpenCV implementation."))
        # OpenCV only accepts np.float32
        return cv2.reprojectImageTo3D(disparityMap.astype(dtype = np.float32, copy = False), reprojection).astype(FLOAT_dtype, copy = False)

    # CPU implementation
    warnings.warn(RuntimeWarning("OpenCL and OpenCV are not available, falling back to CPU implementation."))

    # initialization
    height = disparityMap.shape[0]
    width = disparityMap.shape[1]
    spaceCoords = np.empty((height, width, 4), dtype = FLOAT_dtype)
    spaceCoords_view = spaceCoords
    disparityMap_view = disparityMap.astype(dtype = FLOAT_dtype, copy = False)

    # pick matrix values which are not zero
    q03 = reprojection[0, 3]
    q13 = reprojection[1, 3]
    q32 = reprojection[3, 2]
    q33 = reprojection[3, 3]

    # z is always a constant (before transformation to cartesian)
    fz = reprojection[2, 3]

    # multithreading
    for y in prange(height, schedule = 'static', nogil = True):
        for x in range(width):
            # perform matrix multiplication on homogenuous coordinates
            fx = <FLOAT_t>x + q03
            fy = <FLOAT_t>y + q13
            fw = <FLOAT_t>(1 / (disparityMap_view[y, x]*q32 + q33))     # 4th coordinate, perform 1/x here to avoid repeated divisions afterwards

            # set 3D cartesian coordinates
            spaceCoords_view[y,x,0] = fx * fw
            spaceCoords_view[y,x,1] = fy * fw
            spaceCoords_view[y,x,2] = fz * fw
    spaceCoords = spaceCoords[:,:,:3]
    return spaceCoords


# --- 3D box transformations and other operations ---

cpdef dict transform3DBox(dict box3D, cnp.ndarray[FLOAT_t, ndim=2] transformation):
    """Get transformed 3D box using the given affine transformation

    See :class:`~pydriver.datasets.base.BaseReader` for box format description.
    """
    cdef:
        cnp.ndarray[FLOAT_t, ndim=2] coords
    # reserve space for two 3D vectors
    coords = np.empty((3, 3), dtype = FLOAT_dtype)
    # object center
    coords[0, 0] = box3D['location']['x']
    coords[0, 1] = box3D['location']['y']
    coords[0, 2] = box3D['location']['z']
    # orientation vector (note: right-handed 3D coordinate system)
    coords[1, 0] = libc.math.cos(-box3D['rotation_y'])
    coords[1, 1] = 0
    coords[1, 2] = libc.math.sin(-box3D['rotation_y'])

    # translate orientation vector to object center
    coords[1] += coords[0]
    # transform both points
    coords = affineTransform(coords, transformation)
    # translate orientation vector back to origin
    coords[1] -= coords[0]

    # copy original box
    result = copy.deepcopy(box3D)
    # transformed location
    result['location']['x'] = coords[0, 0]
    result['location']['y'] = coords[0, 1]
    result['location']['z'] = coords[0, 2]
    # transformed rotation_y
    result['rotation_y'] = -libc.math.atan2(coords[1, 2], coords[1, 0])

    return result


cpdef dict project3DBox(dict box3D, cnp.ndarray[FLOAT_t, ndim=2] projection):
    """Get 2D bounding rectangle for a 3D box given the projection matrix

    See :class:`~pydriver.datasets.base.BaseReader` for box format description.

    The resulting dictionary contains the keys *'left'*, *'top'*, *'right'* and *'bottom'*.
    """
    cdef:
        cnp.ndarray[FLOAT_t, ndim=2] vertices3D, vertices2D
        dict box2D
    vertices3D = get3DBoxVertices(box3D)
    vertices2D = affineTransform(vertices3D, projection)
    box2D = {
             'left': vertices2D[:, 0].min(),
             'top': vertices2D[:, 1].min(),
             'right': vertices2D[:, 0].max(),
             'bottom': vertices2D[:, 1].max(),
             }
    return box2D


cpdef cnp.ndarray[FLOAT_t, ndim=2] getNormalizationTransformation(dict box3D):
    """Get NumPy matrix with affine transformation required to normalize a 3D box

    Normalization means that *rotation_y* of the box will be compensated to 0 and box center will be at (0, 0, 0).

    :Parameters:
        box3D: dict
            Dictionary with 3D box information, see :class:`~pydriver.datasets.base.BaseReader` for box format description

    :Returns: transformation matrix
    :Returntype: np.matrix
    """
    cdef:
        cnp.ndarray[FLOAT_t, ndim=2] translation, rotation
        FLOAT_t rsin, rcos
    translation = np.identity(4, dtype = FLOAT_dtype)
    rotation = np.identity(4, dtype = FLOAT_dtype)

    translation[0, 3] = -box3D['location']['x']
    translation[1, 3] = -box3D['location']['y']
    translation[2, 3] = -box3D['location']['z']

    rsin = libc.math.sin(-box3D['rotation_y'])
    rcos = libc.math.cos(-box3D['rotation_y'])
    rotation[0, 0] = rcos
    rotation[0, 2] = rsin
    rotation[2, 0] = -rsin
    rotation[2, 2] = rcos

    return _asMatrix(rotation) * _asMatrix(translation)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef list extractNormalizedOrientedBoxes(FLOAT_t[:, :] xyz, cnp.uint8_t[:, :] rgb, list boxes):
    """Extract normalized oriented boxes at specified location from data

    Normalization means that *rotation_y* of the box will be compensated to 0 and box center will be at (0, 0, 0).

    Function is optimized for speed and parallelized to extract boxes simultaneously. *NaN* values will be removed. For maximum speed, remove all *NaN* values before
    calling this function.

    :Parameters:
        xyz: FLOAT_dtype[:, :]
            Input array with xyz data
        rgb: np.uint8[:, :]
            Input array with rgb data
        boxes: list of dicts
            List with dictionaries describing the 3D boxes, see :class:`~pydriver.datasets.base.BaseReader` for box format description

    :Returns: List with tuples(xyz, rgb)
    :Returntype: list(tuple(np.ndarray[FLOAT_dtype, ndim=2], np.ndarray[np.uint8, ndim=2]))
    """
    cdef:
        int i, nBoxes
        FLOAT_t[:, :] boxes_array       # array with boxes' parameters to use without GIL
        cnp.int_t[:] nExtractedPoints   # array with numbers of extracted points
        FLOAT_t **xyz_results_ptr       # pointer to array with pointers to xyz results
        cnp.uint8_t **rgb_results_ptr   # pointer to array with pointers to rgb results

        dict box3D
        list results    # list of tuples with results
    # basic initialization
    nBoxes = len(boxes)
    results = []

    if xyz.shape[0] == 0:
        # no points in the cloud, take the shortcut
        for i in range(nBoxes):
            results.append((np.empty((0, 3), dtype = FLOAT_dtype), np.empty((0, 3), dtype = np.uint8)))
        return results

    # reserve space
    boxes_array = np.empty((nBoxes, 7), dtype = FLOAT_dtype)
    nExtractedPoints = np.empty(nBoxes, dtype = np.int)
    xyz_results_ptr = <FLOAT_t**>malloc(nBoxes * sizeof(FLOAT_t*))
    rgb_results_ptr = <cnp.uint8_t**>malloc(nBoxes * sizeof(cnp.uint8_t*))

    # copy boxes' parameters to array
    for i in range(nBoxes):
        box3D = boxes[i]
        boxes_array[i, 0] = box3D['location']['x']
        boxes_array[i, 1] = box3D['location']['y']
        boxes_array[i, 2] = box3D['location']['z']
        boxes_array[i, 3] = box3D['dimensions']['width']
        boxes_array[i, 4] = box3D['dimensions']['height']
        boxes_array[i, 5] = box3D['dimensions']['length']
        boxes_array[i, 6] = box3D['rotation_y']

    # multi-threaded extraction of boxes
    for i in prange(nBoxes, schedule = 'static', nogil = True):
        nExtractedPoints[i] = _extractNormalizedOrientedBox(xyz, rgb,
                                                            boxes_array[i, 0], boxes_array[i, 1], boxes_array[i, 2], # location
                                                            boxes_array[i, 3], boxes_array[i, 4], boxes_array[i, 5], # dimensions
                                                            boxes_array[i, 6],                                       # rotation_y
                                                            &xyz_results_ptr[i], &rgb_results_ptr[i],                # pointers to pointers to results, will be set by function
                                                            )

    # extract results
    for i in range(nBoxes):
        if nExtractedPoints[i] > 0:
            # construct NumPy arrays from pointers returned by _extractNormalizedOrientedBox()
            xyz_result = np.asarray(<FLOAT_t[:nExtractedPoints[i], :3]>xyz_results_ptr[i])
            rgb_result = np.asarray(<cnp.uint8_t[:nExtractedPoints[i], :3]>rgb_results_ptr[i])
            # set NumPy ownership of data so it will be freed when the NumPy array gets deallocated
            PyArray_ENABLEFLAGS(xyz_result, cnp.NPY_OWNDATA)
            PyArray_ENABLEFLAGS(rgb_result, cnp.NPY_OWNDATA)
            # append tuple with resulting arrays to results
            results.append((xyz_result, rgb_result))
        else:
            # no points extracted, we can't construct empty cython views, so it's an edge case
            # the memory was already freed by _extractNormalizedOrientedBox()
            results.append((np.empty((0, 3), dtype = FLOAT_dtype), np.empty((0, 3), dtype = np.uint8)))

    # free our arrays with pointers to resulting data
    free(xyz_results_ptr)
    free(rgb_results_ptr)

    return results


cpdef cnp.ndarray[FLOAT_t, ndim=2] get3DBoxVertices(dict box3D):
    """Get array with 8 vertex coordinates of the given 3D box

    Blocks of 4 vertices have the same Y and blocks of 2 have the same X coordinate. Coordinates of blocks are increasing.
    """
    cdef:
        cnp.ndarray[FLOAT_t, ndim=2] vertices
        FLOAT_t x, y, z, width, height, length, rotation_y
        int i
        FLOAT_t cx, cy, cz, rx, ry, rz
    vertices = np.empty((8, 3), dtype = FLOAT_dtype)
    x, y, z = box3D['location']['x'], box3D['location']['y'], box3D['location']['z']
    width, height, length = box3D['dimensions']['width'], box3D['dimensions']['height'], box3D['dimensions']['length']
    rotation_y = box3D['rotation_y']

    i = 0
    for cy in (-height*0.5, +height*0.5):
        for cx in (-length*0.5, +length*0.5):
            for cz in (-width*0.5, +width*0.5):
                # rotate current coordinates
                rx, rz = functions.pyRotate2DXZ(cx, cz, rotation_y)
                ry = cy     # y is not affected by rotation
                # translate rotated coordinates
                vertices[i][0] = rx + x
                vertices[i][1] = ry + y
                vertices[i][2] = rz + z
                i += 1

    return vertices


# --- internal functions ---

cdef cnp.ndarray[FLOAT_t, ndim=2] _asMatrix(cnp.ndarray[FLOAT_t, ndim=2] npArray):
    """Convert NumPy array to matrix"""
    return np.matrix(npArray, copy = False)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int _extractNormalizedOrientedBox(FLOAT_t[:, :] xyz, cnp.uint8_t[:, :] rgb, FLOAT_t x, FLOAT_t y, FLOAT_t z, FLOAT_t width, FLOAT_t height, FLOAT_t depth, FLOAT_t rotation_y, FLOAT_t **xyz_result_ptr, cnp.uint8_t **rgb_result_ptr) nogil:
    """Extract normalized oriented box at specified location from data

    IT LIES IN THE CALLER'S RESPONSIBILITY TO FREE MEMORY AT RESULTING **xyz_result_ptr and **rgb_result_ptr IF FUNCTION RETURNS RESULT > 0.

    Normalization means that the rotation_y will be compensated to 0 and box center will be at (0, 0, 0).

    Function should only be called by extractNormalizedOrientedBoxes. NaN values will be removed.

    :Parameters:
        xyz: FLOAT_t[:, :]
            Input array with xyz data
        rgb: np.uint8[:, :]
            Input array with rgb data
        x: FLOAT_t
            X coordinate of (the center of) the box
        y: FLOAT_t
            Y coordinate of (the center of) the box
        z: FLOAT_t
            Z coordinate of (the center of) the box
        width: FLOAT_t
            Width of the box
        height: FLOAT_t
            Height of the box
        depth: FLOAT_t
            Depth of the box
        rotation_y: FLOAT_t
            Rotation of the box around y-axis
        xyz_result_ptr: FLOAT_t**
            Pointer to pointer to the resulting xyz data (output)
        rgb_result_ptr: np.uint8_t**
            Pointer to pointer to the resulting rgb data (output)

    :Returns: Number of extracted points
    :Returntype: int
    """
    cdef:
        # halved dimensions
        FLOAT_t hwidth = <FLOAT_t>0.5*width, hheight = <FLOAT_t>0.5*height, hdepth = <FLOAT_t>0.5*depth

        FLOAT_t *xyz_temp           # pointer to temporary xyz array
        cnp.uint8_t *rgb_temp       # pointer to temporary rgb array
        FLOAT_t *xyz_result         # pointer to resulting xyz array
        cnp.uint8_t *rgb_result     # pointer to resulting rgb array

        FLOAT_t rsin, rcos          # sin and cos of rotation_y
        FLOAT_t xx, xz              # rotation transformation to obtain normalized x
        FLOAT_t zx, zz              # rotation transformation to obtain normalized z
        FLOAT_t cx, cz              # translated x and z of current point
        FLOAT_t nx, ny, nz          # normalized x, y and z of current point

        int i, nPoints, nExtractedPoints
    # number of points to check
    nPoints = xyz.shape[0]

    # our rotation angle must be -rotation_y to restore normalized orientation
    # but since rotation around y-axis in a right-handed 3D coordinate system is clockwise we use -(-rotation_y) = rotation_y
    rsin = libc.math.sin(rotation_y)
    rcos = libc.math.cos(rotation_y)
    # rotation matrix
    xx = rcos
    xz = -rsin
    zx = rsin
    zz = rcos

    # temporarily reserve maximal needed memory for output
    xyz_temp = <FLOAT_t*>malloc(sizeof(FLOAT_t) * nPoints * 3)
    rgb_temp = <cnp.uint8_t*>malloc(sizeof(cnp.uint8_t) * nPoints * 3)
    if xyz_temp == NULL or rgb_temp == NULL:
        with gil:
            raise MemoryError("Could not reserve %d bytes." % ((sizeof(FLOAT_t)+sizeof(cnp.uint8_t)) * nPoints * 3))

    # reset extracted points to 0
    nExtractedPoints = 0
    for i in range(nPoints):
        ny = xyz[i, 1] - y      # translate y so center of object is at y=0 (no rotation required for normalization)
        if libc.math.fabs(ny) <= hheight:
            # point lies in y-range

            # translated current x and z
            cx = xyz[i, 0] - x
            cz = xyz[i, 2] - z
            # normalized (translated+rotated) current x and z
            nx = xx*cx + xz*cz
            nz = zx*cx + zz*cz
            if libc.math.fabs(nx) <= hwidth and libc.math.fabs(nz) <= hdepth:
                # point lies in x- and z-ranges
                # assign normalized values to resulting array
                xyz_temp[nExtractedPoints*3+0] = nx
                xyz_temp[nExtractedPoints*3+1] = ny
                xyz_temp[nExtractedPoints*3+2] = nz
                # copy rgb data to resulting array
                rgb_temp[nExtractedPoints*3+0] = rgb[i, 0]
                rgb_temp[nExtractedPoints*3+1] = rgb[i, 1]
                rgb_temp[nExtractedPoints*3+2] = rgb[i, 2]
                nExtractedPoints += 1
    # The following code runs out of memory presumably due to an error in realloc(). See http://stackoverflow.com/questions/9164837/realloc-does-not-correctly-free-memory-in-windows.
    # free parts of the originally reserved memory to fit the number of extracted points
    # if nExtractedPoints == 0 the memory will be freed and NULL will be returned
    #xyz_result = <FLOAT_t*>realloc(xyz_temp, sizeof(FLOAT_t) * nExtractedPoints * 3)
    #rgb_result = <cnp.uint8_t*>realloc(rgb_temp, sizeof(cnp.uint8_t) * nExtractedPoints * 3)

    if nExtractedPoints > 0:
        # reserve memory for results
        xyz_result = <FLOAT_t*>malloc(sizeof(FLOAT_t) * nExtractedPoints * 3)
        rgb_result = <cnp.uint8_t*>malloc(sizeof(cnp.uint8_t) * nExtractedPoints * 3)
        if xyz_result == NULL or rgb_result == NULL:
            with gil:
                raise MemoryError("Could not allocate %d bytes." % ((sizeof(FLOAT_t)+sizeof(cnp.uint8_t)) * nExtractedPoints * 3))
        # copy memory from temporary to resulting arrays
        memcpy(xyz_result, xyz_temp, sizeof(FLOAT_t) * nExtractedPoints * 3)
        memcpy(rgb_result, rgb_temp, sizeof(cnp.uint8_t) * nExtractedPoints * 3)
    else:
        xyz_result = NULL
        rgb_result = NULL

    # free temporary memory
    free(xyz_temp)
    free(rgb_temp)

    # set the pointers given to the function to the resulting arrays (or NULL for empty result)
    xyz_result_ptr[0] = xyz_result
    rgb_result_ptr[0] = rgb_result
    # IT LIES IN THE CALLER'S RESPONSIBILITY TO FREE MEMORY AT RESULTING **xyz_result_ptr and **rgb_result_ptr IF FUNCTION RETURNS RESULT > 0.
    return nExtractedPoints


cdef cnp.ndarray[cnp.float32_t, ndim=2] _image2space_cl(cnp.ndarray[cnp.float32_t, ndim=2] disparityMap, cnp.ndarray[FLOAT_t, ndim=2] reprojection, dict _cache = {}):
    """Reproject a full image to 3D space using OpenCL

    Function uses fastest available device (CPU/GPU) supporting OpenCL. Processed data is float32 to ensure OpenCL compatibility.

    :Parameters:
        disparityMap: np.ndarray[np.float32, ndim=2]
            Disparity map
        reprojection: np.ndarray[FLOAT_dtype, ndim=2]
            Matrix used for reprojection
        _cache: dict, *reserved*
            Internal cache, do not use

    :Returns: Array with reprojected x, y and z coordinates
    :Returntype: np.ndarray[np.float32, ndim=2]
    """
    cdef:
        cnp.ndarray[cnp.float32_t, ndim=3] spaceCoords      # resulting 3D coordinates
        cnp.ndarray k_reprojection                          # reprojection matrix passed to kernel
        int itemsize = sizeof(cnp.float32_t)                # size of float32 in bytes
        int width, height                                   # width, height of the image
    # import here so the module can be used without OpenCL
    import pyopencl as cl
    import pyopencl.characterize as cl_characterize
    import pyopencl.array as cl_array

    # get initialized device
    device = opencl_config.PREFERRED_DEVICE['device']
    ctx = opencl_config.PREFERRED_DEVICE['context']
    queue = opencl_config.PREFERRED_DEVICE['queue']

    if _cache.get('initialized', False):
        kernel = _cache['kernel']
    else:
        # define transformation kernel
        kernelCode = """
        __kernel void transform(__global const float *disparityMap_d, int width, float4 q0, float4 q1, float4 q2, float4 q3, __global float *res_d) {
            int i = get_global_id(0);
            int y = i / width;
            int x = i % width;

            float4 org = (float4)(x, y, disparityMap_d[i], 1);
            float4 result = (float4)(dot(org, q0), dot(org, q1), dot(org, q2), 1) / dot(org, q3);

            res_d[i*3] = result.x;
            res_d[i*3+1] = result.y;
            res_d[i*3+2] = result.z;
        }
        """
        with warnings.catch_warnings():
            # suppress the following warning, the output contains a success message
            warnings.filterwarnings(action = 'ignore',
                                    message = 'Non-empty compiler output encountered. Set the environment variable PYOPENCL_COMPILER_OUTPUT=1 to see more.',
                                    category = cl.CompilerWarning,
                                    module = 'pyopencl',
                                    )
            # build transformation kernel
            kernel = cl.Program(ctx, kernelCode).build()
        _cache['kernel'] = kernel
        _cache['initialized'] = True
    # end of initialization

    # get width and height
    height = disparityMap.shape[0]
    width = disparityMap.shape[1]

    # initialize disparity buffer used by device
    disparityMap_d = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.HOST_NO_ACCESS | cl.mem_flags.USE_HOST_PTR, height*width*itemsize, hostbuf = disparityMap)

    # reserve result buffer on device
    spaceCoords_d = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, height*width*3*itemsize)

    # transformation matrix rows as float4 vectors for the kernel
    k_reprojection = np.array([
        tuple(np.array(reprojection)[0]),
        tuple(np.array(reprojection)[1]),
        tuple(np.array(reprojection)[2]),
        tuple(np.array(reprojection)[3]),
    ], dtype = cl_array.vec.float4)

    # execute kernel
    kernel.transform(queue, (height*width,), None,
                     disparityMap_d,
                     np.int32(width),
                     k_reprojection[0],
                     k_reprojection[1],
                     k_reprojection[2],
                     k_reprojection[3],
                     spaceCoords_d,
                     )

    # map device memory for fast reading by host
    spaceCoords, event_mapping = cl.enqueue_map_buffer(queue, spaceCoords_d, flags = cl.mem_flags.READ_ONLY, offset = 0, shape = (height, width, 3), dtype = np.float32, is_blocking = False)
    # wait for mapping to finish
    event_mapping.wait()
    return spaceCoords
