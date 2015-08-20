# This file can be textually included so this code will be linked statically.
# All functions defined here can only be called from C/C++ code, use functions.pyx to use them from Python.

cimport cython
cimport libc.math

# use absolute import since we don't know where it will be included
from pydriver.common.cconstants cimport C_PI
from pydriver.common.structs cimport FLOAT_t


# --- rotation ---

cdef inline void rotate2DXY(FLOAT_t *x, FLOAT_t *y, FLOAT_t angle) nogil:
    """Rotate a point around Z-axis by given angle"""
    cdef:
        FLOAT_t angle_sin = <FLOAT_t>libc.math.sin(angle)
        FLOAT_t angle_cos = <FLOAT_t>libc.math.cos(angle)
    rotate2DXYSinCos(x, y, angle_sin, angle_cos)

cdef inline void rotate2DXZ(FLOAT_t *x, FLOAT_t *z, FLOAT_t angle) nogil:
    """Rotate a point around Y-axis by given angle"""
    rotate2DXY(x, z, -angle)

cdef inline void rotate2DXYSinCos(FLOAT_t *x, FLOAT_t *y, FLOAT_t angle_sin, FLOAT_t angle_cos) nogil:
    """Rotate a point around Z-axis with precomputed sin() and cos()

    This function is much faster than rotate2DXY() if you can precompute angle_sin and angle_cos for many calls.
    """
    cdef:
        FLOAT_t x_old = x[0]
        FLOAT_t y_old = y[0]
    x[0] = x_old * angle_cos - y_old * angle_sin
    y[0] = x_old * angle_sin + y_old * angle_cos

cdef inline void rotate2DXZSinCos(FLOAT_t *x, FLOAT_t *z, FLOAT_t angle_sin, FLOAT_t angle_cos) nogil:
    """Rotate a point around Y-axis with precomputed sin() and cos()

    This function is much faster than rotate2DXZ() if you can precompute angle_sin and angle_cos for many calls.
    """
    rotate2DXYSinCos(x, z, -angle_sin, angle_cos)


# --- angle normalization ---

@cython.cdivision(True)
cdef inline FLOAT_t normalizeAngle(FLOAT_t angle) nogil:
    """Normalize angle to be in [-Pi, Pi)"""
    cdef FLOAT_t result = <FLOAT_t>(angle % <FLOAT_t>(2 * C_PI))
    if (result < -C_PI):
        # this check is only due to rounding errors
        # tests show that the result can be infinitesimally lesser than -C_PI on some CPUs otherwise
        result += <FLOAT_t>(2*C_PI)
    elif (result >= C_PI):
        result -= <FLOAT_t>(2*C_PI)
    return result
