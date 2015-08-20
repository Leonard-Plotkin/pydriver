# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

# ATTENTION:
# enum declarations are true compile-time constants, but float constants will be interpreted as integers by Cython without explicit conversion,
# so use enum declarations only for integer constants or be otherwise careful
# typed declarations (as "double FOO") however are only variables and can not be used in "int bar[FOO]", for example

cdef extern from "cconstants.h":
    # PI constant
    double C_PI

    # length of string containing object category
    enum: C_CATEGORY_LENGTH
