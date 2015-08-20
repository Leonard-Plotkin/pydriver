/*
Changes in this file require a full recompilation of Python AND C/C++ code. This file is also used by pcl_helper.

Only constants which must be available at compile-time or without GIL have to be defined here. These constants
are defined in the "C"-scope, they are not visible to pure (uncompiled) Python code. Every constant should have
the "C_" prefix to mark them as "C"-scope constants. Use constants.pyx to access these constants from Python.
*/

#ifndef COMMON_CCONSTANTS_H
#define COMMON_CCONSTANTS_H

#define C_PI 3.14159265358979323846

// length of string containing object category
#define C_CATEGORY_LENGTH 32

#endif /* COMMON_CCONSTANTS_H */
