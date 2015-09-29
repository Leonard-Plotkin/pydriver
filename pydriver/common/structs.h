/*
Changes in this file require a recompilation of Python AND C/C++ code. This file is also used by pcl_helper.
*/

#ifndef COMMON_STRUCTS_H
#define COMMON_STRUCTS_H

#include "cconstants.h"

// type definition for float values (where they are not required to be of some specific float type)
// must match FLOAT_dtype (in constants.pyx)
typedef float FLOAT_t;

// structure for storing detection positions
// must define an identical structure as Position_dtype (in constants.pyx)
typedef struct Position {
    FLOAT_t x;
    FLOAT_t y;
    FLOAT_t z;
    FLOAT_t rotation_y;
} Position;

// structure for storing object detections
// must define an identical structure as Detection_dtype (in constants.pyx)
typedef struct Detection {
    char category[C_CATEGORY_LENGTH];
    Position position;
    FLOAT_t height;
    FLOAT_t width;
    FLOAT_t length;
    FLOAT_t weight;
} Detection;

// structure for tetragonal prisms: 4 points (defining the base) and height boundaries
// must define an identical structure as TetragonalPrism_dtype (in constants.pyx)
typedef struct TetragonalPrism {
	FLOAT_t xyz[4][3];	// 4 points with 3 coordinates each
	FLOAT_t height_min;
	FLOAT_t height_max;
} TetragonalPrism;

// SHOT feature structure (must be structured like pcl::SHOT352 but can have a different type)
// must define an identical structure as SHOTFeature_dtype (in pcl/pcl.pyx)
typedef struct SHOTFeature {
	FLOAT_t descriptor[352];
	FLOAT_t rf[9];
} SHOTFeature;
// SHOTColor feature structure (must be structured like pcl::SHOT1344 but can have a different type)
// must define an identical structure as SHOTColorFeature_dtype (in pcl/pcl.pyx)
typedef struct SHOTColorFeature {
	FLOAT_t descriptor[1344];
	FLOAT_t rf[9];
} SHOTColorFeature;

#endif /* COMMON_STRUCTS_H */
