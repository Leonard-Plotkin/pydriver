// This header file is used by the library and by Cython code and must be compatible with
// both used compilers. It should not depend on any PCL types or other PCL
// declarations so it can be guaranteed that Cython code does not need to be aware
// of PCL or its version. PCL-specific types are allowed as forward declarations though.

// Notes:
// nullptr is not supported until C++11 (e.g. not by VC++ 9.0)

#ifndef PCL_HELPER_EXPORTS_H
#define PCL_HELPER_EXPORTS_H

#ifdef _MSC_VER
	//  Microsoft
	// The following ifdef block is the standard way of creating macros which make exporting
	// from a library simpler. All files within this library are compiled with the PCL_HELPER_EXPORTS
	// symbol defined on the command line. This symbol should not be defined on any project
	// that uses this library. This way any other project whose source files include this file see
	// PCL_HELPER_API functions as being imported from a library, whereas this library sees symbols
	// defined with this macro as being exported.
	#ifdef PCL_HELPER_EXPORTS
		#define PCL_HELPER_API __declspec(dllexport)
	#else
		#define PCL_HELPER_API __declspec(dllimport)
	#endif
#else
	// not Microsoft
	#define PCL_HELPER_API
#endif

#include <cstddef>					// NULL is declared there
#include "pcl_helper_imports.h"


// make clear that the void pointer used in exported functions refers to a PCLHelper instance
typedef void* PCLHelperPtr;

// exported functions
extern "C" {
	// factory functions that create and release instances of PCLHelper
	PCL_HELPER_API PCLHelperPtr	getPCLHelper();
	PCL_HELPER_API void			freePCLHelper(PCLHelperPtr h);

	// implementation functions
	PCL_HELPER_API void			setNormalsParams(PCLHelperPtr h, int k_search, FLOAT_t radius);

	PCL_HELPER_API size_t		getCloudSize(PCLHelperPtr h, size_t *width, size_t *height);
	PCL_HELPER_API void			fromArray(PCLHelperPtr h, size_t width, size_t height, FLOAT_t *pXYZ, char *pRGB);
	PCL_HELPER_API void			toArray(PCLHelperPtr h, FLOAT_t *pXYZ, char *pRGB);
	PCL_HELPER_API void			copyCloud(PCLHelperPtr h, PCLHelperPtr out);
	PCL_HELPER_API void			addCloud(PCLHelperPtr h, PCLHelperPtr addendHelper);
	PCL_HELPER_API void			removeNaN(PCLHelperPtr h);

	PCL_HELPER_API void			setBGColor(PCLHelperPtr h, FLOAT_t r, FLOAT_t g, FLOAT_t b);
	PCL_HELPER_API void			setCameraPosition(PCLHelperPtr h, FLOAT_t camPosX, FLOAT_t camPosY, FLOAT_t camPosZ, FLOAT_t camViewX, FLOAT_t camViewY, FLOAT_t camViewZ, FLOAT_t camUpX, FLOAT_t camUpY, FLOAT_t camUpZ);
	PCL_HELPER_API void			setKeypointsVisualization(PCLHelperPtr h, FLOAT_t size, FLOAT_t normalLength, FLOAT_t r, FLOAT_t g, FLOAT_t b);
	PCL_HELPER_API void			setDetectionsVisualization(PCLHelperPtr h, FLOAT_t negSize, FLOAT_t negR, FLOAT_t negG, FLOAT_t negB, FLOAT_t posR, FLOAT_t posG, FLOAT_t posB, FLOAT_t GTR, FLOAT_t GTG, FLOAT_t GTB, FLOAT_t GTOptR, FLOAT_t GTOptG, FLOAT_t GTOptB);
	PCL_HELPER_API void			visualize(PCLHelperPtr h, char *title, bool fullscreen = false);
	PCL_HELPER_API void			visualizeKeypoints(PCLHelperPtr h, char *title, PCLHelperPtr keypointsHelper, PCLHelperPtr normalsHelper = NULL, bool fullscreen = false);
	PCL_HELPER_API void			visualizeDetections(PCLHelperPtr h, char *title, size_t nDetections = 0, Detection *detections = NULL, size_t nGroundTruth = 0, Detection *groundTruth = NULL, size_t nGroundTruthOpt = 0, Detection *groundTruthOpt = NULL, bool fullscreen = false);
	PCL_HELPER_API void			saveCloud(PCLHelperPtr h, char *filename);

	PCL_HELPER_API void			downsampleVoxelGrid(PCLHelperPtr h, FLOAT_t leafSize, PCLHelperPtr out);
	PCL_HELPER_API void			restrictViewport(PCLHelperPtr h, FLOAT_t x_min, FLOAT_t x_max, FLOAT_t y_min, FLOAT_t y_max, FLOAT_t z_min, FLOAT_t z_max);

	PCL_HELPER_API void			detectGroundPlane(PCLHelperPtr h, FLOAT_t maxPlaneAngle, FLOAT_t distanceThreshold, FLOAT_t *coefficients, FLOAT_t *transformation);
	PCL_HELPER_API void			removeGroundPlane(PCLHelperPtr h, FLOAT_t distanceThreshold, const FLOAT_t *coefficients);
	PCL_HELPER_API void			transform(PCLHelperPtr h, const FLOAT_t *transformation);

	PCL_HELPER_API size_t		getConnectedComponents(PCLHelperPtr h, FLOAT_t distanceThreshold, void **pLabelsIndices);
	PCL_HELPER_API void			extractConnectedComponents(PCLHelperPtr h, void *pLabelsIndices, PCLHelperPtr *components);

	PCL_HELPER_API void			extractTetragonalPrisms(PCLHelperPtr h, size_t nPrisms, TetragonalPrism *prisms, PCLHelperPtr out, bool invert = false);

	PCL_HELPER_API void			getNormalsOfCloud(PCLHelperPtr h, PCLHelperPtr inputHelper, FLOAT_t radius, FLOAT_t *pNormals);	// estimate normals at positions of "inputHelper", store output in pNormals

	PCL_HELPER_API int			getHarrisPoints(PCLHelperPtr h, FLOAT_t radius, PCLHelperPtr out, bool refine = true, FLOAT_t threshold = 0, int method = 1);
	PCL_HELPER_API int			getISSPoints(PCLHelperPtr h, FLOAT_t salientRadius, FLOAT_t nonMaxRadius, int numberOfThreads, PCLHelperPtr out, int minNeighbors = 5, FLOAT_t threshold21 = 0.975, FLOAT_t threshold32 = 0.975, FLOAT_t angleThreshold = static_cast<FLOAT_t> (C_PI) / 2.0f, FLOAT_t normalRadius = 0, FLOAT_t borderRadius = 0);

	PCL_HELPER_API size_t		computeSHOT(PCLHelperPtr h, PCLHelperPtr keypointHelper, FLOAT_t radius, SHOTFeature *out);
	PCL_HELPER_API size_t		computeSHOTColor(PCLHelperPtr h, PCLHelperPtr keypointHelper, FLOAT_t radius, SHOTColorFeature *out);
}


#endif /* PCL_HELPER_EXPORTS_H */
