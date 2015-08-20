#include "pcl_helper_exports.h"
#include "pcl_helper.h"


// PCL point type for exported C functions
typedef pcl::PointXYZRGB PointTExport;

// macro for converting void* to PCLHelper<PointTExport>*
#define HELPER(ptr) ((PCLHelper<PointTExport>*)ptr)

// factory functions that create and release instances of PCLHelper
PCLHelperPtr getPCLHelper() {
	return new PCLHelper<PointTExport>;
}
void freePCLHelper(PCLHelperPtr h) {
	delete HELPER(h);
}

// implementation functions
void setNormalsParams(PCLHelperPtr h, int k_search, FLOAT_t radius) {
	HELPER(h)->setNormalsParams(k_search, radius);
}
size_t getCloudSize(PCLHelperPtr h, size_t *width, size_t *height) {
	return HELPER(h)->getCloudSize(width, height);
}
void fromArray(PCLHelperPtr h, size_t width, size_t height, FLOAT_t *pXYZ, char *pRGB) {
	HELPER(h)->fromArray(width, height, pXYZ, pRGB);
}
void toArray(PCLHelperPtr h, FLOAT_t *pXYZ, char *pRGB) {
	HELPER(h)->toArray(pXYZ, pRGB);
}
void copyCloud(PCLHelperPtr h, PCLHelperPtr out) {
	HELPER(h)->copyCloud(HELPER(out));
}
void addCloud(PCLHelperPtr h, PCLHelperPtr addendHelper) {
	HELPER(h)->addCloud(HELPER(addendHelper));
}
void removeNaN(PCLHelperPtr h) {
	HELPER(h)->removeNaN();
}
void setBGColor(PCLHelperPtr h, FLOAT_t r, FLOAT_t g, FLOAT_t b) {
	HELPER(h)->setBGColor(r, g, b);
}
void setCameraPosition(PCLHelperPtr h, FLOAT_t camPosX, FLOAT_t camPosY, FLOAT_t camPosZ, FLOAT_t camViewX, FLOAT_t camViewY, FLOAT_t camViewZ, FLOAT_t camUpX, FLOAT_t camUpY, FLOAT_t camUpZ) {
	HELPER(h)->setCameraPosition(camPosX, camPosY, camPosZ, camViewX, camViewY, camViewZ, camUpX, camUpY, camUpZ);
}
void setKeypointsVisualization(PCLHelperPtr h, FLOAT_t size, FLOAT_t normalLength, FLOAT_t r, FLOAT_t g, FLOAT_t b) {
	HELPER(h)->setKeypointsVisualization(size, normalLength, r, g, b);
}
void setDetectionsVisualization(PCLHelperPtr h, FLOAT_t negSize, FLOAT_t negR, FLOAT_t negG, FLOAT_t negB, FLOAT_t posR, FLOAT_t posG, FLOAT_t posB, FLOAT_t GTR, FLOAT_t GTG, FLOAT_t GTB, FLOAT_t GTOptR, FLOAT_t GTOptG, FLOAT_t GTOptB) {
	HELPER(h)->setDetectionsVisualization(negSize, negR, negG, negB, posR, posG, posB, GTR, GTG, GTB, GTOptR, GTOptG, GTOptB);
}
void visualize(PCLHelperPtr h, char *title, bool fullscreen) {
	HELPER(h)->visualize(title, fullscreen);
}
void visualizeKeypoints(PCLHelperPtr h, char *title, PCLHelperPtr keypointsHelper, PCLHelperPtr normalsHelper, bool fullscreen) {
	HELPER(h)->visualizeKeypoints(title, HELPER(keypointsHelper), HELPER(normalsHelper), fullscreen);
}
void visualizeDetections(PCLHelperPtr h, char *title, size_t nDetections, Detection *detections, size_t nGroundTruth, Detection *groundTruth, size_t nGroundTruthOpt, Detection *groundTruthOpt, bool fullscreen) {
	HELPER(h)->visualizeDetections(title, nDetections, detections, nGroundTruth, groundTruth, nGroundTruthOpt, groundTruthOpt, fullscreen);
}
void saveCloud(PCLHelperPtr h, char *filename) {
	HELPER(h)->saveCloud(filename);
}
void downsampleVoxelGrid(PCLHelperPtr h, FLOAT_t leafSize, PCLHelperPtr out) {
	HELPER(h)->downsampleVoxelGrid(leafSize, HELPER(out));
}
void restrictViewport(PCLHelperPtr h, FLOAT_t x_min, FLOAT_t x_max, FLOAT_t y_min, FLOAT_t y_max, FLOAT_t z_min, FLOAT_t z_max) {
	HELPER(h)->restrictViewport(x_min, x_max, y_min, y_max, z_min, z_max);
}
void detectGroundPlane(PCLHelperPtr h, FLOAT_t maxPlaneAngle, FLOAT_t distanceThreshold, FLOAT_t *coefficients, FLOAT_t *transformation) {
	HELPER(h)->detectGroundPlane(maxPlaneAngle, distanceThreshold, coefficients, transformation);
}
void removeGroundPlane(PCLHelperPtr h, FLOAT_t distanceThreshold, const FLOAT_t *coefficients) {
	HELPER(h)->removeGroundPlane(distanceThreshold, coefficients);
}
void transform(PCLHelperPtr h, const FLOAT_t *transformation) {
	HELPER(h)->transform(transformation);
}
size_t getConnectedComponents(PCLHelperPtr h, FLOAT_t distanceThreshold, void **pLabelsIndices) {
	return HELPER(h)->getConnectedComponents(distanceThreshold, pLabelsIndices);
}
void extractConnectedComponents(PCLHelperPtr h, void *pLabelsIndices, PCLHelperPtr *components) {
	HELPER(h)->extractConnectedComponents(pLabelsIndices, (PCLHelper<PointTExport>**)components);
}
void extractTetragonalPrisms(PCLHelperPtr h, size_t nPrisms, TetragonalPrism *prisms, PCLHelperPtr out, bool invert) {
	HELPER(h)->extractTetragonalPrisms(nPrisms, prisms, HELPER(out), invert);
}
void getNormalsOfCloud(PCLHelperPtr h, PCLHelperPtr inputHelper, FLOAT_t radius, FLOAT_t *pNormals) {
	HELPER(h)->getNormalsOfCloud(HELPER(inputHelper), radius, pNormals);
}
int getHarrisPoints(PCLHelperPtr h, FLOAT_t radius, PCLHelperPtr out, bool refine, FLOAT_t threshold, int method) {
	return HELPER(h)->getHarrisPoints(radius, HELPER(out), refine, threshold, method);
}
int getISSPoints(PCLHelperPtr h, FLOAT_t salientRadius, FLOAT_t nonMaxRadius, int numberOfThreads, PCLHelperPtr out, int minNeighbors, FLOAT_t threshold21, FLOAT_t threshold32, FLOAT_t angleThreshold, FLOAT_t normalRadius, FLOAT_t borderRadius) {
	return HELPER(h)->getISSPoints(salientRadius, nonMaxRadius, numberOfThreads, HELPER(out), minNeighbors, threshold21, threshold32, angleThreshold, normalRadius, borderRadius);
}
size_t computeSHOT(PCLHelperPtr h, PCLHelperPtr keypointHelper, FLOAT_t radius, SHOTFeature *out) {
	return HELPER(h)->computeSHOT(HELPER(keypointHelper), radius, out);
}
size_t computeSHOTColor(PCLHelperPtr h, PCLHelperPtr keypointHelper, FLOAT_t radius, SHOTColorFeature *out) {
	return HELPER(h)->computeSHOTColor(HELPER(keypointHelper), radius, out);
}
