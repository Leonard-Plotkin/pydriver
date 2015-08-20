#ifndef PCL_HELPER_H
#define PCL_HELPER_H

#include "precompiled.h"			// explicit include if precompiled headers are not supported
#include <cstddef>					// NULL is declared there
#include "pcl_helper_imports.h"


template <typename PointT> class PCLHelper
{
	public:
		PCLHelper();
		~PCLHelper();

		void setNormalsParams(int k_search, FLOAT_t radius);

		size_t getCloudSize(size_t *width, size_t *height);
		void fromArray(size_t width, size_t height, FLOAT_t *pXYZ);
		void fromArray(size_t width, size_t height, FLOAT_t *pXYZ, char *pRGB);
		void toArray(FLOAT_t *pXYZ);
		void toArray(FLOAT_t *pXYZ, char *pRGB);
		void copyCloud(PCLHelper<PointT> *out);
		void addCloud(PCLHelper<PointT> *addendHelper);
		void removeNaN();

		void setBGColor(FLOAT_t r, FLOAT_t g, FLOAT_t b);
		void setCameraPosition(FLOAT_t camPosX, FLOAT_t camPosY, FLOAT_t camPosZ, FLOAT_t camViewX, FLOAT_t camViewY, FLOAT_t camViewZ, FLOAT_t camUpX, FLOAT_t camUpY, FLOAT_t camUpZ);
		void setKeypointsVisualization(FLOAT_t size, FLOAT_t normalLength, FLOAT_t r, FLOAT_t g, FLOAT_t b);
		void setDetectionsVisualization(FLOAT_t negSize, FLOAT_t negR, FLOAT_t negG, FLOAT_t negB, FLOAT_t posR, FLOAT_t posG, FLOAT_t posB, FLOAT_t GTR, FLOAT_t GTG, FLOAT_t GTB, FLOAT_t GTOptR, FLOAT_t GTOptG, FLOAT_t GTOptB);
		void visualize(char *title, bool fullscreen = false);
		void visualizeKeypoints(char *title, PCLHelper<PointT> *keypointsHelper, PCLHelper<PointT> *normalsHelper = NULL, bool fullscreen = false);
		void visualizeDetections(char *title, size_t nDetections = 0, Detection *detections = NULL, size_t nGroundTruth = 0, Detection *groundTruth = NULL, size_t nGroundTruthOpt = 0, Detection *groundTruthOpt = NULL, bool fullscreen = false);
		void saveCloud(char *filename);

		void downsampleVoxelGrid(FLOAT_t leafSize, PCLHelper<PointT> *out);
		void restrictViewport(FLOAT_t x_min, FLOAT_t x_max, FLOAT_t y_min, FLOAT_t y_max, FLOAT_t z_min, FLOAT_t z_max);

		void detectGroundPlane(FLOAT_t maxPlaneAngle, FLOAT_t distanceThreshold, FLOAT_t *coefficients, FLOAT_t *transformation);
		void removeGroundPlane(FLOAT_t distanceThreshold, const FLOAT_t *coefficients);
		void transform(const FLOAT_t *transformation);

		size_t getConnectedComponents(FLOAT_t distanceThreshold, void **pLabelsIndices);
		void extractConnectedComponents(void *pLabelsIndices, PCLHelper<PointT> **components);

		void extractTetragonalPrisms(size_t nPrisms, TetragonalPrism *prisms, PCLHelper<PointT> *out, bool invert = false);

		void getNormalsOfCloud(PCLHelper<PointT> *inputHelper, FLOAT_t radius, FLOAT_t *pNormals);	// estimate normals at positions of "inputHelper", store output in pNormals

		int getHarrisPoints(FLOAT_t radius, PCLHelper<pcl::PointXYZRGB> *out, bool refine = true, FLOAT_t threshold = 0, int method = 1);	// pcl::HarrisKeypoint3D<>::HARRIS=1 method by default
		int getISSPoints(FLOAT_t salientRadius, FLOAT_t nonMaxRadius, int numberOfThreads, PCLHelper<PointT> *out, int minNeighbors = 5, FLOAT_t threshold21 = 0.975, FLOAT_t threshold32 = 0.975, FLOAT_t angleThreshold = static_cast<FLOAT_t> (C_PI) / 2.0f, FLOAT_t normalRadius = 0, FLOAT_t borderRadius = 0);

		size_t computeSHOT(PCLHelper<PointT> *keypointHelper, FLOAT_t radius, SHOTFeature *out);
		size_t computeSHOTColor(PCLHelper<PointT> *keypointHelper, FLOAT_t radius, SHOTColorFeature *out);
	protected:
		// internal PCL objects
		typename pcl::PointCloud<PointT>::Ptr pCloud;
		pcl::PointCloud<pcl::Normal>::Ptr pCloudNormals;
		typename pcl::search::KdTree<PointT>::Ptr pKdTree;

		// visualizer background color
		FLOAT_t bgR, bgG, bgB;
		// visualizer camera position
		FLOAT_t camPosX, camPosY, camPosZ, camViewX, camViewY, camViewZ, camUpX, camUpY, camUpZ;
		// keypoints visualization: keypoint size (in pixels), normal length, RGB color
		FLOAT_t kpSize, kpNormalLength, kpR, kpG, kpB;
		// edge length of negative detections cubes
		FLOAT_t detNegSize;
		// detections RGB colors (negative, positive, ground truth, optional ground truth)
		FLOAT_t detNegR, detNegG, detNegB, detPosR, detPosG, detPosB, detGTR, detGTG, detGTB, detGTOptR, detGTOptG, detGTOptB;

		// normals estimation parameters
		// the one which is >0 will be used: either kNN with the given k or a sphere with the given radius
		int normals_k_search;
		FLOAT_t normals_radius;

		// use this function instead of manipulating pCloud directly
		void setCloud(typename pcl::PointCloud<PointT>::Ptr newCloud);

		pcl::visualization::PCLVisualizer getVisualizer(const std::string &title, bool addCloud = true, bool fullscreen = false);
		void addCloudToVisualizer(pcl::visualization::PCLVisualizer &viewer, typename pcl::PointCloud<PointT>::Ptr pCloudToAdd);
		void showVisualizer(pcl::visualization::PCLVisualizer &viewer);

		// initialize normals and kdTree
		void initKdTree();
		void initNormals();
		// reset normals and kdTree
		void resetCache();
};


#endif /* PCL_HELPER_H */
