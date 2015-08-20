// patch for missing to_string support in some compilers
#include <string>
#include <sstream>
namespace patch
{
    template <typename T> std::string to_string(const T &n)
    {
        std::ostringstream stm;
        stm << n;
        return(stm.str());
    }
}


#include "pcl_helper.h"

using namespace std;


// ********** public functions **********

template <typename PointT> PCLHelper<PointT>::PCLHelper()
{
	// initialize an empty cloud
	pCloud = typename pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>());
}

template <typename PointT> PCLHelper<PointT>::~PCLHelper()
{
}

template <typename PointT> void PCLHelper<PointT>::setNormalsParams(int k_search, FLOAT_t radius)
{
	normals_k_search = k_search;
	normals_radius = radius;
	resetCache();
}

template <typename PointT> size_t PCLHelper<PointT>::getCloudSize(size_t *width, size_t *height)
{
	if (width != NULL)
		*width = pCloud->width;
	if (height != NULL)
		*height = pCloud->height;
	return(pCloud->points.size());
}

template <typename PointT> void PCLHelper<PointT>::fromArray(size_t width, size_t height, FLOAT_t *pXYZ)
{
	pCloud->width		= width;
	pCloud->height		= height;
	pCloud->is_dense	= false;
	pCloud->points.resize(pCloud->width * pCloud->height);

	for (size_t i = 0; i < pCloud->points.size(); i++)
	{
		pCloud->points[i].x = pXYZ[3*i + 0];
		pCloud->points[i].y = pXYZ[3*i + 1];
		pCloud->points[i].z = pXYZ[3*i + 2];
	}

	resetCache();
}

// PointXYZRGB specialization
template <> void PCLHelper<pcl::PointXYZRGB>::fromArray(size_t width, size_t height, FLOAT_t *pXYZ, char *pRGB)
{
	fromArray(width, height, pXYZ);

	if(pRGB != NULL)
	{
		for (size_t i = 0; i < pCloud->points.size(); i++)
		{
			pCloud->points[i].r = pRGB[3*i + 0];
			pCloud->points[i].g = pRGB[3*i + 1];
			pCloud->points[i].b = pRGB[3*i + 2];
		}

		resetCache();
	}
}

// universal implementation
template <typename PointT> void PCLHelper<PointT>::fromArray(size_t width, size_t height, FLOAT_t *pXYZ, char *pRGB)
{
	if(pRGB != NULL) {
		cerr << "WARNING in PCLHelper::fromArray(): RGB data received, but the point type does not support it." << endl;
	}
	fromArray(width, height, pXYZ);
}

template <typename PointT> void PCLHelper<PointT>::toArray(FLOAT_t *pXYZ)
{
	for (size_t i = 0; i < pCloud->points.size(); i++)
	{
		pXYZ[3*i + 0] = pCloud->points[i].x;
		pXYZ[3*i + 1] = pCloud->points[i].y;
		pXYZ[3*i + 2] = pCloud->points[i].z;
	}
}

// PointXYZRGB specialization
template <> void PCLHelper<pcl::PointXYZRGB>::toArray(FLOAT_t *pXYZ, char *pRGB)
{
	toArray(pXYZ);

	if(pRGB != NULL)
	{
		for (size_t i = 0; i < pCloud->points.size(); i++)
		{
			pRGB[3*i + 0] = pCloud->points[i].r;
			pRGB[3*i + 1] = pCloud->points[i].g;
			pRGB[3*i + 2] = pCloud->points[i].b;
		}
	}
}

// universal implementation
template <typename PointT> void PCLHelper<PointT>::toArray(FLOAT_t *pXYZ, char *pRGB)
{
	if(pRGB != NULL) {
		cerr << "WARNING in PCLHelper::toArray(): RGB data requested, but the point type does not support it." << endl;
	}
	toArray(pXYZ);
}

template <typename PointT> void PCLHelper<PointT>::copyCloud(PCLHelper<PointT> *out)
{
	out->setCloud(typename pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>(*pCloud)));
}

template <typename PointT> void PCLHelper<PointT>::addCloud(PCLHelper<PointT> *addendHelper)
{
	*(pCloud) += *(addendHelper->pCloud);
	resetCache();
}

template <typename PointT> void PCLHelper<PointT>::removeNaN()
{
	std::vector<int> indices;
	pcl::removeNaNFromPointCloud(*pCloud, *pCloud, indices);
	resetCache();
}

template <typename PointT> void PCLHelper<PointT>::setBGColor(FLOAT_t r, FLOAT_t g, FLOAT_t b)
{
	bgR = r;
	bgG = g;
	bgB = b;
}

template <typename PointT> void PCLHelper<PointT>::setCameraPosition(FLOAT_t camPosX, FLOAT_t camPosY, FLOAT_t camPosZ, FLOAT_t camViewX, FLOAT_t camViewY, FLOAT_t camViewZ, FLOAT_t camUpX, FLOAT_t camUpY, FLOAT_t camUpZ)
{
	this->camPosX = camPosX;
	this->camPosY = camPosY;
	this->camPosZ = camPosZ;
	this->camViewX = camViewX;
	this->camViewY = camViewY;
	this->camViewZ = camViewZ;
	this->camUpX = camUpX;
	this->camUpY = camUpY;
	this->camUpZ = camUpZ;
}

template <typename PointT> void PCLHelper<PointT>::setKeypointsVisualization(FLOAT_t size, FLOAT_t normalLength, FLOAT_t r, FLOAT_t g, FLOAT_t b)
{
	kpSize = size;
	kpNormalLength = normalLength;
	kpR = r;
	kpG = g;
	kpB = b;
}

template <typename PointT> void PCLHelper<PointT>::setDetectionsVisualization(FLOAT_t negSize, FLOAT_t negR, FLOAT_t negG, FLOAT_t negB, FLOAT_t posR, FLOAT_t posG, FLOAT_t posB, FLOAT_t GTR, FLOAT_t GTG, FLOAT_t GTB, FLOAT_t GTOptR, FLOAT_t GTOptG, FLOAT_t GTOptB)
{
	// edge length of negative detections cubes
	detNegSize = negSize;
	// negative detections color
	detNegR = negR;
	detNegG = negG;
	detNegB = negB;
	// positive detections color
	detPosR = posR;
	detPosG = posG;
	detPosB = posB;
	// ground truth color
	detGTR = GTR;
	detGTG = GTG;
	detGTB = GTB;
	// optional ground truth color
	detGTOptR = GTOptR;
	detGTOptG = GTOptG;
	detGTOptB = GTOptB;
}

template <typename PointT> void PCLHelper<PointT>::visualize(char *title, bool fullscreen)
{
	pcl::visualization::PCLVisualizer viewer = getVisualizer(title, true, fullscreen);
	showVisualizer(viewer);
}

template <typename PointT> void PCLHelper<PointT>::visualizeKeypoints(char *title, PCLHelper<PointT> *keypointsHelper, PCLHelper<PointT> *normalsHelper, bool fullscreen)
{
	pcl::visualization::PCLVisualizer viewer = getVisualizer(title, true, fullscreen);

	// add keypoints
	viewer.addPointCloud<PointT>(keypointsHelper->pCloud, "keypoints");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, kpSize, "keypoints");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, kpR, kpG, kpB, "keypoints");

	if (normalsHelper != NULL)
	{
		size_t nKeypoints = keypointsHelper->pCloud->points.size();
		if (normalsHelper->pCloud->points.size() != nKeypoints)
		{
			cerr << "ERROR in PCLHelper::visualizeKeypoints(): Numbers of points in 'keypointsHelper' and 'normalsHelper' must match." << endl;
			throw length_error("Numbers of points in 'keypointsHelper' and 'normalsHelper' must match.");
		}

		// construct normal cloud of type pcl::Normal
		pcl::PointCloud<pcl::Normal>::Ptr pKeypointNormals(new pcl::PointCloud<pcl::Normal>);
		pcl::Normal normal;
		for (size_t i = 0; i < nKeypoints; i++)
		{
			normal.normal_x = normalsHelper->pCloud->points[i].x;
			normal.normal_y = normalsHelper->pCloud->points[i].y;
			normal.normal_z = normalsHelper->pCloud->points[i].z;
			pKeypointNormals->points.push_back(normal);
		}

		// add normals
		viewer.addPointCloudNormals<PointT, pcl::Normal>(keypointsHelper->pCloud, pKeypointNormals, 1, kpNormalLength, "normals");
		viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, kpR, kpG, kpB, "normals");
	}

	showVisualizer(viewer);
}

template <typename PointT> void PCLHelper<PointT>::visualizeDetections(char *title, size_t nDetections, Detection *detections, size_t nGroundTruth, Detection *groundTruth, size_t nGroundTruthOpt, Detection *groundTruthOpt, bool fullscreen)
{
	string id;
	FLOAT_t rotation_y;
	pcl::visualization::PCLVisualizer viewer = getVisualizer(title, true, fullscreen);

	for (size_t i = 0; i < nDetections; i++)
	{
		Detection& Detection = detections[i];
		// correct for our convention that Detection.position.rotation_y = 0 points to the positive x-direction
		// rotation_y = 0 will point to positive z-direction
		rotation_y = Detection.position.rotation_y + 0.5*C_PI;
		id = "Detection"+patch::to_string(static_cast<long long>(i));
		if(string(Detection.category) == "negative")
		{
			// negative Detection
			viewer.addCube(
				Eigen::Vector3f(Detection.position.x, Detection.position.y, Detection.position.z),							// translation
				Eigen::Quaternionf(cos(0.5*rotation_y), 0, sin(0.5*rotation_y), 0),											// rotation
				detNegSize, detNegSize, detNegSize,																			// dimensions
				id																											// id
			);
			viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, detNegR, detNegG, detNegB, id);	// color of the box
			// show orientation arrow
			viewer.addArrow<pcl::PointXYZ, pcl::PointXYZ>(
				pcl::PointXYZ(																								// orientation direction
					Detection.position.x + sin(rotation_y)*0.5*detNegSize,
					Detection.position.y,
					Detection.position.z + cos(rotation_y)*0.5*detNegSize
				),
				pcl::PointXYZ(Detection.position.x, Detection.position.y, Detection.position.z),							// object center
				detNegR, detNegG, detNegB,																					// color
				false,																										// don't display length
				id+"_arrow"																									// id
			);
		} else {
			// positive Detection
			viewer.addCube(
				Eigen::Vector3f(Detection.position.x, Detection.position.y, Detection.position.z),							// translation
				Eigen::Quaternionf(cos(0.5*rotation_y), 0, sin(0.5*rotation_y), 0),											// rotation
				Detection.width, Detection.height, Detection.length,														// dimensions
				id																											// id
			);
			viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, detPosR, detPosG, detPosB, id);	// color of the box
			// show orientation arrow
			viewer.addArrow<pcl::PointXYZ, pcl::PointXYZ>(
				pcl::PointXYZ(																								// orientation direction
					Detection.position.x + sin(rotation_y)*0.5*Detection.length,
					Detection.position.y,
					Detection.position.z + cos(rotation_y)*0.5*Detection.length
				),
				pcl::PointXYZ(Detection.position.x, Detection.position.y, Detection.position.z),							// object center
				detPosR, detPosG, detPosB,																					// color
				false,																										// don't display length
				id+"_arrow"																									// id
			);
		}
	}
	for (size_t i = 0; i < nGroundTruth; i++)
	{
		// ground truth
		Detection& Detection = groundTruth[i];
		// correct for our convention that Detection.position.rotation_y = 0 points to the positive x-direction
		// rotation_y = 0 will point to positive z-direction
		rotation_y = Detection.position.rotation_y + 0.5*C_PI;
		id = "GroundTruth"+patch::to_string(static_cast<long long>(i));
		viewer.addCube(
			Eigen::Vector3f(Detection.position.x, Detection.position.y, Detection.position.z),								// translation
			Eigen::Quaternionf(cos(0.5*rotation_y), 0, sin(0.5*rotation_y), 0),												// rotation
			Detection.width, Detection.height, Detection.length,															// dimensions
			id																												// id
		);
		viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, detGTR, detGTG, detGTB, id);			// color of the box
		// show orientation arrow
		viewer.addArrow<pcl::PointXYZ, pcl::PointXYZ>(
			pcl::PointXYZ(																									// orientation direction
				Detection.position.x + sin(rotation_y)*0.5*Detection.length,
				Detection.position.y,
				Detection.position.z + cos(rotation_y)*0.5*Detection.length
			),
			pcl::PointXYZ(Detection.position.x, Detection.position.y, Detection.position.z),								// object center
			detGTR, detGTG, detGTB,																							// color
			false,																											// don't display length
			id+"_arrow"																										// id
		);
	}
	for (size_t i = 0; i < nGroundTruthOpt; i++)
	{
		// optional ground truth
		Detection& Detection = groundTruthOpt[i];
		// correct for our convention that Detection.position.rotation_y = 0 points to the positive x-direction
		// rotation_y = 0 will point to positive z-direction
		rotation_y = Detection.position.rotation_y + 0.5*C_PI;
		id = "GroundTruthOpt"+patch::to_string(static_cast<long long>(i));
		viewer.addCube(
			Eigen::Vector3f(Detection.position.x, Detection.position.y, Detection.position.z),								// translation
			Eigen::Quaternionf(cos(0.5*rotation_y), 0, sin(0.5*rotation_y), 0),												// rotation
			Detection.width, Detection.height, Detection.length,															// dimensions
			id																												// id
		);
		viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, detGTOptR, detGTOptG, detGTOptB, id);	// color of the box
		// show orientation arrow
		viewer.addArrow<pcl::PointXYZ, pcl::PointXYZ>(
			pcl::PointXYZ(																									// orientation direction
				Detection.position.x + sin(rotation_y)*0.5*Detection.length,
				Detection.position.y,
				Detection.position.z + cos(rotation_y)*0.5*Detection.length
			),
			pcl::PointXYZ(Detection.position.x, Detection.position.y, Detection.position.z),								// object center
			detGTOptR, detGTOptG, detGTOptB,																				// color
			false,																											// don't display length
			id+"_arrow"																										// id
		);
	}

	showVisualizer(viewer);
}

template <typename PointT> void PCLHelper<PointT>::saveCloud(char *filename)
{
	pcl::PCLPointCloud2 cloud2;
	const Eigen::Quaternionf &orientation = Eigen::Quaternionf(0, 0, 0, 1);

	// convert PointCloud to PointCloud2
	pcl::toPCLPointCloud2(*pCloud, cloud2);
	// save PointCloud2 as binary PCD file with given orientation
	pcl::io::savePCDFile(filename, cloud2, Eigen::Matrix<float, 4, 1, 0, 4, 1>::Zero(), orientation, true);
}

template <typename PointT> void PCLHelper<PointT>::downsampleVoxelGrid(FLOAT_t leafSize, PCLHelper<PointT> *out)
{
	pcl::VoxelGrid<PointT> vgFilter;
	typename pcl::PointCloud<PointT>::Ptr cloudDownsampled(new pcl::PointCloud<PointT>);
	vgFilter.setInputCloud(pCloud);
	vgFilter.setLeafSize(leafSize, leafSize, leafSize);
	vgFilter.filter(*cloudDownsampled);

	out->setCloud(cloudDownsampled);
}

template <typename PointT> void PCLHelper<PointT>::restrictViewport(FLOAT_t x_min, FLOAT_t x_max, FLOAT_t y_min, FLOAT_t y_max, FLOAT_t z_min, FLOAT_t z_max)
{
	// restrict viewport while keeping the cloud organized

	// create range condition(s)
	typename pcl::ConditionAnd<PointT>::Ptr rangeCond(new pcl::ConditionAnd<PointT>());
	rangeCond->addComparison(typename pcl::FieldComparison<PointT>::Ptr(new pcl::FieldComparison<PointT>("x", pcl::ComparisonOps::GE, x_min)));
	rangeCond->addComparison(typename pcl::FieldComparison<PointT>::Ptr(new pcl::FieldComparison<PointT>("x", pcl::ComparisonOps::LE, x_max)));
	rangeCond->addComparison(typename pcl::FieldComparison<PointT>::Ptr(new pcl::FieldComparison<PointT>("y", pcl::ComparisonOps::GE, y_min)));
	rangeCond->addComparison(typename pcl::FieldComparison<PointT>::Ptr(new pcl::FieldComparison<PointT>("y", pcl::ComparisonOps::LE, y_max)));
	rangeCond->addComparison(typename pcl::FieldComparison<PointT>::Ptr(new pcl::FieldComparison<PointT>("z", pcl::ComparisonOps::GE, z_min)));
	rangeCond->addComparison(typename pcl::FieldComparison<PointT>::Ptr(new pcl::FieldComparison<PointT>("z", pcl::ComparisonOps::LE, z_max)));

	// create and apply filter
	pcl::ConditionalRemoval<PointT> rangeFilter;
	rangeFilter.setInputCloud(pCloud);
	rangeFilter.setCondition(rangeCond);
	rangeFilter.setKeepOrganized(true);
	rangeFilter.filter(*pCloud);

	resetCache();
}

template <typename PointT> void PCLHelper<PointT>::detectGroundPlane(FLOAT_t maxPlaneAngle, FLOAT_t distanceThreshold, FLOAT_t *coefficients, FLOAT_t *transformation)
{
	// detect ground plane, save ground plane model coefficients to "coefficients"

	// ground plane estimation
	typename pcl::SampleConsensusModelPerpendicularPlane<PointT>::Ptr planeModel(new pcl::SampleConsensusModelPerpendicularPlane<PointT>(pCloud));
	planeModel->setAxis(Eigen::Vector3f(0, 1, 0));	// ground plane should be perpendicular to Y-axis
	planeModel->setEpsAngle(maxPlaneAngle);			// maximal angle to given axis
	pcl::RandomSampleConsensus<PointT> ransac(planeModel);
	ransac.setDistanceThreshold(distanceThreshold);
	ransac.computeModel();

	// get ground plane coefficients
	Eigen::VectorXf planeCoefficients;				// (4,1)-matrix/vector, plane: a0*x + a1*y + a2*z + a3 = 0
	ransac.getModelCoefficients(planeCoefficients);
	if(planeCoefficients[1] < 0)
		planeCoefficients = -1 * planeCoefficients;	// let plane normal point to positive y-direction

	// copy plane coefficients to provided buffer
	for (int i=0; i<4; i++)
		coefficients[i] = planeCoefficients[i];

	// *** compute ground plane transformation ***

	// compute y- and z-axis according to ground plane
	Eigen::Vector3f y_axis, z_axis;
	y_axis = Eigen::Vector3f(coefficients[0], coefficients[1], coefficients[2]);	// plane normal pointing to positive y-direction
	z_axis = Eigen::Vector3f(1, 0, 0).cross(y_axis);								// project z-axis onto ground plane (cross-product between ideal x-axis and real y-axis)

	// transformation for point cloud to get ground normal pointing to (0, 1, 0), z-axis to (0, 0, 1) and the point on ground under the camera to be in (0, 0, 0)
	Eigen::Affine3f gpTransformation;
	pcl::getTransformationFromTwoUnitVectorsAndOrigin(y_axis, z_axis, y_axis * -coefficients[3], gpTransformation);

	// copy transformation to provided buffer
	for (int row=0; row<4; row++)
		for (int col=0; col<4; col++)
			transformation[row*4+col] = gpTransformation(row, col);
}

template <typename PointT> void PCLHelper<PointT>::removeGroundPlane(FLOAT_t distanceThreshold, const FLOAT_t *coefficients)
{
	// remove ground plane while keeping the cloud organized

	// get indices of plane inliers and everything below the plane
	boost::shared_ptr< vector<int> > planeIndicesPtr(new vector<int>);
	vector<int> &planeIndices = *(planeIndicesPtr.get());
	for (size_t i=0; i<pCloud->points.size(); i++)
	{
		// compute signed distance
		FLOAT_t dist =	pCloud->points[i].x * coefficients[0] +
						pCloud->points[i].y * coefficients[1] +
						pCloud->points[i].z * coefficients[2] +
											 coefficients[3];
		// positive distance is under the plane since the plane normal points to the positive y-direction which points down,
		// account for threshold
		if(dist > -distanceThreshold)
			planeIndices.push_back(i);
	}
	// create filter and apply it
	pcl::ExtractIndices<PointT> planeFilter;
	planeFilter.setIndices(planeIndicesPtr);	// set indices of ground plane
	planeFilter.setNegative(true);				// remove the given indices (instead of keeping them)
	planeFilter.filterDirectly(pCloud);			// keep the cloud organized, set filtered points to NaN

	resetCache();
}

template <typename PointT> void PCLHelper<PointT>::transform(const FLOAT_t *transformation)
{
	Eigen::Affine3f eigenTransformation;
	for (int row=0; row<4; row++)
		for (int col=0; col<4; col++)
			eigenTransformation(row, col) = transformation[row*4+col];

	pcl::transformPointCloud(*pCloud, *pCloud, eigenTransformation);
	resetCache();
}

template <typename PointT> size_t PCLHelper<PointT>::getConnectedComponents(FLOAT_t distanceThreshold, void **pLabelsIndices)
{
	if (!pCloud->isOrganized())
	{
		cerr << "ERROR in PCLHelper::getConnectedComponents(): Cloud needs to be organized in order to extract connected components." << endl;
		throw domain_error("Cloud needs to be organized in order to extract connected components.");
	}

	typename pcl::EuclideanClusterComparator<PointT, pcl::Normal, pcl::Label>::Ptr
		comparator(new pcl::EuclideanClusterComparator<PointT, pcl::Normal, pcl::Label>());
	comparator->setDistanceThreshold(distanceThreshold, true);
	comparator->setInputCloud(pCloud);

	// set labelling of the entire scene to false...
	pcl::Label l; l.label = 0;
	pcl::PointCloud<pcl::Label>::Ptr scene(new pcl::PointCloud<pcl::Label> (pCloud->width, pCloud->height, l));
	// ... and valid points (objects located on top of the plane) to true
	for (int i = 0; i < pCloud->points.size(); i++)
	{
		if (pcl::isFinite<PointT>(pCloud->points[i]))
		{
			scene->points[i].label = 1;
		}
	}
	// set labels
	comparator->setLabels(scene);
	// exclude invalid points (labeled "0")
	vector<bool> exclude_labels(2);  exclude_labels[0] = true; exclude_labels[1] = false;
	comparator->setExcludeLabels(exclude_labels);

	// setup segmentation
	pcl::OrganizedConnectedComponentSegmentation<PointT, pcl::Label> segmenter(comparator);
	segmenter.setInputCloud(pCloud);

	// execute segmentation
	vector<pcl::PointIndices> &labelsIndices = *(new vector<pcl::PointIndices>());	// reserve space on heap (for later extraction)
	pcl::PointCloud<pcl::Label> labels;
	segmenter.segment(labels, labelsIndices);

	// result that goes to Python (pointer to vector<pcl::PointIndices>)
	*pLabelsIndices = (void*)&labelsIndices;

	// return number of clusters
	return(labelsIndices.size());
}

template <typename PointT> void PCLHelper<PointT>::extractConnectedComponents(void *pLabelsIndices, PCLHelper<PointT> **components)
{
	// extract segments and free memory

	vector<pcl::PointIndices> &labelsIndices = *((vector<pcl::PointIndices>*)pLabelsIndices);

	for (int i = 0; i < labelsIndices.size(); i++) {
		// extract cloud
		typename pcl::PointCloud<PointT>::Ptr pCloud(new pcl::PointCloud<PointT>);
		pcl::copyPointCloud(*this->pCloud, labelsIndices[i], *pCloud);

		// assign resulting segment
		components[i]->setCloud(pCloud);
	}

	// free memory
	delete &labelsIndices;
}

template <typename PointT> void PCLHelper<PointT>::extractTetragonalPrisms(size_t nPrisms, TetragonalPrism *prisms, PCLHelper<PointT> *out, bool invert)
{
	pcl::PointIndicesPtr pAllIndices(new pcl::PointIndices);

	for (size_t iPrism = 0; iPrism < nPrisms; iPrism++)
	{
		// create cloud with given points which define the planar hull
		typename pcl::PointCloud<PointT>::Ptr pHullCloud(new pcl::PointCloud<PointT>);
		PointT point;
		for (size_t i = 0; i < 4; i++)
		{
			point.x = prisms[iPrism].xyz[i][0];
			point.y = prisms[iPrism].xyz[i][1];
			point.z = prisms[iPrism].xyz[i][2];
			pHullCloud->push_back(point);
		}
		// FIX for PCL 1.7 bug (https://github.com/PointCloudLibrary/pcl/issues/552)
		// add the first point again explicitly closing the polygon
		point.x = prisms[iPrism].xyz[0][0];
		point.y = prisms[iPrism].xyz[0][1];
		point.z = prisms[iPrism].xyz[0][2];
		pHullCloud->push_back(point);
		// END OF FIX

		// use vertices of convex hull to define prism
		pcl::ExtractPolygonalPrismData<PointT> prism;
		prism.setInputCloud(pCloud);
		prism.setInputPlanarHull(pHullCloud);
		prism.setHeightLimits(prisms[iPrism].height_min, prisms[iPrism].height_max);

		// extract indices of points inside the prism
		pcl::PointIndicesPtr pPrismIndices(new pcl::PointIndices);
		prism.segment(*pPrismIndices);

		// add indices to pAllIndices
		pAllIndices->indices.reserve(pAllIndices->indices.size() + pPrismIndices->indices.size());
		pAllIndices->indices.insert(pAllIndices->indices.end(), pPrismIndices->indices.begin(), pPrismIndices->indices.end());
	}

	// make indices unique since the prisms can overlap (sorting only for speed)
	sort(pAllIndices->indices.begin(), pAllIndices->indices.end());
	pAllIndices->indices.erase(unique(pAllIndices->indices.begin(), pAllIndices->indices.end()), pAllIndices->indices.end());

	// extract indexed points into pPrismCloud
	typename pcl::PointCloud<PointT>::Ptr pPrismCloud(new pcl::PointCloud<PointT>);
	pcl::ExtractIndices<PointT> extractor;
	extractor.setInputCloud(pCloud);
	extractor.setIndices(pAllIndices);
	extractor.setNegative(invert);
	extractor.filter(*pPrismCloud);

	// replace out's cloud with the new one
	out->setCloud(pPrismCloud);

	if (invert) {
		// there can still be NaNs in the inverted cloud, remove them since the cloud is not organized anyway
		out->removeNaN();
	}
}

template <typename PointT> void PCLHelper<PointT>::getNormalsOfCloud(PCLHelper<PointT> *inputHelper, FLOAT_t radius, FLOAT_t *pNormals)
{
	initKdTree();

	pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
	ne.setInputCloud(inputHelper->pCloud);	// cloud for which to estimate normals
	ne.setSearchSurface(pCloud);			// cloud used to estimate normals
	ne.setSearchMethod(pKdTree);
	ne.setRadiusSearch(radius);
	pcl::PointCloud<pcl::Normal>::Ptr pCloudNormals(new pcl::PointCloud<pcl::Normal>);
	ne.compute(*pCloudNormals);

	// copy result to output array
	for (size_t i = 0; i < pCloudNormals->points.size(); i++)
	{
		pNormals[3*i + 0] = (FLOAT_t)pCloudNormals->points[i].normal_x;
		pNormals[3*i + 1] = (FLOAT_t)pCloudNormals->points[i].normal_y;
		pNormals[3*i + 2] = (FLOAT_t)pCloudNormals->points[i].normal_z;
	}
}

// PointXYZRGB specialization
template <> int PCLHelper<pcl::PointXYZRGB>::getHarrisPoints(FLOAT_t radius, PCLHelper<pcl::PointXYZRGB> *out, bool refine, FLOAT_t threshold, int method)
{
	typedef pcl::PointXYZRGB PointT;
	typedef pcl::PointXYZI keyPointT;

	pcl::HarrisKeypoint3D<PointT,keyPointT> harris3D((pcl::HarrisKeypoint3D<PointT,keyPointT>::ResponseMethod)method);
	harris3D.setRadius(radius);
	harris3D.setRefine(refine);
	if (threshold > 0)
		harris3D.setThreshold(threshold);

	harris3D.setNonMaxSupression(true);
	harris3D.setInputCloud(pCloud);

	pcl::PointCloud<keyPointT>::Ptr keypoints_temp(new pcl::PointCloud<keyPointT>);
	harris3D.compute(*keypoints_temp);
	pcl::PointCloud<PointT>::Ptr keypoints(new pcl::PointCloud<PointT>);
	pcl::copyPointCloud(*keypoints_temp , *keypoints);

	// store result
	out->setCloud(keypoints);

	return(keypoints->points.size());
}

// universal implementation
template <typename PointT> int PCLHelper<PointT>::getHarrisPoints(FLOAT_t radius, PCLHelper<pcl::PointXYZRGB> *out, bool refine, FLOAT_t threshold, int method)
{
	cerr << "ERROR in PCLHelper::getHarrisPoints(): Harris keypoints are only implemented for PointXYZRGB." << endl;
	throw domain_error("Harris keypoints are only implemented for PointXYZRGB.");
}

template <typename PointT> int PCLHelper<PointT>::getISSPoints(FLOAT_t salientRadius, FLOAT_t nonMaxRadius, int numberOfThreads, PCLHelper<PointT> *out, int minNeighbors, FLOAT_t threshold21, FLOAT_t threshold32, FLOAT_t angleThreshold, FLOAT_t normalRadius, FLOAT_t borderRadius)
{
	typename pcl::PointCloud<PointT>::Ptr keypoints(new pcl::PointCloud<PointT>);

	if(pCloud->points.size() == 0) {
		// special case for empty clouds
		out->setCloud(keypoints);
		return(0);
	}

	initKdTree();

	pcl::ISSKeypoint3D<PointT, PointT> iss3D;
	iss3D.setNumberOfThreads(numberOfThreads);	// crashes without initialization or when set to 0
	iss3D.setSalientRadius(salientRadius);
	iss3D.setNonMaxRadius(nonMaxRadius);
	iss3D.setMinNeighbors(minNeighbors);
	iss3D.setThreshold21(threshold21);
	iss3D.setThreshold32(threshold32);

	// only used for ISS with boundary estimation
	iss3D.setAngleThreshold(angleThreshold);
	iss3D.setNormalRadius(normalRadius);		// normals can be precomputed with .setNormals()
	iss3D.setBorderRadius(borderRadius);

	iss3D.setInputCloud(pCloud);
	iss3D.setSearchMethod(pKdTree);
	iss3D.compute(*keypoints);

	// store result
	out->setCloud(keypoints);

	return(keypoints->points.size());
}

template <typename PointT> size_t PCLHelper<PointT>::computeSHOT(PCLHelper<PointT> *keypointHelper, FLOAT_t radius, SHOTFeature *out)
{
	typename pcl::PointCloud<PointT>::Ptr keypointCloud = keypointHelper->pCloud;

	initKdTree();
	initNormals();

	pcl::SHOTEstimationOMP<PointT, pcl::Normal, pcl::SHOT352> se;
	se.setInputCloud(keypointCloud);
	se.setSearchSurface(pCloud);
	se.setInputNormals(pCloudNormals);
	se.setSearchMethod(pKdTree);
	se.setRadiusSearch(radius);

	pcl::PointCloud<pcl::SHOT352>::Ptr shotFeatures(new pcl::PointCloud<pcl::SHOT352>);
	se.compute(*shotFeatures);

	for (size_t i=0; i < shotFeatures->points.size(); i++)
	{
		for (size_t j=0; j < 352; j++)
			out[i].descriptor[j] = (FLOAT_t)shotFeatures->points[i].descriptor[j];
		for (size_t j=0; j < 9; j++)
			out[i].rf[j] = (FLOAT_t)shotFeatures->points[i].rf[j];
	}

	return(shotFeatures->points.size());
}

// PointXYZRGB specialization
template <> size_t PCLHelper<pcl::PointXYZRGB>::computeSHOTColor(PCLHelper<pcl::PointXYZRGB> *keypointHelper, FLOAT_t radius, SHOTColorFeature *out)
{
	typedef pcl::PointXYZRGB PointT;

	pcl::PointCloud<PointT>::Ptr keypointCloud = keypointHelper->pCloud;

	initKdTree();
	initNormals();

	pcl::SHOTColorEstimationOMP<PointT, pcl::Normal, pcl::SHOT1344> se;
	se.setInputCloud(keypointCloud);
	se.setSearchSurface(pCloud);
	se.setInputNormals(pCloudNormals);
	se.setSearchMethod(pKdTree);
	se.setRadiusSearch(radius);

	pcl::PointCloud<pcl::SHOT1344>::Ptr shotFeatures(new pcl::PointCloud<pcl::SHOT1344>);
	se.compute(*shotFeatures);

	for (size_t i=0; i < shotFeatures->points.size(); i++)
	{
		for (size_t j=0; j < 1344; j++)
			out[i].descriptor[j] = (FLOAT_t)shotFeatures->points[i].descriptor[j];
		for (size_t j=0; j < 9; j++)
			out[i].rf[j] = (FLOAT_t)shotFeatures->points[i].rf[j];
	}

	return(shotFeatures->points.size());
}

// universal implementation
template <typename PointT> size_t PCLHelper<PointT>::computeSHOTColor(PCLHelper<PointT> *keypointHelper, FLOAT_t radius, SHOTColorFeature *out)
{
	cerr << "ERROR in PCLHelper::computeSHOTColor(): SHOTColor is only implemented for PointXYZRGB." << endl;
	throw domain_error("SHOTColor is only implemented for PointXYZRGB.");
}


// ********** protected functions **********

template <typename PointT> void PCLHelper<PointT>::initKdTree()
{
	if (!pKdTree) {
		pKdTree = typename pcl::search::KdTree<PointT>::Ptr(new pcl::search::KdTree<PointT>);
		pKdTree->setInputCloud(pCloud);
	}
}

template <typename PointT> void PCLHelper<PointT>::initNormals()
{
	if (!pCloudNormals) {
		initKdTree();

		pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
		ne.setInputCloud(pCloud);
		ne.setSearchMethod(pKdTree);
		if (normals_radius > 0) {
			ne.setRadiusSearch(normals_radius);
		} else if (normals_k_search > 0) {
			ne.setKSearch(normals_k_search);
		} else {
			cerr << "ERROR in PCLHelper::initNormals(): normals_k_search and normals_radius are both invalid." << endl;
			throw invalid_argument("normals_k_search and normals_radius are both invalid.");
		}

		pCloudNormals = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>);
		ne.compute(*pCloudNormals);
	}
}

template <typename PointT> void PCLHelper<PointT>::resetCache()
{
	pKdTree.reset();
	pCloudNormals.reset();
}

template <typename PointT> void PCLHelper<PointT>::setCloud(typename pcl::PointCloud<PointT>::Ptr newCloud)
{
	pCloud = newCloud;
	pKdTree.reset();
	pCloudNormals.reset();
}

template <typename PointT> pcl::visualization::PCLVisualizer PCLHelper<PointT>::getVisualizer(const std::string &title, bool addCloud, bool fullscreen)
{
	pcl::visualization::PCLVisualizer viewer(title);
	viewer.setBackgroundColor(bgR, bgG, bgB);
	viewer.addCoordinateSystem();
	viewer.initCameraParameters();
	viewer.setCameraPosition(camPosX, camPosY, camPosZ, camViewX, camViewY, camViewZ, camUpX, camUpY, camUpZ);
	if (fullscreen)
	{
		viewer.setFullScreen(true);
		viewer.createInteractor();		// is needed after setFullScreen(true) due to a bug in PCL 1.7
	}

	if (addCloud)
		addCloudToVisualizer(viewer, pCloud);

	return(viewer);
}

// PointXYZRGB specialization
template <> void PCLHelper<pcl::PointXYZRGB>::addCloudToVisualizer(pcl::visualization::PCLVisualizer &viewer, typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr pCloudToAdd)
{
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(pCloudToAdd);
	viewer.addPointCloud<pcl::PointXYZRGB>(pCloudToAdd, rgb, "cloud");
}

// universal implementation
template <typename PointT> void PCLHelper<PointT>::addCloudToVisualizer(pcl::visualization::PCLVisualizer &viewer, typename pcl::PointCloud<PointT>::Ptr pCloudToAdd)
{
	viewer.addPointCloud<PointT>(pCloudToAdd, "cloud");
}

template <typename PointT> void PCLHelper<PointT>::showVisualizer(pcl::visualization::PCLVisualizer &viewer)
{
	while (!viewer.wasStopped())
	{
		viewer.spinOnce(100);
	}
	viewer.close();
}


// explicit instantiations
template class PCLHelper<pcl::PointXYZ>;
template class PCLHelper<pcl::PointXYZRGB>;
