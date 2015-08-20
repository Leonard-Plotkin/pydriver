#include <iostream>

#include <Eigen/Core>

#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>

#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/crop_hull.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/keypoints/harris_3d.h>
#include <pcl/keypoints/iss_3d.h>

#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_perpendicular_plane.h>

#include <pcl/segmentation/euclidean_cluster_comparator.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/segmentation/organized_connected_component_segmentation.h>

#include <pcl/surface/mls.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
