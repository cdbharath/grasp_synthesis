#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include <iostream>

#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/surface/concave_hull.h>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>
#include <top_surface_algo/GraspPrediction.h>


class PtCloudClass{
  public:
    PtCloudClass(ros::NodeHandle& nh) : n(nh){
        pub = n.advertise<sensor_msgs::PointCloud2>("filtered_cloud", 3);
        // markers_pub = n.advertise<visualization_msgs::Marker>("grasp_points", 3);
        ros::ServiceServer service = n.advertiseService("coords_in_cam", &PtCloudClass::getGrasp, this);
        cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer ("Object Clusters Top Surfaces"));
        hullViewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer ("2D Concave hulls"));
        graspViewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer ("Grasps"));
        pt_cloud_sub = n.subscribe<sensor_msgs::PointCloud2>("/camera/depth/color/points", 5, &PtCloudClass::ptCloudCallback, this);
        initializeViewers();
        ros::spin();
    }

    void ptCloudCallback(const sensor_msgs::PointCloud2ConstPtr& in_cloud);
    bool getGrasp(top_surface_algo::GraspPrediction::Request  &req, top_surface_algo::GraspPrediction::Response &res);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filter_roi(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> getObjectClusters(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> getPassthroughFilteredClouds(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>);
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> getConvexHulls(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr calculateHull(pcl::PointCloud<pcl::PointXYZ>::Ptr CloudPtr);
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> getGrasp(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds);
    void addCentroid(pcl::PointCloud<pcl::PointXYZ>::Ptr CloudPtr);
    void addMarker(int id, float point1_x, float point1_y, float point1_z, float point2_x, float point2_y, float point2_z);
    void initializeViewers();
    
  private:
    ros::NodeHandle n;
    ros::Subscriber pt_cloud_sub;
    ros::Publisher pub;
    // ros::Publisher markers_pub;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    pcl::visualization::PCLVisualizer::Ptr viewer;
    pcl::visualization::PCLVisualizer::Ptr hullViewer;
    pcl::visualization::PCLVisualizer::Ptr graspViewer;
    // visualization_msgs::Marker marker;
};

bool PtCloudClass::getGrasp(top_surface_algo::GraspPrediction::Request  &req, top_surface_algo::GraspPrediction::Response &res)
{
    ROS_INFO("Request Recieved");

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_roi = filter_roi(cloud);
    
    // std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> obj_clusters = getObjectClusters(cloud_roi);
    // if (obj_clusters.size() < 1)
    // {
    //     return false;
    // }

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> obj_clusters{cloud_roi};
    pcl::PointCloud<pcl::PointXYZRGB> finalCloud;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> passthroughClusters = getPassthroughFilteredClouds(obj_clusters);
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> convexHulls = getConvexHulls(passthroughClusters);
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> graspClouds = getGrasp(convexHulls);
    if (convexHulls.size() > 0){
        finalCloud = *graspClouds[0];
        for (std::size_t i = 1; i < graspClouds.size(); i++){
            finalCloud += *graspClouds[i];
        }
    }
    res.best_grasp.pose.position.x = finalCloud[finalCloud.size()-1].x;
    res.best_grasp.pose.position.y = finalCloud[finalCloud.size()-1].y;
    res.best_grasp.pose.position.z = finalCloud[finalCloud.size()-1].z;
    res.best_grasp.pose.orientation.w = 1;

    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(finalCloud, output);
    output.header.frame_id = "camera_depth_optical_frame";
    output.header.stamp = ros::Time::now();
    pub.publish(output);
    return true;
}

void PtCloudClass::initializeViewers(){
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addCoordinateSystem (0.1);
    viewer->initCameraParameters ();
    hullViewer->setBackgroundColor (0, 0, 0);
    hullViewer->addCoordinateSystem (0.1);
    hullViewer->initCameraParameters ();
    graspViewer->setBackgroundColor (0, 0, 0);
    graspViewer->addCoordinateSystem (0.1);
    graspViewer->initCameraParameters ();
}

void PtCloudClass::addCentroid(pcl::PointCloud<pcl::PointXYZ>::Ptr CloudPtr){
    Eigen::Matrix< float, 4, 1 > centroid;
    pcl::PointXYZ centroidpoint;

    pcl::compute3DCentroid(*CloudPtr, centroid); 
    centroidpoint.x = centroid[0];
    centroidpoint.y = centroid[1];
    centroidpoint.z = centroid[2];

    CloudPtr->push_back(centroidpoint);
}

// void PtCloudClass::addMarker(int id, float point1_x, float point1_y, float point1_z, float point2_x, float point2_y, float point2_z){
//     visualization_msgs::Marker marker;
//     marker.header.frame_id = "panda_camera_optical_link";
//     marker.header.stamp = ros::Time();
//     marker.id = 0;
//     marker.type = visualization_msgs::Marker::POINTS;
//     marker.action = visualization_msgs::Marker::ADD;
//     marker.ns = "object";
//     marker.scale.x = 0.01;
//     marker.scale.y = 0.01;
//     marker.scale.z = 0.01;
//     marker.color.g = 1.0f;
//     marker.color.a = 1.0;
//     geometry_msgs::Point point1, point2;
//     point1.x = point1_x;
//     point1.y = point1_y;
//     point1.z = point1_z;
//     point2.x = point2_x;
//     point2.y = point2_y;
//     point2.z = point2_z;
//     marker.points.push_back(point1);
//     marker.points.push_back(point2);
//     markers_pub.publish(marker);
//     // marker_array.markers.push_back(marker);
// }

void PtCloudClass::ptCloudCallback(const sensor_msgs::PointCloud2ConstPtr& in_cloud){
    std::cout << "---------------------------------------------" << std::endl;
    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // sensor_msgs::PointCloud2 output;
    pcl::fromROSMsg(*in_cloud, *cloud);
    // std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> obj_clusters = getObjectClusters(cloud);
    // pcl::PointCloud<pcl::PointXYZRGB> finalCloud;
    // std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> passthroughClusters = getPassthroughFilteredClouds(obj_clusters);
    // std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> convexHulls = getConvexHulls(passthroughClusters);
    // std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> graspClouds = getGrasp(convexHulls);
    // if (convexHulls.size() > 0){
    //     finalCloud = *graspClouds[0];
    //     for (std::size_t i = 1; i < graspClouds.size(); i++){
    //         finalCloud += *graspClouds[i];
    //     }
    // }
    // pcl::toROSMsg(finalCloud, output);
    // output.header.frame_id = "panda_camera_optical_link";
    // output.header.stamp = ros::Time::now();
    // pub.publish(output);
}

std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> PtCloudClass::getObjectClusters(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> object_clusters;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
    std::cout << "PointCloud before filtering has: " << cloud->size () << " data points." << std::endl;
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
    vg.setInputCloud (cloud);
    vg.setLeafSize (0.008f, 0.008f, 0.008f);
    vg.filter (*cloud_filtered);
    std::cout << "PointCloud after filtering has: " << cloud_filtered->size ()  << " data points." << std::endl; //*

    // Create the segmentation object for the planar model and set all the parameters
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (100);
    seg.setDistanceThreshold (0.02);

    int nr_points = (int) cloud_filtered->size ();
    while (cloud_filtered->size () > 0.3 * nr_points)
    {
        // Segment the largest planar component from the remaining cloud
        seg.setInputCloud (cloud_filtered);
        seg.segment (*inliers, *coefficients);
        if (inliers->indices.size () == 0)
        {
            std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
            break;
        }

        // Extract the planar inliers from the input cloud
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud (cloud_filtered);
        extract.setIndices (inliers);
        extract.setNegative (false);

        // Get the points associated with the planar surface
        extract.filter (*cloud_plane);
        std::cout << "PointCloud representing the planar component: " << cloud_plane->size () << " data points." << std::endl;

        // Remove the planar inliers, extract the rest
        extract.setNegative (true);
        extract.filter (*cloud_f);
        if (cloud_f->size() < 1)
        {
            return object_clusters;
        }
        *cloud_filtered = *cloud_f;
    }

    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud_filtered);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance (0.02); // 2cm
    ec.setMinClusterSize (100);
    ec.setMaxClusterSize (25000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud_filtered);
    ec.extract (cluster_indices);

    int j = 0;
    for (const auto& cluster : cluster_indices)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& idx : cluster.indices) {
            cloud_cluster->push_back((*cloud_filtered)[idx]);
        }
        cloud_cluster->width = cloud_cluster->size ();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;
        std::cout << "PointCloud representing the Cluster: " << cloud_cluster->size () << " data points." << std::endl;
        object_clusters.push_back(cloud_cluster);
        j++;
    }
    return object_clusters;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr PtCloudClass::filter_roi(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
    float min_x = -0.36;
    float max_x = 0.36;
    float min_y = -0.22;
    float max_y = 0.22;
    float min_z = 0.0;
    float max_z = 0.7;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (min_z, max_z);
    pass.filter (*cloud_filtered);

    pass.setInputCloud (cloud_filtered);
    pass.setFilterFieldName ("y");
    pass.setFilterLimits (min_y, max_y);
    pass.filter (*cloud_filtered);

    pass.setInputCloud (cloud_filtered);
    pass.setFilterFieldName ("x");
    pass.setFilterLimits (min_x, max_x);
    pass.filter (*cloud_filtered);

    pcl::visualization::PCLVisualizer roiViewer("ROI Viewer");

    roiViewer.addPointCloud<pcl::PointXYZ> (cloud_filtered, "cloud_filtered");
    roiViewer.setBackgroundColor(0, 0, 0);

    roiViewer.spinOnce();

    // while(!roiViewer.wasStopped()){
    //     roiViewer.spinOnce();
    // }

    return cloud_filtered;
}

std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> PtCloudClass::getPassthroughFilteredClouds(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds){
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> fl_clouds;
    viewer->removeAllPointClouds();
    viewer->removeAllShapes();
    int count = 0;
    for (auto cloud : clouds){
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointXYZ minPt, maxPt;
        pcl::getMinMax3D (*cloud, minPt, maxPt);
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud (cloud);
        pass.setFilterFieldName ("z");
        pass.setFilterLimits (minPt.z, minPt.z+0.01);
        pass.filter (*cloud_filtered);
        pcl::getMinMax3D (*cloud_filtered, minPt, maxPt);
        std::cout << "Min z: " << minPt.z << std::endl;
        std::cout << "Max z: " << maxPt.z << std::endl;
        Eigen::Vector4f centroid;
        Eigen::Vector4f pcaCentroid;
        Eigen::Matrix3f covariance_matrix;
        pcl::compute3DCentroid(*cloud_filtered, pcaCentroid);
        viewer->addPointCloud<pcl::PointXYZ> (cloud_filtered, "cloud_"+std::to_string(count));
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_"+std::to_string(count));
        viewer->addSphere(pcl::PointXYZ( pcaCentroid[0], pcaCentroid[1],pcaCentroid[2]), 0.005, 0.5, 0.5, 0.0, "sphere"+std::to_string(count));
        fl_clouds.push_back(cloud_filtered);
        count++;
        // while (!viewer->wasStopped ()){
        //     viewer->spinOnce (100);
        //     // boost::this_thread::sleep (boost::posix_time::microseconds (100000));
        // }
    }
    viewer->spinOnce (10);
    return fl_clouds;
    
}

std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> PtCloudClass::getConvexHulls(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds){
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> fl_clouds;
    hullViewer->removeAllPointClouds();
    hullViewer->removeAllShapes();
    int count = 0;
    for (auto cloud : clouds){
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::ModelCoefficients::Ptr cluster_coefficients (new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr cluster_inliers (new pcl::PointIndices);
        // Create the segmentation object
        pcl::SACSegmentation<pcl::PointXYZ> cluster_seg;
        // Optional
        cluster_seg.setOptimizeCoefficients (true);
        // Mandatory
        cluster_seg.setModelType (pcl::SACMODEL_PLANE);
        cluster_seg.setMethodType (pcl::SAC_RANSAC);
        cluster_seg.setDistanceThreshold (0.01);

        cluster_seg.setInputCloud (cloud);
        cluster_seg.segment (*cluster_inliers, *cluster_coefficients);

        // Project the model inliers
        pcl::ProjectInliers<pcl::PointXYZ> proj;
        proj.setModelType (pcl::SACMODEL_PLANE);
        proj.setInputCloud (cloud);
        proj.setModelCoefficients (cluster_coefficients);
        proj.filter (*cloud_projected);

        /************************* Create a Concave Hull representation of the projected inliers******************/
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull (new pcl::PointCloud<pcl::PointXYZ>);
        cloud_hull = calculateHull(cloud_projected);
        Eigen::Vector4f centroid;
        Eigen::Vector4f pcaCentroid;
        Eigen::Matrix3f covariance_matrix;
        pcl::compute3DCentroid(*cloud_hull, pcaCentroid);
        hullViewer->addPointCloud<pcl::PointXYZ> (cloud_hull, "cloud_"+std::to_string(count));
        hullViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_"+std::to_string(count));
        hullViewer->addSphere(pcl::PointXYZ( pcaCentroid[0], pcaCentroid[1],pcaCentroid[2]), 0.005, 0.5, 0.5, 0.0, "sphere"+std::to_string(count));
        fl_clouds.push_back(cloud_hull);
        count++;
    }
    hullViewer->spinOnce (10);
    return fl_clouds;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr PtCloudClass::calculateHull(pcl::PointCloud<pcl::PointXYZ>::Ptr CloudPtr){
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ConcaveHull<pcl::PointXYZ> chull;
    chull.setInputCloud (CloudPtr);
    chull.setAlpha (0.1);
    chull.reconstruct (*cloud_hull);
    return cloud_hull;  
}

std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> PtCloudClass::getGrasp(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds){
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> fl_clouds;
    graspViewer->removeAllPointClouds();
    graspViewer->removeAllShapes();
    int count = 0;
    for (auto CloudPtr : clouds){
        addCentroid(CloudPtr);
        float min_dis = FLT_MAX;
        int index_closest_point,index_opposite_point;

        for (std::size_t i = 0; i < (CloudPtr->points.size() - 1); ++i)
        {
            float dist_x = CloudPtr->points[(CloudPtr->points.size()-1)].x - CloudPtr->points[i].x;
            float dist_y = CloudPtr->points[(CloudPtr->points.size()-1)].y - CloudPtr->points[i].y;
            float dis = sqrt(dist_x*dist_x + dist_y*dist_y);

            if (dis < min_dis)
            {
                min_dis = dis;
                index_closest_point = i;
            }

        }
        pcl::PointXYZ mirrorpoint;
        mirrorpoint.x = (2*CloudPtr->points[(CloudPtr->points.size()-1)].x) - CloudPtr->points[index_closest_point].x;
        mirrorpoint.y = (2*CloudPtr->points[(CloudPtr->points.size()-1)].y) - CloudPtr->points[index_closest_point].y;
        
        for (std::size_t i = 0; i < (CloudPtr->points.size() - 1); ++i)
        {
            float dist_x = mirrorpoint.x - CloudPtr->points[i].x;
            float dist_y = mirrorpoint.y - CloudPtr->points[i].y;
            float dis = sqrt(dist_x*dist_x + dist_y*dist_y);

            if (dis < min_dis)
            {
                min_dis = dis;
                index_opposite_point = i;
            }
        }
        
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_vis (new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::copyPointCloud(*CloudPtr, *cloud_vis);

        for (std::size_t i = 0; i < cloud_vis->points.size(); ++i){
            cloud_vis->points[i].r = 255;
            cloud_vis->points[i].b = 255;
            cloud_vis->points[i].g = 255;
        }

        cloud_vis->points[index_closest_point].r = 255;
        cloud_vis->points[index_closest_point].b = 0;
        cloud_vis->points[index_closest_point].g = 0;

        cloud_vis->points[cloud_vis->points.size()-1].r = 0;
        cloud_vis->points[cloud_vis->points.size()-1].g = 0;
        cloud_vis->points[cloud_vis->points.size()-1].b = 255;

        cloud_vis->points[index_opposite_point].r = 255;
        cloud_vis->points[index_opposite_point].b = 0;
        cloud_vis->points[index_opposite_point].g = 0;
        // addMarker(count, cloud_vis->points[index_closest_point].x, cloud_vis->points[index_closest_point].y, cloud_vis->points[index_closest_point].z, cloud_vis->points[index_opposite_point].x, cloud_vis->points[index_opposite_point].y, cloud_vis->points[index_opposite_point].z);
        graspViewer->addPointCloud<pcl::PointXYZRGB> (cloud_vis, "cloud_"+std::to_string(count));
        graspViewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud_"+std::to_string(count));
        fl_clouds.push_back(cloud_vis);
        count++;
    }
    graspViewer->spinOnce (10);
    std::cout << "getGrasp finished!" << std::endl;
    return fl_clouds;
    // return cloud_vis;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "grasp_synthesis_node");
  ros::NodeHandle n;
  PtCloudClass PtCloudObj(n);
  return 0;
}