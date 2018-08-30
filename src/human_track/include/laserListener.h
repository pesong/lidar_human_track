
#ifndef HDETECT_LASERLISTENER_H
#define HDETECT_LASERLISTENER_H

#include <csm/csm_all.h>
#include <message_filters/subscriber.h>
#include <std_msgs/Float32.h>
#include <human_track/ClusteredScan.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/Image.h>
#include <newmat/newmatap.h>
#include <boost/bind.hpp>
#include "lgeometry.hpp"
#include <recognizer.hpp>
#include <header.hpp>
#include <laserLib.hpp>
#include <human_track/ClusterClass.h>
#include <tracker.h>
#include "std_msgs/Float32MultiArray.h"
#include <cv_bridge/cv_bridge.h>

#include "HumanTrack.h"
#include "laser.h"

using namespace Header;

tf::Matrix3x3 &Odom_Amcl();
//double &laserX();


class label_finder  
{  
public:  
    label_finder( int l ) : label(l) {}  
    bool operator () (const pair<int,int> paired)  
    {  
        return paired.second == label;  
    }  
private:  
    int label;  
};

class index_finder  
{  
public:  
    index_finder( int l ) : index(l) {}  
    bool operator () (const pair<int,int> paired)  
    {  
        return paired.first == index;  
    }  
private:  
    int index;  
};


class laserListener {
public:
    laserListener(tracker *trk, std::string frame_name);

    float getTimestamp();

    ~laserListener();

    void laserFilter_tb(std::vector<Point3D_container> &clusters);

    void laserFilter_st(std::vector<Point3D_container> &clusters);

    void laserCallback_(const sensor_msgs::LaserScan::ConstPtr &message);

    //void laserInvoke(const sensor_msgs::LaserScan::ConstPtr &message);

    //void syncCallback(const sensor_msgs::LaserScan::ConstPtr &message, const geometry_msgs::PoseWithCovarianceStamped &pose);
    void syncCallback(const sensor_msgs::LaserScan::ConstPtr &message, const geometry_msgs::PoseWithCovarianceStamped::ConstPtr &pose);
    
    // void mapCallback(const sensor_msgs::Image::ConstPtr &scoremap_);
    
    // void mapInfoCallback(const std_msgs::Float32MultiArray &map_msg);

    //void laserXCallback(const std_msgs::Float32 &laserX_msg);

    void poseCallback(const nav_msgs::Odometry::ConstPtr &pose);
   // void amclInvoke(const geometry_msgs::PoseWithCovarianceStamped &pose);
    void amclCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr &pose);

    double wrongDistribution(std::vector<Point3D_str> &pts, Point3D_str &center);
    void Laser2Amcl(tf::Matrix3x3 odom_amcl, double xi, double yi, double &xo, double &yo);

    bool ToMap(tf::Matrix3x3 odom_amcl, double xi, double yi, double &xoo, double &yoo);


    //sensor_msgs::LaserScan::ConstPtr _message;

    void plicpInit();
    bool isMapInit; 
    bool isMapInfo; 
    ros::Publisher rviz_points_pub_;
    

private:
    //filters::FilterChain<sensor_msgs::LaserScan> laserFilter;

    ///CSM
    bool LaserInit;

    sensor_msgs::LaserScan::ConstPtr last_message;
    geometry_msgs::PoseWithCovarianceStamped::ConstPtr last_pose;

    sm_params m_input_;      // the input scan structure

    sm_result m_output_;       // the final output structure

    double plicp_min_reading, plicp_max_reading, plicp_angular_correction, plicp_linear_correction;
    LDP m_previous_ldp_scan_;


    /// The filtered scan
    sensor_msgs::LaserScan message;
    std::list<sensor_msgs::LaserScan> message_list;
    std::list<std::vector<Point3D_container> > cluster_list;
    filters::MultiChannelFilterChain<float> *range_filter_;
    //接受map
    cv::Mat scoremap;
    

    //sensor_msgs::LaserScan filtScan;
    ros::NodeHandle nh;
    sensor_msgs::Image::ConstPtr image_input;
    // The adaboost detector for the laser
    cv::Ptr<cv::ml::Boost> boost;
    std::deque<fb_model> feed_back;
    // The laser feature matrix
    cv::Mat lFeatures;
    lengine *libEngine;
    std::vector<std::vector<float> > descriptor;
    double laser_x;
    int window_, neighbors_;
    double max_angle_, min_angle_;
    /// Subsciber to the laser scan topic
    /// Subsciber to the laser scan topic
    //message_filters::Subscriber <sensor_msgs::LaserScan> *laserScan_sub_;
    ros::Subscriber *laserScan_sub_;
    //message_filters::Subscriber <
    tracker *tracker_;
    std::deque<Observation> observations;
    boost::shared_ptr<clusterFrame> Frame;
    std::string laserTopic;
    std::string laser_frame_id;
    tf::Matrix3x3 icp_base;

    double v_x, v_y, pose_x, pose_y, pose_theta, sin_, cos_;
    double icp_v_x, icp_v_y, icp_pose_x, icp_pose_y, icp_pose_theta, icp_sin_, icp_cos_;

    int frame_id;

    void clusterFeature(std::vector<Point3D_container> &clusters);

    void segmentMerge(std::vector<Point3D_container> &segments, int &label);

    void segmentFilter(std::vector<Point3D_container> &segments);

    int segmentClustering(std::vector <Point3D_container> &segments);

    void processModels(float deltaT);

    void LaserShow(sensor_msgs::LaserScan::ConstPtr message_, geometry_msgs::PoseWithCovarianceStamped::ConstPtr pose);


    tf::Matrix3x3 icp_odom;

    ///// Configurations for segmenting and clustering /////

    double segment_dist_thres, cluster_dist_thres, max_seg_length, max_density, min_dist, max_dist;
    float min_model_dist, max_model_dist;
    int min_seg_size, max_seg_size, min_clu_size, max_clu_size;

    ///////////////////////////////////////////////////////////

    double v_theta;
    bool plicp_init;

    void laserScanToLDP(const sensor_msgs::LaserScan &scan, LDP &ldp);

    double icp_delta_y;
    double icp_delta_x;
    bool odom_received;
    double icp_delta_cos_;
    double icp_delta_sin_;
    double icp_delta_theta;
    vector<int> clust_id_st, clust_id_tb;
    ros::Publisher st_laser_pub_;
    ros::Publisher tb_laser_pub_;
    ros::Publisher icp_pose_2d_pub_;
    ros::Publisher laser_filt_pub_;
    ros::Publisher laser_sync;
    //ros::Publisher amcl_sync;
    bool use_image_;
    bool tb_filter;
    bool st_filter;
};//
// Created by song on 16-4-9.
//



#endif //HDETECT_LASERLISTENER_H
