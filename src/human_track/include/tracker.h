#ifndef TRACKER_H
#define TRACKER_H


#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/LaserScan.h>
#include <visualization_msgs/MarkerArray.h>
#include <boost/bind.hpp>
#include <human.hpp>
#include <observation.hpp>
#include <object_tracking.hpp>
#include <recognizer.hpp>
#include <header.hpp>
#include <human_track/ClusterClass.h>
#include <clusterFrame.h>

#include "HumanTrack.h"
#include "laser.h"


using std_msgs::ColorRGBA;
using cv::HOGDescriptor;
struct fb_model {
    float x, y, v_x, v_y, length;
    int id;
};

class tracker {
public:
    tracker();

    ~tracker();

    void setTheta(double theta);

    void extractModels(std::deque<fb_model> &models,std::vector<int> &clust_id_tb,std::vector<int> &clust_id_st);

    void update(boost::shared_ptr<clusterFrame>frame_);

    void publish(string laser_frame_id);

    void initColor();

    void setMatrix(tf::Matrix3x3 x3, tf::Matrix3x3 icp_base);

    void setLaser(double d);

protected:
    cv_bridge::CvImagePtr output_cv_ptr;
private:
    ros::NodeHandle nh;
    std::deque<cv::Scalar> colors;
    ros::Publisher rviz_pub_;
    ros::Publisher rviz_pub_clu_;
    cv::Mat outputImage;
    ros::Publisher humans_pub_;
   // ros::Publisher cluster_pub;
    std::deque<HumanCluster> humans;
    cv::HOGDescriptor hog;
    cv::CascadeClassifier face;

    image_transport::ImageTransport it_;
    image_transport::Publisher it_pub_;

    //void fusion(sensor_msgs::Image::ConstPtr image_);

    std_msgs::ColorRGBA getColorRgba(int index, float a);

    uint getId(int index, int category);

    std::string toString(float num);

    cv::Scalar getColor(int index);

    /// Camera matrix
    cv::Mat K;
    /// Distortion coeffs
    cv::Mat D;
    double theta;
    std::map<int, int> pairs;
    enum Rviz {
        RVIZ_NONE = 0,
        RVIZ_HUMAN,
        RVIZ_ID,
        RVIZ_LINE,
        RVIZ_SCAN,
        RVIZ_TOTAL
    };

    bool projected(HumanCluster &human, int &x, int &y);

    int image_height;
    int image_width;


    cv::Rect getBound(HumanCluster &human, int &x, int &y);

    double icp_odom_x, icp_pose_x;
    double icp_odom_y, icp_pose_y;
    double icp_odom_sin, icp_sin_;
    double icp_odom_cos, icp_cos_;
    double laser_x;

};


//
// Created by song on 16-4-9.
//

#endif //HDETECT_TRACKER_H
