#ifndef __Laser_tangwb
#define __Laser_tangwb

#include <iostream>
#include <sensor_msgs/LaserScan.h> 
#include <std_msgs/Float32MultiArray.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <cv.h>
#include <highgui.h>
#include <vector>
#include <map>
#include <time.h>
#include <math.h>
#include <string>
#include <tuple>
#include <nav_msgs/Odometry.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv/cv.hpp>


#include "score_map.h"
#include "track_utils.h"
#include "human.h"
#include "HumanTrack.h"
#include "connect.h"

template<typename T>
inline double distance_l2(T x1, T y1, T x2, T y2)
{	
	double dx = x1-x2;
	double dy = y1-y2;
	return sqrt(dx*dx+dy*dy);
};

template<typename T>
inline T distance_l1(T x1, T y1, T x2, T y2)
{	
	return fabs(x1-x2)+fabs(y1-y2);
};


// double& icpPoseX();
// double& icpPoseY();
// double& icpPoseSin();
// double& icpPoseCos();
// double& amclPoseX();
// double& amclPoseY();
// double& amclPoseSin();
// double& amclPoseCos();
// double& robotPoseX();
// double& robotPoseY();
// double& robotPoseZ();
// double& robotVel();
// double& robotYaw();
// std::tuple<double,double>  &robotPose();

// sensor_msgs::LaserScan & message_pub();
// geometry_msgs::Quaternion& robotOrientation();



// //把world平面转到amcl平面
// void ToAmcl (double xi, double yi, double &xo, double &yo);

std::vector<std::pair<double,double>> aroundLaser(std::vector<std::pair<double, double>> laser, double x_0, double y_0);

/* 
 inline void coorInvTrans(double xi, double yi, double &xo, double &yo,
                         double dx, double dy, double sin, double cos)
{
    xo = cos*(xi - dx) + sin*(yi - dy);
    yo = cos*(yi - dy) - sin*(xi - dx);
}
    auto theworld = [&](double xi, double yi, double &xo, double &yo)
    {
        xo = pose_cos * xi - pose_sin * yi + pose_x;
        yo = pose_cos * yi + pose_sin * xi + pose_y;
    };

*/
void clusterCallback(std_msgs::Float32MultiArray &cluster_msg);


std::vector<std::pair<double,double> >& humanCluster();
std::list<Human>& trackedHuman();
std::list<Human>& realHuman();

std::vector<std::pair<double,double> >& transedLaser();
void laserCallback(sensor_msgs::LaserScan::ConstPtr &message_);
std::list<Human>& trackLaser(std::vector<std::pair<double,double> > &new_humans, 
                  std::vector<std::pair<double,double> > &laser);

std::vector<std::pair<double,double> >& getCluster();
std::vector<std::pair<double,double> > getClusterClone();

//void poseCallback(const std_msgs::Float32MultiArray & icp2d_msg);

void robotCallback(const geometry_msgs::PoseWithCovarianceStamped & robot_msg);
//void odomCallback(const nav_msgs::Odometry::ConstPtr &pose);

void visualscoremap(std::list<Human> &tracked_huamns,
                             std::vector<std::pair<double, double>> &transed_laser);
//void printLaserpoints(std::vector<std::pair<double, double>> transed_laser);

struct ROBOT_STATES
{
    ROBOT_STATE first_state_;
    ROBOT_STATE second_state_;
    ROBOT_STATE robot_state_;
    ROBOT_SHOW_STATE robot_show_state_;
    ROBOT_SHOW_STATE robot_show_state_last;
};

struct TIME_PUB
{
    ros::Time pub_time_last;
    ros::Time pub_time_now;
};

struct TARGET_INFO
{
    Human target;
    int targetID;
    double tx; //amcl平面上target的x,y坐標
    double ty;
    std::vector<std::tuple<double, double, double, double, double>> predict_path;
};

#endif
