#ifndef HTRACK_TRACKUTILS
#define HTRACK_TRACKUTILS

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
#include <tf/transform_datatypes.h>
#include <geometry_msgs/PolygonStamped.h>

//#include　<tf/transform_broadcaster.h>

double& icpPoseX();
double& icpPoseY();
double& icpPoseSin();
double& icpPoseCos();
double& amclPoseX();
double& amclPoseY();
double& amclPoseSin();
double& amclPoseCos();
double& robotPoseX();
double& robotPoseY();
double& robotPoseZ();
double& robotVel();
double& robot_Theta();
double& robotYaw();
bool& useGreet();
double& laserX(); 
bool& ret(); //全局reset标志位
std::tuple<double,double>  &robotPose();
int& MODE();
int& free_MODE();
bool &greet_track_MODE();
int &detectAngle();
bool &use_camera();

double getTimestamp();
sensor_msgs::LaserScan & message_pub();
geometry_msgs::Quaternion& robotOrientation();

std::vector<std::tuple<double, double>> &Poly_robot();
void robotShapeCallback(const geometry_msgs::PolygonStamped &robotshape_msg);

//把world平面转到amcl平面
void ToAmcl (double xi, double yi, double &xo, double &yo);
//void Toangle(geometry_msgs::Quaternion quat, double &roll, double &pitch, double &yaw);
void htrack_poseCallback(std_msgs::Float32MultiArray & icp2d_msg);


void Toangle(geometry_msgs::Quaternion quat, double &roll, double &pitch, double &yaw);

//void robotCallback(const geometry_msgs::PoseWithCovarianceStamped & robot_msg);
void odomCallback(const nav_msgs::Odometry::ConstPtr &pose);

//计算人与机器人的夹角
double angle2robot(double humanx, double humany, double robotx, double roboty);

double disTwoPoints(double x1, double y1, double x2, double y2);


enum ROBOT_STATE
{
    INACTIVE = 0,
    ACTIVE_R,
    TRACK,
    LOST,
    STAY,
    STAY_SEARCH
};

enum ROBOT_SHOW_STATE
{
    FREE = 0,
    GOODBYE,
    TRACK_FAR,
    TRACK_NEAR,
    TRACK_STAY,
    TRACK_BACK,
    OUT,
    TRACKER_TRACK,
    TRACKER_FIND,
    TRACKER_LOST
};


#endif