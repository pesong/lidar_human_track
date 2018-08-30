#ifndef _HUMANTRACK
#define _HumanTRACK

#include <std_msgs/Float32MultiArray.h>
#include "track_utils.h"

//#include "laser.h"

//costmap & scoremap
cv::Mat& globalCostMap();
cv::Mat& globalScoreMap();
cv::Mat& globalCostMapProto();
cv::Mat& globalScoreMapProto();
cv::Mat& globalOrientMap();

// the map information from costmap occupanygrid message
int& mapWidth();
int& mapHeight();
double& mapResolution();
double& mapOriginX();
double& mapOriginY();
double& mapOriginZ();
double& mapOrientX();
double& mapOrientY();
double& mapOrientZ();
double& mapOrientW();

std::string toString(float num);
std_msgs::Float32MultiArray& clusterMsg();
std_msgs::Float32MultiArray icp2dMsg();
sensor_msgs::LaserScan::ConstPtr &laserMessage();
 
std::vector<cv::Rect>& yoloHumans();
std::vector<cv::Rect>& laserHumans();




#endif