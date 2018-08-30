#include "HumanTrack.h"

std_msgs::Float32MultiArray& clusterMsg(){static std_msgs::Float32MultiArray v; return v;}
std_msgs::Float32MultiArray icp2dMsg(){static std_msgs::Float32MultiArray v; return v;}

sensor_msgs::LaserScan::ConstPtr &laserMessage()
{
    static sensor_msgs::LaserScan::ConstPtr v;
    return v;
}

std::string toString(float num)
{
    std::ostringstream buf;

    buf << num;

    return buf.str();
}

std::vector<cv::Rect>& yoloHumans()
{
    static std::vector<cv::Rect> v;
    return v;
}

std::vector<cv::Rect>& laserHumans()
{
    static std::vector<cv::Rect> v;
    return v;
}


//原始的接收的costmap数据
cv::Mat& globalCostMap(){static cv::Mat v; return v;}

//处理后的map数据
cv::Mat& globalScoreMap(){static cv::Mat v; return v;}

//第一次接收的costmap数据
cv::Mat& globalCostMapProto(){static cv::Mat v; return v;}

//第一次处理后的map数据
cv::Mat& globalScoreMapProto(){static cv::Mat v; return v;}

//第一次处理后的map数据的梯度方向
cv::Mat& globalOrientMap(){static cv::Mat v; return v;}

// the map information from costmap occupanygrid message
int& mapWidth(){static int v; return v;}
int& mapHeight(){static int v; return v;}
double& mapResolution(){static double v; return v;}
double& mapOriginX(){static double v; return v;}
double& mapOriginY(){static double v; return v;}
double& mapOriginZ(){static double v; return v;}
double& mapOrientX(){static double v; return v;}
double& mapOrientY(){static double v; return v;}
double& mapOrientZ(){static double v; return v;}
double& mapOrientW(){static double v; return v;}