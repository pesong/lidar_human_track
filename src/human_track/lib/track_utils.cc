#include "track_utils.h"

//把四元素转化成roll,pitch,yaw,角度是 0-3.14，-3.14-0
void Toangle(geometry_msgs::Quaternion quat, double &roll, double &pitch, double &yaw)
{
    tf::Quaternion q((quat.x), (quat.y), (quat.z), (quat.w));
    tf::Matrix3x3 m(q);
    m.getRPY(roll, pitch, yaw);
}

int &MODE()
{
    static int v;
    return v;
}

int &free_MODE()
{
    static int v;
    return v;
}

bool &greet_track_MODE()
{
    static bool v;
    return v;
}

bool &use_camera()
{
    static bool v;
    return v;
}

int &detectAngle()
{
    static int v;
    return v;
}


double &icpPoseX()
{
    static double v;
    return v;
}
double &icpPoseY()
{
    static double v;
    return v;
}
double &icpPoseSin()
{
    static double v;
    return v;
}
double &icpPoseCos()
{
    static double v;
    return v;
}
double &amclPoseX()
{
    static double v;
    return v;
}
double &amclPoseY()
{
    static double v;
    return v;
}
double &amclPoseSin()
{
    static double v;
    return v;
}
double &amclPoseCos()
{
    static double v;
    return v;
}
double &robotPoseX()
{
    static double v;
    return v;
}
double &robotPoseY()
{
    static double v;
    return v;
}
double &robotPoseZ()
{
    static double v;
    return v;
}
double &robotVel()
{
    static double v;
    return v;
}

double &robot_Theta()
{
    static double v;
    return v;
}

double &laserX()
{
    static double v;
    return v;
}


double &robotYaw()
{
    static double v;
    return v;
}

geometry_msgs::Quaternion &robotOrientation()
{
    static geometry_msgs::Quaternion v;
    return v;
}

sensor_msgs::LaserScan &message_pub()
{
    static sensor_msgs::LaserScan l;
    return l;
}

std::tuple<double, double> &robotPose()
{
    static std::tuple<double, double> v;
    return v;
}

bool &useGreet()
{
    static bool v;
    return v;
}

bool &ret()
{
    static bool v;
    return v;
}


double angle2robot(double humanx, double humany, double robotx, double roboty)
{
    return cv::fastAtan2((roboty - humany), (robotx - humanx)) * 0.017453292519943;
}

double getTimestamp()
{
    timeval t;
    gettimeofday(&t, NULL);
    static int constSec = t.tv_sec;
    return t.tv_sec - constSec + (double)t.tv_usec / 1000000;  
}


//get the amcl pose 接收icp 坐标系下的二维位姿
void htrack_poseCallback(std_msgs::Float32MultiArray &icp2d_msg)
{
   // double time_seq;
    //icp has the true orient & odom amcl has the true offset
    amclPoseX() = icp2d_msg.data[0];   //x
    amclPoseY() = icp2d_msg.data[1];   //y
    amclPoseSin() = icp2d_msg.data[2]; //sin
    amclPoseCos() = icp2d_msg.data[3]; //cos
    icpPoseX() = icp2d_msg.data[4];   //x
    icpPoseY() = icp2d_msg.data[5];   //y
    icpPoseSin() = icp2d_msg.data[6]; //sin
    icpPoseCos() = icp2d_msg.data[7]; //cos
    //std::cout << "htrack poseCallback amclPose: " << amclPoseX() << std::endl;
    //time_seq = icp2d_msg.data[8];
    //std::cout << "****callback seq: " << time_seq << std::endl;
}
// //把四元素转化成roll,pitch,yaw,角度是 0-3.14，-3.14-0
// void Toangle(geometry_msgs::Quaternion quat, double &roll, double &pitch, double &yaw)
// {
//     tf::Quaternion q((quat.x), (quat.y), (quat.z), (quat.w));
//     tf::Matrix3x3 m(q);
//     m.getRPY(roll, pitch, yaw);
// }

// //接受机器人现在的位置
// void robotCallback(const geometry_msgs::PoseWithCovarianceStamped &robot_msg)
// {
//     robotPoseX() = robot_msg.pose.pose.position.x;
//     robotPoseY() = robot_msg.pose.pose.position.y;
//     robotPoseZ() = robot_msg.pose.pose.position.z;
//     robotOrientation() = robot_msg.pose.pose.orientation;
//     robotPose() = std::make_tuple(robotPoseX(), robotPoseY());

//     double r, p, y;
//     Toangle(robotOrientation(), r, p, y);
//     robotYaw() = y;

//     // std::cout << " yaw: " << y << std::endl;
//     //std::cout << " z:" << robotOrientation().z << " w: " << robotOrientation().w << std::endl;
//     // ROS_INFO("robotPoseX(): %f, robotPoseY(): %f",robotPoseX(),robotPoseY());
//     // ROS_INFO("z: %f, w: %f", robotOrientation().z, robotOrientation().w);
// }

double disTwoPoints(double x1, double y1, double x2, double y2)
{
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
};


void odomCallback(const nav_msgs::Odometry::ConstPtr &pose)
{
   // ROS_INFO("odomCallback");
    double pose_x, pose_y, v_x, v_y, v_theta;
    // pose_x = pose->pose.pose.position.x;
    // pose_y = pose->pose.pose.position.y;
    v_x = pose->twist.twist.linear.x;
    // v_y = pose->twist.twist.linear.y;
    v_theta = pose->twist.twist.angular.z;
    //ROS_WARN("v_x: %f, v_y: %f, v_theta: %f", v_x,v_y,v_theta);
    robotVel() = v_x;
    robot_Theta() = v_theta;
}

//把world平面转到amcl平面
void ToAmcl(double xi, double yi, double &xo, double &yo)
{
    double pose_cos = icpPoseCos();
    double pose_sin = icpPoseSin();
    double pose_x = amclPoseX();
    double pose_y = amclPoseY();
    double amcl_pos_sin = amclPoseSin();
    double amcl_pos_cos = amclPoseCos();
    double pub_xx, pub_yy;
    pub_xx = pose_cos * (xi - pose_x) + pose_sin * (yi - pose_y);
    pub_yy = pose_cos * (yi - pose_y) - pose_sin * (xi - pose_x);
    xo = amcl_pos_cos * pub_xx - amcl_pos_sin * pub_yy + pose_x;
    yo = amcl_pos_cos * pub_yy + amcl_pos_sin * pub_xx + pose_y;
};

std::vector<std::tuple<double, double>> &Poly_robot()
{
    static std::vector<std::tuple<double, double>> v;
    return v;
}

void robotShapeCallback(const geometry_msgs::PolygonStamped &robotshape_msg)
{
   // ROS_INFO("IN robotShape");
    int count = robotshape_msg.polygon.points.size();
    std::vector<std::tuple<double, double>> v;
    for(int i=0; i<count; i++)
    {
        v.push_back(std::make_tuple(robotshape_msg.polygon.points[i].x, robotshape_msg.polygon.points[i].y));
        //std::cout << "robotshape.x: " << robotshape_msg.polygon.points[i].x << " robotshape.y: " << robotshape_msg.polygon.points[i].y << std::endl;
    }
    Poly_robot() = v;
    
};
