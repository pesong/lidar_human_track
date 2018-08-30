#ifndef __Human_ReadParams
#define __Human_ReadParams
#include <tuple>
#include <vector>
#include <geometry_msgs/Quaternion.h>
#include <cv.h>
#include "score_map.h"
#include <array>
#include <fstream>
#include <jsoncpp/json/json.h> 
#include <autoscrubber_services/MovebaseGoal.h>
#include "track_utils.h"
#include "autoscrubber_services/StringArray.h"
#include <std_msgs/Bool.h>
#include <autoscrubber_services/Resume.h>
#include <autoscrubber_services/Pause.h>
#include <autoscrubber_services/Skip.h>
#include <autoscrubber_services/Detect.h>
#include "std_srvs/SetBool.h"

//ros::Publisher pos_pub, watch_pose;
geometry_msgs::Quaternion &greetOrientation();
geometry_msgs::Quaternion &oriDirection();
std::tuple<double,double> &Origin();
std::vector<std::tuple<double,double> > &Poly();
std::vector<std::tuple<double, double>> &Door();
std::vector<std::tuple<double, double>> &doorPoly();
std::vector<std::tuple<double, double>> &Poly_wall();
std::vector<double> &Importance();
bool &PauseFlag();
bool &ResumeFlag();
bool &skipFlag();
bool &detectFlag();
bool pauseService(autoscrubber_services::Pause::Request &req, autoscrubber_services::Pause::Response &res);
bool resumeService(autoscrubber_services::Resume::Request &req, autoscrubber_services::Resume::Response &res);
bool skipService(autoscrubber_services::Skip::Request &req, autoscrubber_services::Skip::Response &res);
bool detectService(autoscrubber_services::Detect::Request &req, autoscrubber_services::Detect::Response &res);
double &Angle();
void paramCallback(const autoscrubber_services::StringArray &param_title);
void getParam(std::vector<std::tuple<double, double>> door, std::tuple<double,double> origin, geometry_msgs::Quaternion greet, std::vector<std::tuple<double,double>> poly, std::vector<std::tuple<double,double>> poly_wall, double angle, int imp);
bool PtInPolygon (std::tuple<double,double> p, std::vector<std::tuple<double,double> > ptPolygon, int nCount);
bool init_range(std::tuple<double, double> p);//TODO
void read_json();
void robot_initializing(double &x, double &y, geometry_msgs::Quaternion &v, ros::Publisher pos_pub, ros::Publisher watch_pose);
double distToRobotPose(double x, double y);
double distToDoorPose(double x, double y);
void imp_analyzing(int imp);
bool PtInPolygon(std::tuple<double, double> p, std::vector<std::tuple<double, double>> ptPolygon);

#endif