#ifndef __ScoreMap
#define __ScoreMap

#include <cv.h>
#include <highgui.h>
#include "std_msgs/String.h"
#include <costmap_2d/costmap_2d_publisher.h>
#include <costmap_2d/cost_values.h>
#include <malloc.h>
#include <iostream>
#include <cv_bridge/cv_bridge.h> 
#include "HumanTrack.h"

bool projectToMap(double xi, double yi, double &xo, double &yo);

void pasteRoi(cv::Mat &full, cv::Mat &crop, cv::Rect &roi);
cv::Mat transCostMap(cv::Mat &signed_map);
cv::Mat transScoreMap(cv::Mat &costmap);

bool& isMapInited();
void protoMapCallback(const nav_msgs::OccupancyGrid & map);
void updateMapCallback(const map_msgs::OccupancyGridUpdate & update);


#endif
