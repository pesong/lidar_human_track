#ifndef OBSERVATION_HPP
#define OBSERVATION_HPP

#include <opencv2/highgui/highgui.hpp>
#include <geometry_msgs/Point32.h>

#include <newmat/newmat.h>
#include <stdio.h>

class Observation {
public:
    Observation(double x, double y, int scan_index, float prob, bool camera_detected, geometry_msgs::Point32 &pos, double dist,
                double clust_size, double density, double pca, double length);

    Observation(int scan_index, float prob, bool camera_detected, geometry_msgs::Point32 &pos);

    Observation(int scan_index, float prob, bool camera_detected, geometry_msgs::Point32 &pos, cv::Mat &image,
                cv::Rect &rect);

    int scan_index;

    float prob, base_x, base_y;
    double pca, length, density, size, distance,shape;
    bool camera_detected;

    NEWMAT::ColumnVector state;

    cv::Mat image;
    cv::Rect rect;
};

#endif // OBSERVATION_HPP
