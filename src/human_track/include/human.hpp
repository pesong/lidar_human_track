#ifndef HUMANCLUSTER_HPP
#define HUMANCLUSTER_HPP

#include <newmat/newmat.h>
#include <geometry_msgs/Point.h>
#include <stdio.h>
#include <ros/ros.h>


class HumanCluster{
public:
    int id, pair_, observe_, clust_id;
    list<int> clust_id_list;
    bool validated, initialized, isHuman, paired;

    int detect;
    float score, base_x, base_y;
    float init_x, init_y, prev_x, prev_y, prv2_x, prv2_y;
    NEWMAT::ColumnVector state;
    NEWMAT::Matrix cov;

    NEWMAT::ColumnVector preState;
    float preTimestamp;
    double shape, pca, length, distance;

    bool validate();

    // Variable for scoring process
    ros::Time firstTimestamp; // First detection time

    HumanCluster(int id, float score, NEWMAT::ColumnVector state, NEWMAT::Matrix cov, int preTimestamp);

    HumanCluster(double x, double y, int id, float score, NEWMAT::ColumnVector state, NEWMAT::Matrix cov, int preTimestamp, double pca, double shape, double length, double distance, float prob, int clust_id);

    geometry_msgs::Point toPoint();

    float prob;

///////////////////////////////////////////////////////////////////
//

};

#endif // HUMAN_HPP
