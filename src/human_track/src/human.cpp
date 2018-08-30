#include <human.hpp>


//using namespace NEWMAT;
using NEWMAT::Matrix;
using NEWMAT::ColumnVector;

using geometry_msgs::Point;

HumanCluster::HumanCluster(double x, double y, int id, float score, ColumnVector state, Matrix cov, int preTimestamp, double pca,
             double shape, double length, double distance, float prob, int clust_id) {
    this->id = id;
    this->pair_ = 0;
    this->base_x = x;
    this->base_y = y;
    this->observe_ = 0;
    this->validated = true;
    this->initialized = false;
    this->score = score;
    this->init_x = state(1);
    this->init_y = state(2);
    this->length = length;
    prev_x = init_x;
    prev_y = init_y;
    this->state = state;
    this->cov = cov;
    this->shape = shape;
    this->pca = pca;
    this->distance = distance;
    this->preState = state;
    this->preTimestamp = preTimestamp;
    this->firstTimestamp = ros::Time::now();
    this->prob = prob;
    this->detect = 0;
    this->clust_id = clust_id;
    if (prob > 0.97) {
        this->initialized = true;
        if (prob > 0.99)
            this->isHuman = true;
    }
    //this->isHuman = false;
}

HumanCluster::HumanCluster(int id, float score, ColumnVector state, Matrix cov, int preTimestamp) {
    this->id = id;
    this->pair_ = 0;
    this->observe_ = 0;
    this->validated = true;
    this->initialized = false;
    this->score = score;
    this->init_x = state(1);
    this->init_y = state(2);
    prev_x = init_x;
    prev_y = init_y;
    this->state = state;
    this->cov = cov;
    this->preState = state;
    this->preTimestamp = preTimestamp;
    this->firstTimestamp = ros::Time::now();

}

bool HumanCluster::validate() {
    if (observe_ == 0)
        return true;
    if (observe_ < 4) {
        if (observe_ != pair_) {
            validated = false;
            return false;
        } else
            return true;
    }
    else if (observe_ < 16) {
        //ROS_ERROR("DIS:%lf", (state(1) - init_x) * (state(1) - init_x) + (state(2) - init_y) * (state(2) - init_y));
        if ((state(1) - init_x) * (state(1) - init_x) + (state(2) - init_y) * (state(2) - init_y) > 0.45) {
            initialized = true;
            ROS_ERROR("INITIALIZED!");

        }
        return true;
    }
    validated = false;
    return false;
}

Point HumanCluster::toPoint() {
    Point pos;

    pos.x = (state(1) + prev_x + prv2_x) / 3 - (state(1) - prv2_x) * 0.06 - (state(1) - prev_x) * 0.1;
    pos.y = (state(2) + prev_y + prv2_y) / 3 - (state(2) - prv2_y) * 0.06 - (state(2) - prev_y) * 0.1;
    pos.z = 0;

    return pos;
}
