#include <observation.hpp>


//using namespace cv;
using cv::Mat;
using cv::Rect;
//using namespace NEWMAT;
using NEWMAT::ColumnVector;

Observation::Observation(int scan_index, float prob, bool camera_detected, geometry_msgs::Point32 &pos){
    this->scan_index = scan_index;

    this->prob = prob;
    this->camera_detected = camera_detected;

    this->state = ColumnVector(4);
    this->state << pos.x <<
    pos.y <<
    0 <<
    0;
}
Observation::Observation(double x, double y, int scan_index, float prob, bool camera_detected, geometry_msgs::Point32 &pos, double dist,double clust_size,double density,double pca, double length)
{
    this->scan_index = scan_index;
    this->base_x = x;
    this->base_y = y;
    this->prob = prob;
    this->distance = dist;
    this->density = density;
    this->pca = pca;
    this->size = clust_size;
    this->length = length;
    this->camera_detected = camera_detected;
    shape = clust_size+length+density+dist;
    this->state = ColumnVector(4);
    this->state << pos.x <<
                   pos.y <<
                   0 <<
                   0;
}

Observation::Observation(int scan_index, float prob, bool camera_detected, geometry_msgs::Point32 &pos, Mat &image, Rect &rect)
{
    this->scan_index = scan_index;

    this->prob = prob;

    this->camera_detected = camera_detected;

    this->state = ColumnVector(4);
    this->state << pos.x <<
                   pos.y <<
                   0 <<
                   0;

    this->image = image;
    this->rect = rect;
}
