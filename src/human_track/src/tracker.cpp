#include <tracker.h>//
#include <boost/foreach.hpp>
#include <string>
#include <sstream>


using cv::Mat;
using cv::Scalar;
using cv::Rect;
#define RVIZ_ID_SIZE 1000

tracker::tracker() : it_(nh) {
    //ROS_INFO("tracker created");
    initColor();
    theta = 0;
    it_pub_ = it_.advertise("rectangular", 1);


    image_height = 480;
    image_width = 640;


    D = Mat::zeros(1, 5, CV_64FC1);
    K = Mat::zeros(3, 3, CV_64FC1);


    K.at<double>(0, 0) = 622.45996;
    K.at<double>(1, 1) = 628.57257;
    K.at<double>(0, 2) = 338.63428;
    K.at<double>(1, 2) = 257.66031;
    K.at<double>(2, 2) = 1;
    //hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    //face.load("/home/totti/pedestrain_track_reborn/haarcascade_frontalface_alt.xml");


    ObjectTracking::loadCfg("");

    rviz_pub_ = nh.advertise<visualization_msgs::MarkerArray>("markers", 1);
    rviz_pub_clu_ = nh.advertise<visualization_msgs::MarkerArray>("markers_clu", 1);
    humans_pub_ = nh.advertise<std_msgs::Float32MultiArray>("old_humans", 3);
    //cluster_pub = nh.advertise<std_msgs::Float32MultiArray>("cluster_position",3);
    output_cv_ptr.reset(new cv_bridge::CvImage);

    output_cv_ptr->encoding = "rgb8";

it_pub_ = it_.advertise("output_video", 1);
///////////////////////////////////////////////////////////
}

void tracker::initColor() {
    colors.push_back(Scalar(191, 0, 0));
    colors.push_back(Scalar(0, 191, 0));
    colors.push_back(Scalar(0, 0, 191));
    colors.push_back(Scalar(255, 127, 0));
    colors.push_back(Scalar(127, 255, 0));
    colors.push_back(Scalar(255, 0, 127));
    colors.push_back(Scalar(127, 0, 255));
    colors.push_back(Scalar(0, 255, 127));
    colors.push_back(Scalar(0, 127, 255));
    colors.push_back(Scalar(255, 255, 0));
    colors.push_back(Scalar(255, 0, 255));
    colors.push_back(Scalar(0, 255, 255));
    colors.push_back(Scalar(101, 0, 0));
    colors.push_back(Scalar(0, 101, 0));
    colors.push_back(Scalar(0, 0, 101));
    colors.push_back(Scalar(205, 127, 0));
    colors.push_back(Scalar(127, 205, 0));
    colors.push_back(Scalar(205, 0, 127));
    colors.push_back(Scalar(107, 0, 205));
    colors.push_back(Scalar(0, 205, 127));
    colors.push_back(Scalar(0, 127, 255));
    colors.push_back(Scalar(255, 25, 0));
    colors.push_back(Scalar(205, 0, 255));
    colors.push_back(Scalar(0, 205, 205));
    colors.push_back(Scalar(191, 0, 0));
    colors.push_back(Scalar(0, 191, 0));
    colors.push_back(Scalar(0, 0, 191));
    colors.push_back(Scalar(255, 127, 0));
    colors.push_back(Scalar(127, 255, 0));
    colors.push_back(Scalar(5, 0, 127));
    colors.push_back(Scalar(127, 0, 55));
    colors.push_back(Scalar(0, 75, 177));
    colors.push_back(Scalar(0, 127, 255));
    colors.push_back(Scalar(75, 125, 0));
    colors.push_back(Scalar(2, 0, 25));
    colors.push_back(Scalar(0, 125, 115));
    colors.push_back(Scalar(191, 0, 0));
    colors.push_back(Scalar(0, 191, 0));
    colors.push_back(Scalar(0, 0, 191));
    colors.push_back(Scalar(255, 127, 0));
    colors.push_back(Scalar(127, 255, 0));
    colors.push_back(Scalar(255, 0, 127));
    colors.push_back(Scalar(127, 0, 255));
    colors.push_back(Scalar(0, 255, 127));
    colors.push_back(Scalar(0, 127, 255));
    colors.push_back(Scalar(255, 255, 0));
    colors.push_back(Scalar(255, 0, 255));
    colors.push_back(Scalar(0, 255, 255));
    colors.push_back(Scalar(101, 0, 0));
    colors.push_back(Scalar(0, 101, 0));
    colors.push_back(Scalar(0, 0, 101));
    colors.push_back(Scalar(205, 127, 0));
    colors.push_back(Scalar(127, 205, 0));
    colors.push_back(Scalar(205, 0, 127));
    colors.push_back(Scalar(107, 0, 205));
    colors.push_back(Scalar(0, 205, 127));
    colors.push_back(Scalar(0, 127, 255));
    colors.push_back(Scalar(255, 25, 0));
    colors.push_back(Scalar(205, 0, 255));
    colors.push_back(Scalar(0, 205, 205));
    colors.push_back(Scalar(191, 0, 0));
    colors.push_back(Scalar(0, 191, 0));
    colors.push_back(Scalar(0, 0, 191));
    colors.push_back(Scalar(255, 127, 0));
    colors.push_back(Scalar(127, 255, 0));
    colors.push_back(Scalar(5, 0, 127));
    colors.push_back(Scalar(127, 0, 55));
    colors.push_back(Scalar(0, 75, 177));
    colors.push_back(Scalar(0, 127, 255));
    colors.push_back(Scalar(75, 125, 0));
    colors.push_back(Scalar(2, 0, 25));
    colors.push_back(Scalar(0, 125, 115));
    colors.push_back(Scalar(191, 0, 0));
    colors.push_back(Scalar(0, 191, 0));
    colors.push_back(Scalar(0, 0, 191));
    colors.push_back(Scalar(255, 127, 0));
    colors.push_back(Scalar(127, 255, 0));
    colors.push_back(Scalar(255, 0, 127));
    colors.push_back(Scalar(127, 0, 255));
    colors.push_back(Scalar(0, 255, 127));
    colors.push_back(Scalar(0, 127, 255));
    colors.push_back(Scalar(255, 255, 0));
    colors.push_back(Scalar(255, 0, 255));
    colors.push_back(Scalar(0, 255, 255));
    colors.push_back(Scalar(101, 0, 0));
    colors.push_back(Scalar(0, 101, 0));
    colors.push_back(Scalar(0, 0, 101));
    colors.push_back(Scalar(205, 127, 0));
    colors.push_back(Scalar(127, 205, 0));
    colors.push_back(Scalar(205, 0, 127));
    colors.push_back(Scalar(107, 0, 205));
    colors.push_back(Scalar(0, 205, 127));
    colors.push_back(Scalar(0, 127, 255));
    colors.push_back(Scalar(255, 25, 0));
    colors.push_back(Scalar(205, 0, 255));
    colors.push_back(Scalar(0, 205, 205));
    colors.push_back(Scalar(191, 0, 0));
    colors.push_back(Scalar(0, 191, 0));
    colors.push_back(Scalar(0, 0, 191));
    colors.push_back(Scalar(255, 127, 0));
    colors.push_back(Scalar(127, 255, 0));
    colors.push_back(Scalar(5, 0, 127));
    colors.push_back(Scalar(127, 0, 55));
    colors.push_back(Scalar(0, 75, 177));
    colors.push_back(Scalar(0, 127, 255));
    colors.push_back(Scalar(75, 125, 0));
    colors.push_back(Scalar(2, 0, 25));
    colors.push_back(Scalar(0, 125, 115));
    colors.push_back(Scalar(191, 0, 0));
    colors.push_back(Scalar(0, 191, 0));
    colors.push_back(Scalar(0, 0, 191));
    colors.push_back(Scalar(255, 127, 0));
    colors.push_back(Scalar(127, 255, 0));
    colors.push_back(Scalar(255, 0, 127));
    colors.push_back(Scalar(127, 0, 255));
    colors.push_back(Scalar(0, 255, 127));
    colors.push_back(Scalar(0, 127, 255));
    colors.push_back(Scalar(255, 255, 0));
    colors.push_back(Scalar(255, 0, 255));
    colors.push_back(Scalar(0, 255, 255));
    colors.push_back(Scalar(101, 0, 0));
    colors.push_back(Scalar(0, 101, 0));
    colors.push_back(Scalar(0, 0, 101));
    colors.push_back(Scalar(205, 127, 0));
    colors.push_back(Scalar(127, 205, 0));
    colors.push_back(Scalar(205, 0, 127));
    colors.push_back(Scalar(107, 0, 205));
    colors.push_back(Scalar(0, 205, 127));
    colors.push_back(Scalar(0, 127, 255));
    colors.push_back(Scalar(255, 25, 0));
    colors.push_back(Scalar(205, 0, 255));
    colors.push_back(Scalar(0, 205, 205));
    colors.push_back(Scalar(191, 0, 0));
    colors.push_back(Scalar(0, 191, 0));
    colors.push_back(Scalar(0, 0, 191));
    colors.push_back(Scalar(255, 127, 0));
    colors.push_back(Scalar(127, 255, 0));
    colors.push_back(Scalar(5, 0, 127));
    colors.push_back(Scalar(127, 0, 55));
    colors.push_back(Scalar(0, 75, 177));
    colors.push_back(Scalar(0, 127, 255));
    colors.push_back(Scalar(75, 125, 0));
    colors.push_back(Scalar(2, 0, 25));
    colors.push_back(Scalar(0, 125, 115));
    colors.push_back(Scalar(191, 0, 0));
    colors.push_back(Scalar(0, 191, 0));
    colors.push_back(Scalar(0, 0, 191));
    colors.push_back(Scalar(255, 127, 0));
    colors.push_back(Scalar(127, 255, 0));
    colors.push_back(Scalar(255, 0, 127));
    colors.push_back(Scalar(127, 0, 255));
    colors.push_back(Scalar(0, 255, 127));
    colors.push_back(Scalar(0, 127, 255));
    colors.push_back(Scalar(255, 255, 0));
    colors.push_back(Scalar(255, 0, 255));
    colors.push_back(Scalar(0, 255, 255));
    colors.push_back(Scalar(101, 0, 0));
    colors.push_back(Scalar(0, 101, 0));
    colors.push_back(Scalar(0, 0, 101));
    colors.push_back(Scalar(205, 127, 0));
    colors.push_back(Scalar(127, 205, 0));
    colors.push_back(Scalar(205, 0, 127));
    colors.push_back(Scalar(107, 0, 205));
    colors.push_back(Scalar(0, 205, 127));
    colors.push_back(Scalar(0, 127, 255));
    colors.push_back(Scalar(255, 25, 0));
    colors.push_back(Scalar(205, 0, 255));
    colors.push_back(Scalar(0, 205, 205));
    colors.push_back(Scalar(191, 0, 0));
    colors.push_back(Scalar(0, 191, 0));
    colors.push_back(Scalar(0, 0, 191));
    colors.push_back(Scalar(255, 127, 0));
    colors.push_back(Scalar(127, 255, 0));
    colors.push_back(Scalar(5, 0, 127));
    colors.push_back(Scalar(127, 0, 55));
    colors.push_back(Scalar(0, 75, 177));
    colors.push_back(Scalar(0, 127, 255));
    colors.push_back(Scalar(75, 125, 0));
    colors.push_back(Scalar(2, 0, 25));
    colors.push_back(Scalar(0, 125, 115));
    colors.push_back(Scalar(191, 0, 0));
    colors.push_back(Scalar(0, 191, 0));
    colors.push_back(Scalar(0, 0, 191));
    colors.push_back(Scalar(255, 127, 0));
    colors.push_back(Scalar(127, 255, 0));
    colors.push_back(Scalar(255, 0, 127));
    colors.push_back(Scalar(127, 0, 255));
    colors.push_back(Scalar(0, 255, 127));
    colors.push_back(Scalar(0, 127, 255));
    colors.push_back(Scalar(255, 255, 0));
    colors.push_back(Scalar(255, 0, 255));
    colors.push_back(Scalar(0, 255, 255));
    colors.push_back(Scalar(101, 0, 0));
    colors.push_back(Scalar(0, 101, 0));
    colors.push_back(Scalar(0, 0, 101));
    colors.push_back(Scalar(205, 127, 0));
    colors.push_back(Scalar(127, 205, 0));
    colors.push_back(Scalar(205, 0, 127));
    colors.push_back(Scalar(107, 0, 205));
    colors.push_back(Scalar(0, 205, 127));
    colors.push_back(Scalar(0, 127, 255));
    colors.push_back(Scalar(255, 25, 0));
    colors.push_back(Scalar(205, 0, 255));
    colors.push_back(Scalar(0, 205, 205));
    colors.push_back(Scalar(191, 0, 0));
    colors.push_back(Scalar(0, 191, 0));
    colors.push_back(Scalar(0, 0, 191));
    colors.push_back(Scalar(255, 127, 0));
    colors.push_back(Scalar(127, 255, 0));
    colors.push_back(Scalar(5, 0, 127));
    colors.push_back(Scalar(127, 0, 55));
    colors.push_back(Scalar(0, 75, 177));
    colors.push_back(Scalar(0, 127, 255));
    colors.push_back(Scalar(75, 125, 0));
    colors.push_back(Scalar(2, 0, 25));
    colors.push_back(Scalar(0, 125, 115));
    colors.push_back(Scalar(191, 0, 0));
    colors.push_back(Scalar(0, 191, 0));
    colors.push_back(Scalar(0, 0, 191));
    colors.push_back(Scalar(255, 127, 0));
    colors.push_back(Scalar(127, 255, 0));
    colors.push_back(Scalar(255, 0, 127));
    colors.push_back(Scalar(127, 0, 255));
    colors.push_back(Scalar(0, 255, 127));
    colors.push_back(Scalar(0, 127, 255));
    colors.push_back(Scalar(255, 255, 0));
    colors.push_back(Scalar(255, 0, 255));
    colors.push_back(Scalar(0, 255, 255));
    colors.push_back(Scalar(21, 15, 200));
    colors.push_back(Scalar(0, 161, 0));
    colors.push_back(Scalar(0, 0, 191));
    colors.push_back(Scalar(165, 127, 0));
    colors.push_back(Scalar(133, 205, 0));
    colors.push_back(Scalar(55, 0, 127));
    colors.push_back(Scalar(107, 0, 205));
    colors.push_back(Scalar(0, 205, 127));
    colors.push_back(Scalar(0, 127, 255));
    colors.push_back(Scalar(32, 25, 0));
    colors.push_back(Scalar(123, 0, 255));
    colors.push_back(Scalar(0, 205, 205));
    colors.push_back(Scalar(51, 0, 0));
    colors.push_back(Scalar(0, 31, 0));
    colors.push_back(Scalar(0, 0, 181));
    colors.push_back(Scalar(155, 27, 0));

}

uint tracker::getId(int index, int category) {
    return index % RVIZ_ID_SIZE + category * RVIZ_ID_SIZE;
}

Scalar tracker::getColor(int index) {
    return colors[index % colors.size()];
}

string tracker::toString(float num) {
    ostringstream buf;

    buf << num;

    return buf.str();
}

std_msgs::ColorRGBA tracker::getColorRgba(int index, float a) {
    ColorRGBA colorRgba;
    Scalar color = getColor(index);
    colorRgba.r = color[0] / 255.0;
    colorRgba.g = color[1] / 255.0;
    colorRgba.b = color[2] / 255.0;
    colorRgba.a = a;
    return colorRgba;
}

void tracker::publish(string laser_frame_id) {
    //ROS_INFO("publishing");

   // std_msgs::Float32MultiArray cluster_msg;
    std_msgs::Float32MultiArray humans_msg;
    int msg_size = humans.size();
    humans_msg.layout.data_offset = 0;
    humans_msg.data.resize(9 * msg_size);
    humans_msg.layout.dim.resize(2);
    humans_msg.layout.dim[0].label = "height";
    humans_msg.layout.dim[0].size = msg_size;
    humans_msg.layout.dim[0].stride = 9 * msg_size;

    humans_msg.layout.dim[1].label = "width";
    humans_msg.layout.dim[1].size = 9;
    humans_msg.layout.dim[1].stride = 9;

    visualization_msgs::Marker temp_human;
    visualization_msgs::Marker temp_id;
    visualization_msgs::Marker temp_line;
    visualization_msgs::Marker temp_scan;
    visualization_msgs::MarkerArray rviz_markers;

    temp_human.header.frame_id = "world";
    temp_human.header.stamp = ros::Time();
    temp_human.type = visualization_msgs::Marker::SPHERE;
    temp_human.action = visualization_msgs::Marker::ADD;
    temp_human.lifetime = ros::Duration(0.5);


    temp_id.header.frame_id = "world";
    temp_id.header.stamp = ros::Time();
    temp_id.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    temp_id.action = visualization_msgs::Marker::ADD;
    temp_id.lifetime = ros::Duration(0.5);

    temp_id.scale.x = 0.3;
    temp_id.scale.y = 0.3;
    temp_id.scale.z = 0.3;

    temp_id.color.r = 0.7;
    temp_id.color.g = 0.7;
    temp_id.color.b = 0.7;
    temp_id.color.a = 0.8;


    int j = 0;
    if(humans.size()>0){
    for (uint i = 0; i < humans.size(); i++, j++) {
        // if (!humans[i].initialized) {
        //     msg_size--;
        //     humans_msg.data.resize(9 * msg_size);
        //     humans_msg.layout.dim[0].size = msg_size;
        //     humans_msg.layout.dim[0].stride = 9 * msg_size;
        //     j--;
        //     continue;
        // }
        humans_msg.data[j * 9] =
                min((1 - humans[i].cov(1, 1)), 0.9) * min((1 - humans[i].cov(2, 2)), 0.8) + 0.1;
        humans_msg.data[j * 9 + 1] = humans[i].toPoint().x;
        humans_msg.data[j * 9 + 2] = humans[i].toPoint().y;
        humans_msg.data[j * 9 + 6] = humans[i].id;


        double v_x = humans[i].state(3);
        double v_y = humans[i].state(4);
        double v_ = sqrt(v_x * v_x + v_y * v_y);
        double theta_;
        double degree_;
        if (v_ < 0.1) {
            humans_msg.data[j * 9 + 3] = 0;
            humans_msg.data[j * 9 + 4] = 0;
        } else {

            humans_msg.data[j * 9 + 3] = v_;
            theta_ = v_x > 0 ? atan(v_y / v_x) - theta : atan(v_y / v_x) + 3.1415926 - theta;
            degree_ = theta_ / 3.1415926 * 180;
            while (degree_ < 0)
                degree_ += 360;
            while (degree_ > 360)
                degree_ -= 360;
            humans_msg.data[j * 9 + 4] = degree_;
        }
        humans_msg.data[j * 9 + 5] = humans[i].isHuman;
        float x_ = icp_cos_ * humans[i].toPoint().x - icp_sin_ * humans[i].toPoint().y + icp_pose_x;
        float y_ = icp_cos_ * humans[i].toPoint().y + icp_sin_ * humans[i].toPoint().x + icp_pose_y;
        humans_msg.data[j * 9 + 7] = sqrt(x_ * x_ + y_ * y_);
        theta_ = x_ > 0 ? atan(y_ / x_) : atan(y_ / x_) + 3.1415926;
        degree_ = theta_ / 3.1415926 * 180;
        while (degree_ < -180)
            degree_ += 360;
        while (degree_ > 180)
            degree_ -= 360;
        humans_msg.data[j * 9 + 8] = degree_;

        if (humans[i].cov(1, 1) > 0.4 || humans[i].cov(2, 2) > 0.4) {
            continue;
        }

        temp_human.id = getId(humans[i].id, RVIZ_HUMAN);

//        temp_human.pose.position = humans[i].toPoint();
        temp_human.pose.position.x =
                (icp_odom_cos * humans[i].toPoint().x - icp_odom_sin * humans[i].toPoint().y) + icp_odom_x;
        temp_human.pose.position.y =
                (icp_odom_cos * humans[i].toPoint().y + icp_odom_sin * humans[i].toPoint().x) + icp_odom_y;
        temp_human.pose.orientation.w = 1.0;
        temp_human.pose.orientation.x = 0.0;
        temp_human.pose.orientation.y = 0.0;
        temp_human.pose.orientation.z = 0.0;


        if (humans[i].isHuman) {
            temp_human.scale.x = 1.2;

            temp_human.scale.y = 1.2;
            temp_human.scale.z = 1.2;
        }
        else {
            temp_human.scale.x = 0.3;

            temp_human.scale.y = 0.3;
            temp_human.scale.z = 0.3;
        }

        temp_human.color = getColorRgba(humans[i].id, 0.5);

        temp_id.id = getId(humans[i].id, RVIZ_ID);

        temp_id.text = toString(humans[i].id);

        temp_id.pose.position.x =
                (icp_odom_cos * humans[i].toPoint().x - icp_odom_sin * humans[i].toPoint().y) + icp_odom_x;
        temp_id.pose.position.y =
                (icp_odom_cos * humans[i].toPoint().y + icp_odom_sin * humans[i].toPoint().x) + icp_odom_y;

        rviz_markers.markers.push_back(temp_human);
        rviz_markers.markers.push_back(temp_id);

        // float xx = humans[i].state(1);
        // float yy = humans[i].state(2);
        // cluster_msg.data.push_back(xx);
        // cluster_msg.data.push_back(yy); 
        // cluster_pub.publish(cluster_msg);
        
    }
    //ROS_INFO("Model size: %d", msg_size);

    if (rviz_markers.markers.size() > 0) {
        rviz_pub_.publish(rviz_markers);
    }
    //humans_pub_.publish(humans_msg);
    //ROS_INFO("published");
}
}
void tracker::extractModels(std::deque<fb_model> &models, std::vector<int> &clust_id_tb,
                            std::vector<int> &clust_id_st) {
    clust_id_st.clear();
    clust_id_tb.clear();

    models.clear();
    
    if (humans.size() > 0) {
        int size = 0;
        BOOST_FOREACH(HumanCluster &human, humans) { //BOOST_FOREACH：简化遍历 humans are in icp plane
                        // if (human.initialized && (human.paired || human.isHuman) && human.state(3) < 3 &&
                        //     human.state(4) < 3) 
                        {
                            models.resize(++size);
                            models[size - 1].x = human.toPoint().x;
                            models[size - 1].y = human.toPoint().y;
                            models[size - 1].v_x = human.state(3);
                            models[size - 1].v_y = human.state(4);
                            models[size - 1].id = human.id;
                            models[size - 1].length = human.length;
                            //ROS_ERROR("extract model: %ld, x: %f, y: %f", human.id,human.toPoint().x,human.toPoint().y);
                            if (human.paired)
                                clust_id_st.insert(clust_id_st.end(), human.clust_id);
                        }


                        if (human.initialized || human.isHuman) {
                            clust_id_tb.insert(clust_id_tb.end(), human.clust_id_list.back());
                        }
                    }
        }
    if (models.size() > 0){

    }


}

void tracker::update(boost::shared_ptr<clusterFrame> frame_) {
    //ROS_INFO("updating");
///*
    visualization_msgs::Marker temp_human;
    visualization_msgs::MarkerArray rviz_markers;

    temp_human.header.frame_id = "world";
    temp_human.header.stamp = ros::Time();
    temp_human.type = visualization_msgs::Marker::SPHERE;//visualization_msgs::Marker::TEXT_VIEW_FACING;
    temp_human.action = visualization_msgs::Marker::ADD;
    temp_human.lifetime = ros::Duration(1);
   

//    ROS_INFO("Model size: %d", humans.size());

    std_msgs::Float32MultiArray cluster_msg;

    for (uint i = 0; i < frame_->observations.size(); i++) {
        //if(!humans[i].initialized || humans[i].cov(1, 1) > 1)
        {
            //continue;
        }
        //temp_human.text = toString(frame_->observations[i].size);
        temp_human.id = getId(i, RVIZ_HUMAN);
        temp_human.pose.position.x =
                (icp_odom_cos * frame_->observations[i].state(1) - icp_odom_sin * frame_->observations[i].state(2)) +
                icp_odom_x;
        temp_human.pose.position.y =
                (icp_odom_cos * frame_->observations[i].state(2) + icp_odom_sin * frame_->observations[i].state(1)) +
                icp_odom_y;
//        temp_human.pose.position.x = frame_->observations[i].state(1);
//        temp_human.pose.position.y = frame_->observations[i].state(2);
        temp_human.pose.position.z = 0;
        temp_human.pose.orientation.w = 1.0;
        temp_human.pose.orientation.x = 0.0;
        temp_human.pose.orientation.y = 0.0;
        temp_human.pose.orientation.z = 0.0;

        temp_human.scale.x = 0.2;

        temp_human.scale.y = 0.2;
        temp_human.scale.z = 0.2;

        temp_human.color.r = 0.7;
        temp_human.color.g = 0.7;
        temp_human.color.b = 0.7;
        temp_human.color.a = 0.8;

        //float xx = temp_human.pose.position.x;
        //float yy = temp_human.pose.position.y;
        float xx = frame_->observations[i].state(1);
        float yy = frame_->observations[i].state(2);
        cluster_msg.data.push_back(xx);
        cluster_msg.data.push_back(yy); 
    
    rviz_markers.markers.push_back(temp_human);
    }

    //cluster_pub.publish(cluster_msg); //icp
    //clusterCallback(cluster_msg);
    clusterMsg() = cluster_msg;
    clusterCallback(clusterMsg());
    //laserCallback(laserMessage());
    if (rviz_markers.markers.size() > 0) {
        rviz_pub_clu_.publish(rviz_markers);
    }
//*/
    
    pairs.clear();
    ObjectTracking::eliminate(humans);
    //ROS_INFO("predict");
    ObjectTracking::predict(humans);
    //ROS_INFO("pair");
    ObjectTracking::pair(humans, frame_->observations, pairs);
    //ROS_INFO("update");
    ObjectTracking::update(humans, frame_->observations, pairs);


    //ROS_INFO("publish");
    publish(frame_->laser_frame_name);
    frame_->observations.clear();
    //delete frame_;
}

tracker::~tracker() {
}

cv::Rect tracker::getBound(HumanCluster &human, int &x, int &y) {
    Rect rect;
    double length = human.length;
    double distance = human.distance;
    int width = 800 * max(length, 0.6) / distance;////1000 for xtion
    int height = 1080 / distance; ////1200 for xtion

    rect.x = max(x - width / 2, 1);

    rect.y = max(y - height, 1);
    rect.width = width - max(rect.x + width - image_width, 0);
    if (rect.width < 75) {
        rect.width = 64;
        rect.height = 128;
        //ROS_ERROR("bound width %d  height %d  x %d  y %d", width, height, x, y);
        return rect;
    }
    //ROS_ERROR("bound width %d  height %d  x %d  y %d", width, height, x, y);
    rect.height = max(128, height - max(rect.y + height - image_height, 0));
    return rect;

}

bool tracker::projected(HumanCluster &human, int &x, int &y) {
    //if (human.base_x < 1 || human.initialized == false || abs(human.base_x) / abs(human.base_y) < 1.2) {
    if (human.base_x < 1 || abs(human.base_x) / abs(human.base_y) < 1.2) {
        return false;
    }
    Mat pIn(1, 3, CV_64FC1);
    Mat pOut(1, 3, CV_64FC1);
    Mat rvec = Mat::zeros(3, 1, CV_64FC1);
    Mat tvec = Mat::zeros(3, 1, CV_64FC1);

    pIn.at<double>(0) = -human.base_y;
    pIn.at<double>(1) = 0.9;
    pIn.at<double>(2) = human.base_x;


    tvec.at<double>(0, 0) = 0;
    tvec.at<double>(0, 1) = -1;
    //tvec.at<double>(0, 2) = -0.47;
    tvec.at<double>(0, 2) = 0.47;//0.47

    rvec.at<double>(2) =0.0;//0.1
    rvec.at<double>(1) =0.0;//0.05
    rvec.at<double>(0) =0.0;//0.1
    //rvec.at<double>(1) = -0.075;
    //rvec.at<double>(0) = -0.075;
    cv::projectPoints(pIn, rvec, tvec, K, D, pOut);
    x = (int) pOut.at<double>(0, 0);
    y = (int) pOut.at<double>(0, 1);
    //ROS_ERROR("**************************** x= %d  y= %d   pt.x= %lf pt.y= %lf", x, y,
              //human.base_x, human.base_y);
    if (/*std::signbit(pOut.at<float>(0, 0) - image_width / 2) != std::signbit(y) &&*/ x > 32 && x < image_width - 32 &&
        y > 0)// &&
        //y < image_height)
    {
        //ROS_INFO("accepted");
        return true;
    }
    //pointOut.x = pOut.at<double>(0, 0);
    //pointOut.y = pOut.at<double>(0, 1);

    return false;
}

void tracker::setTheta(double theta) {
    this->theta = theta;
}

void tracker::setMatrix(tf::Matrix3x3 x3, tf::Matrix3x3 icp_base) {
    icp_odom_x = x3.getColumn(2).getX();
    icp_odom_y = x3.getColumn(2).getY();
    icp_odom_sin = x3.getColumn(0).getY();
    icp_odom_cos = x3.getColumn(0).getX();
    icp_pose_x = icp_base.getColumn(2).getX();
    icp_pose_y = icp_base.getColumn(2).getY();
    icp_sin_ = icp_base.getColumn(0).getY();
    icp_cos_ = icp_base.getColumn(0).getX();

}

void tracker::setLaser(double d) {
    laser_x = d;
}









// Created by song on 16-4-12.
//
