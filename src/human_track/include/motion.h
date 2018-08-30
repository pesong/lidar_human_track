#ifndef __motion_MOtion
#define __motion_MOtion

#include "laser.h"
#include "camera.h"
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <thread>
#include <boost/thread/thread.hpp>
#include <std_msgs/UInt32.h>
#include <geometry_msgs/PoseStamped.h>
#include <chrono>
#include <autoscrubber_services/MovebaseGoal.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <sstream>
#include <iostream>
#include "readParams.h"
#include <chrono>

class MOtion
{
  private:
    double detectx;
    double detecty;
    geometry_msgs::Quaternion detectv;
    //int MODE;
    //int free_MODE;
    double last_x; //追踪时上次的x,y
    double last_y;
    geometry_msgs::Quaternion last_direction;
    cv::Mat scoremap = globalScoreMapProto();
    std::vector<std::pair<double,double>> random_points; //随机行走点
    double random_x_last; //上一次的随机行走点
    double random_y_last; //上一次的随机行走点
    bool pause_flag; //pause标志位
    

  public:
    int max_back;
    int back_cnt;
    double frz;
    double random_frz;
    bool init_flag; //是否发出initializing指令，发出后置false
    bool init_lock ;//初始化完成的锁，初始化未完成为false
    bool isBack;//是否已经发了回原点的命令，为避免多次发回原点命令
    bool lock_pub; //定时发送位置的锁，和发送时间间隔有关
    ros::NodeHandle motion_nh_;
    ros::Publisher points_pub_;
    ros::Publisher pos_pub_, watch_pose_, stop_pub_;
    
    boost::thread stop_thread;
    void back_init(TIME_PUB &tp, TARGET_INFO &ti, ROBOT_STATES &rs, bool &stay_flag, bool &no_task, bool &neighbour_flag);
    void back(ros::Time &pub_time_last, ROBOT_SHOW_STATE &robot_show_state_, ROBOT_STATES &rs);
    void norm_generator(double xin, double yin, std::vector<std::pair<double,double>> &points);
    void random_search(double x, double y, TIME_PUB &tp, TARGET_INFO &ti, ROBOT_STATES &rs, bool &stay_flag, bool &no_task, bool &neighbour_flag);
    void random_walk(double x, double y, ROBOT_STATES &rs, ros::Time &pub_time_last);
    void publishMove(ros::Time *pub_time_, geometry_msgs::PoseStamped &pose, bool *lock_pub, ROBOT_STATES &rs);
    void tracking(TIME_PUB &tp, TARGET_INFO &ti, ROBOT_STATES &rs, bool no_task, bool out_flag);
    void tracker_tracking(TIME_PUB &tp, TARGET_INFO &ti, ROBOT_STATES &rs, bool no_task, bool out_flag, bool path_tracking_flag);
    void pause_react(ROBOT_STATES &rs);
    //HumanExaminer motion_examiner_; 

    MOtion(){};
    MOtion(double dx, double dy, geometry_msgs::Quaternion dv)
    {
      random_x_last = 0;
      random_y_last = 0;
      //motion_examiner_ = HumanExaminer();
      last_x = robotPoseX();
      last_y = robotPoseY();
      last_direction = robotOrientation();
      back_cnt = 0;
      max_back = 20;
      frz = 1.2; //1秒1次
      random_frz = 1.2;
      detectx = dx;
      detecty = dy;
      detectv = dv;
      init_flag = true; //是否发出initializing指令，发出后置false
      init_lock = true;
      isBack = false;
      lock_pub = false;
      points_pub_ = motion_nh_.advertise<visualization_msgs::Marker>("random_points", 1);
      watch_pose_ = motion_nh_.advertise<geometry_msgs::PoseStamped>("/watch_pose", 1);
      pos_pub_ = motion_nh_.advertise<autoscrubber_services::MovebaseGoal>("/move_base/movebase_goal", 1);
      stop_pub_ = motion_nh_.advertise<std_msgs::UInt32>("/move_base/gaussian_cancel", 1);
    };  
};
  bool PtInLine(double x1, double y1, double x2, double y2, double ptx, double pty);


  #endif