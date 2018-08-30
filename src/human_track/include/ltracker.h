#ifndef __LTracker
#define __LTracker
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
#include "Ferns.h"
#include "readParams.h"
#include "kcftracker.hpp"
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <chrono>
#include "motion.h"
//#include "hyolo/yoloArray.h"
#include <string.h>
#include <Eigen/Dense>
#include "connect.h"
#include "facenet.h"
#include <algorithm>




bool isHuman(cv::Mat &show, Human it, bool draw);
void getNegtiveBox(cv::Rect &pos, cv::Rect &neg);
void fern_sampling(cv::Rect result_box, Ferns &fern, cv::Mat frame);



class Ltracker
{
  private:
    //int MODE; //迎人模式： MODE: 0： 普通迎人模式 1：宽送版迎人模式 2：发传单模式
    //int free_MODE; //闲置状态的机器人模式：free_MODE: 0：从原点处迎人 1. 随机行走迎人 2. 巡航模式
    double detect_region_x;
    double detect_region_y;
    bool isReplaced; //是否在first_state_decision阶段找到掉的ID，默认false,找到为true;
    cv::Mat frame;
    std::vector<std::tuple<double, double>> poly;
    geometry_msgs::Quaternion detect_rigion_v;
    cv::Mat scoremap = globalScoreMapProto();
    std::vector<std::tuple<std::string, double, double, double, double>> yolo_items; //yolo检测结果
    bool path_tracking_flag;
    bool use_greet;
    bool use_map;
    bool isTarget;
    bool isHUman;
    bool mode_one;
    bool Tar_dis; //target discard flag
    bool stay_flag;
    int nCount;
    int first_found_cnt; //用于inactive状态第一次发现
    Ferns fern;
    Ferns fern_ref;
    KCFTracker kcf;
    int track_cnt;
    int target_width_proto;
    int target_height_proto;
    std::tuple<double, double> Init_pos;
    bool no_task;        //作为开启送宾状态的标志位，在initialize完 并且没有在执行任务的时候，才会送宾
    bool neighbour_flag; //很近就置true,默认是false；
    ros::Time timer_start, timer_end; //用于发传单模式的计时器
    ROBOT_STATES rs; //记录机器人状态结构体
    TIME_PUB tp;//记录发送时间的结构体
    TARGET_INFO ti;//记录目标信息结构体
    bool tracker_loss_flag; //初始默认目标是没有丢失，置false
    int tracker_loss_cnt;
    int second_find_cnt;
    bool second_find_flag;
    int no_target_find_cnt;
    std::vector<cv::Rect> Faces;
    //define predict_list structure
    struct last_candidate
    {
        Human human;
        int cnt;
    };
    std::list<last_candidate> last_candidates;
    void greeter();
    void tracker();
    void tracker_camera();
    void humanPredictor();
    void classifier_init(cv::Mat show, bool draw);
    void moving_state(cv::Mat show, bool draw, int frz, int max_back, int &pr_cnt);
    bool stayMode(std::list<Human> tracked_huamns);
    void first_state_decision(std::list<Human> &tracked_huamns, double time_th, int &pr_cnt, cv::Mat show, bool draw);
    void first_state_decision(std::list<Human> &tracked_huamns, double time_th);
    void second_state_decision(std::list<Human> &tracked_huamns);
    void human_out(std::list<Human> tracked_huamns, double frz);
    void fill_candidates(std::list<Human> &tracked_huamns);
    void rviz_visualization(bool use_greet, std::list<Human> &tracked_huamns);
    void human_predict_path(std::list<Human> &tracked_huamns);
    void human_publishment(std::list<Human> tracked_huamns);
    void freeOperation(double x, double y);
    void yoloBoundingBox(cv::Mat &show);
    void yoloDetect(cv::Mat &show);
    bool choose_target_tracker(std::list<Human> tracked_huamns, double time_th);
    bool stayMode_track(std::list<Human> tracked_huamns);
    bool estimate_direction(double tx, double ty, double &pub_o);

    //ncs
    void init_ncs(ros::NodeHandle ph_);


    // The HoG detector for the image
    cv::HOGDescriptor hog;
    // Vectors to hold the class and the probability of the ROI
    std::vector<cv::Rect> hogFound;
    std::vector<double> hogPred;
    std::list<Human>::iterator inheritor;
    int color_cnt;

    //visual part
    void visualStart(cv::Rect &target_rect, std::list<Human> tracked_huamns);
    void visualTrack(std::list<Human> tracked_huamns);
    void laserTrack_first_state_decision(std::list<Human> &tracked_huamns);
    void laserTrack_second_state_decision(std::list<Human> &tracked_huamns);
    void laserTrack_moving_state(int frz);
    void laserTrack(std::list<Human> tracked_huamns);
    void runWithVisual();
    float HueHistFeature(cv::Rect verifyRect); //recovery部分的直方图对比
    void FaceDetection();
    bool FaceFilter(std::vector<cv::Rect> &FacesRaw);
    bool catched;
    bool missed;
    cv::Mat show;
    cv::Rect targetRect;
    std::pair<double,double> correct_reproject;
    double max_reproject_dist_small;
    double max_reproject_dist_big;
    int buffer_cnt0; //摄像头没掉，laser掉了； 1. 可能id错了，2 .可能摄像头暂时性错了 3.可能project错了，要给冗余缓冲时间buffer_cnt0
    // bool ref_is_human; //true: 付给ti的target是人, false: 是红点
    cv::Mat color_ref_Rect; //用来当HSV这一帧目标参照，只在KCF不丢时更新
    bool recovery_combination(std::list<Human> tracked_huamns,std::vector<cameraRect> tmpRects, cv::Rect& recovery_rect);
    bool FaceFindHuman(std::list <Human> tracked_huamns, std::vector <cameraRect> tmpRects, cv::Rect face, cv::Rect& HumanRect);

    //face
    cv::CascadeClassifier face_cascade;
    cv::Mat refFace;
    char * GRAPH_FILE_NAME_face;
    float *T_resultData32; //ref face feature result
    float *V_resultData32; //chosen face feature result
    void face_detect_num_control(); //limit face num to two

  public:
    ros::NodeHandle nh_;
    ros::Publisher stop_track_pub_;
    ros::Publisher pre_pos_pub_, visual_pub_, state_pub_, image_pub_, rviz_pub_, stop_pub_, rviz_target_pub_, rviz_target_id_pub_, rviz_range_, rviz_poly_door_, rviz_poly_wall_, points_pub_;

    ros::Publisher humans_pub_, yolo_pub_;
    ros::Publisher rviz_path_pub_;

    ros::Publisher predict_path_pointcloud_publisher_;

    ros::Publisher tracked_path_pointcloud_publisher_;


    MOtion motion_;

    //void yoloCallback(const hyolo::yoloArray &results);

    Ltracker();
    Ltracker(const Ltracker& ltracker){

    };

    Ltracker(double dx, double dy, geometry_msgs::Quaternion dv, bool uv, bool um, cv::CascadeClassifier face, ros::NodeHandle ph_)
    {
        detectFlag() = false;
        motion_ = MOtion(dx,dy,dv);
        timer_start = ros::Time::now();
        timer_end =  ros::Time::now();
        last_candidates.clear();

        //facenet的movidius相关
        init_ncs(ph_);
        T_resultData32 = (float *) malloc(128 * sizeof(*T_resultData32));
        V_resultData32 = (float *) malloc(128 * sizeof(*V_resultData32));
        //runWithVisual
        catched = false;
        missed = false;
        max_reproject_dist_small = 0.6;
        max_reproject_dist_big = 0.95;
       // ref_is_human = true;

        //laser tracker:
        first_found_cnt = 0;
        second_find_cnt = 0;
        second_find_flag = false;
        tracker_loss_cnt = 0;
        no_target_find_cnt = 0;

        //laser greeter:
        no_task = true;
        use_greet = useGreet();
        detect_region_x = dx; //用于回指定初始点的x
        detect_region_y = dy; //用于回指定初始点的y
        detect_rigion_v = dv; //用于回指定初始点的方向
        use_map = um;
        neighbour_flag = false;
        stay_flag = false;
        Tar_dis = false;
        isTarget = false;
        isHUman = false;
        tracker_loss_flag = false;
        nCount = Poly().size();
        track_cnt = 0;
        ti.predict_path.clear();
        ti.targetID = 0;
        ti.tx = 0;
        ti.ty = 0;
        path_tracking_flag = false;
        Init_pos = std::make_tuple(0, 0);
        rs.first_state_ = INACTIVE;
        rs.second_state_ = INACTIVE;
        rs.robot_state_ = STAY;
        if(greet_track_MODE())
            rs.robot_show_state_ = FREE;
        else
            rs.robot_show_state_ = TRACKER_FIND;
        face_cascade = face;


        kcf = KCFTracker(true /* hog*/, true /* fixed_window*/, true /* multiscale*/, true /* lab*/);
        tp.pub_time_last = ros::Time::now();
        tp.pub_time_now = ros::Time::now();
        yolo_pub_ = nh_.advertise<sensor_msgs::Image>("/greeter/startYolo", 1);
        humans_pub_ = nh_.advertise<std_msgs::Float32MultiArray>("humans", 3);
        image_pub_ = nh_.advertise<sensor_msgs::Image>("detection_show", 1);
        rviz_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("tracked_humans", 1);
        rviz_target_pub_ = nh_.advertise<visualization_msgs::Marker>("target", 1);
        rviz_target_id_pub_ = nh_.advertise<visualization_msgs::Marker>("target_id", 1);
        stop_pub_ = nh_.advertise<std_msgs::UInt32>("/move_base/gaussian_cancel", 1);
        state_pub_ = nh_.advertise<std_msgs::UInt32>("/greeter/robot_state", 1);
        rviz_range_ = nh_.advertise<visualization_msgs::Marker>("Wall", 1);
        rviz_poly_door_ = nh_.advertise<visualization_msgs::Marker>("Door", 1);
        rviz_poly_wall_ = nh_.advertise<visualization_msgs::Marker>("Outer_wall", 1);
        predict_path_pointcloud_publisher_ = nh_.advertise<sensor_msgs::PointCloud2>("predict_human_path", 1);
        tracked_path_pointcloud_publisher_ = nh_.advertise<sensor_msgs::PointCloud2>("tracked_human_path", 1);
        stop_track_pub_ = nh_.advertise<std_msgs::UInt32>("/move_base/gaussian_cancel", 1);


        //points_pub_ = nh_.advertise<visualization_msgs::Marker>("random_points", 1);
    }

    ~Ltracker();

    void run();

    //define predict_list structure
    struct track_list
    {
        std::vector<std::tuple<double, double, double, double, double>> path;
        int humanID;
        bool fromDoor;
        double greet_score;
    };

    double distToDetectRegion(double x, double y)
    {
        return sqrt((x - this->detect_region_x) * (x - this->detect_region_x) + (y - this->detect_region_y) * (y - this->detect_region_y));
    };


    void tracked_human_path_visual_test(std::list<Human> &tracked_huamns)
    {
        sensor_msgs::PointCloud2 msg_pointcloud;
        msg_pointcloud.width = 100;
        msg_pointcloud.height = 1;
        msg_pointcloud.header.stamp = ros::Time::now();
        msg_pointcloud.is_dense = true;
        msg_pointcloud.is_bigendian = false;
        sensor_msgs::PointCloud2Modifier modifier(msg_pointcloud);
        modifier.setPointCloud2FieldsByString(1, "xyz");
        sensor_msgs::PointCloud2Iterator<float> iter_x(msg_pointcloud, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(msg_pointcloud, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(msg_pointcloud, "z");
        int count = 0;
        for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++)
        {
            std::deque<std::tuple<double, double, double>> s = it->get_amclStates();
            if (s.size() < 4)
                continue;
            //s = it->regression(s, 0);
            for (int i = 0; i < s.size(); i++)
            {
                double px = std::get<0>(s[i]);
                double py = std::get<1>(s[i]);
                //ToAmcl(px, py, px, py);
                *iter_x = px;
                *iter_y = py;
                *iter_z = 0;
                ++iter_x, ++iter_y, ++iter_z;
                count++;
            }
        }
    };
};

#endif
