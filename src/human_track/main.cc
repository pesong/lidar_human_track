//#include <message_filters/subscriber.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/UInt32.h>
#include <std_msgs/Float32.h>
#include <csignal>
#include <geometry_msgs/PoseStamped.h>
#include <tuple>
#include <vector>
#include "extra.h"
#include "readParams.h"
#include <tf/transform_listener.h>
#include "laserListener.h"
#include "HumanTrack.h"
#include "laser.h"
#include "tracker.h"

#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <SSD_Detector.hpp>

ros::Publisher *stop_pub_ptr;

void Cs(int n)
{
    if (n == SIGINT)
    {
        std::cerr << "Get Ctrl+C" << std::endl;
        std_msgs::UInt32 stop;
        stop.data = 1;
        (*stop_pub_ptr).publish(stop);
        std::cerr << "Before shutdown" << std::endl;
        ros::shutdown();
        exit(0);
    }
}

/*关于各种模式的设置
迎人模式 MODE // MODE: 0： 普通迎人模式 1：宽送版迎人模式 2：发传单模式
闲置状态的机器人模式 free_MODE //free_MODE: 0：从原点处迎人 1. 随机行走迎人 2. 巡航模式

MODE目前是代码在main.cc初始写死
free_MODE是可从app设置，置位相关代码在readParams.cc里
*/

int main(int argc, char *argv[])
{
    ros::Publisher stop_pub;
    stop_pub_ptr = &(stop_pub);

//
    ros::init(argc, argv, "human_track");
    ros::NodeHandle nh, priv_nh("~");
    signal(SIGINT, Cs);
;
    stop_pub = nh.advertise<std_msgs::UInt32>("/move_base/gaussian_cancel", 1);
    std::string camera_topic("//rgb_image_hd"); //摄像头图像
    std::string laser_topic("/synchronous_scan"); //激光雷达
    std::string map_proto_topic("/move_base/global_costmap/inflationed_static_map"); //从机器人发过来的costmap 坐标系
    std::string robot_shape_topic("/move_base/global_costmap/footprint_master");

    ros::Subscriber sub_proto_map = nh.subscribe(map_proto_topic, 3, protoMapCallback);

    ROS_WARN("The version of human_track is 1.1.0");

    //darknet
//    std::string graphModel;
//    priv_nh.param("yolo_model/graph_file/name", graphModel, std::string("tiny-yolo-voc_graph"));
//    priv_nh.param("graph_path", graphModel, std::string("tiny-yolo-voc_graph"));
//    std::cout << "YOLO MODEL GRAPH: " << graphModel << std::endl;
    ssd_ros::SSD_Detector ssd_Detector(priv_nh);
    std::string face_cascade_name;
    priv_nh.param("face_cascade", face_cascade_name, std::string("/home/ziwei/human_track_dl/src/human_track/config/haarcascade_frontalface_alt2.xml"));
    cv::CascadeClassifier face_cascade;
    // Load the cascade
    if (!face_cascade.load(face_cascade_name)){
        printf("--(!)Error loading\n");
        return (-1);
    }

    //after get scoremap hdetect
    tracker tracker_;
    laserListener listener_(&tracker_, laser_topic);
    ROS_INFO("IN hdetect!");

//    //hdetect
    message_filters::Subscriber<sensor_msgs::LaserScan> laser_sub(nh,laser_topic,1);
    message_filters::Subscriber<geometry_msgs::PoseWithCovarianceStamped> pose_sub(nh,"/synchronous_pose",1);//(nh,"/synchronous_pose",1);
    message_filters::TimeSynchronizer<sensor_msgs::LaserScan, geometry_msgs::PoseWithCovarianceStamped> sync(laser_sub, pose_sub, 10);
    sync.registerCallback(boost::bind(&laserListener::syncCallback, &listener_,  _1, _2));

    //hdetect
    //flag initialization
    PauseFlag() = false;
    ResumeFlag() = false;
    skipFlag() = false;

    //确定是普通模式还是迎宾模式
    bool use_greet;
    nh.param("/strategy/greeter/use_greeter", use_greet, false);
    ROS_INFO("usegreet: %d", use_greet);

    useGreet() = use_greet;
    std::string robot_topic("/amcl_pose"); ///amcl_sync
    std::string robot_odom("/odom");
    ros::Subscriber sub_robot = nh.subscribe(robot_topic, 1, robotCallback);

    ros::Subscriber sub_robot_shape = nh.subscribe(robot_shape_topic, 1, robotShapeCallback);

    ros::ServiceServer srv_pause = nh.advertiseService("/greeter/pause", pauseService);
    ros::ServiceServer srv_resume = nh.advertiseService("/greeter/resume", resumeService);
    ros::ServiceServer srv_skip = nh.advertiseService("/greeter/skip", skipService);
    ros::ServiceServer srv_detect = nh.advertiseService("/greeter/detect", detectService);
    //ros::Subscriber sub_laser = nh.subscribe(laser_topic, 1, laserCallback);       //接收激光雷达的点云
    ros::Subscriber sub_image = nh.subscribe(camera_topic, 1, cameraCallback);     //将ros发过来的摄像头数据转换成opencv里的mat,并存储到全局变量里
    ros::Subscriber sub_odom = nh.subscribe(robot_odom, 1, odomCallback);

    ROS_ERROR("IN htrack Start");
    ros::Publisher pos_pub = nh.advertise<autoscrubber_services::MovebaseGoal>("/move_base/movebase_goal", 1);
    ros::Publisher watch_pose = nh.advertise<geometry_msgs::PoseStamped>("/watch_pose", 1);
    static bool init_Flag = false;
    double x, y;
    geometry_msgs::Quaternion v;
   // std::cout << " x: " << robotPoseX() << " y: " << robotPoseY() << std::endl;
    while ((robotPoseX() == 0) && (robotPoseY() == 0))
    {

        ROS_INFO("Waitting for the robotPose...");
        ros::Duration(0.2).sleep();
        ros::spinOnce();
    }

    if(use_greet)
    {
        read_json();
        if(greet_track_MODE())
            robot_initializing(x, y, v, pos_pub, watch_pose);
    }
    else
    {
        x = robotPoseX();
        y = robotPoseY();
        v = robotOrientation();
    }

    //MODE: 0： 普通迎人模式 1：宽送版迎人模式 2：发传单模式 3: 打分模式
    MODE() = 3;
    Ltracker lt = Ltracker(x,y,v,false,true, face_cascade, priv_nh);

//    Ltracker B =lt;

    ros::Rate loop_rate(7);
    ros::Publisher state_pub_ = nh.advertise<std_msgs::UInt32>("robot_state", 1);
    std_msgs::UInt32 show_state;
    while (ros::ok())
    {
        //std::cout << "ros ok" << std::endl;
        x = robotPoseX();
        y = robotPoseY();
        if(!projectToMap(x, y, x, y))
        {
            show_state.data = 6;
            ROS_INFO("The robot position is out of map.");
            state_pub_.publish(show_state);
            ros::Duration(0.5).sleep();
        }
        else
        {
            lt.run();
            loop_rate.sleep();
            //std::cout << "aaaaa" << std::endl;
            ros::spinOnce();
            //std::cout << "ddddd" << std::endl;
        }
    }

    return 0;
}