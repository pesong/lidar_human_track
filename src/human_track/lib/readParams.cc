//read parameters, the origin point, polygon region, door position
#include "readParams.h"
#define Pi 3.141592653
#define IMP 52431

double distToRobotPose(double x, double y)
{
    return sqrt((x - robotPoseX()) * (x - robotPoseX()) + (y - robotPoseY()) * (y - robotPoseY()));
}

double distToDoorPose(double x, double y)
{
    double doorx = 0.5 * (std::get<0>(Door()[0]) + std::get<0>(Door()[1]));
    double doory = 0.5 * (std::get<1>(Door()[0]) + std::get<1>(Door()[1]));
    return sqrt((x - doorx) * (x - doorx) + (y - doory) * (y - doory));
}

//解析重要系数
void imp_analyzing(int imp)
{
    std::cout << "imp: " << imp << std::endl;
    std::vector<int> imp_vec;
    if (imp > 100000) //只能是5位数
    {
        ROS_ERROR("Imported Importance is invaild!");
        imp = IMP;
    }
    else
    {
        imp_vec.push_back(imp / 10000);
        imp_vec.push_back((imp / 1000) % 10);
        imp_vec.push_back((imp / 100) % 10);
        imp_vec.push_back((imp / 10) % 10);
        imp_vec.push_back(imp % 10);
        auto max_it = max_element(std::begin(imp_vec), std::end(imp_vec));
        if (*max_it > 5)
        {
            ROS_ERROR("Imported Importance is invaild!");
            imp_vec.push_back(IMP / 10000);
            imp_vec.push_back((IMP / 1000) % 10);
            imp_vec.push_back((IMP / 100) % 10);
            imp_vec.push_back((IMP / 10) % 10);
            imp_vec.push_back(IMP % 10);
        }
    }
    if (Importance().size() == 0)
    {
        Importance().push_back(imp_vec[0]);
        Importance().push_back(imp_vec[1]);
        Importance().push_back(imp_vec[2]);
        Importance().push_back(imp_vec[3]);
        Importance().push_back(imp_vec[4]);
    }
    else
    {
        int i = 0;
        for (auto it = Importance().begin(); it != Importance().end(); it++)
        {
            *it = imp_vec[i];
            i++;
        }
    }
}

//解析free_mode
void read_free_mode(int fm)
{
    free_MODE() = fm;
}

//解析greet_track_mode
void read_greet_track_mode(bool gtm)
{
    greet_track_MODE() = gtm;
    ROS_WARN("greet_track_MODE: %d", greet_track_MODE());
}

void paramCallback(const autoscrubber_services::StringArray &param_title)
{
    //std::cout << "*********************angleCallback" << std::endl;
    ros::NodeHandle nh0;
    double angle;
    int imp;
    int fm;
    int da;
    bool gtm;
    for (int i = 0; i < param_title.strings.size(); i++)
    {
        if (param_title.strings[i] == "/strategy/greeter/angle_threshold")
        {
            nh0.param("/strategy/greeter/angle_threshold", angle, 20.0);
            Angle() = (double(angle) / 180) * 3.1416;
            //ROS_WARN("Angle threshold changes to %f", angle);
        }
        if (param_title.strings[i] == "/strategy/greeter/importance")
        {
            nh0.param("/strategy/greeter/importance", imp, IMP);
            imp_analyzing(imp);
            //ROS_WARN("Imp changes to %d", imp);
        }
        if (param_title.strings[i] == "/strategy/greeter/free_mode")
        {
            nh0.param("/strategy/greeter/free_mode", fm, 0);
            read_free_mode(fm);
        }
        if (param_title.strings[i] == "/strategy/greeter/greet_track_mode")
        {
            //true: 迎人 false: 跟踪 默认迎人
            nh0.param("/strategy/greeter/greet_track_mode", gtm, true);
            read_greet_track_mode(gtm);
        }
        if (param_title.strings[i] == "/strategy/greeter/detect_angle")
        {
            nh0.param("/strategy/greeter/detect_angle", da, 45);
            std::cout << "detect Angle: " << da << std::endl;
            detectAngle() = da;
        }

        // if(param_title.strings[i] == "/strategy/greeter/mode")
        // {
        //   nh0.param("/strategy/greeter/mode", m, 3);
        //   read__mode(imp);
        // }
    }
    
}

bool &PauseFlag()
{
    static bool v;
    return v;
}

bool &ResumeFlag()
{
    static bool v;
    return v;
}

bool &skipFlag()
{
    static bool v;
    return v;
}

bool &detectFlag()
{
    static bool v;
    return v;
}


geometry_msgs::Quaternion &oriDirection()
{
    static geometry_msgs::Quaternion v;
    return v;
}

geometry_msgs::Quaternion &greetOrientation()
{
    static geometry_msgs::Quaternion v;
    return v;
}

std::tuple<double, double> &Origin()
{
    static std::tuple<double, double> v;
    return v;
}

std::vector<std::tuple<double, double>> &Door()
{
    static std::vector<std::tuple<double, double>> v;
    return v;
}

std::vector<std::tuple<double, double>> &Poly()
{
    static std::vector<std::tuple<double, double>> v;
    return v;
}

std::vector<std::tuple<double, double>> &Poly_wall()
{
    static std::vector<std::tuple<double, double>> v;
    return v;
}

double &Angle()
{
    static double v;
    return v;
}

std::vector<std::tuple<double, double>> &doorPoly()
{
    static std::vector<std::tuple<double, double>> v;
    return v;
}

std::vector<double> &Importance()
{
    static std::vector<double> v;
    return v;
}

bool pauseService(autoscrubber_services::Pause::Request &req, autoscrubber_services::Pause::Response &res)
{
    std::cout << "Recieve Pause" << std::endl;
    PauseFlag() = true; //req.data;                       // res.success = true;
}

bool skipService(autoscrubber_services::Skip::Request &req, autoscrubber_services::Skip::Response &res)
{
    // skipFlag() = req.data;
    // res.success = true;
    //std::cout << "asssssssssssssssssssss" << std::endl;
    skipFlag() = true;
}


bool detectService(autoscrubber_services::Detect::Request &req, autoscrubber_services::Detect::Response &res)
{
    detectFlag() = true;
}


bool resumeService(autoscrubber_services::Resume::Request &req, autoscrubber_services::Resume::Response &res)
{
    PauseFlag() = false; //!req.data;
    ResumeFlag() = true; //req.data;
                         // res.success = true;
}

void getParam(std::vector<std::tuple<double, double>> door, std::tuple<double, double> origin, geometry_msgs::Quaternion greet, std::vector<std::tuple<double, double>> poly, std::vector<std::tuple<double, double>> poly_wall, double angle, int imp)
{
    double orx = std::get<0>(origin);
    double ory = std::get<1>(origin);
    double l = 0.5; //门宽
    Door() = door;
    Origin() = std::make_tuple(orx, ory);
    greetOrientation() = greet;
    Poly() = poly;
    Poly_wall() = poly_wall;
    Angle() = angle;
    double x3, y3, x4, y4;
    double x1 = std::get<0>(door[0]);
    double y1 = std::get<1>(door[0]);
    double x2 = std::get<0>(door[1]);
    double y2 = std::get<1>(door[1]);
    double theta = atan2(y2 - y1, x2 - x1);
    if (fabs(theta - 0.5 * Pi) < 0.08)
    {
        x3 = x2 - l;
        y3 = y2;
        x4 = x1 - l;
        y4 = y1;
    }
    else
    {
        x3 = x2 - l * sin(theta);
        y3 = y2 + l * cos(theta);
        x4 = x1 - l * sin(theta);
        y4 = y1 + l * cos(theta);
    }
    std::vector<std::tuple<double, double>> poly_door;
    poly_door.push_back(std::make_tuple(x1, y1));
    poly_door.push_back(std::make_tuple(x2, y2));
    poly_door.push_back(std::make_tuple(x3, y3));
    poly_door.push_back(std::make_tuple(x4, y4));
    doorPoly() = poly_door;
    //解析重要系数
    imp_analyzing(imp);
}

bool PtInPolygon(std::tuple<double, double> p, std::vector<std::tuple<double, double>> ptPolygon)
{
    int nCount = ptPolygon.size();
    int nCross = 0;
    double px = std::get<0>(p);
    double py = std::get<1>(p);
    for (int i = 0; i < nCount; i++)
    {
        std::tuple<double, double> p1 = ptPolygon[i];
        std::tuple<double, double> p2 = ptPolygon[(i + 1) % nCount];
        double p1x = std::get<0>(p1);
        double p1y = std::get<1>(p1);
        double p2x = std::get<0>(p2);
        double p2y = std::get<1>(p2);
        if (p1y == p2y) //p1p2 与 y=p0.y平行
            continue;
        if (py < std::min(p1y, p2y)) //交点在p1p2延长线上
            continue;
        if (py >= std::max(p1y, p2y)) //交点在p1p2延长线上
            continue;
        //求交点的 X 坐标
        double x = (double)(py - p1y) * (double)(p2x - p1x) / (double)(p2y - p1y) + p1x;
        if (x > px)
            nCross++; //只统计单边交点
    }
    //单边交点为偶数，点在多边形之外
    return (nCross % 2 == 1);
    //-------------------------------------//
    //return true;
}



bool PtInPolygon(std::tuple<double, double> p, std::vector<std::tuple<double, double>> ptPolygon, int nCount)
{
    //--------------//
    if (!useGreet())
        return true;
    int nCross = 0;
    double px = std::get<0>(p);
    double py = std::get<1>(p);
    for (int i = 0; i < nCount; i++)
    {
        std::tuple<double, double> p1 = ptPolygon[i];
        std::tuple<double, double> p2 = ptPolygon[(i + 1) % nCount];
        double p1x = std::get<0>(p1);
        double p1y = std::get<1>(p1);
        double p2x = std::get<0>(p2);
        double p2y = std::get<1>(p2);
        if (p1y == p2y) //p1p2 与 y=p0.y平行
            continue;
        if (py < std::min(p1y, p2y)) //交点在p1p2延长线上
            continue;
        if (py >= std::max(p1y, p2y)) //交点在p1p2延长线上
            continue;
        //求交点的 X 坐标
        double x = (double)(py - p1y) * (double)(p2x - p1x) / (double)(p2y - p1y) + p1x;
        if (x > px)
            nCross++; //只统计单边交点
    }
    //单边交点为偶数，点在多边形之外
    return (nCross % 2 == 1);
    //-------------------------------------//
    //return true;
}

bool init_range(std::tuple<double, double> p)
{

    bool flag;
    if (useGreet())
    {
        if (doorPoly().size() > 0)
            flag = PtInPolygon(p, doorPoly(), 4);
        else
            flag = true;
    }
    else
        flag = true;
    return flag;
}

void read_json()
{
    
    Json::Reader reader;
    Json::Value root;
    ros::NodeHandle param_nh("~");
    ros::NodeHandle nh0;

    std::ifstream is;
    std::tuple<double, double> origin;
    geometry_msgs::Quaternion greet, ori_direction;
    std::vector<std::tuple<double, double>> poly_wall, poly, door;
    //std::cout << "door size: " << door.size() << std::endl;
    double angle;
    int imp;
    int fm;
    int da;
    bool gtm;
    // std::cout << "aaaa" <<std::endl;
    ROS_INFO("Read greeter area.");
    std::string address;
    //param_nh.param(std::string("file_address"), address, std::string("/home/ziwei/human_track_camera/run.map.greeter01.json"));
    param_nh.param(std::string("file_address"), address, std::string("/root/GAUSSIAN_RUNTIME_DIR/run.map.greeter"));
    
    is.open(address);
    std::cout << address << std::endl;

    nh0.param("/strategy/greeter/greet_track_mode", gtm, false);
    read_greet_track_mode(gtm);
   // ROS_INFO("sssssssssss");
    if (gtm)
    {
        if (reader.parse(is, root))
        {
            double orx = root["originPosition"]["worldPose"]["position"]["x"].asDouble();
            double ory = root["originPosition"]["worldPose"]["position"]["y"].asDouble();
            //std::cout << " orx: " << orx << " ory: " << ory << std::endl;
            origin = std::make_tuple(orx, ory);
            ori_direction.w = root["originPosition"]["worldPose"]["orientation"]["w"].asDouble();
            ori_direction.x = root["originPosition"]["worldPose"]["orientation"]["x"].asDouble();
            ori_direction.y = root["originPosition"]["worldPose"]["orientation"]["y"].asDouble();
            ori_direction.z = root["originPosition"]["worldPose"]["orientation"]["z"].asDouble();
            oriDirection() = ori_direction;
            //std::cout << " ori_direction.w: " << ori_direction.w << " ori_direction.z: " << ori_direction.z << std::endl;
            greet.w = root["direction"]["worldPose"]["orientation"]["w"].asDouble();
            greet.x = root["direction"]["worldPose"]["orientation"]["x"].asDouble();
            greet.y = root["direction"]["worldPose"]["orientation"]["y"].asDouble();
            greet.z = root["direction"]["worldPose"]["orientation"]["z"].asDouble();

            int polywall_size = root["detectArea"].size();
            for (int i = 0; i < polywall_size; i++)
            {
                double x = root["detectArea"][i]["worldPose"]["position"]["x"].asDouble();
                double y = root["detectArea"][i]["worldPose"]["position"]["y"].asDouble();
                //std::cout << " detect x: " <<  x << " detect y: " << y << std::endl;
                poly_wall.push_back(std::make_tuple(x, y));
            }

            int poly_size = root["greetArea"].size();
            for (int i = 0; i < poly_size; i++)
            {
                double x = root["greetArea"][i]["worldPose"]["position"]["x"].asDouble();
                double y = root["greetArea"][i]["worldPose"]["position"]["y"].asDouble();
                std::cout << " greet x: " << x << " greet y: " << y << std::endl;
                poly.push_back(std::make_tuple(x, y));
            }

            int door_size = root["door"].size();
            double x, y;
            x = root["door"]["left"]["worldPose"]["position"]["x"].asDouble();
            y = root["door"]["left"]["worldPose"]["position"]["y"].asDouble();
            //std::cout << " door0 x: " <<  x << " door0 y: " << y << std::endl;
            door.push_back(std::make_tuple(x, y));
            x = root["door"]["right"]["worldPose"]["position"]["x"].asDouble();
            y = root["door"]["right"]["worldPose"]["position"]["y"].asDouble();
            //std::cout << " door1 x: " <<  x << " door1 y: " << y << std::endl;
            door.push_back(std::make_tuple(x, y));
            //angle = (double(20)/180)*3.1416;
        }

        nh0.param("/strategy/greeter/angle_threshold", angle, 20.0);
        std::cout << "angle: " << angle << std::endl;
        angle = (double(angle) / 180) * 3.1416;

        nh0.param("/strategy/greeter/importance", imp, IMP);
        std::cout << "importance: " << imp << std::endl;

        nh0.param("/strategy/greeter/free_mode", fm, 0);
        std::cout << "free Mode: " << fm << std::endl;
        free_MODE() = fm;

        //  nh0.param("/strategy/greeter/detect_angle", da, 45);
        //  std::cout << "detect Angle: " << da << std::endl;
        //  detectAngle() = da;
        getParam(door, origin, greet, poly, poly_wall, angle, imp);
        //std::cout << "getParam" << std::endl;
    }
    //std::cout << "getParam" << std::endl;
}

void robot_initializing(double &x, double &y, geometry_msgs::Quaternion &v, ros::Publisher pos_pub, ros::Publisher watch_pose)
{
    //机器人行走
    double Orx = std::get<0>(Origin());
    double Ory = std::get<1>(Origin());
    ROS_INFO("originx: %f, originy: %f",std::get<0>(Origin()),std::get<1>(Origin()));

    int init_Flag = false;
    ros::Rate loop_rate(10);

    autoscrubber_services::MovebaseGoal pose_track;
    geometry_msgs::PoseStamped pose;
    pose.header.stamp = ros::Time::now();
    pose.header.frame_id = "world";
    pose.pose.position.x = Orx;
    pose.pose.position.y = Ory;
    pose.pose.orientation = oriDirection();
    pose_track.type = 3;
    pose_track.pose = pose;
    pos_pub.publish(pose_track);
    watch_pose.publish(pose);

    //是否行至初始点判断
    while ((robotVel() != 0) || (robot_Theta() != 0) || (fabs(robotPoseX() - Orx) > 0.2))
    {
        if ((robotVel() == 0) && (robotVel() == 0) && (fabs(robotPoseX() - Orx) > 0.2))
        {
            pos_pub.publish(pose_track);
            watch_pose.publish(pose);
        }
        loop_rate.sleep();
        ros::spinOnce();
    }
    if (!init_Flag)
    {
        //ROS_WARN("robotPoseX(): %f, robotPoseY(): %f",robotPoseX(),robotPoseY());
        x = Orx;
        y = Ory;
        v = oriDirection();
        init_Flag = true;
    }
}