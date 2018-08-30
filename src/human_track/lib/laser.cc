#include "laser.h"

//接受机器人现在的位置
void robotCallback(const geometry_msgs::PoseWithCovarianceStamped &robot_msg)
{
    //std::cout << "***robot_msg: " << robot_msg.header.stamp << std::endl;
    //ROS_INFO("robotCallback");
    robotPoseX() = robot_msg.pose.pose.position.x;
    robotPoseY() = robot_msg.pose.pose.position.y;
    robotPoseZ() = robot_msg.pose.pose.position.z;
    robotOrientation() = robot_msg.pose.pose.orientation;
    robotPose() = std::make_tuple(robotPoseX(), robotPoseY());

    double r, p, y;
    Toangle(robotOrientation(), r, p, y);
    robotYaw() = y;
}

std::vector<std::pair<double, double>> aroundLaser(std::vector<std::pair<double, double>> laser, double x_0, double y_0)
{

    double dist_threshold = Human::track_distance;
    double wall_threshold = Human::meanshift_wall;
    double x, y, dx, dy, amcl_x, amcl_y, xs, ys, s;
    std::vector<std::pair<double, double>> around_laser;
    // std::cout << " x0: " << x_0 << " y0: " << y_0 << std::endl;

    for (int i = 0; i < laser.size(); i++)
    {
        x = laser.at(i).first;
        y = laser.at(i).second;
        std::tuple<double, double> p = std::make_tuple(x,y);
        dx = fabs(x - x_0);
        dy = fabs(y - y_0);
        if (dx > dist_threshold || dy > dist_threshold)
            continue;
        if (PtInPolygon(p, Poly_robot()))
            continue;
        ToAmcl(x, y, amcl_x, amcl_y);
        if (!projectToMap(amcl_x, amcl_y, xs, ys))
            continue;
        s = globalScoreMapProto().at<unsigned char>((int)ys, (int)xs);
        if (s > wall_threshold)
            continue;
        if ((fabs(x - robotPoseX()) < 0.06) && (fabs(y - robotPoseY()) < 0.06))
            continue;
        if(around_laser.size()>0)
        {
            if( x == (around_laser.end()-1)->first && y==(around_laser.end()-1)->second)
                continue;
        }
        // std::cout << " ax: " << x << " ay: " << y;
        around_laser.push_back(std::make_pair(x, y));
    }
    // std::cout <<  std::endl;
    return around_laser;
}

std::pair<double, double> getCenter(std::vector<std::pair<double, double>> laser)
{
    double sumx = 0;
    double sumy = 0;
    for (int i = 0; i < laser.size(); i++)
    {
        sumx = sumx + std::get<0>(laser[i]);
        sumy = sumy + std::get<1>(laser[i]);
    }
    sumx = sumx / (laser.size());
    sumy = sumy / (laser.size());
    return std::make_pair(sumx, sumy);
}


//DEBUG:transed_laser在base_laser平面上
// void printLaserpoints(std::vector<std::pair<double, double>> transed_laser)
// {
//     double x, y, x_amcl, y_amcl, x_icp, y_icp;
//     std::vector<std::pair<double, double>> laser_amcl, laser_icp;
//     for(auto it=transed_laser.begin(); it!=transed_laser.end(); it++)
//     {
//         x = it->first;
//         y = it->second;
//         x_amcl = amclPoseCos() * x - amclPoseSin() * y + amclPoseX();
//         y_amcl = amclPoseCos() * y + amclPoseSin() * x + amclPoseY();
//         x_icp = icpPoseCos() * x - icpPoseSin() * y + icpPoseX();
//         y_icp = icpPoseCos() * y + icpPoseSin() * x + icpPoseY();
//         laser_amcl.push_back(std::make_pair(x_amcl, y_amcl));
//         laser_icp.push_back(std::make_pair(x_icp, y_icp));
//     }
//std::cout << "htrack tarnslaser amclPose: " << amclPoseX() << std::endl;
//  std::cout << "base_link_x" << " ";
// for(int i=0; i<laser_amcl.size(); i++)
// {
//     std::cout << laser_amcl[i].first << " ";
// }
// std::cout << std::endl;


// std::cout << "laser_amclx" << " ";
// for(int i=0; i<laser_amcl.size(); i++)
// {
//     std::cout << laser_amcl[i].first << " ";
// }
// std::cout << std::endl;

// std::cout << "laser_amcly" << " ";
// for(int i=0; i<laser_amcl.size(); i++)
// {
//     std::cout << laser_amcl[i].second << " ";
// }
// std::cout << std::endl;

// std::cout << "laser_icpx" << " ";
// for(int i=0; i<laser_icp.size(); i++)
// {
//     std::cout << laser_icp[i].first << " ";
// }
// std::cout << std::endl;

// std::cout << "laser_icpy" << " ";
// for(int i=0; i<laser_icp.size(); i++)
// {
//     std::cout << laser_icp[i].second << " ";
// }
// std::cout << std::endl;
// }


//基于激光雷达的目标跟踪，将断掉的人物坐标点连接起来
//
std::list<Human> &trackLaser(std::vector<std::pair<double, double>> &new_humans,
                             std::vector<std::pair<double, double>> &laser)
{
    //std::cout << "BUG6" <<std::endl;

    double time_now = getTimestamp();
    cv::Mat scoremap = globalScoreMapProto();

    //meanshift找重心
    auto meanshift = [&laser, &scoremap](double &x_0, double &y_0, std::vector<std::pair<double, double>> last_laser, std::vector<std::pair<double, double>> &now_laser)
    {
        double dist_threshold = Human::track_distance;
        double wall_threshold = Human::meanshift_wall;
        double init_x = x_0;
        double init_y = y_0;

        //上一帧和这一帧的激光位移变化

        now_laser = aroundLaser(laser, x_0, y_0);
        // std::cout << "now_all_laser: " << std::endl;
        // for(int i=0; i<laser.size(); i++)
        // {
        //     std::cout << " (" << std::get<0>(laser[i]) << ", " << std::get<1>(laser[i]) << ")" ;
        // }
        // std::cout << std::endl;
        // std::cout << "now_all_laser: " << laser.size() << std::endl;
        std::pair<double, double> oldC = getCenter(last_laser);
        std::pair<double, double> newC = getCenter(now_laser);
        //print around laser
        // std::cout << "last_laser: "<< std::endl;
        // for(int i=0; i<last_laser.size(); i++)
        // {
        //     std::cout << " (" << std::get<0>(last_laser[i]) << ", " << std::get<1>(last_laser[i]) << ")" ;
        // }
        //std::cout << std::endl;
        // std::cout << "last_laser: " << last_laser.size() << std::endl;
        //print around laser
        // std::cout << "x0: " << x_0 << " y0: " << y_0 << std::endl;
        // std::cout << "now_laser: "<< std::endl;
        // for(int i=0; i< now_laser.size(); i++)
        // {
        //     std::cout << " (" << std::get<0>(now_laser[i]) << ", " << std::get<1>(now_laser[i]) << ")" ;
        // }
        // std::cout << std::endl;


        if ((abs(now_laser.size() - last_laser.size()) < last_laser.size() * 0.25) && (last_laser.size() > 0))
        {
            //std::cout << "laser 1" << std::endl;
            // ROS_WARN("laserXX 1");
            double new_x = newC.first;
            double new_y = newC.second;
            double old_x = oldC.first;
            double old_y = oldC.second;
            // std::cout << "new_x: " << newC.first << " new_y: " << newC.second << " old_x: " << oldC.first << " old_y: " << oldC.second << std::endl;
            // std::cout << "move distance: " << distance_l2(newC.first, newC.second, oldC.first, oldC.second) << std::endl;
            x_0 = x_0+(new_x-old_x);
            y_0 = y_0+(new_y-old_y);

        }
        else
        {
            // ROS_WARN("laser 2");
            //std::cout << "laser 2" << std::endl;
            // std::cout << " changes: " << abs(now_laser.size() - last_laser.size()) << " change_th: " << last_laser.size() * 0.25 << std::endl;
            for (int iter = 0; iter < 5; iter++)
            {
                //printf("iter: %d x0: %f, y0: %f \n", iter,x_0,y_0);
                double a_x = 0, a_y = 0, b_x = 0, b_y = 0;
                double x, y, xs, ys, dx, dy, s, amcl_x, amcl_y;
                //std::cout << "th: " << dist_threshold << std::endl;
                // ROS_WARN("laser size: %d", laser.size());
                // std::cout << "trackLaser mapResolution(): " << mapResolution() << " mapWidth(): "<< mapWidth() << std::endl;
                for (int i = 0; i < laser.size(); i++)
                {
                    x = laser.at(i).first;
                    y = laser.at(i).second;

                    dx = fabs(x - x_0);
                    dy = fabs(y - y_0);
                    if (dx > dist_threshold || dy > dist_threshold)
                        continue;
                    ToAmcl(x, y, amcl_x, amcl_y);
                    if (!projectToMap(amcl_x, amcl_y, xs, ys))
                        continue;
                    s = scoremap.at<unsigned char>((int)ys, (int)xs);
                    if (s > wall_threshold)
                        continue;
                    if ((fabs(x - robotPoseX()) < 0.2) && (fabs(y - robotPoseY()) < 0.2))
                        continue;
                    // std::cout << " x: " << x << " y: " << y ;

                    s = 1.0 - s / 255.0; //s:离墙越近越小
                    s = s * s;
                    dx = 1.0 - dx / dist_threshold; //closer to center -> weight greater
                    dx = dx * dx * s;
                    dy = 1.0 - dy / dist_threshold;
                    dy = dy * dy * s; //核函数：d_center*d_center*d_wall*d_wall
                    a_x += x * dx;    //all激光点除以它所对应的核函数的和
                    a_y += y * dy;
                    b_x += dx;
                    b_y += dy;
                }
                // std::cout << std::endl;
                // std::cout << " x_0: " << x_0 <<" y_0: " << y_0 << std::endl;

                if (b_x == 0 || b_y == 0)
                {
                    x_0 = 0;
                    y_0 = 0;
                    break;
                }
                else
                {
                    x_0 = a_x / b_x;
                    y_0 = a_y / b_y;
                }
            }
        }
        if((fabs(x_0 - robotPoseX()) < 0.2) && (fabs(y_0 - robotPoseY()) < 0.2))
        {
            x_0 = 0;
            y_0 = 0;
        }
    };
    //std::cout << "BUG7" <<std::endl;

    std::list<Human> &humans = trackedHuman();
    //std::cout << "human0 size: " << trackedHuman().size() <<std::endl;
    //std::cout << "human_back ID1: " << trackedHuman().back().getId() << " x:" << trackedHuman().back().getCurrentX() << " y: " << trackedHuman().back().getCurrentY() << std::endl;
    //ROS_ERROR("renew human!");
    //tracking human using meanshift
    for (auto it = humans.begin(); it != humans.end();)
    {
        //ROS_WARN("human.cnt: %d",it->getCnt());
        std::tuple<double, double> p = std::make_tuple(it->getAmclX(),it->getAmclY());
        //ROS_WARN("human_id: %d, cnt: %d", it->getId(), it->getCnt());
        if (it->isOverTrack() || PtInPolygon(p, Poly_robot())) //cnt > max_track_cnt
        {
            ROS_INFO("OverTrack or inRobot: delete ID: %d", it->getId());
            it = humans.erase(it);
            //ROS_ERROR("Erase!");
        }
        else
        {
            double x = it->getCurrentX(); //base_laser ping mian
            double y = it->getCurrentY();
            //std::cout << "ID: " << it->getId() << std::endl;
            //std::cout << x << " " << y << std::endl;
            //std::cout << "meanshift Id: " << it->getId() << std::endl;
            //  std::cout <<" x: "<<  x <<" y: "<< y << std::endl;
            double amcl_x = it->getAmclX();
            double amcl_y = it->getAmclY();
            //ToAmcl(x,y,amcl_x,amcl_y);
            // std::cout <<" amclx: "<<  amcl_x <<" amcly: "<< amcl_y << std::endl;
            // std::cout << "robotx " << robotPoseX() << " roboty " << robotPoseY()<< std::endl;

            // ROS_ERROR("ID %d meanshift x: %f, y: %f", it->getId(),x,y);
            double ori_x = x;
            double ori_y = y;
            //std::cout << "human ID: " << it->getId()  << std::endl;
            //std::cout << "xxxtrackLaser mapResolution(): " << mapResolution() << " mapWidth(): "<< mapWidth() << std::endl;
            //std::cout << "meanshift_x: " << x << " meanshift_y: " << y << std::endl;
            std::vector<std::pair<double, double>> now_laser;
            meanshift(x, y, it->getLaser(), now_laser); //预测出重心移动位置
            //std::cout << "after_meanshift_x: "<< x << " after_meanshift_y: " << y << std::endl;
            it->meanshift_ = distance_l2(x, y, ori_x, ori_y);
            //ROS_ERROR("ID %d new x: %f, y: %f, dis: %f", it->getId(),x,y,distance_l2(x, y, ori_x, ori_y));
            double xa, ya;
            ToAmcl(x, y, xa, ya);
            std::tuple<double, double> pa = std::make_tuple(xa, ya);
            if (x == 0 && y == 0)
            {
                ROS_WARN("Meanshift no laser delete ID: %d", it->getId());
                it = humans.erase(it);
            }
            else if (PtInPolygon(pa, Poly_robot()))
            {
                ROS_WARN("Meanshift to robot delete ID: %d", it->getId());
                it = humans.erase(it);
            }
            else if(distance_l2(x, y, ori_x, ori_y)> Human::meanshift_invaild)
            {
                ROS_WARN("Invaild meanshift ID: %d, moving %f, delete", it->getId(), distance_l2(x, y, ori_x, ori_y));
                it = humans.erase(it);
            }
            else
            {
                double xx, yy;
                ToAmcl(x, y, xx, yy);
                std::tuple<double, double> p = std::make_tuple(xx, yy);

                if (it->first_position_flag == false)
                {
                    //std::cout << "aaaaaa" << std::endl;
                    //ROS_ERROR("Update ID: %d", it->getId());
                    if ((init_range(p)) || (PtInPolygon(p, Poly(), Poly().size())))
                    {
                        it->init_position = p;
                        it->first_position_flag = true;
                        if(init_range(p)){
                            ROS_INFO("From door ID: %d", it->getId());
                            it->from_door = true;
                        }else
                            ROS_INFO("Not from door ID: %d", it->getId());
                    }
                }
                //std::vector<std::pair<double, double>> laser_around;
                // laser_around = aroundLaser(laser, x, y);

                //print laser
                // std::cout << "around_all_laser: " << std::endl;
                // for(int i=0; i<laser.size(); i++)
                // {
                //     std::cout << " (" << std::get<0>(laser[i]) << ", " << std::get<1>(laser[i]) << ")" ;
                // }
                // std::cout << std::endl;
                // std::cout << "around_all_laser size: " << laser.size() << std::endl;
                // std::cout << "x: " << x << " y: " << y << std::endl;
                // std::cout << "laser_around: "<< std::endl;

                // for(int i=0; i< now_laser.size(); i++)
                // {
                //    std::cout << " (" << std::get<0>(now_laser[i]) << ", " << std::get<1>(now_laser[i]) << ")" ;
                // }
                // std::cout << std::endl;
                // ///
                // std::cout << "laser_around size: " << now_laser.size() <<std::endl;
                it->update(x, y, time_now, now_laser); //pushback到这个人的states和speeds里

                it++;
            }
        }
    }


    // //get new human & fusion
    //std::cout << "new humans: " << new_humans.size() << std::endl;
    //std::cout << "humans: " << humans.size() << std::endl;
    if (new_humans.size())
    {
        auto the_fusion_one_new = new_humans.end();
        for (auto it1 = new_humans.begin(); it1 != new_humans.end(); it1++)
        {
            double x = it1->first;
            double y = it1->second;
            auto the_fusion_one = humans.end(); //humans是预测到的人，the_fusion_one 最后一个预测到的人
            int tID = 66555;
            float min_dist = Human::fusion_distance;
            //通过将上一帧的目标进行meanshift后的坐标来计算与当前帧的目标的距离，来判断上一帧的目标与当前帧是否相同
            for (auto it2 = humans.begin(); it2 != humans.end(); it2++)
            {
                //std::cout << "11111" << std::endl;
                double xx = it2->getCurrentX();
                double yy = it2->getCurrentY();

                //TODO bug
                if (distance_l2(x, y, xx, yy) < min_dist) //在上一次预测到的observations里匹配的那个human
                {
                    the_fusion_one = it2;
                    the_fusion_one_new = it1;
                    tID = it2->getId();
                    min_dist = distance_l2(x, y, xx, yy);
                    // std::cout<< "x: " << x << " y: " << y << " ID: " << it2->getId() << std::endl;
                    //ROS_ERROR("tID ID: %d",it2->getId());
                }
            }
            // ROS_WARN("Human ID: %d", the_fusion_one->getId());
            //if (the_fusion_one != humans.end()) //证明找到了,找到了是上一帧的人，不是这一帧的
            //在上一帧和这一帧距离最近的两个human，上一帧找的是白点，这一帧找到的数字，看他们是不是一个human
            if(fabs(min_dist-Human::fusion_distance)>0.01)
            {
                // std::cout << "min_dist: " << min_dist << " fusion_dis: " << Human::fusion_distance << std::endl;
                //ROS_WARN("renew!");
                the_fusion_one->resetTrackCnt(); //此时the_fusion_one是上一次detect到的人，cnt = 0;
                double time_ = the_fusion_one->getCurrentTime();
                the_fusion_one->renew_position(the_fusion_one_new->first, the_fusion_one_new->second, time_);
                continue;
            }
                // //TODO
                // else if( the_fusion_one->getId() == tID ){
                //         the_fusion_one->resetTrackCnt();
                //         continue;
                // }
            else
            { //没找到human+1
                //std::cout << "has new human" << std::endl;
                double xx, yy;
                std::vector<std::pair<double, double>> laser_around;
                laser_around = aroundLaser(laser, x, y);

                ToAmcl(x, y, xx, yy);
                std::tuple<double, double> p = std::make_tuple(xx, yy);
                //std::cout << "before human push" << std::endl;
                // std::cout << "laser_around: " << laser_around.size() <<std::endl;
                humans.push_back(Human(x, y, time_now, laser_around));
            }
        }
    }

//     // DEBUG----------------------------//
//      humans.clear();
//      for (auto it = new_humans.begin(); it != new_humans.end(); it++)
//      {
//         double x = it->first;
//         double y = it->second;
//         double xx, yy;
//         std::vector<std::pair<double, double>> laser_around;
//         laser_around = aroundLaser(laser, x, y);
//         ToAmcl(x, y, xx, yy);
//         std::tuple<double, double> p = std::make_tuple(xx, yy);
//         // std::cout << "laser_around: " << laser_around.size() <<std::endl;
//         humans.push_back(Human(x, y, time_now, laser_around));
//      }
//    // ----------------DEBUG----------------------//




    // erase the human who is in the wall
    // ROS_WARN("human size: %d", humans.size());
    for (auto it = humans.begin(); it != humans.end();)
    {
        double xx, yy;
        // double x = it->getCurrentX();
        // double y = it->getCurrentY();
        // ToAmcl(x, y, xx, yy);
        double x = it->getAmclX();
        double y = it->getAmclY();

        //std::cout << "xx: " << xx << " yy: " << yy << std::endl;
        //std::cout << "amclx: " << x << " amcly: " << y << std::endl;
        //ROS_INFO("human ID: %d",it->getId());
        if (!projectToMap(x, y, x, y))
        {
            //std::cout << " project x: " << x << " project y: " << y << std::endl;
            ROS_INFO("Cannot project to map delete ID: %d", it->getId());
            ret() = true;
            it = humans.erase(it);
        }
        else if (scoremap.at<unsigned char>((int)y, (int)x) > Human::wall_threshold)
        {
            //std::cout << " project x: " << x << " project y: " << y << std::endl;
            ROS_INFO("In wall delete ID: %d, score: %d", it->getId(), scoremap.at<unsigned char>((int)y, (int)x));
            // double xx = fabs(xx - robotPoseX());
            // double yy = fabs(yy - robotPoseY());
            // std::cout << "dis from robot: " << sqrt(xx*xx+yy*yy) << std::endl;
            it = humans.erase(it);
        }
        else
        {
            //ROS_INFO("human ID: %d, score: %d", it->getId(), scoremap.at<unsigned char>((int)y, (int)x));
            it++;
        }
    }

    // fusion adject tracked human
    humans.sort([](Human &a, Human &b) { return a.getId() < b.getId(); }); //使humans按id从小到大排序
    for (auto it1 = humans.begin(); it1 != humans.end(); it1++)            //对每个human,计算相邻两个human的距离
    {
        for (auto it2 = humans.begin(); it2 != humans.end();)
        {
            if (it1->getId() == it2->getId())
            {
                it2++;
                continue;
            }
            double x1 = it1->getCurrentX();
            double y1 = it1->getCurrentY();
            double x2 = it2->getCurrentX();
            double y2 = it2->getCurrentY();
            //std::cout << " ID: " << it1->getId() << " ID: " << it2->getId() << " dis:" <<distance_l2(x1, y1, x2, y2) << std::endl;

            if ((distance_l2(x1, y1, x2, y2) < Human::fusion_distance) && (it1->meanshift_ < it2->meanshift_)) //两个human离太近了删掉一个
            {
                //以防白点消失时，数字被吸到邻近已有数字的白点上，从而导致一个白点上有两个数字
                if (distance_l2(x1, y1, x2, y2) > 0.05)
                {
                    if ((it2->getStates().size() > it2->stable_cnt) && (it1->getStates().size() > it1->stable_cnt))
                    {
                        double it1_x = std::get<0>(it1->getInitxy());
                        double it1_y = std::get<1>(it1->getInitxy());
                        double it2_x = std::get<0>(it2->getInitxy());
                        double it2_y = std::get<1>(it2->getInitxy());
                        it2++;
                        // if (distance_l2(it1_x, it1_y, it2_x, it2_y) > it1->init_distance)
                        //     it2++;
                        // else
                        // {
                        //     ROS_INFO("To close delete ID: %d，the initial position is closed", it2->getId());
                        //     ROS_INFO("it1_x: %f, it1_y: %f, it2_x: %f, it2_y: %f", it1_x, it1_y, it2_x, it2_y);
                        //     it2 = humans.erase(it2);
                        // }
                    }
                    else
                    {
                        ROS_INFO("To close delete ID: %d, getCnt is smaller than stable, it2 cnt: %d, it1 cnt: %d", it2->getId(), it2->getStates().size(), it1->getStates().size());
                        it2 = humans.erase(it2);
                    }
                }
                else
                {
                    ROS_INFO("To close delete ID: %d, distance: %f", it2->getId(), distance_l2(x1, y1, x2, y2));
                    it2 = humans.erase(it2);
                }

            }
            else
                it2++;
        }
    }
    return humans;
}

//laser.cc 中接收人的坐标
void clusterCallback(std_msgs::Float32MultiArray &cluster_msg) //observation[i].state(1),state(2)
{
    // x y  x y  ...
    //laser coordination;
    //std::cout << "BUG1" << std::endl;
    // std::cout << "icpPoseCos(): " << icpPoseCos() << std::endl;
    double icp_pose_cos = icpPoseCos();
    double icp_pose_sin = icpPoseSin();
    double icp_pose_x = icpPoseX();
    double icp_pose_y = icpPoseY();
    double amcl_pose_cos = amclPoseCos();
    double amcl_pose_sin = amclPoseSin();
    double amcl_pose_x = amclPoseX();
    double amcl_pose_y = amclPoseY();
    //人传进来在world坐标系下，要转到icp+amcl
    auto theworld = [&](double xi, double yi, double &xo, double &yo) //相当于把icp坐标系转到world坐标系
    {
        double xt, yt;                                                            //
        xt = icp_pose_cos * (xi - icp_pose_x) + icp_pose_sin * (yi - icp_pose_y); //转回base_link坐标系
        yt = icp_pose_cos * (yi - icp_pose_y) - icp_pose_sin * (xi - icp_pose_x);
        xo = icp_pose_cos * xt - icp_pose_sin * yt + amcl_pose_x; //用icp的角度，amcl的平移量转到world坐标系
        yo = icp_pose_cos * yt + icp_pose_sin * xt + amcl_pose_y;
    };
    std::vector<std::pair<double, double>> tmp;
    auto &data = cluster_msg.data; //human is in amcl coord
    double x, y;
    for (int i = 0; i < data.size();) //把observations的x,y转到world平面下，赋给humanCluster
    {
        x = data[i++];
        y = data[i++];        //已经是在icp坐标系下
        // std::cout << " xx: " << x << " yy: " << y << std::endl;
        theworld(x, y, x, y); //转到world坐标系
        if (x != 0 || y != 0)
        {
            tmp.push_back(std::make_pair(x, y));
        }
    }
    //ROS_INFO("human size: %d", tmp.size());
    //std::cout << "BUG3" << std::endl;
    humanCluster().swap(tmp);
    //ROS_INFO("humanCluster: %d", humanCluster().size());
    // std::cout << "human size: " << humanCluster().size() << std::endl;
    // for (int i = 0; i < humanCluster().size(); i++){
    //    std::cout << " x: " << std::get<0>(humanCluster()[i]) << " y: " << std::get<1>(humanCluster()[i]) << std::endl;
    // }
    //std::cout << "BUG2" << std::endl;
}

//接收激光雷达的点云
void laserCallback(sensor_msgs::LaserScan::ConstPtr &message_)
{
    // laserTime = message_->header.stamp;
    double pose_cos = icpPoseCos();
    double pose_sin = icpPoseSin();
    double pose_x = amclPoseX();
    double pose_y = amclPoseY();
    cv::Mat scoremap = globalScoreMap();
    static bool laserInit = false;
    //laser是从base_laser转到icp+amcl
    auto theworld = [&](double xi, double yi, double &xo, double &yo) {
        xo = pose_cos * xi - pose_sin * yi + pose_x;
        yo = pose_cos * yi + pose_sin * xi + pose_y;
    };
    double angle;
    auto rotate = [](double xi, double yi, double &xo, double &yo, double angle) {
        xo = -yi * sin(angle) + xi * cos(angle);
        yo = xi * sin(angle) + yi * cos(angle);
    };

    std::vector<std::pair<double, double>> transed_laser,base_laser;
    sensor_msgs::LaserScan current_message;
    static sensor_msgs::LaserScan message;
    current_message = *message_;
    message = current_message;
    // if (laserInit == false)
    // {
    //     message = current_message;
    //     laserInit = true;
    //     return;
    // }
    //message = *message_;

    transed_laser.reserve(message.ranges.size());
    base_laser.reserve(message.ranges.size());
    // printf("\n \n raw laser message");
    for (int i = 0; i < message.ranges.size(); i++)
    {
        double angle = message_->angle_min + i * message_->angle_increment;
        double x = message.ranges[i];
        double y = 0;
        rotate(x, y, x, y, angle);                     //现在在base_laser平面上
        x = x + laserX(); //转到base_link
        base_laser.push_back(std::make_pair(x, y));
        theworld(x, y, x, y);                          //转换到world,icp+amcl平面上,
        transed_laser.push_back(std::make_pair(x, y)); //transed_laser收集了所有的转换到world平面上的点,amcl+icp
    }
    // std::cout << "htrack_lasertime: " << message_->header.stamp << std::endl;
    // std::cout << "htrack_amclpose: " << amclPoseX() << " corresponding laser: " << message_->header.stamp << std::endl;
    // std::cout << "htrack laser time: " << message.header.stamp << std::endl;
    // std::cout << "htrack laserCalllback amclPose: " << amclPoseX() << std::endl;
    //printLaserpoints(base_laser);
    //std::vector<std::pair<double,double> >& humanCluster()
    //ROS_INFO("Has pass human");
    auto new_humans(humanCluster()); //grammar ??? std::vector<std::pair<double,double> >& humanCluster()
    humanCluster().clear();
    //ROS_INFO("new_humans: %d", new_humans.size());
    //std::cout << "new_humans: " << new_humans.size() <<std::endl;
    // std::cout << "BUG6" <<std::endl;
    auto &tmp_humans = trackLaser(new_humans, transed_laser);

    //auto camera_humans = laser_human_extraction(tracked_huamns);
    laser_human_extraction(tmp_humans);

    auto &tracked_huamns = tmp_humans;

    // ROS_INFO("tracked_huamns size: %d", tracked_huamns.size());
    //std::cout << "tracked_huamns size: " << tracked_huamns.size() <<std::endl;
    //output message laser:
    // ros::NodeHandle _nh;
    // ros::Publisher laser_pub;
    message_pub() = message;
    //message = current_message;
    transed_laser.swap(transedLaser());
}

std::vector<std::pair<double, double>> &transedLaser()
{
    static std::vector<std::pair<double, double>> v;
    return v;
}

std::vector<std::pair<double, double>> &humanCluster()
{
    static std::vector<std::pair<double, double>> v;
    return v;
}

std::list<Human> &trackedHuman()
{
    static std::list<Human> v;
    return v;
}

std::list<Human> &realHuman()
{
    static std::list<Human> v;
    return v;
}

//maybe Bug
void visualscoremap(std::list<Human> &tracked_huamns,
                    std::vector<std::pair<double, double>> &transed_laser)
{
    auto new_humans(humanCluster());
    cv::Mat scoremap = globalScoreMapProto();
    cv::Mat show;
    cv::cvtColor(scoremap, show, CV_GRAY2RGB);
    for (int i = 0; i < transed_laser.size(); i++)
    {
        double x = transed_laser.at(i).first;
        double y = transed_laser.at(i).second;
        // ToAmcl (x,y,x,y);
        if (projectToMap(x, y, x, y))
        {
            cv::circle(show, cv::Point((int)x, (int)y), 1, cv::Scalar(0, 255, 0), -1);
        }
    }
    for (auto it = new_humans.begin(); it != new_humans.end(); it++)
    {
        double x = it->first;
        double y = it->second;
        ToAmcl(x, y, x, y);
        // ToAmcl (x,y,x,y);
        if (projectToMap(x, y, x, y))
            cv::circle(show, cv::Point((int)x, (int)y), 3, cv::Scalar(0, 0, 255), -1);
    }
    for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++)
    {
        // double x = it->getCurrentX();
        // double y = it->getCurrentY();
        // ToAmcl(x, y, x, y);
        double x = it->getAmclX();
        double y = it->getAmclY();
        //ToAmcl (x,y,x,y);
        if (projectToMap(x, y, x, y))
        {
            cv::circle(show, cv::Point((int)x, (int)y), 3, cv::Scalar(255, 0, 0), -1);
            putText(show, std::to_string(it->getId()), cv::Point(int(x), int(y)),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 0, 0), 1, CV_AA);
        }
    }

    for(auto it=tracked_huamns.begin(); it!=tracked_huamns.end(); it++)
    {
        auto pp = it->predict(5,1);
        for(int i=0; i<pp.size(); i++)
        {
            double x = std::get<0>(pp[i]);
            double y = std::get<1>(pp[i]);
            ToAmcl(x, y, x, y);
            if(projectToMap(x,y,x,y))
            {
                cv::circle(show,cv::Point((int)x,(int)y), 1, cv::Scalar(255,255-255/pp.size()*i,0),-1);
            }
        }
    }
    cv::imshow("1", show);
    cv::waitKey(5);
}