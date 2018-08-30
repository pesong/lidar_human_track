#include "motion.h"

//判断点是否在直线上，(x1,y1)和(x2,y2)是直线上的点，（ptx,pty）是中间点的坐标
bool PtInLine(double x1, double y1, double x2, double y2, double ptx, double pty)
{
  //std::cout << fabs(disTwoPoints(x1, y1, x2, y2) - (disTwoPoints(x1, y1, ptx, pty) + disTwoPoints(x2, y2, ptx, pty))) << std::endl;
  if (fabs(disTwoPoints(x1, y1, x2, y2) - (disTwoPoints(x1, y1, ptx, pty) + disTwoPoints(x2, y2, ptx, pty))) < 0.01)
    return true;
  else
    return false;
};

void MOtion::pause_react(ROBOT_STATES &rs)
{
  //std::cout << "asdd" << std::endl;
  if(robotVel()!=0 || robot_Theta()!=0)
  {
      std_msgs::UInt32 stop;
      stop.data = 1;
      stop_pub_.publish(stop);
  }
}

//二维高斯分布随机数生成器
void MOtion::norm_generator(double xin, double yin, std::vector<std::pair<double,double>> &points)
{
  //std::cout << "norm_generator" << std::endl;
  double x,y,l,theta,sigma;
  sigma = 3;
  // construct a trivial random generator engine from a time-based seed:
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator (seed);

  std::normal_distribution<double> distribution (0.0,sigma);
  std::default_random_engine random(seed);
  std::uniform_int_distribution<int> dis(0, 360);


  visualization_msgs::Marker points_rviz;
  points_rviz.header.frame_id = "world";
  points_rviz.header.stamp = ros::Time();
  points_rviz.type =  visualization_msgs::Marker::POINTS;
  points_rviz.action = visualization_msgs::Marker::ADD;
  points_rviz.lifetime = ros::Duration(0.5);

  points_rviz.scale.x = 0.1;
  points_rviz.scale.y = 0.1;
  points_rviz.color.r = 0.7;
  points_rviz.color.g = 0.7;
  points_rviz.color.b = 0;
  points_rviz.color.a = 1;

  for (int i = 0; i < 100; i++)
  {
    l = distribution(generator);
    theta = (double(dis(random)) / 180) * 3.1415;
    x = xin + l * cos(theta);
    y = yin + l * sin(theta);
    geometry_msgs::Point p; 
    p.x = x;
    p.y = y;

    points.push_back(std::make_pair(x,y));
    
    points_rviz.points.push_back(p); 
  }
  points_pub_.publish(points_rviz);
} 

void MOtion::publishMove(ros::Time *pub_time_, geometry_msgs::PoseStamped &pose, bool *lock_pub, ROBOT_STATES &rs)
{
  //std::cout << "PauseFlag: " << PauseFlag() << std::endl;
  if(PauseFlag() == true)
  {
    pause_react(rs);
    return;
  }
  (*lock_pub) = true;
  autoscrubber_services::MovebaseGoal pose_track;
  pose_track.type = 3;
  pose_track.pose = pose;
  ROS_ERROR("Take Effect!");
  pos_pub_.publish(pose_track);
  watch_pose_.publish(pose);
  last_x = pose.pose.position.x;
  last_y = pose.pose.position.y;
  last_direction = pose.pose.orientation;
  *pub_time_ = ros::Time::now();
  (*lock_pub) = false;
}

void MOtion::tracking(TIME_PUB &tp, TARGET_INFO &ti, ROBOT_STATES &rs, bool no_task, bool out_flag)
{
  ROS_WARN("In tracking");
  if(MODE() != 0 && !out_flag)
  {
    //std::cout << "*******" << std::endl;
  if((ti.target.directionOK_cnt < ti.target.max_direction_cnt) && (!ti.target.isStatic()))
    return;
  }
  
  if (!out_flag && (ti.target.isFar() || ti.target.isfar))
    return;
//  std::cout << "*******11" << std::endl;
  std::vector<std::tuple<double, double, double, double, double>> predict_path;
  this->isBack = false;
  this->back_cnt = 0;
  double predict_position_x, predict_position_y, predict_x, predict_y, predict_map_x, predict_map_y;
  predict_position_x = ti.target.getCurrentX();
  predict_position_y = ti.target.getCurrentY();
  ToAmcl(predict_position_x, predict_position_y, predict_x, predict_y);

  //std::cout << "*******22" << std::endl;
  //目标在1.5米外发预测轨迹，否则直接发目标位置
  if ((distToRobotPose(predict_x, predict_y) > 2) && (!out_flag))
  {
    std::cout << "It is predict position!" << std::endl;
    //预测路径根据目标离中心点远近预测长短
    double weight = (distToRobotPose(predict_x, predict_y) / 3);
    int predict_time = ceil(weight * 2); //最远预测四秒内的路径
    predict_path = ti.target.predict(predict_time, 1);
    predict_position_x = std::get<0>(predict_path.back());
    predict_position_y = std::get<1>(predict_path.back());
    predict_x = predict_position_x;
    predict_y = predict_position_y;
  }
  //std::cout << "*******33" << std::endl;
  projectToMap(predict_x, predict_y, predict_map_x, predict_map_y);
  //计算转向角
  double pub_oo = cv::fastAtan2((predict_y - robotPoseY()), (predict_x - robotPoseX())) * 0.017453292519943; //0-360
  double adjust_x = cos(pub_oo) * 0.8;
  double adjust_y = sin(pub_oo) * 0.8;
  double robot_aim_x = predict_x - adjust_x;
  double robot_aim_y = predict_y - adjust_y;
  double pub_adjust_o = cv::fastAtan2((robot_aim_y - robotPoseY()), (robot_aim_x - robotPoseX())) * 0.017453292519943;

  geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromYaw(pub_oo);
  //std::cout << "*******44" << std::endl;
  //位置变化小，并且角度变化也小，不发新跟踪点
  double r, p, last_yy;
  Toangle(last_direction, r, p, last_yy);
  if (last_yy < 0) //统一到0-360
    last_yy = (3.1416 - fabs(last_yy)) + 3.1416;
  double dis2last = sqrt((robot_aim_x - last_x) * (robot_aim_x - last_x) + (robot_aim_y - last_y) * (robot_aim_y - last_y));
  if (((fabs(last_yy - pub_adjust_o) < 0.2) || (fabs(last_yy - pub_adjust_o) > 6.08)) && (dis2last < 0.2))
  {
    ROS_ERROR("change small!");
    return;
  }

  if ((!PtInLine(robotPoseX(), robotPoseY(), predict_x, predict_y, robot_aim_x, robot_aim_y)) && (distToRobotPose(robot_aim_x, robot_aim_y) < 0.3))
  {
    if (fabs(robotVel()) != 0.0)
    {
      ROS_ERROR("Behind it!");
      return;
    }
  }
  //std::cout << "*******55" << std::endl;
  std::tuple<double, double> targetPos;
  int nCount = Poly().size();
  if ((scoremap.at<unsigned char>((int)predict_map_y, (int)predict_map_x) < Human::wall_threshold) && (PtInPolygon(std::make_tuple(predict_x, predict_y), Poly(), nCount) || (no_task)))
  {
    geometry_msgs::PoseStamped pose_amcl;
    ros::Time current_time = ros::Time::now();
    pose_amcl.header.stamp = current_time;
    pose_amcl.header.frame_id = "world";
    std::cout << "predict_x: " << predict_x << " predict_y: " << predict_y << std::endl;
    if (((distToRobotPose(predict_x, predict_y) <= 0.8) && (fabs(robotVel()) == 0.0)) || (out_flag))
    {
      ROS_WARN("Turn around!");
      pose_amcl.pose.position.x = robotPoseX();
      pose_amcl.pose.position.y = robotPoseY();
    }
    else
    {
      //ROS_WARN("Pose!");
      pose_amcl.pose.position.x = robot_aim_x;
      pose_amcl.pose.position.y = robot_aim_y;
    }
    pose_amcl.pose.orientation = odom_quat;
    stop_thread.join();

    //跟踪阶段更新robot_show_state_的状态
    if (rs.robot_state_ == STAY)
      rs.robot_show_state_ = TRACK_STAY;
    else if (out_flag)
      rs.robot_show_state_ = GOODBYE;
    else
    {
      if (ti.target.isClose)
        rs.robot_show_state_ = TRACK_NEAR;
      else
        rs.robot_show_state_ = TRACK_FAR;
    }
    ROS_ERROR("ID: %d, direction_cnt: %d", ti.target.getId(),ti.target.directionOK_cnt);

    //随机行走所有参数初始化 
    this->random_x_last = 0;
    this->random_y_last = 0;
    this->back_cnt = 0;
    stop_thread = boost::thread(boost::bind(&MOtion::publishMove, this, &tp.pub_time_last, pose_amcl, &lock_pub,rs));
  }
}

//tracker的追踪就是直接追踪目标点，不是追踪轨迹
void MOtion::tracker_tracking(TIME_PUB &tp, TARGET_INFO &ti, ROBOT_STATES &rs, bool no_task, bool out_flag, bool path_tracking_flag)
{
  ROS_WARN("In tracking");
  std::vector<std::tuple<double, double, double, double, double>> predict_path;
  double predict_position_x, predict_position_y, predict_x, predict_y, predict_map_x, predict_map_y;
  // predict_position_x = ti.target.getCurrentX();
  // predict_position_y = ti.target.getCurrentY();
  // ToAmcl(predict_position_x, predict_position_y, predict_x, predict_y);
  predict_x = ti.tx;
  predict_y = ti.ty;

  projectToMap(predict_x, predict_y, predict_map_x, predict_map_y);
  //计算转向角
  double pub_oo = cv::fastAtan2((predict_y - robotPoseY()), (predict_x - robotPoseX())) * 0.017453292519943; //0-360
  double adjust_x = cos(pub_oo) * 0.8;
  double adjust_y = sin(pub_oo) * 0.8;
  double robot_aim_x , robot_aim_y;
  if(path_tracking_flag)
  {
    robot_aim_x = predict_x;
    robot_aim_y = predict_y;
  }
  else
  {
    robot_aim_x = predict_x - adjust_x;
    robot_aim_y = predict_y - adjust_y;
  }
  double pub_adjust_o = cv::fastAtan2((robot_aim_y - robotPoseY()), (robot_aim_x - robotPoseX())) * 0.017453292519943;

  geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromYaw(pub_oo);
  //std::cout << "*******44" << std::endl;
  //位置变化小，并且角度变化也小，不发新跟踪点
  double r, p, last_yy;
  Toangle(last_direction, r, p, last_yy);
  if (last_yy < 0) //统一到0-360
    last_yy = (3.1416 - fabs(last_yy)) + 3.1416;
  double dis2last = sqrt((robot_aim_x - last_x) * (robot_aim_x - last_x) + (robot_aim_y - last_y) * (robot_aim_y - last_y));
  if (((fabs(last_yy - pub_adjust_o) < 0.2) || (fabs(last_yy - pub_adjust_o) > 6.08)) && (dis2last < 0.2))
  {
    ROS_ERROR("change small!");
    return;
  }
  //跟踪点在机器人后面（人不停往机器人靠近，减去固定长度后，目标点就在机器人后面，此时机器人应保持不动）
  if ((!PtInLine(robotPoseX(), robotPoseY(), predict_x, predict_y, robot_aim_x, robot_aim_y)) && (distToRobotPose(robot_aim_x, robot_aim_y) < 0.85))
  {
    if (fabs(robotVel()) != 0.0)
    {
      ROS_ERROR("Behind it!");
      return;
    }
  }

  std::tuple<double, double> targetPos;
  if ((scoremap.at<unsigned char>((int)predict_map_y, (int)predict_map_x) < Human::wall_threshold) &&  no_task)
  {
    geometry_msgs::PoseStamped pose_amcl;
    ros::Time current_time = ros::Time::now();
    pose_amcl.header.stamp = current_time;
    pose_amcl.header.frame_id = "world";
    ROS_INFO("goalx: %f, goaly: %f",predict_x, predict_y);
    //std::cout << "predict_x: " << predict_x << " predict_y: " << predict_y << std::endl;
    if (((distToRobotPose(predict_x, predict_y) <= 0.8) && (fabs(robotVel()) == 0.0)) || (out_flag))
    {
      ROS_WARN("Turn around!");
      pose_amcl.pose.position.x = robotPoseX();
      pose_amcl.pose.position.y = robotPoseY();
    }
    else
    {
      //ROS_WARN("Pose!");
      pose_amcl.pose.position.x = robot_aim_x;
      pose_amcl.pose.position.y = robot_aim_y;
    }
    pose_amcl.pose.orientation = odom_quat;
    stop_thread.join();

    //跟踪阶段更新robot_show_state_的状态
    rs.robot_show_state_ = TRACKER_TRACK;
    ROS_ERROR("TRACKER ID: %d", ti.target.getId());

    stop_thread = boost::thread(boost::bind(&MOtion::publishMove, this, &tp.pub_time_last, pose_amcl, &lock_pub,rs));
  }
}


//随机行走
void MOtion::random_search(double x, double y, TIME_PUB &tp, TARGET_INFO &ti, ROBOT_STATES &rs, bool &stay_flag, bool &no_task, bool &neighbour_flag)
{
    stay_flag = false;
    rs.first_state_ = INACTIVE;
    std::cout << "random_points:" << random_points.size() << std::endl;
   // std::cout << "robot_state_" << rs.robot_state_ << std::endl;
    if (rs.robot_state_ == LOST)
    {
      this->back_cnt++;
      // std::cout << "back_cnt: " << this->back_cnt << std::endl;
      tp.pub_time_now = ros::Time::now();
      if ((this->back_cnt > this->max_back) && ((tp.pub_time_now.toSec() - tp.pub_time_last.toSec()) > random_frz))
      {
        ROS_ERROR("Lost Target!");
        random_walk( x, y, rs, tp.pub_time_last);
        neighbour_flag = false;
        rs.first_state_ = INACTIVE;
      }
    }
}

void MOtion::random_walk(double x, double y, ROBOT_STATES &rs, ros::Time &pub_time_last)
{
  this->back_cnt = 0;
  
  if (random_points.size() == 0)
    norm_generator(x, y, random_points);
  
  //ROS_WARN("random points size0: %d", random_points.size());
  auto it = random_points.begin();
  //下一个目标点与当前机器人位置距离一定要大于1米
  while (disTwoPoints(it->first, it->second, robotPoseX(), robotPoseY()) < 1)
    it++;
  std::tuple<double, double> p = std::make_tuple(it->first,it->second);

  while (!PtInPolygon( p, Poly(), Poly().size()))
  {
    it = random_points.erase(it);
    p = std::make_tuple(it->first,it->second);;
  }
  if (((disTwoPoints(robotPoseX(), robotPoseY(), random_x_last , random_y_last) < 0.3))
      || ((random_x_last == 0) && (random_y_last == 0)))
  {
    double pub_oo = cv::fastAtan2((it->second - robotPoseY()), (it->first - robotPoseX())) * 0.017453292519943; //0-360
    geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromYaw(pub_oo);
    geometry_msgs::PoseStamped pose_amcl;
    ros::Time current_time = ros::Time::now();
    pose_amcl.header.stamp = current_time;
    pose_amcl.header.frame_id = "world";
    pose_amcl.pose.position.x = it->first;
    pose_amcl.pose.position.y = it->second;
    pose_amcl.pose.orientation = odom_quat;
    stop_thread.join();
    stop_thread = boost::thread(boost::bind(&MOtion::publishMove, this, &pub_time_last, pose_amcl, &lock_pub, rs));
    rs.robot_show_state_ = TRACK_BACK;
    // autoscrubber_services::MovebaseGoal pose_track;
    // pose_track.type = 3;
    // pose_track.pose = pose_amcl;
    ROS_ERROR("Random traveling: Go to next position!");
    // pos_pub_.publish(pose_track);
    // watch_pose_.publish(pose_amcl);
    random_x_last = it->first;
    random_y_last = it->second;
    random_points.erase(it);
    ROS_WARN("last random points size: %d", random_points.size());
    //if(disTwoPoints(robotPoseX(), robotPoseY(), it->first, it->second) < 0.2)     
  }
}


void MOtion::back(ros::Time &pub_time_last, ROBOT_SHOW_STATE &robot_show_state_, ROBOT_STATES &rs)
{
  if(PauseFlag() == true)
  {
    pause_react(rs);
    return;
  }
  this->isBack = true;
  this->back_cnt = 0;
  double pub_oo = cv::fastAtan2((this->detecty - robotPoseY()), (this->detectx - robotPoseX())) * 0.017453292519943;
  geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromYaw(pub_oo);
  geometry_msgs::PoseStamped pose_amcl;
  ros::Time current_time = ros::Time::now();
  pose_amcl.header.stamp = current_time;
  pose_amcl.header.frame_id = "world";
  pose_amcl.pose.position.x = this->detectx;
  pose_amcl.pose.position.y = this->detecty;
  pose_amcl.pose.orientation = this->detectv;
  stop_thread.join();
  stop_thread = boost::thread(boost::bind(&MOtion::publishMove, this, &pub_time_last, pose_amcl, &lock_pub, rs));
  robot_show_state_ = TRACK_BACK;
  ROS_WARN("Missing...Going back...");
  ROS_WARN("robot distance: %f", sqrt((robotPoseX() - this->detectx) * (robotPoseX() - this->detectx) + (robotPoseY() - this->detecty) * (robotPoseY() - this->detecty)));
  ROS_WARN("x: %f,y: %f", this->detectx, this->detecty);
  ROS_WARN("robotx: %f,roboty: %f", robotPoseX(), robotPoseY());
}

void MOtion::back_init(TIME_PUB &tp, TARGET_INFO &ti, ROBOT_STATES &rs, bool &stay_flag, bool &no_task, bool &neighbour_flag)
{
  //std::cout << "******back init" << std::endl;
  stay_flag = false;
  rs.first_state_ = INACTIVE;
  //ROS_WARN("In back!");
  bool direction_flag = false; //和初始方向相同为true,否则为false
  double robot_to_detect_dist = sqrt((robotPoseX() - this->detectx) * (robotPoseX() - this->detectx) + (robotPoseY() - this->detecty) * (robotPoseY() - this->detecty));
  double r, p, y, r0, p0, y0;
  Toangle(robotOrientation(), r, p, y);
  Toangle(this->detectv, r0, p0, y0);
  //if ((fabs(y - y0) < 0.2) || (fabs(y) + fabs(y0) > 6.10))
    direction_flag = true;
  if ((robot_to_detect_dist < 0.3) && (direction_flag))
  { //回到原点，初始化
    //ROS_WARN("first");
    //std::cout << "init_flag: " << init_flag << " isBack: " << isBack << std::endl;
   // ROS_INFO("robotVel: %f, robotTheta: %f", robotVel(), robot_Theta());
    //std::cout << "robotVel: " << robotVel() << " robot_Theta: " << robot_Theta() << std::endl;
    if ((this->init_flag == false) && (this->isBack == true))
    {
      ROS_WARN("Initializing...");
      this->init_lock = false;
      //ROS_INFO("y-y0: %f", fabs(y - y0));
      //if ((fabs(y - y0) < 0.3) || (fabs(y) + fabs(y0) > 6.00))
      if ((robotVel() ==0 && robot_Theta()==0) && ((fabs(y - y0) < 0.3) || (fabs(y) + fabs(y0) > 6.00)))
      {
        ti.tx = 0;
        ti.ty = 0;
        this->init_flag = true;
        this->init_lock = true;
        rs.first_state_ = INACTIVE;
        neighbour_flag = false;
        this->isBack == false;
        this->back_cnt = 0;
        ROS_INFO("y-y0: %f", fabs(y - y0));
        ROS_ERROR("Initialize finished!");
        no_task = true;
        sleep(1);
        rs.robot_show_state_ = FREE;
      }
    }
    else
    {
      rs.robot_show_state_ = FREE;
    }
  }
  else //if (robot_to_detect_dist > 0.09)
  {
    if (rs.robot_state_ == LOST)
    {
      this->back_cnt++;
      // std::cout << "back_cnt: " << back_cnt << std::endl;
      tp.pub_time_now = ros::Time::now();
      if ((this->back_cnt > this->max_back) && ((tp.pub_time_now.toSec() - tp.pub_time_last.toSec()) > frz) && (this->lock_pub == false) && (this->isBack == false))
      {
        ROS_ERROR("Lost Target!");
        back(tp.pub_time_last, rs.robot_show_state_,rs);
        neighbour_flag = false;
        rs.first_state_ = INACTIVE;
      }
    }
  }
}

