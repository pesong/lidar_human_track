#include "HumanExaminer.h"

//double distToDoorPose(double x, double y)
//{
//    double doorx = 0.5 * (std::get<0>(Door()[0]) + std::get<0>(Door()[1]));
//    double doory = 0.5 * (std::get<1>(Door()[0]) + std::get<1>(Door()[1]));
//    return sqrt((x - doorx) * (x - doorx) + (y - doory) * (y - doory));
//}

//没有转到amcl平面，但不影响，icp改变的是角度
bool HumanExaminer::isFar(std::_List_iterator<Human> &it)
{
  std::deque<std::tuple<double, double, double>> &s = it->get_amclStates();
  int num = s.size();
  std::deque<std::tuple<double, double>> path;
  if (num < 18)
  {
    if (it->isfar)
      return true;
    else
      return false;
  }

  for (int i = num - 1; i >= 0; i = i - 4)
  {
    double x1 = std::get<0>(s[i]);
    double y1 = std::get<1>(s[i]);
    //ToAmcl(x1, y1, x1, y1);
    path.push_back(std::make_tuple(x1, y1));
  }
  // printf("human_path size=%d\n",(int)path.size());
  int compare_cnt = 0;
  int pn = path.size();
  for (int i = 0; i < pn - 2; i++)
  {
    double x1 = std::get<0>(path[i]);
    double y1 = std::get<1>(path[i]);
    double x2 = std::get<0>(path[i + 1]);
    double y2 = std::get<1>(path[i + 1]);
    double dis_cur = distToRobotPose(x1, y1);
    double dis = distToRobotPose(x2, y2);
    if ((dis_cur - dis) > 0)
      compare_cnt = compare_cnt + 1;
  }
  double x0 = std::get<0>(path[0]);
  double y0 = std::get<1>(path[0]);
  double xn = std::get<0>(path[pn - 1]);
  double yn = std::get<1>(path[pn - 1]);
  double first_dis = distToRobotPose(x0, y0);
  double last_dis = distToRobotPose(xn, yn);
  //std::cout << " ****far dis:" << (first_dis - last_dis) <<std::endl;
  if ((compare_cnt >= (path.size() - 2)) && ((first_dis - last_dis) > 0.6))
  {
    // it->isfar = true;
    return true;
  }
  else
  {
    // it->isfar = false;
    return false;
  }
}

bool HumanExaminer::isFar(Human it)
{
  std::deque<std::tuple<double, double, double>> &s = it.get_amclStates();
  int num = s.size();
  std::deque<std::tuple<double, double>> path;
  if (num < 18)
  {
    if (it.isfar)
      return true;
    else
      return false;
  }

  for (int i = num - 1; i >= 0; i = i - 4)
  {
    double x1 = std::get<0>(s[i]);
    double y1 = std::get<1>(s[i]);
    //ToAmcl(x1, y1, x1, y1);
    path.push_back(std::make_tuple(x1, y1));
  }
  // printf("human_path size=%d\n",(int)path.size());
  int compare_cnt = 0;
  int pn = path.size();
  for (int i = 0; i < pn - 2; i++)
  {
    double x1 = std::get<0>(path[i]);
    double y1 = std::get<1>(path[i]);
    double x2 = std::get<0>(path[i + 1]);
    double y2 = std::get<1>(path[i + 1]);
    double dis_cur = distToRobotPose(x1, y1);
    double dis = distToRobotPose(x2, y2);
    if ((dis_cur - dis) > 0)
      compare_cnt = compare_cnt + 1;
  }
  double x0 = std::get<0>(path[0]);
  double y0 = std::get<1>(path[0]);
  double xn = std::get<0>(path[pn - 1]);
  double yn = std::get<1>(path[pn - 1]);
  double first_dis = distToRobotPose(x0, y0);
  double last_dis = distToRobotPose(xn, yn);
  //std::cout << " ****far dis:" << (first_dis - last_dis) <<std::endl;
  if ((compare_cnt >= (path.size() - 2)) && ((first_dis - last_dis) > 0.6))
  {
    // it->isfar = true;
    return true;
  }
  else
  {
    // it->isfar = false;
    return false;
  }
}

//必须转到amcl平面
bool HumanExaminer::is_close_to_door(std::_List_iterator<Human> it)
{
  std::deque<std::tuple<double, double, double>> &s = it->getStates();
  int num = s.size();
  std::deque<std::tuple<double, double>> path;
  if (num < 28)
    return false;
  for (int i = num - 1; i >= 0; i = i - 4)
  {
    double x1 = std::get<0>(s[i]);
    double y1 = std::get<1>(s[i]);
    ToAmcl(x1, y1, x1, y1);
    path.push_back(std::make_tuple(x1, y1));
  }
  int compare_cnt = 0;
  int pn = path.size();
  for (int i = 0; i < pn - 2; i++)
  {
    double x1 = std::get<0>(path[i]);
    double y1 = std::get<1>(path[i]);
    double x2 = std::get<0>(path[i + 1]);
    double y2 = std::get<1>(path[i + 1]);
    double dis_cur = distToDoorPose(x1, y1);
    double dis = distToDoorPose(x2, y2);
    if ((dis - dis_cur) > 0)
      compare_cnt = compare_cnt + 1;
  }
  double x0 = std::get<0>(path[0]);
  double y0 = std::get<1>(path[0]);
  double xn = std::get<0>(path[pn - 1]);
  double yn = std::get<1>(path[pn - 1]);
  double first_dis = distToDoorPose(x0, y0);
  double last_dis = distToDoorPose(xn, yn);
  if ((compare_cnt >= (path.size() - 2)) && ((last_dis - first_dis) > 1))
    return true;
  else
    return false;
}

bool HumanExaminer::isDirection_ok(std::_List_iterator<Human> it)
{
  bool result;
  if (MODE == 2 || MODE == 3) //MODE3对人没有运动方向要求
    result = true;
  if (MODE != 2)
  {
    std::pair<double, double> s = Speed_amcl(std::make_tuple(it->getCurrentX(), it->getCurrentY()), it->getCurrent_speed());
    double vx = s.first;
    double vy = s.second;
    double pub_oo = cv::fastAtan2(vy, vx) * 0.017453292519943;

    //geometry_msgs::Quaternion direction = tf::createQuaternionMsgFromYaw(pub_oo);
    double r, p, greet_y;
    if (MODE == 0)
      Toangle(greetOrientation(), r, p, greet_y);
    else if (MODE == 1)
    {
      double x, y;
      ToAmcl(it->getCurrentX(), it->getCurrentY(), x, y);
      greet_y = angle2robot(x, y, robotPoseX(), robotPoseY());
    }

    double ang = Angle();
    if (greet_y < 0) //统一到0-360
      greet_y = (3.1416 - fabs(greet_y)) + 3.1416;
    if (!it->isClose) 
    {
      //角度偏差0.3弧度
      if ((fabs(greet_y - pub_oo) < ang) || (fabs(greet_y - pub_oo) > (6.28 - ang)))
      {
        //std::cout << it->getId() << " direction ok!" << std::endl;
        result = true;
        //return true;
      }
      else
      {
        //std::cout << it->getId() << "direction not ok" << std::endl;
        result = false;
        //return false;
      }
    }
    else
    {
      if (Angle() < 0.7)
      {
        ang = 0.7;
      }
      if ((fabs(greet_y - pub_oo) < ang) || (fabs(greet_y - pub_oo) > (6.28 - ang)))
      {
        //std::cout << it->getId() << " direction ok!" << std::endl;
        result = true;
        // return true;
      }
      else
      {
        //std::cout << it->getId() << "direction not ok" << std::endl;
        result = false;
        //return false;
      }
    }
  }

  if(result == true)
    it->directionOK_cnt++;
  else 
    it->directionOK_cnt = 0;

  return result;
}

bool HumanExaminer::isStatic(std::_List_iterator<Human> it)
{
  std::pair<double, double> s = Speed_amcl(std::make_tuple(it->getCurrentX(), it->getCurrentY()), it->getCurrent_speed());
  double vx = s.first;
  double vy = s.second;
  double v_threhold;
  if (!useGreet())
  {
    v_threhold = v_th_;
  }
  else
  {
    if (!it->isClose) //(!neighbour_flag)
      v_threhold = v_th_;
    else
      v_threhold = v_th_ * 2;
  }
  // std::cout << "***********" << std::endl;
  // std::cout << "v_th: " << v_threhold << std::endl;
  if (sqrt(vx * vx + vy * vy) > v_threhold)
  {
    // std::cout << "not static" << std::endl;
    return false;
  }
  else
  {
    //  std::cout << "Static" << std::endl;
    return true;
  }
}

bool HumanExaminer::isStatic(Human it)
{
  std::pair<double, double> s = Speed_amcl(std::make_tuple(it.getCurrentX(), it.getCurrentY()), it.getCurrent_speed());
  double vx = s.first;
  double vy = s.second;
  double v_threhold;
  if (!useGreet())
  {
    v_threhold = v_th_;
  }
  else
  {
    if (!it.isClose) 
      v_threhold = v_th_;
    else
      v_threhold = v_th_ * 2;
  }

  if (sqrt(vx * vx + vy * vy) > v_threhold)
  {
    // std::cout << "not static" << std::endl;
    return false;
  }
  else
  {
    //  std::cout << "Static" << std::endl;
    return true;
  }
}