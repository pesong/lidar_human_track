#include "human.h"
#include "score_map.h"
using namespace Eigen;

int Human::id_cnt = 0;
double Human::v_th_ = 0.15;
double Human::fusion_distance = 0.5; //0.5
double Human::meanshift_invaild = 1;
double Human::track_distance = 0.45;
double Human::init_distance = 1;
unsigned char Human::wall_threshold = 230; //230
unsigned char Human::meanshift_wall = 245;
int Human::max_state_size = 20; //30
int Human::max_part_of_state_size = 6;
int Human::min_predict_size = 3; // 5
int Human::max_track_cnt = 10;   // 70
double Human::predict_step = 0.2;
double Human::init_range_ = 1.0;
double Human::dis_boundary = 1.5; //远近距离范围
int Human::max_direction_cnt = 8;
int Human::max_not_direction_cnt = 8;
int Human::stable_cnt = 3;
int Human::max_human_cnt_true =2;
int Human::max_human_cnt_false =30;


using std::random_shuffle;

std::pair<double, double> Speed_amcl(std::tuple<double, double> state, std::tuple<double, double> speed)
{
    double x1 = std::get<0>(state);
    double y1 = std::get<1>(state);
    ToAmcl(x1, y1, x1, y1);
    double x2 = x1 + std::get<0>(speed);
    double y2 = y1 + std::get<1>(speed);
    double vx = x2 - x1;
    double vy = y2 - y1;
    return std::make_pair(vx, vy);
}

//线性回归滤波 a=(Y^T * Y)^(-1)*Y^T*f, case_作用是看是什么情况调用regression，不同情况返回值不同
std::deque<std::tuple<double, double, double>> Human::regression(std::deque<std::tuple<double, double, double>> list, int case_)
{
    //std::cout << "regression" << std::endl;
    // regression
    int num = list.size();
    MatrixXd Y(num, 2);
    MatrixXd fx(num, 1);
    MatrixXd fy(num, 1);
    MatrixXd fxx(num, 1); //理论值 fxx,fyy
    MatrixXd fyy(num, 1);
    MatrixXd a1(2, 1);
    MatrixXd a2(2, 1);
    std::deque<std::tuple<double, double, double>> filtered_list;

    for (int i = 0; i < num; i++)
    {
        fx(i, 0) = std::get<0>(list[i]);       //x
        fy(i, 0) = std::get<1>(list[i]);       //y;
        Y(i, 0) = std::get<2>(list[i]) * 1000; //t,时间都是扩大到了毫秒级
        Y(i, 1) = 1;
        //  ToAmcl(fx(i, 0),fy(i, 0),amcl_x,amcl_y); //debug
        //  std::cout << " amcl_x: " << amcl_x << " amcl_y: " << amcl_y << " time: " << Y(i, 0) ; //debug
    }
    a1 = (((Y.transpose()) * Y).inverse()) * (Y.transpose()) * fx;
    a2 = (((Y.transpose()) * Y).inverse()) * (Y.transpose()) * fy;
    //std::cout << "a1: " << a1(0,0) << std::endl;
    //std::cout << "a2: " << a2(0,0) << std::endl;
    // a1 = Y.colPivHouseholderQr().solve(fx);
    // a2 = Y.colPivHouseholderQr().solve(fy);
    // std::cout << "new a1: " << a1(0,0) << std::endl;
    // std::cout << "new a2: " << a2(0,0) << std::endl;

    //std::cout << std::endl;//debug
    fxx = Y * a1;
    fyy = Y * a2;

    //case0: 和理论值偏差过大过滤
    if (case_ == 0)
    {
        double x, y, t;
        for (int i = 0; i < num; i++)
        {
            double dis = sqrt((fx(i, 0) - fxx(i, 0)) * (fx(i, 0) - fxx(i, 0)) + (fy(i, 0) - fyy(i, 0)) * (fy(i, 0) - fyy(i, 0)));
            //std::cout << " " << dis;
            // std::cout << " dis: " << dis << " truex: " << fx(i,0)<< " x: " << fxx(i,0) << " truey: " << fy(i,0) <<" y: "<<fyy(i,0);
            if (dis < 0.1)
            {
                x = std::get<0>(list[i]);
                y = std::get<1>(list[i]);
                t = std::get<2>(list[i]);
                filtered_list.push_back(std::make_tuple(x, y, t));
            }
        }
    }

    //case1: 和理论值偏差过大,留下理论值
    if (case_ == 1)
    {
        double x, y, t;
        for (int i = 0; i < num; i++)
        {
            double dis = sqrt((fx(i, 0) - fxx(i, 0)) * (fx(i, 0) - fxx(i, 0)) + (fy(i, 0) - fyy(i, 0)) * (fy(i, 0) - fyy(i, 0)));
            //std::cout << " " << dis;
            // std::cout << " dis: " << dis << " truex: " << fx(i,0)<< " x: " << fxx(i,0) << " truey: " << fy(i,0) <<" y: "<<fyy(i,0);
            if (dis < 0.1)
            {
                x = std::get<0>(list[i]);
                y = std::get<1>(list[i]);
                t = std::get<2>(list[i]);
                filtered_list.push_back(std::make_tuple(x, y, t));
            }
            else
            {
                x = fxx(i, 0);
                y = fyy(i, 0);
                t = std::get<2>(list[i]);
                filtered_list.push_back(std::make_tuple(x, y, t));
            }
        }
    }
    return filtered_list;
}


//获取目标行走的速度
std::tuple<double, double, double> Human::getSpeed()
{
    //保存目标的位置，x,y,time
    std::deque<std::tuple<double, double, double>> s = get_amclStates();
    //std::cout << " ID: " << getId() << std::endl;
    //ROS_WARN("s size: %d", s.size());
    if (s.size() > 4)
    {
        s = regression(s, 0);
    }
    if (s.size() == 0)
        s = get_amclStates();
    //ROS_WARN("filtered s size: %d", s.size());
    double vx, vy;
    //std::cout << "before calculate V" << std::endl;
    //给定目标的两个位置，计算他的速度
    auto quick = [&](double &vx, double &vy, int i1, int i2) {
        double x1, y1, x2, y2, delta_t;
        x1 = std::get<0>(s[i1]);
        y1 = std::get<1>(s[i1]);
        x2 = std::get<0>(s[i2]);
        y2 = std::get<1>(s[i2]);
        delta_t = std::get<2>(s[i2]) - std::get<2>(s[i1]);
        //std::cout << " x1: " << x1 << " x2: " << x2 << " y1: " << y1 << " y2: "<< y2 << " delta_t: " << delta_t << std::endl;
        if (delta_t == 0)
        {
            vx = 0;
            vy = 0;
        }
        else
        {
            vx = (x2 - x1) / delta_t;
            vy = (y2 - y1) / delta_t;
        }
        // std::cout << "vx: " << vx << "vy: " << vy << std::endl;
    };

    //用最小二乘计算多个点的速度，平滑速度值
    auto minsquare = [&](std::vector<std::pair<int, int>> &pairs, double &vx, double &xy) {
        int id1, id2;
        double x1, x2, y1, y2, t1, t2, sx = 0, sy = 0, l = 0;
        for (auto it = pairs.begin(); it != pairs.end(); it++)
        {
            id1 = it->first;
            id2 = it->second;
            x1 = std::get<0>(s[id1]);
            y1 = std::get<1>(s[id1]);
            t1 = std::get<2>(s[id1]);
            x2 = std::get<0>(s[id2]);
            y2 = std::get<1>(s[id2]);
            t2 = std::get<2>(s[id2]);
            sx += (x2 - x1) / (t2 - t1);
            sy += (y2 - y1) / (t2 - t1);
            l += 1;
        }
        vx = sx / l;
        vy = sy / l;
    };
    //当保存的目标位置少于阈值，速度直接给0
    if (s.size() < min_predict_size) //s是同一个人物之前所有的state集合
    {
        //quick(vx, vy, 0 , s.size()-1);
        //return std::make_tuple(vx,vy,std::get<2>(s.back()));
        return std::make_tuple(0, 0, std::get<2>(s.back()));
    }

    static std::default_random_engine e;
    int full_size = s.size();
    //std::normal_distribution<double> d(0.5,0.25);
    std::uniform_int_distribution<int> d(1, unsigned(time(NULL)));

    double pick_rate = 0.3;
    double thres_rate = 0.4;
    double thres_error = 0.05;
    int pick_size = pick_rate * full_size;
    int thres_size = thres_rate * full_size;
    auto inliner = std::vector<std::pair<int, int>>();
    double min_error = 99999;
    std::vector<std::pair<int, int>> full;
    std::vector<std::pair<int, int>> pick;
    pick.resize(pick_size);

    //full[(1,2),(2,3),(3,4),(4,5)]
    for (int i = 0; i < full_size - 3; i++)
        full.push_back(std::make_pair(i, i + 3));

    std::vector<int> index_list;
    for (int i = 0; i < full_size - 3; i++)
    {
        index_list.push_back(i);
    }

    for (int k = 10; k; k--)
    {
        int a = d(e);
        //std::cout << a << std::endl;
        srand(a);
        random_shuffle(index_list.begin(), index_list.end()); /* 打乱元素 */
        for (int i = 0; i < pick_size; i++)
        {
            //int cc = d(e);//取一个0到full_size-2的随机数
            int cc = index_list.at(i); //ransac随机抽样一致算法，调用最小二乘算法平滑目标速度值
            // std::cout << cc << " ";
            pick.at(i) = std::make_pair(cc, cc + 3);
        }
        // std::cout << std::endl;

        double vx1, vy1, vx2, vy2, error, ierror = 0;
        minsquare(pick, vx1, vy1); //vx1,vy1速度平滑值
        std::vector<std::pair<int, int>> temp;
        for (int i = 0; i < full.size(); i++) //对每个state
        {
            quick(vx2, vy2, full.at(i).first, full.at(i).second);
            error = sqrt((vx2 - vx1) * (vx2 - vx1) + (vy2 - vy1) * (vy2 - vy1));
            if (error > thres_error)
                continue;
            ierror += error;
            temp.push_back(full.at(i));
        } //一旦两种方法算出来的速度不一样，就停止循环
        // std::cout << "temp size: " << temp.size() << "threshole size: " << thres_size <<std::endl;
        if (temp.size() > thres_size && ierror < min_error)
        {
            inliner.swap(temp);
            min_error = ierror;
            vx = vx1; //最后还是用最小二乘算出来的速度
            vy = vy1;
        }
    }

    if (!inliner.size())
    {
        //std::cout << "quick!" <<std::endl;
        quick(vx, vy, 0, s.size() - 1);
        int num = get_amclStates().size();
        //std::cout << " final vx: " << vx << " final vy: " << vy <<std::endl;

        //std::cout << "quick_s[0]x: " << std::get<0>(getStates()[0]) << " quick_s[0]y: " << std::get<1>(getStates()[0]);
        //std::cout << "quick_s[num]x: " << std::get<0>(getStates()[num - 1]) << " quick_s[num]y: " << std::get<1>(getStates()[num - 1]) << std::endl;
        return std::make_tuple(vx, vy, std::get<2>(s.back()));
    }
    //std::cout << "minsquare!" <<std::endl;
    //std::cout << " final vx: " << vx << " final vy: " << vy <<std::endl;
    // std::cout << "min_vx: " << vx << " min_vy: " << vy <<std::endl;
    return std::make_tuple(vx, vy, std::get<2>(s.back()));
}

// }

double Human::getCurrentX()
{
    return std::get<0>(states.back());
}

double Human::getCurrentY()
{
    return std::get<1>(states.back());
}

double Human::getAmclX()
{
    return std::get<0>(amcl_states.back());
}

double Human::getAmclY()
{
    return std::get<1>(amcl_states.back());
}

std::tuple<double, double> Human::getInitxy()
{
    return std::make_tuple(init_x, init_y);
}


bool Human::operator==(const Human &rhs) const
{
    return (id == rhs.id);
}

//根据前面算出来的速度，计算目标的轨迹，future_time代表几秒内目标的轨迹， use_map代表是否用到【处理后的map】，用来预测目标轨迹（转弯，直行）。
std::vector<std::tuple<double, double, double, double, double>> Human::predict(double future_time, bool use_map)
{
    std::vector<std::tuple<double, double, double, double, double>> ret;

    double vx = std::get<0>(speeds.back());
    double vy = std::get<1>(speeds.back());

    double v = sqrt(vx * vx + vy * vy);
    if (v > 0.4)
        v = 0.4;
    double inte_x = getAmclX(); // getCurrentX();
    double inte_y = getAmclY();// getCurrentY();
    double inte_t = getCurrentTime();
    double inte_o = cv::fastAtan2((float)vy, (float)vx) * 0.017453292519943; //目标方向

    double dxt = vx, dyt = vy;

    cv::Mat omap = globalOrientMap();
    cv::Mat smap = globalScoreMapProto();
    cv::Mat smap_show;
    cv::cvtColor(smap, smap_show, CV_GRAY2RGB);

    // std::cout << "time: " << getCurrentTime() << std::endl;
    //当角度大于360度的时候，减去360度
    auto check = [](double &o) {
        if (o > 6.2831853071796)
            o -= 6.2831853071796;
        if (o < 0)
            o += 6.2831853071796;
    };

    double goal_time = inte_t + future_time;
    double dxh = 0, dyh = 0;

    double predictx_nomap = inte_x + v * future_time * cos(inte_o);
    double predicty_nomap = inte_y + v * future_time * sin(inte_o);
    projectToMap(predictx_nomap, predicty_nomap, predictx_nomap, predicty_nomap);
    if (smap.at<unsigned char>((int)predicty_nomap, (int)predictx_nomap) < wall_threshold)
    {
        use_map = false;
    }

    while (inte_t < goal_time) //用时间预测
    {
        if (use_map)
        {
            double mx, my;
            double temp_o = inte_o;
            if (projectToMap(inte_x, inte_y, mx, my))
            {
                circle(smap_show, cvPoint(mx, my), 3, CV_RGB(255, 0, 0), -1);

                double weight = smap.at<unsigned char>((int)my, (int)mx) / 255.0;
                double o = omap.at<float>((int)my, (int)mx) + 3.1415926535898; //反向转180度！注意！现在方向转向图像变暗的方向了！！！！
                check(o);
                double dx2 = cos(temp_o); //上次运动方向
                double dy2 = sin(temp_o);
                double dx1 = cos(o); //地图梯度方向，指向图像变暗的方向！！！！！！
                double dy1 = sin(o);
                //下一次运动方向由上次运动方向，地图梯度方向（共0.3），这次计算出的方向共同决定 0.7
                if(greet_track_MODE())
                {
                    dxt = dx2 * 0.9 + 0.1 * (dxh * (1 - weight) + weight * dx1); //把速度和往墙的方向加weight
                    dyt = dy2 * 0.9 + 0.1 * (dyh * (1 - weight) + weight * dy1); //上一次的速度和离墙距离
                }
                else
                {
                    dxt = dx2 * 0.7 + 0.3 * (dxh * (1 - weight) + weight * dx1); //把速度和往墙的方向加weight
                    dyt = dy2 * 0.7 + 0.3 * (dyh * (1 - weight) + weight * dy1); //上一次的速度和离墙距离
                }
                dxh = dx2;
                dyh = dy2;
                temp_o = cv::fastAtan2((float)dyt, (float)dxt) * 0.017453292519943; //运动方向调整
                check(temp_o);
            }
            inte_o = temp_o;
        }
        inte_t += predict_step;
        inte_x += v * predict_step * cos(inte_o); //速度*时间*方向
        inte_y += v * predict_step * sin(inte_o);
        //  ROS_INFO("inte_x: %f, inte_y %f",inte_x,inte_y);
        ret.push_back(std::make_tuple(inte_x, inte_y, inte_t, dxt, dyt));
    }

    if (0)
    {
        imshow("smap", smap_show);
        cv::waitKey(1);
    }
    return ret;
}

double Human::doorScore(double full_mark)
{
    //std::tuple <double,double> init_position;
    std::tuple<double, double> p = init_position;
    if (PtInPolygon(p, doorPoly(), doorPoly().size()))
        return full_mark;
    else
        return 0;
}

double Human::targetScore(double full_mark)
{
    return 0;
}

double Human::distScore(double full_mark)
{
    double dist = distToRobotPose(std::get<0>(amcl_states.back()), std::get<1>(amcl_states.back()));
    if (dist > 5)
        return 0;
    else
        return (1 - dist / 5) * full_mark;
    // else if ((dist < 5) && (dist > 3))
    //     return full_mark * 0.25;
    // else if ((dist < 3) && (dist > 2))
    //     return full_mark * 0.5;
    // else if ((dist < 2) && (dist > 1))
    //     return full_mark * 0.75;
    // else
    //     return full_mark;
}

double Human::robotAngleScore(double full_mark)
{
    if (this->isStatic())
    {
        return 0;
    }
    std::tuple<double, double> v = std::make_tuple(std::get<0>(speeds.back()), std::get<1>(speeds.back()));
    std::tuple<double, double> pos = std::make_tuple(std::get<0>(states.back()), std::get<1>(states.back()));
    std::pair<double, double> s = Speed_amcl(pos, v);
    double vx = s.first;
    double vy = s.second;
    double pub_oo = cv::fastAtan2(vy, vx); //运动角度
    double greet_y = cv::fastAtan2((robotPoseY() - std::get<1>(amcl_states.back())), (robotPoseX() - std::get<0>(amcl_states.back())));
    if (greet_y < 0) //统一到0-360
        greet_y = (180 - fabs(greet_y)) + 180;

    double final_angle = fabs(greet_y - pub_oo);
    if ((final_angle < 22.5) || (final_angle > (360 - 22.5)))
        return full_mark;
    else if ((final_angle < 45) || (final_angle > (360 - 45)))
        return full_mark * 0.75;
    else if ((final_angle < 67.5) || (final_angle > (360 - 67.5)))
        return full_mark * 0.5;
    else if ((final_angle < 90) || (final_angle > (360 - 90)))
        return full_mark * 0.25;
    else
        return 0;
}

double Human::mainAngleScore(double full_mark)
{
    if (this->isStatic())
    {
        return 0;
    }
    std::tuple<double, double> v = std::make_tuple(std::get<0>(speeds.back()), std::get<1>(speeds.back()));
    std::tuple<double, double> pos = std::make_tuple(std::get<0>(states.back()), std::get<1>(states.back()));
    std::pair<double, double> s = Speed_amcl(pos, v);
    double vx = s.first;
    double vy = s.second;
    double pub_oo = cv::fastAtan2(vy, vx); //运动角度

    double r, p, greet_y;
    Toangle(greetOrientation(), r, p, greet_y);
    greet_y = greet_y * 57.3;
    if (greet_y < 0) //统一到0-360
        greet_y = (180 - fabs(greet_y)) + 180;

    double final_angle = fabs(greet_y - pub_oo);
    if ((final_angle < 22.5) || (final_angle > (360 - 22.5)))
        return full_mark;
    else if ((final_angle < 45) || (final_angle > (360 - 45)))
        return full_mark * 0.75;
    else if ((final_angle < 67.5) || (final_angle > (360 - 67.5)))
        return full_mark * 0.5;
    else if ((final_angle < 90) || (final_angle > (360 - 90)))
        return full_mark * 0.25;
    else
        return 0;
}

double Human::greetScore(std::vector<double> imp)
{
    double greet_score = doorScore(imp[0]) + targetScore(imp[1]) + distScore(imp[2]) + robotAngleScore(imp[3]) + mainAngleScore(imp[4]);
    ROS_INFO("ID: %d, doorScore: %f, distScore: %f, robotAngle: %f, mainAngle: %f, SumScore: %f", this->getId(), doorScore(imp[0]), distScore(imp[2]), robotAngleScore(imp[3]), mainAngleScore(imp[4]), greet_score);
    ROS_WARN("ID: %d, doorScore: %f, targetScore: %f, distScore: %f, robotAngle: %f, mainAngle: %f", this->getId(), imp[0], imp[1], imp[2], imp[3], imp[4]);
    //std::cout << "ID: " << this->getId() << " doorScore: " << doorScore(imp[0]) << " disScore: " << distScore(imp[2]) << " robotAngle: " << robotAngleScore(imp[3])
    //<< " mainAngle: " << mainAngleScore(imp[4]) << " Score: " << greet_score << std::endl;
    return greet_score;
}

//没有转到amcl平面，但不影响，icp改变的是角度
bool Human::isFar()
{
    std::deque<std::tuple<double, double, double>> &s = this->get_amclStates();
    int num = s.size();
    std::deque<std::tuple<double, double>> path;
    if (num < 12)
    {
        if (this->isfar)
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

bool Human::isDirection_ok()
{
    bool result;
    if (MODE() == 2) //发传单对方向无限制
        result = true;
    if (MODE() == 0 || MODE() == 1)
    {
        std::pair<double, double> s = Speed_amcl(std::make_tuple(this->getCurrentX(), this->getCurrentY()), this->getCurrent_speed());
        double vx = s.first;
        double vy = s.second;
        double pub_oo = cv::fastAtan2(vy, vx) * 0.017453292519943;

        //geometry_msgs::Quaternion direction = tf::createQuaternionMsgFromYaw(pub_oo);
        double r, p, greet_y;
        if (MODE() == 0)
            Toangle(greetOrientation(), r, p, greet_y);
        else if (MODE() == 1)
        {
            double x, y;
            //ToAmcl(this->getCurrentX(), this->getCurrentY(), x, y);
            x = this->getAmclX();
            y = this->getAmclY();
            greet_y = angle2robot(x, y, robotPoseX(), robotPoseY());
        }

        double ang = Angle();
        if (greet_y < 0) //统一到0-360
            greet_y = (3.1416 - fabs(greet_y)) + 3.1416;
        if (!this->isClose)
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
    if (MODE() == 3) //MODE3的direction_ok的定义是和主方向并且和机器人夹角小于等于90度
    {
        std::pair<double, double> s = Speed_amcl(std::make_tuple(this->getCurrentX(), this->getCurrentY()), this->getCurrent_speed());
        double vx = s.first;
        double vy = s.second;
        double pub_oo = cv::fastAtan2(vy, vx) * 0.017453292519943;

        //geometry_msgs::Quaternion direction = tf::createQuaternionMsgFromYaw(pub_oo);
        double r, p, x, y, greet_y_1, greet_y_2;
        Toangle(greetOrientation(), r, p, greet_y_1);
        x = this->getAmclX();
        y = this->getAmclY();
        // ToAmcl(this->getCurrentX(), this->getCurrentY(), x, y);
        greet_y_2 = angle2robot(x, y, robotPoseX(), robotPoseY());

        double ang = 1.571;
        if (greet_y_1 < 0) //统一到0-360
            greet_y_1 = (3.1416 - fabs(greet_y_1)) + 3.1416;
        if (greet_y_2 < 0) //统一到0-360
            greet_y_2 = (3.1416 - fabs(greet_y_2)) + 3.1416;

        //角度偏差90度
        if (((fabs(greet_y_1 - pub_oo) < ang) || (fabs(greet_y_1 - pub_oo) > (6.28 - ang))) || ((fabs(greet_y_2 - pub_oo) < ang) || (fabs(greet_y_2 - pub_oo) > (6.28 - ang))))
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

    if ((result == true) && (!isStatic()))
        this->directionOK_cnt++;
    else if (result == false)
        this->directionOK_cnt = 0;

    if ((result == false) && (!isStatic()))
        this->direction_not_OK_cnt++;
    else if ((result == true) || isStatic())
        this->direction_not_OK_cnt = 0;

    return result;
}

bool Human::isStatic()
{
    std::pair<double, double> s = Speed_amcl(std::make_tuple(this->getCurrentX(), this->getCurrentY()), this->getCurrent_speed());
    double vx = s.first;
    double vy = s.second;
    double v_threhold;
    if (!useGreet())
    {
        v_threhold = this->v_th_;
    }
    else
    {
        if (!this->isClose)
            v_threhold = this->v_th_;
        else
            v_threhold = this->v_th_ * 2;
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

//必须转到amcl平面
bool Human::is_close_to_door()
{
    std::deque<std::tuple<double, double, double>> &s = this->getStates();
    int num = s.size();
    std::deque<std::tuple<double, double>> path;
    if (num < 20)
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