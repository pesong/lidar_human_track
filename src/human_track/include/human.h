#ifndef __Human_ziwei
#define __Human_ziwei

#include <vector>
#include <map>
#include <tuple>
#include <cv.h>
#include <math.h>
#include <random>
#include "readParams.h"
#include <algorithm>
#include <ctime>
#include <time.h>
#include <Eigen/Dense>
#include "track_utils.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv/cv.hpp>



std::pair<double, double> Speed_amcl(std::tuple<double, double> state, std::tuple<double, double> speed);
// double distToRobotPose(double x, double y);
// double distToDoorPose(double x, double y);

class Human
{
private:
    int id;
    int score;
    int wall_score;
    int cnt;//跟踪次数,用于控制human消失时长
    std::deque<std::tuple<double,double,double> > states;//x,y,time
    std::deque<std::tuple<double,double,double> > speeds;//x,y,time,speeds就是在amcl平面上
    std::deque<std::tuple<double,double,double> > amcl_states;//x,y,time,已转换到amcl平面上的states
    std::deque<std::tuple<double,double,double> > part_of_states;//x,y,time,后一部分的位置，用于计算Current postion
    cv::Rect box;
    cv::Rect camera_box;
    static int id_cnt;
    double init_x,init_y,init_time_;
    std::tuple<double,double,double> getSpeed();
    std::vector<std::pair<double, double>> laser_around;
    int label;




public:
    //parameters
    double meanshift_;
    double greet_score;
    double towards_robot;
    int directionOK_cnt;
    int direction_not_OK_cnt;
    static double init_range_;
    std::tuple <double,double> init_position;
    bool first_position_flag; //是否需要更新初始位置标志位, 如果在范围外之后就要更新
    int cnt_combo;
    double toRobot;
    double toTarget;
    bool isDiscard;
    bool from_door;
    bool timer_ok;       //是否在有效跟踪时间内
    static double v_th_;
    static double fusion_distance;//0.5
    static double track_distance;
    static double meanshift_invaild;
    static unsigned char wall_threshold;//245
    static unsigned char meanshift_wall;
    static int stable_cnt; //3
    static int max_state_size;//30
    static int max_part_of_state_size;//30
    static int min_predict_size;// 5
    static int max_track_cnt;// 70
    static double predict_step;
    static double dis_boundary;
    static int max_direction_cnt;
    static int max_not_direction_cnt;
    static double init_distance;
    static int max_human_cnt_false;
    static int max_human_cnt_true;
    bool isfar;
    bool isClose;
    bool isReplaced;
    bool move_flag;
    bool isHuman;
    bool isHuman_tmp;
    bool isProject;
    int human_cnt;


    std::vector<std::pair<double, double>> getLaser(){
        return laser_around;
    }

    int getLabel(){
        return label;
    }


    void update(double x, double y, double t, std::vector<std::pair<double,double>> laser)
    {
        //std::cout << "update" << std::endl;
        double amcl_x,amcl_y;
        cnt++;
        states.push_back(std::make_tuple(x,y,t));
        ToAmcl(x,y,amcl_x,amcl_y);
        amcl_states.push_back(std::make_tuple(amcl_x,amcl_y,t));
        part_of_states.push_back(std::make_tuple(x,y,t));
        speeds.push_back(getSpeed()); //speed已经全部转到amcl平面上了
        //std::cout << "update speed size: " << speeds.size() <<std::endl;

        double vx = std::get<0>(getSpeed());
        double vy = std::get<1>(getSpeed());
        if((sqrt(vx*vx+vy*vy)>0.05)&&(getMD()>0.2))
            move_flag = true;
        if(states.size()>max_state_size)
        {
            states.pop_front(); //一个人很多次的states和速度
            speeds.pop_front();
            amcl_states.pop_front();
        }
        if(part_of_states.size()>max_part_of_state_size)
        {
            part_of_states.pop_front(); //一个人很多次的states和速度
        }

        laser_around = laser;
        double predict_x,predict_y;
        ToAmcl(x,y,predict_x,predict_y);
        double dis = sqrt((predict_x-robotPoseX())*(predict_x-robotPoseX())+(predict_y-robotPoseY())*(predict_y-robotPoseY()));
        if (dis>dis_boundary)
            isClose = false;
        else
            isClose = true;
        if(useGreet() && greet_track_MODE())
        {
            greet_score = greetScore(Importance());
            //ROS_INFO("1: %d, 2: %d, 3: %d, 4: %d, 5: %d", Importance()[0],Importance()[1],Importance()[2],Importance()[3],Importance()[4]);
        }

        //std::cout << " human_cnt: " << human_cnt << " isHuman: " << isHuman << " isHuman_tmp: "<< isHuman_tmp << std::endl;
        if(isHuman != isHuman_tmp)
        {
            human_cnt++; //
            int max_human_cnt;
            if(isHuman && !isHuman_tmp)
                max_human_cnt = max_human_cnt_false;
            if(!isHuman && isHuman_tmp)
                max_human_cnt = max_human_cnt_true;

            if(human_cnt > max_human_cnt)
            {
                human_cnt = 0;
                isHuman = isHuman_tmp;
            }
        }
        else
            human_cnt = 0;
    };

    void IDoffset(Human target)
    {
        double amcl_x, amcl_y;
        init_position = target.init_position;
        first_position_flag = target.first_position_flag;
        from_door = target.from_door;
        directionOK_cnt = target.directionOK_cnt;
        direction_not_OK_cnt = target.direction_not_OK_cnt;
        double x = this->getCurrentX();
        double y = this->getCurrentY();
        //std::cout << "xxx: " << x << " yyy: " << y << std::endl;
        this->states = target.states;
        this->amcl_states = target.amcl_states;
        this->part_of_states = target.part_of_states;
        this->speeds = target.speeds;
        this->isDiscard = target.isDiscard;
        ToAmcl(x,y,amcl_x,amcl_y);
        double t = getTimestamp();
        states.push_back(std::make_tuple(x,y,t));
        amcl_states.push_back(std::make_tuple(amcl_x,amcl_y,t));
        part_of_states.push_back(std::make_tuple(x,y,t));
        speeds.push_back(getSpeed()); //speed已经全部转到amcl平面上了
        //std::cout << "update speed size: " << speeds.size() <<std::endl;
        if(states.size()>max_state_size)
        {
            states.pop_front(); //一个人很多次的states和速度
            speeds.pop_front();
            amcl_states.pop_front();
        }
        //std::cout << "Currentxxx: " << this->getCurrentX() << " Currentyyy: " << this->getCurrentY() << std::endl;
        if(part_of_states.size()>max_part_of_state_size)
        {
            part_of_states.pop_front(); //一个人很多次的states和速度
        }
    };

    std::deque<std::tuple<double, double, double>> regression(std::deque<std::tuple<double, double, double>> list, int case_);

    void renew_position(double x, double y, double t)
    {
        //double tt = std::get<3>(states.back());
        states.pop_back();
        states.push_back(std::make_tuple(x,y,t));
    };



    int getNewestID()
    {
        return id_cnt;
    };

    void addScore()
    {
        score = score+2;
    };

    void reduceScore()
    {
        score = score-1;
    };

    int getCnt()
    {
        return cnt;
    }

    int getScore(){
        return wall_score;
    }

    double getMD() //get move distance
    {
        double dx,dy,distance;
        dx = std::get<0>(states.back())-init_x;
        dy = std::get<1>(states.back())-init_y;
        distance = sqrt(dx*dx+dy*dy);
        return distance;
    }

    double getAvrspeed()
    {
        double sx,sy,ss;
        sx = std::get<0>(speeds.back());
        sy = std::get<1>(speeds.back());
        ss = sqrt(sx*sx+sy*sy);
        return ss;
    }

    std::tuple<double,double> getCurrent_speed() //amcl平面上
    {
        std::tuple<double,double> s;
        s = std::make_tuple(std::get<0>(speeds.back()), std::get<1>(speeds.back()));
        return s;
    }

    double getCurrentX();

    double getCurrentY();

    double getAmclX();

    double getAmclY();



    double getCurrentTime()
    {
        return std::get<2>(states.back());
    }

    double getLastTime()
    {
        return (std::get<2>(states.back()) - init_time_);
    }


    cv::Rect& getBoundingBox()
    {
        return box;
    };

    cv::Rect& getcameraBoundingBox()
    {
        return camera_box;

    };

    void updateBoundingBox(cv::Rect& r)
    {
        box = r;
    };

    void updateCameraBoundingBox(cv::Rect& r)
    {
        camera_box = r;
    };

    bool isOverTrack()
    {
        return cnt > max_track_cnt;
    };

    bool scoreIvalid()
    {
        return score < 0;
    };

    void resetTrackCnt()
    {
        cnt = 0;
    };

    int getId()
    {
        return id;
    }

    std::deque<std::tuple<double,double,double> >& getStates()
    {
        return states;
    }

    std::deque<std::tuple<double,double,double> >& get_amclStates()
    {
        return amcl_states;
    }

    std::deque<std::tuple<double,double,double> >& get_part_of_States()
    {
        return part_of_states;
    }

    std::tuple<double, double> getInitxy();

    bool operator==(const Human &rhs) const;

    double doorScore(double full_mark);
    double targetScore(double full_mark);
    double distScore(double full_mark);
    double robotAngleScore(double full_mark);
    double mainAngleScore(double full_mark);
    double greetScore(std::vector<double> imp);


    //from human examiner
    bool isFar();
    bool isDirection_ok();
    bool isStatic();
    bool is_close_to_door();


    Human(){};
    Human(double _x, double _y, double _t, std::vector<std::pair<double, double>> laser)
    {
        id = id_cnt++;
        cnt = 0;
        directionOK_cnt = 0;
        direction_not_OK_cnt = 0;
        //cnt_combo =0;
        score = 20;
        update( _x, _y, _t, laser);
        //std::cout << "update_laser " << laser.size() << std::endl;
        init_x = _x;
        init_y = _y;
        init_time_ = _t;
        timer_ok = true;
        isDiscard = false;
        from_door = false;
        wall_score = 0;
        first_position_flag = false;
        init_position = std::make_tuple(0, 0);
        isClose = false;
        isfar = false;
        isReplaced = false;
        move_flag = false;
        label = 0;
        isHuman = false;
        isHuman_tmp = false;
        human_cnt = 0;
        isProject = false;

    };

    std::vector<std::tuple<double,double,double,double,double> > predict(double future_time, bool use_map);
};


#endif