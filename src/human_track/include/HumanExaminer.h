#include "laser.h"
#include "readParams.h"
#include <ros/ros.h>

class HumanExaminer
{
    public:
    double v_th_;
    int MODE;
    bool isFar(std::_List_iterator<Human> &it);
    bool isFar(Human it);
    bool isDirection_ok(std::_List_iterator<Human> it);
    bool isStatic(std::_List_iterator<Human> it);
    bool isStatic(Human it);
    bool is_close_to_door(std::_List_iterator<Human> it);

    HumanExaminer(){};
    HumanExaminer(int M)
    {
        MODE = M;
        v_th_ = 0.1;
    };
};

//double distToRobotPose(double x, double y);
//double distToDoorPose(double x, double y);