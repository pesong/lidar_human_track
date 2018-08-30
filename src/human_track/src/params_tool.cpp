#include "params_tool.h"

double &laser_xx()
{
    static double v;
    return v;
}