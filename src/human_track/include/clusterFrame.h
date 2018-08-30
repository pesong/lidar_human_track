#ifndef HDETECT_CLUSTERFRAME_H
#define HDETECT_CLUSTERFRAME_H

#include <recognizer.hpp>
#include <header.hpp>
#include <sensor_msgs/LaserScan.h>
#include "lgeometry.hpp"
class clusterFrame{
public:
    clusterFrame(std::string laser_frame_id){
        laser_frame_name = laser_frame_id;
    };
    ~clusterFrame(){
    };
    std::deque <Observation> observations;
    uint frame_id;
    std::string laser_frame_name;


};//
// Created by song on 16-4-9.
//


#endif //HDETECT_CLUSTERFRAME_H
