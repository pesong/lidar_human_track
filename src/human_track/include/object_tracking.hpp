#ifndef OBJECTDETECTION_HPP
#define OBJECTDETECTION_HPP

#include <newmat/newmat.h>
#include <stdio.h>
#include <string>
#include <deque>
#include <map>

#include <header.hpp>
#include <human.hpp>
#include <observation.hpp>
#include <detector.hpp>

namespace ObjectTracking
{
    void loadCfg(detectorParameters params);
    void loadCfg(std::string cfg);

    void eliminate(deque<HumanCluster> &humans);
    void predict(deque<HumanCluster> &humans);
    void pair(deque<HumanCluster> &humans, deque<Observation> &observations, map<int, int> &pairs);
    void update(deque<HumanCluster> &humans, deque<Observation> &observations, map<int, int> &pairs);

    float calculateMahDis(Observation &observation, HumanCluster &human);
    float calculateEucDis(Observation &observation, HumanCluster &human);
}

#endif // OBJECTDETECTION_HPP
