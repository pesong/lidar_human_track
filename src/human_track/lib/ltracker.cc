#include "ltracker.h"
#include "Ferns.h"
#include "laser.h"

using cv::HOGDescriptor;
using cv::Rect;
using namespace Eigen;

inline float getOverlap(const cv::Rect &b1, const cv::Rect &b2) {
#define min___(a, b) (a > b ? b : a)
#define max___(a, b) (a < b ? b : a)
    int ws1 = min___(b1.x + b1.width, b2.x + b2.width) - max___(b1.x, b2.x);
    int hs1 = min___(b1.y + b1.height, b2.y + b2.height) - max___(b1.y, b2.y);
    float o = max___(0, ws1) * max___(0, hs1);
    o = o / (b1.width * b1.height + b2.width * b2.height - o);
    return o;
}

inline double dis2line(std::pair<double, double> a, std::pair<double, double> b, std::pair<double, double> s) {
    double ab = sqrt((a.first - b.first) * (a.first - b.first) + (a.second - b.second) * (a.second - b.second));
    double as = sqrt((a.first - s.first) * (a.first - s.first) + (a.second - s.second) * (a.second - s.second));
    double bs = sqrt((s.first - b.first) * (s.first - b.first) + (s.second - b.second) * (s.second - b.second));

    double cos_A = (pow(as, 2.0) + pow(ab, 2.0) - pow(bs, 2.0)) / (2 * ab * as);
    double sin_A = sqrt(1 - pow(cos_A, 2.0));
    return as * sin_A;
}


inline void theBaseLink(double xi, double yi, double &xo, double &yo) {
    double pose_x = amclPoseX();
    double pose_y = amclPoseY();
    double amcl_pos_sin = amclPoseSin();
    double amcl_pos_cos = amclPoseCos();
    xo = amcl_pos_cos * (xi - pose_x) + amcl_pos_sin * (yi - pose_y);
    yo = amcl_pos_cos * (yi - pose_y) - amcl_pos_sin * (xi - pose_x);
}

//返回面积比
inline float area(cv::Rect A, cv::Rect B) {
    cv::Rect max = (A.width * A.height > B.width * B.height ? A : B);
    cv::Rect min = (A.width * A.height < B.width * B.height ? A : B);
    float o = (float) (min.width * min.height) / (max.width * max.height); //0.3一下肯定不是对应
    return o;
}

double dist2Reproject(cv::Rect M, Human hum) {
    std::pair<double, double> point = ReProject(M, hum);
    double xo, yo;
    //theBaseLink(hum.getAmclX(), hum.getAmclY(), xo, yo);
    double dx = hum.getAmclX() - point.first;
    double dy = hum.getAmclY() - point.second;
    double distance = sqrt(dx * dx + dy * dy);
    return distance;
}

void Ltracker::init_ncs(ros::NodeHandle ph_) {
    retCode_face = mvncGetDeviceName(1, devName_face, NAME_SIZE);
    if (retCode_face != MVNC_OK) {   // failed to get device name, maybe none plugged in.
        printf("No NCS face devices found\n");
        exit(-1);
    }

    // Try to open the NCS device via the device name
    retCode_face = mvncOpenDevice(devName_face, &deviceHandle_face);
    if (retCode_face != MVNC_OK) {   // failed to open the device.
        printf("Could not open NCS device\n");
        exit(-1);
    }

    // deviceHandle is ready to use now.
    // Pass it to other NC API calls as needed and close it when finished.
    printf("Successfully opened NCS device!\n");

    // import graph to ncs
    std::string graphPath;
    std::string graphModel;
    // Path to graph file.
    ph_.param("facenet_model/graph_file/name", graphModel, std::string("facenet_graph01"));
    ph_.param("graph_path", graphPath, std::string("/default"));
    std::cout << "facenet_model: " << graphModel << std::endl;
    graphPath += "/" + graphModel;
    GRAPH_FILE_NAME_face = new char[graphPath.length() + 1];
    strcpy(GRAPH_FILE_NAME_face, graphPath.c_str());

    // Now read in a graph file
    graphFileBuf_face = LoadFile(GRAPH_FILE_NAME_face, &graphFileLen_face);

    // allocate the graph
    retCode_face = mvncAllocateGraph(deviceHandle_face, &graphHandle_face, graphFileBuf_face, graphFileLen_face);
    if (retCode_face != MVNC_OK) {   // error allocating graph
        printf("Could not allocate graph for file: %s\n", GRAPH_FILE_NAME_face);
        printf("Error from mvncAllocateGraph is: %d\n", retCode_face);
    } else {
        printf("Successfully allocate face graph for file: %s\n", GRAPH_FILE_NAME_face);
        g_graph_Success_face = true;
    }
}

void Ltracker::freeOperation(double x, double y) {
    //std::cout << "freeOperation" << std::endl;
    switch (free_MODE()) {
        case 0: //回原点模式
        {
            // std::cout << "back_init" << std::endl;
            motion_.back_init(tp, ti, rs, stay_flag, no_task, neighbour_flag);
        }
            break;
        case 1: {
            //std::cout << "random_search" << std::endl;
            motion_.random_search(x, y, tp, ti, rs, stay_flag, no_task, neighbour_flag);
        }
            break;
    }
}

//initialize classifiers
void Ltracker::classifier_init(cv::Mat show, bool draw) {
    //std::cout << "roi1:" << target.getBoundingBox() << std::endl;
    cv::Mat roi = frame(ti.target.getBoundingBox());
    cv::resize(roi, roi, cv::Size(64, 128));
    // std::cout << "roi2:" << target.getBoundingBox() << std::endl;
    if (draw)
        cv::rectangle(show, ti.target.getBoundingBox(), cv::Scalar(0, 255, 255), 2);
    target_width_proto = ti.target.getBoundingBox().width;
    target_height_proto = ti.target.getBoundingBox().height;
    kcf.init(ti.target.getBoundingBox(), frame); //kcf初始化，提取目标特征，用来判断目标是否跟丢。
    fern.initParams();                           //初始化随机蕨，是个分类器，当跟丢时，通过该分类器判断新目标是否是原目标。
    std::vector <cv::Size> tmp_size = {cv::Size(64, 128)};
    fern.init(tmp_size);
}

//fern sampling function
void fern_sampling(cv::Rect result_box, Ferns &fern, cv::Mat frame) {
    std::vector < std::pair < std::vector < int > , int >> fern_data; //(一个vector, 一个数)
    //std::cout<<result_box<<"\t 1"<<std::endl;
    // std::cout << "x " << result_box.x << " y " << result_box.y << std::endl;
    if (result_box.x > 0 && result_box.y > 0 && (result_box.x + result_box.width) < frame.cols &&
        (result_box.y + result_box.height) < frame.rows) {
        cv::Mat patch = frame(result_box); //当跟踪目标的矩形框超过图像范围，有可能报错
        //std::cout<<result_box<<"\t 2"<<std::endl;
        cv::resize(patch, patch, cv::Size(64 + 24, 128 + 24));
        cv::cvtColor(patch, patch, CV_RGB2GRAY);
        std::vector<int> fern_result;
        fern_result.resize(fern.getNstructs());
        //采取正样本
        for (int x = 0; x <= 24; x += 4)
            for (int y = 0; y <= 24; y += 4) {
                cv::Mat tpatch = patch(cv::Rect(x, y, 64, 128));
                fern.calcFeatures(tpatch, 0, fern_result);
                fern_data.push_back(std::make_pair(fern_result, true));
            }
        //采取负样本
        for (int i = 0; i < 49; i++) {
            cv::Rect neg_box;
            getNegtiveBox(result_box, neg_box);
            cv::Mat tpatch = frame(neg_box);
            cv::resize(tpatch, tpatch, cv::Size(64, 128));
            cv::cvtColor(tpatch, tpatch, CV_RGB2GRAY);
            fern.calcFeatures(tpatch, 0, fern_result);
            fern_data.push_back(std::make_pair(fern_result, false));
        }
        //正负样本进行训练，目的为了找回跟丢目标
        fern.trainF(fern_data, 1);
    }
}

//////判断目标是否为人（HOG）
//bool isHuman(cv::Mat &show, Human it, bool draw)
//{
//  // The HoG detector for the image
//  cv::HOGDescriptor hog;
//  // Vectors to hold the class and the probability of the ROI
//  std::vector<cv::Rect> hogFound;
//  std::vector<double> hogPred;
//  //ROS_INFO("HOG!");
//  if (draw)
//    cv::rectangle(show, it.getBoundingBox(), cv::Scalar(255, 0, 255), 2);
//  if (draw)
//    cv::putText(show, std::to_string(it.getId()),
//                cv::Point(it.getBoundingBox().x, it.getBoundingBox().y),
//                cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 255), 1, CV_AA); //粉色的框 ,判断人的背景
//  //ROS_INFO("HOG1!");
//  // Set the default detector
//  Rect rect0 = it.getBoundingBox();
//  std::vector<Rect> foundM;
//  std::vector<double> weightM;
//  cv::Mat crop = show(rect0);
//  //double wid = hogDetector.getWinSize().width;
//  size_t hig = hog.getDescriptorSize();
//  //ROS_INFO("hig: %d", hig);
//  hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
//  hog.detectMultiScale(crop, foundM, weightM, 0,
//                       cv::Size(4, 4), cv::Size(0, 0), 1.2, 0, false);
//  //ROS_INFO("HOG2!");
//  //检测出的人画绿框
//  for (int i = 0; i < foundM.size(); i++)
//  {
//    foundM[i].x = rect0.x + foundM[i].x;
//    foundM[i].y = rect0.y + foundM[i].y;
//    if (weightM[i] > 0.5)
//    {
//      if (draw)
//        cv::rectangle(show, foundM[i], cv::Scalar(0, 255, 0), 2);
//    }
//  }
//  if (weightM.size() > 0)
//  {
//    double prob = *max_element(weightM.begin(), weightM.end());
//    ROS_INFO("max probability %f", prob);
//    if (prob > 0.5)
//      return true;
//    else
//      return false;
//  }
//}

//随机采样的负样本
void getNegtiveBox(cv::Rect &pos, cv::Rect &neg) {
    static std::default_random_engine e;
    int cols = frameWidth();
    int rows = frameHeight();
    std::uniform_int_distribution<int> dc(1, cols - 1 - pos.width);
    std::uniform_int_distribution<int> dr(1, rows - 1 - pos.height);
    while (1) {
        neg = cv::Rect(dc(e), dr(e), pos.width, pos.height);
        if (getOverlap(neg, pos) < 0.3)
            break;
    }
}

//void Ltracker::yoloCallback(const hyolo::yoloArray &results)
//{
//  if (!results.labels.size())
//    return;
//  int num = results.labels.size();
//  std::vector<std::tuple<std::string, double, double, double, double>> items;
//  for (int i = 0; i < num; i++)
//  {
//    std::tuple<std::string, double, double, double, double> item;
//    item = std::make_tuple(results.labels[i], results.x[i], results.y[i], results.w[i], results.h[i]);
//   items.push_back(item);
//  }
//  // std::cout << "item size:" << items.size() <<std::endl;
//  this->yolo_items = items;
//}

//void Ltracker::yoloBoundingBox(cv::Mat &show)
//{
//  std::vector<cv::Rect> rects;
//  cv::Rect rect;
//  //std::cout << "yolo_items:" << yolo_items.size() <<std::endl;
//  if (this->yolo_items.size())
//  {
//    for (auto it = this->yolo_items.begin(); it != this->yolo_items.end(); it++)
//    {
//      rect.width = std::get<3>(*it);
//      rect.height = std::get<4>(*it);
//      rect.x = std::get<1>(*it) - 0.5 * rect.width;
//      rect.y = std::get<2>(*it) - 0.5 * rect.height;
//      std::cout << rect.x << ", " << rect.y << std::endl;
//      rects.push_back(rect);
//      cv::rectangle(show, rect, cv::Scalar(255, 0, 255), 2);
//      cv::putText(show, std::get<0>(*it),
//                  cv::Point(rect.x, rect.y + rect.height),
//                  cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 255), 1, CV_AA);
//    }
//  }
//  if (true)
//  {
//    sensor_msgs::ImagePtr show_msg_ptr = cv_bridge::CvImage(std_msgs::Header(), "bgr8", show).toImageMsg();
//    sensor_msgs::Image show_msg = *show_msg_ptr;
//    image_pub_.publish(show_msg);
//  }
//}

//void Ltracker::yoloDetect(cv::Mat &show)
//{
//  // cv::Mat detect_img = cv::imread("/home/ziwei/caffe2_yolo/images/cowboy-hat.jpg");
//  sensor_msgs::ImagePtr yolo_msg_ptr = cv_bridge::CvImage(std_msgs::Header(), "bgr8", show).toImageMsg();
//  yolo_msg_ptr->width = show.cols;
//  yolo_msg_ptr->height = show.rows;
//  yolo_pub_.publish(yolo_msg_ptr);
//}

//first_state_decision(tracked_huamns, time_th, pr_cnt, show, draw);
void Ltracker::first_state_decision(std::list <Human> &tracked_huamns, double time_th) {
    //tracker:
    switch (rs.first_state_) {
        case (INACTIVE): //
        {
            //证明已到上次发送位置的点
            first_found_cnt = first_found_cnt + 1;
            rs.second_state_ = INACTIVE;
            int max_cnt0;
            if (path_tracking_flag)
                max_cnt0 = 25;
            else
                max_cnt0 = 15;
            ROS_INFO("first_found_cnt: %d, path_tracking_flag: %d", first_found_cnt, path_tracking_flag);
            if ((first_found_cnt < max_cnt0) && (ti.tx != 0 && ti.ty != 0)) {
                //补偿掉的ID：先找有没有近的点，有就更新init_position，没有就算了；
                //目标在机器人附近，静止阈值会很大，远离机器人必须考虑方向
                double min_dis;
                double min_th;
                if (!path_tracking_flag) {
                    min_dis = 1;
                    min_th = 1;
                } else {
                    min_dis = 2;
                    min_th = 2;
                }
                // auto inheritor = tracked_huamns.end();
                for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
                    // double x, y;
                    // ToAmcl(it->getCurrentX(), it->getCurrentY(), x, y);
                    double x = it->getAmclX();
                    double y = it->getAmclY();
                    //在上一个目标附近找
                    if ((sqrt((ti.tx - x) * (ti.tx - x) + (ti.ty - y) * (ti.ty - y)) < min_dis) &&
                        (it->getId() > (it->getNewestID() - 5))) {
                        min_dis = sqrt((ti.tx - x) * (ti.tx - x) + (ti.ty - y) * (ti.ty - y));
                        inheritor = it;
                    }
                }
                if (min_dis < min_th) {
                    ROS_WARN("Change1 ID! from %d to %d", ti.targetID, inheritor->getId());
                    // ROS_WARN("ID: %d, isDiscard: %d", ti.targetID, ti.target.isDiscard);
                    std::list<Human>::iterator it_find_id = find(tracked_huamns.begin(), tracked_huamns.end(),
                                                                 *inheritor);
                    it_find_id->IDoffset(ti.target);
                    inheritor->IDoffset(ti.target);

                    ROS_WARN("ID: %d, isDiscard: %d", inheritor->getId(), inheritor->isDiscard);
                    // ROS_WARN("ID: %d, direction_cnt: %d; ID: %d, direction_cnt: %d, isDiscard: %d", ti.targetID, ti.target.directionOK_cnt, inheritor->getId(), inheritor->directionOK_cnt, inheritor->isDiscard);
                    //std::cout << "init_posx: " << std::get<0>(inheritor->init_position) << " init_posy: " << std::get<1>(inheritor->init_position) << std::endl;
                    it_find_id->isReplaced = true;
                    inheritor->isReplaced = true;
                    motion_.back_cnt = 0;
                    first_found_cnt = max_cnt0 + 1;
                    ti.target = *inheritor;
                    ti.targetID = inheritor->getId();
                    ti.tx = inheritor->getAmclX();
                    ti.ty = inheritor->getAmclY();
                    Init_pos = inheritor->init_position;
                    rs.second_state_ = ACTIVE_R;
                    motion_.back_cnt = 0;
                    first_found_cnt = 0;
                    second_find_cnt = 0;
                    //_target_find_flag = 0;
                    second_find_flag = false;
                    tracker_loss_flag = false;
                    tracker_loss_cnt = 0;
                    path_tracking_flag = false;
                }
            } else {
                first_found_cnt = max_cnt0 + 1;
                ti.tx = 0;
                ti.ty = 0;
                //去跟踪新目标
                choose_target_tracker(tracked_huamns, time_th);
            }
        }
            break;
        case ACTIVE_R: //有目标状态
        {
            //std::cout << "first state active" << std::endl;
            double txx, tyy;
            double min_dis = 1000;
            double min_th;
            if (!path_tracking_flag)
                min_th = 0.8;
            else
                min_th = 2;

            if (rs.second_state_ != STAY_SEARCH) {
                //second_state=ACTIVE_R
                for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
                    txx = it->getAmclX();
                    tyy = it->getAmclY();
                    if (it->getId() == ti.targetID) {
                        ti.target = *it;
                        ti.tx = txx;
                        ti.ty = tyy;
                        second_find_cnt = 0;
                        second_find_flag = false;
                        rs.second_state_ = ACTIVE_R;
                        //std::cout << "sssfind the id" << std::endl;
                        break;
                    } else {
                        //ID没找到,
                        it->toTarget = sqrt((ti.tx - txx) * (ti.tx - txx) + (ti.ty - tyy) * (ti.ty - tyy)); //到目标ID的距离
                        if (it->toTarget < min_th) {
                            if ((min_dis > it->toTarget) && (it->getId() > (it->getNewestID() - 5)))
                                min_dis = it->toTarget;
                        }
                    }
                }
                //这里考虑的是激光遮挡导致掉ID
                //if (rs.second_state_ == INACTIVE) {
                    second_find_cnt = second_find_cnt + 1;
                    ROS_INFO("second_find_cnt: %d", second_find_cnt);
                    //std::cout << "second_find_cnt: " << second_find_cnt << std::endl;
                    if (second_find_cnt < 15) {
                        second_find_flag = true;
                        if ((min_dis != 1000)) //&& (!path_tracking_flag) //在上次目标消失位置的0.8米内有替代目标
                        {                      //找target,换一个target，只要在target 0.8米范围内，离target最近的
                            for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
                                // txx = it->getCurrentX();
                                // tyy = it->getCurrentY();
                                // ToAmcl(txx, tyy, txx, tyy);
                                txx = it->getAmclX();
                                tyy = it->getAmclY();
                                if (it->toTarget == min_dis) {
                                    ROS_WARN("Change2 ID! from %d to %d", ti.targetID, it->getId());
                                    ROS_WARN("ID: %d, isDiscard: %d", ti.targetID, ti.target.isDiscard);
                                    ti.targetID = it->getId();
                                    ti.tx = txx;
                                    ti.ty = tyy;
                                    it->IDoffset(ti.target);
                                    ti.target = *it;
                                    rs.second_state_ = ACTIVE_R;
                                    second_find_cnt = 0;
                                    second_find_flag = false;
                                    path_tracking_flag = false;
                                    ROS_WARN("ID: %d, isDiscard: %d", ti.targetID, ti.target.isDiscard);
                                    ROS_INFO("targetID1: %d, from_door: %d", ti.targetID, it->from_door);
                                    break;
                                }
                            }
                        }
                        //rs.first_state_ = ACTIVE_R;
                    } else
                        //min_dis还是1000，证明上次目标0.8米内没有人；
                        //0.8米范围没有可能人拐弯了，这时候目标id不变，直接走到路径在墙外的最后一点
                    {
                        //std::cout << "second: " << second_find_cnt << std::endl;
                        //todo_ziwei: 理论上路径最后一点不会在墙里，但仍要实验论证
                        second_find_flag = false;
                        int path_size = ti.predict_path.size() - 1;
                        double txx = std::get<0>(ti.predict_path[path_size]);
                        double tyy = std::get<1>(ti.predict_path[path_size]);
                        double map_x, map_y;
                        bool a = projectToMap(txx, txx, map_x, map_y);
                        while ((((int) scoremap.at<unsigned char>((int) map_y, (int) map_x) > 240)) &&
                               (path_size > 0)) {
                            path_size = path_size - 1;
                            double txx = std::get<0>(ti.predict_path[path_size]);
                            double tyy = std::get<1>(ti.predict_path[path_size]);
                            a = projectToMap(txx, tyy, map_x, map_y);
                        }
                        if (path_size == 0) {
                            ROS_INFO("target predict path is in wall, tx=0, ty=0");
                            ti.tx = 0;
                            ti.ty = 0;
                            path_tracking_flag = false;
                        } else {
                            double min_th = 2;
                            double min_dis = 1000;
                            for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
                                double hx = it->getAmclX();
                                double hy = it->getAmclY();
                                double dis2tar = sqrt((ti.tx - hx) * (ti.tx - hx) + (ti.ty - hy) * (ti.ty - hy));
                                if ((dis2tar < min_th) && (dis2tar < min_dis))
                                    min_dis = dis2tar;
                            }
                            std::cout << "min_dis: " << min_dis << std::endl;
                            std::cout << "min_th: " << min_th << std::endl;

                            if ((distToRobotPose(txx, tyy) > 0.5) && (min_dis > min_th)) {
                                ti.tx = txx;
                                ti.ty = tyy;
                                rs.second_state_ = ACTIVE_R;
                                path_tracking_flag = true;
                                ROS_WARN("Track the path end!");
                            } else {
                                if (min_dis < min_th)
                                    path_tracking_flag = true;
                                //path_tracking_flag = false;
                                ROS_WARN("Still can not find target Id and the substitute, Lost.");
                            }
                        }
                    }

            }
            if (rs.second_state_ == STAY_SEARCH) //
            {
                //std::cout << "STAY_SEARCH" << std::endl;
                rs.second_state_ = INACTIVE;
                double min_robot = 1000;
                double txx, tyy;
                for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
                    if (it->getId() == ti.target.getId()) {
                        //ToAmcl(it->getCurrentX(), it->getCurrentY(), txx, tyy);
                        txx = it->getAmclX();
                        tyy = it->getAmclY();
                        ti.targetID = it->getId();
                        ti.target = *it;
                        ti.tx = txx;
                        ti.ty = tyy;
                        Init_pos = it->init_position;
                        rs.second_state_ = ACTIVE_R;
                        //pr_cnt = 0;
                        ROS_WARN("Stay searched targetID: %d", ti.targetID);
                        return;
                    }
                }
                //程序走到这里证明stay_search没找到id一样的，可能人走出激光范围了，再回来id变了
                //激光范围200度，走出激光范围不寻找，直接丢弃
                // if (min_robot != 1000)
                // { //找target,换一个target，只要在target 1米范围内，离target最近的
                //   for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++)
                //   {
                //     // txx = it->getCurrentX();
                //     // tyy = it->getCurrentY();
                //     // ToAmcl(txx, tyy, txx, tyy);
                //     txx = it->getAmclX();
                //     tyy = it->getAmclY();
                //     if (it->toRobot == min_robot)
                //     {
                //       ti.targetID = it->getId();
                //       ti.target = *it;
                //       ti.tx = txx;
                //       ti.ty = tyy;
                //       Init_pos = it->init_position;
                //       rs.second_state_ = ACTIVE;
                //       pr_cnt = 0;
                //       ROS_INFO("Stay search targetID: %d", ti.targetID);
                //     }
                //   }
                // }
                //to_do推送丢失状态
                ROS_WARN("STAY Can not find target and substitute in Search, Lost.");
            }
        }
            break;
    }
}

void Ltracker::first_state_decision(std::list <Human> &tracked_huamns, double time_th, int &pr_cnt, cv::Mat show,
                                    bool draw) {
    switch (rs.first_state_) {
        case (INACTIVE): {
            first_found_cnt = first_found_cnt + 1;
            rs.second_state_ = INACTIVE;
            std::vector <track_list> predict_path_list;
            std::vector <track_list> predict_path_list_from_door;
            std::vector <track_list> predict_path_list_not_from_door;
            auto candidates = tracked_huamns;
            candidates.clear();
            int max_cnt = 15;

            if (PtInPolygon(robotPose(), Poly(), nCount)) //机器人在小圈范围内
            {
                if ((first_found_cnt < max_cnt) && (ti.tx != 0 && ti.ty != 0)) {
                    //补偿掉的ID：先找有没有近的点，有就更新init_position，没有就算了；
                    //目标在机器人附近，静止阈值会很大，远离机器人必须考虑方向
                    double min_dis = 1;
                    // auto inheritor = tracked_huamns.end();
                    for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
                        // double x, y;
                        // ToAmcl(it->getCurrentX(), it->getCurrentY(), x, y);
                        double x = it->getAmclX();
                        double y = it->getAmclY();
                        //在上一个目标附近找
                        if ((sqrt((ti.tx - x) * (ti.tx - x) + (ti.ty - y) * (ti.ty - y)) < min_dis) &&
                            (it->getId() > (it->getNewestID() - 5))) {
                            min_dis = sqrt((ti.tx - x) * (ti.tx - x) + (ti.ty - y) * (ti.ty - y));
                            inheritor = it;
                        }
                    }
                    if (min_dis != 1) {
                        ROS_WARN("Change1 ID! from %d to %d", ti.targetID, inheritor->getId());
                        // ROS_WARN("ID: %d, isDiscard: %d", ti.targetID, ti.target.isDiscard);
                        std::list<Human>::iterator it_find_id = find(tracked_huamns.begin(), tracked_huamns.end(),
                                                                     *inheritor);
                        it_find_id->IDoffset(ti.target);
                        inheritor->IDoffset(ti.target);

                        ROS_WARN("ID: %d, isDiscard: %d", inheritor->getId(), inheritor->isDiscard);
                        // ROS_WARN("ID: %d, direction_cnt: %d; ID: %d, direction_cnt: %d, isDiscard: %d", ti.targetID, ti.target.directionOK_cnt, inheritor->getId(), inheritor->directionOK_cnt, inheritor->isDiscard);
                        //std::cout << "init_posx: " << std::get<0>(inheritor->init_position) << " init_posy: " << std::get<1>(inheritor->init_position) << std::endl;
                        it_find_id->isReplaced = true;
                        inheritor->isReplaced = true;
                        path_tracking_flag = false;
                        motion_.back_cnt = 0;
                        first_found_cnt = max_cnt + 1;
                    }
                } else
                    first_found_cnt = max_cnt + 1;

                //候选跟踪对象必须满足初始点在设定范围内，运动方向为设定方向
                for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
                    //ROS_INFO("HumanID: %d", it->getId());
                    //Mode 0 和 Mode 1，2 区别在 门口进来跟不跟
                    switch (MODE()) {
                        case 0: //MODE 0:普通模式
                        {
                            if (!init_range(
                                    it->init_position)) //初始点是否在设定范围内（是否是从门口走过）                                 //std::cout << "it->id: " << it->getId() << std::endl;
                            {
                                //ROS_INFO("!init_range(it->init_position)");
                                continue;
                            }
                        }
                            break;
                        case 1: //MODE 1：第二种迎人模式
                            break;
                        case 2: //MODE 2：发传单模式
                            break;
                        case 3: //MODE 3:打分模式
                            break;
                    }
                    if (MODE() == 1 || MODE() == 3) //MODE2和MODE3对运动方向无限制，因此需要对抗点阵抖动产生的速度干扰
                    {
                        //当direction_ok方向出现次数少于max_direction_cnt，视为无效
                        if ((it->directionOK_cnt < it->max_direction_cnt) && (!it->isStatic())) {
                            //ROS_INFO("directionOK_cnt : %d  smaller than max_direction_cnt and not static", it->directionOK_cnt);
                            continue;
                        }
                    }

                    if ((!it->isDirection_ok()) && (!it->isClose)) //当目标离机器人较远时，方向是否在设定角度范围内
                    {
                        //ROS_INFO("(!isDirection_ok())&&(!isClose)");
                        continue;
                    }

                    if ((!it->isReplaced) && (!it->isClose)) {
                        if (it->isStatic()) {
                            //ROS_INFO("has not been Replaced but static");
                            continue;
                        }
                    }
                    if (it->isFar()) //人是否正在走远，或者模式2中跟踪时间到了
                    {
                        //ROS_INFO("isFar()");
                        continue;
                    }
                    if (it->isfar) //人一旦有走远历史，就不在寻找范围内
                    {
                        //ROS_INFO("isfar");
                        continue;
                    }
                    if ((!it->timer_ok) && (MODE() == 2)) //模式二中跟踪时间到了
                    {
                        //ROS_INFO("time_ok and mode = 2");
                        continue;
                    }
                    if (it->isDiscard) {
                        //ROS_INFO("isDiscard");
                        continue;
                    }
                    // std::cout << "ID: " << it->getId() << " direction_cnt: " << it->directionOK_cnt << " isClose: " << it->isClose << " isDiscard: "<< it->isDiscard << std::endl;
                    candidates.push_back(*it);
                }

                //isReplaced = false; //判断条件使用完，恢复isReplaced置位

                for (auto it = candidates.begin(); it != candidates.end(); it++) {
                    std::vector <std::tuple<double, double, double, double, double>> predict_path; //inte_x,inte_y,inte_t,vx,vy
                    track_list track_;
                    double t = it->getLastTime();
                    double x = it->getAmclX();
                    double y = it->getAmclY();
                    // ToAmcl(x, y, x, y);
                    if ((t > time_th) && PtInPolygon(std::make_tuple(x, y), Poly(), nCount)) { //目标持续时间大于时间阈值，预测路径
                        //预测路径根据目标离中心点远近预测长短
                        double weight = (distToRobotPose(x, y)) / 3;
                        double predict_time = weight * 3; //最远预测四秒内的路径
                        predict_path = it->predict(predict_time, 1);
                    }
                    // std::cout << "predict path size(): " << predict_path.size() << std::endl;
                    if (predict_path.size() > 0) {
                        track_.path = predict_path;
                        track_.humanID = it->getId();
                        track_.fromDoor = init_range(it->init_position);
                        track_.greet_score = it->greet_score;
                        predict_path_list.push_back(track_);
                        if (track_.fromDoor)
                            predict_path_list_from_door.push_back(track_);
                        else
                            predict_path_list_not_from_door.push_back(track_);
                    }
                }
            }

            if (predict_path_list.size() > 0) {
                //判断哪个目标的轨迹最末端离中心最近
                double disx, disy, dis;
                auto nearest = predict_path_list.end();
                double nearest_dis = 1000;
                double max_greet_score = 0;

                //MODE 0 普通模式： 只跟从门进的，之前已处理过，这里不用处理
                //MODE 1 迎人模式：先跟从门进的 > 不从门进的也跟，需判断优先级

                //这一步是通过switch，选择出不同模式下的潜在目标点nearest
                switch (MODE()) {
                    case 0: //离机器人最近
                    {
                        for (auto it = predict_path_list.begin(); it != predict_path_list.end(); it++) {
                            if (it->path.size() > 0) { //detect_region_x
                                double disxx = std::get<0>(it->path.back());
                                double disyy = std::get<1>(it->path.back());
                                ToAmcl(disxx, disyy, disx, disy);
                                dis = distToRobotPose(disx, disy);
                                if (dis < nearest_dis) {
                                    nearest = it;
                                    nearest_dis = dis;
                                }
                            }
                        }
                    }
                        break;
                    case 1: //也是从门／和不从门最近的
                    {
                        for (auto it = predict_path_list_from_door.begin();
                             it != predict_path_list_from_door.end(); it++) {
                            if (it->path.size() > 0) { //detect_region_x
                                double disxx = std::get<0>(it->path.back());
                                double disyy = std::get<1>(it->path.back());
                                ToAmcl(disxx, disyy, disx, disy);
                                dis = distToRobotPose(disx, disy);
                                if (dis < nearest_dis) {
                                    nearest = it;
                                    nearest_dis = dis;
                                }
                            }
                        }
                        if (nearest_dis != 1000)
                            break;

                        for (auto it = predict_path_list_not_from_door.begin();
                             it != predict_path_list_not_from_door.end(); it++) {
                            if (it->path.size() > 0) { //detect_region_x
                                double disxx = std::get<0>(it->path.back());
                                double disyy = std::get<1>(it->path.back());
                                ToAmcl(disxx, disyy, disx, disy);
                                dis = distToRobotPose(disx, disy);
                                if (dis < nearest_dis) {
                                    nearest = it;
                                    nearest_dis = dis;
                                }
                            }
                        }
                    }
                        break;
                    case 2: {
                        for (auto it = predict_path_list.begin(); it != predict_path_list.end(); it++) {
                            if (it->path.size() > 0) { //detect_region_x
                                double disxx = std::get<0>(it->path.back());
                                double disyy = std::get<1>(it->path.back());
                                ToAmcl(disxx, disyy, disx, disy);
                                dis = distToRobotPose(disx, disy);
                                if (dis < nearest_dis) {
                                    nearest = it;
                                    nearest_dis = dis;
                                }
                            }
                        }
                    }
                        break;
                    case 3: //分数最高的
                    {
                        //std::cout << "*******case 3" << std::endl;
                        for (auto it = predict_path_list.begin(); it != predict_path_list.end(); it++) {
                            if (it->greet_score > max_greet_score) {
                                nearest = it;
                                max_greet_score = it->greet_score;
                            }
                        }
                    }
                        break;
                }

                for (auto it = candidates.begin(); it != candidates.end(); it++) {
                    if (it->getId() == nearest->humanID) {
                        it->isDiscard = false;
                        ti.targetID = nearest->humanID;
                        ti.target = *it;
                        ti.tx = it->getAmclX();
                        ti.ty = it->getAmclY();
                        //ToAmcl(it->getCurrentX(), it->getCurrentY(), ti.tx, ti.ty);
                        Init_pos = it->init_position;
                        this->Tar_dis = false; //target discard
                        this->isHUman = false;
                        rs.second_state_ = ACTIVE_R;
                        pr_cnt = 0;
                        motion_.back_cnt = 0;
                        first_found_cnt = 0;
                        ROS_WARN("New find target");
                        ROS_WARN("Target ID: %d", ti.targetID);
                        break;
                    }
                }
            }
        }
            break;
        case ACTIVE_R: {
            if (rs.second_state_ != STAY_SEARCH) {
                //std::cout << "not STAY_SEARCH" << std::endl;
                rs.second_state_ = INACTIVE;
                double min_dis = 1000;
                double min_robot = 1000;
                double txx, tyy;
                //加附近flag改变搜寻方式
                auto candidates = tracked_huamns;
                candidates.clear(); //静止或者方向ok的属于考虑目标
                for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
                    //std::cout << "ID: " << it->getId() << " static: " << examiner_.isStatic(it) << " direction ok: " << examiner_.isDirection_ok(it) << " isFar: " << it->isfar << " isDiscard: " << it->isDiscard << std::endl;
                    if ((it->isStatic() || (it->direction_not_OK_cnt < it->max_not_direction_cnt)) && (!it->isfar) &&
                        (it->timer_ok) && (!it->isDiscard)) //方向是否在设定角度范围内 或者 静止
                    {
                        //std::cout << "IDX: " << it->getId() << std::endl;
                        candidates.push_back(*it);
                    }
                }
                //std::cout << "candidate size: " << candidates.size() <<std::endl;
                for (auto it = candidates.begin(); it != candidates.end(); it++) {
                    // txx = it->getCurrentX();
                    // tyy = it->getCurrentY();
                    // ToAmcl(txx, tyy, txx, tyy);
                    txx = it->getAmclX();
                    tyy = it->getAmclY();
                    if (it->getId() == ti.targetID) {
                        //std::cout << "ID: " << it->getId() << "it->getId == targetID" << std::endl;
                        if (this->Tar_dis == true) {
                            it->isDiscard = true;
                        }
                        if (it->isDiscard == false) {
                            if (!this->isHUman) {
                                ti.target = *it;
                                ti.tx = txx;
                                ti.ty = tyy;
                                rs.second_state_ = ACTIVE_R;
                                break;
                            } else { //ishuman
                                if (projectBoxInCamera(*it)) {                                  //fixed size rect
                                    kcf._roi = it->getBoundingBox(); //detected human in current frame
                                    float scale_addtion = sqrtf((float) target_width_proto * target_height_proto /
                                                                (it->getBoundingBox().width *
                                                                 it->getBoundingBox().height));
                                    double score;
                                    bool success;
                                    //frame是整个大图像，目标检测区域是kcf._roi，score是得分，success是判断是不是跟丢
                                    cv::Rect result_box = kcf.update(frame, scale_addtion, score, success);
                                    if (!success || getOverlap(result_box, it->getBoundingBox()) < 0.7)
                                        continue;
                                    else {
                                        ti.target = *it;
                                        ti.tx = txx;
                                        ti.ty = tyy;
                                        rs.second_state_ = ACTIVE_R;
                                        break;
                                    }
                                } else {
                                    ti.target = *it;
                                    ti.tx = txx;
                                    ti.ty = tyy;
                                    rs.second_state_ = ACTIVE_R;
                                    break;
                                }
                            }
                        } else //丢弃后有缓冲时间找其他点，缓冲时间过了就回去
                        {
                            ROS_WARN("The Discard is true, first_state become inactive.");
                            rs.first_state_ = INACTIVE;
                        }
                    } else {
                        //ID没找到
                        it->toTarget = sqrt((ti.tx - txx) * (ti.tx - txx) + (ti.ty - tyy) * (ti.ty - tyy)); //到目标ID的距离
                        if (it->toTarget < 1) {
                            if ((min_dis > it->toTarget) && (it->getId() > (it->getNewestID() - 5)))
                                min_dis = it->toTarget;
                        }
                    }
                }
                if (!this->isHUman) {
                    //  std::cout << "not human" << std::endl;
                    if (rs.second_state_ == INACTIVE) {
                        if (min_dis != 1000) { //找target,换一个target，只要在target 1米范围内，离target最近的
                            for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
                                // txx = it->getCurrentX();
                                // tyy = it->getCurrentY();
                                // ToAmcl(txx, tyy, txx, tyy);
                                txx = it->getAmclX();
                                tyy = it->getAmclY();
                                if (it->toTarget == min_dis) {
                                    ROS_WARN("Change2 ID! from %d to %d", ti.targetID, it->getId());
                                    ROS_WARN("ID: %d, isDiscard: %d", ti.targetID, ti.target.isDiscard);
                                    ti.targetID = it->getId();
                                    ti.tx = txx;
                                    ti.ty = tyy;
                                    it->IDoffset(ti.target);
                                    ti.target = *it;
                                    rs.second_state_ = ACTIVE_R;
                                    pr_cnt = 0;
                                    ROS_WARN("ID: %d, isDiscard: %d", ti.targetID, ti.target.isDiscard);
                                    ROS_INFO("targetID1: %d, from_door: %d", ti.targetID, it->from_door);
                                }
                            }
                        } else
                            ROS_WARN("Can not find target Id and the substitute, Lost.");
                    }
                } else { //this->isHUman = true
                    if (rs.second_state_ == INACTIVE) {
                        float max_score = fern.getThreshFern();
                        //ROS_INFO("fern: %f", max_score);
                        for (auto it = candidates.begin(); it != candidates.end(); it++) {
                            //ROS_INFO("BUG2");
                            if (projectBoxInCamera(*it)) {
                                cv::Mat patch = frame(it->getBoundingBox());
                                cv::cvtColor(patch, patch, CV_RGB2GRAY);
                                cv::resize(patch, patch, cv::Size(64, 128));
                                std::vector<int> fern_result;
                                fern_result.resize(fern.getNstructs());
                                fern.calcFeatures(patch, 0, fern_result);
                                float score = fern.measure_forest(fern_result) / fern.getNstructs();
                                //ROS_INFO("score : %f", score);
                                if (draw)
                                    cv::putText(show, std::to_string(score),
                                                cv::Point(it->getBoundingBox().x,
                                                          it->getBoundingBox().y + it->getBoundingBox().height),
                                                cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 255), 1, CV_AA);
                                if (score > max_score) {
                                    max_score = score;
                                    ti.target = *it;
                                }
                            }
                        }
                        if (max_score != fern.getThreshFern()) { //maybe you wenti TODO
                            this->isHUman = true;
                        } else {
                            this->isHUman = true;
                            rs.second_state_ = INACTIVE;
                            ti.targetID = ti.target.getId();
                            this->Tar_dis = false;
                            ti.tx = txx;
                            ti.ty = tyy;
                            Init_pos = ti.target.init_position;
                            pr_cnt = 0;
                            ROS_INFO("targetID3: %d", ti.targetID);
                        }
                    }
                }
            }
            if (rs.second_state_ == STAY_SEARCH) //只找机器人1.2米内的目标，否则作为找不到目标处理
            {
                //std::cout << "STAY_SEARCH" << std::endl;
                rs.second_state_ = INACTIVE;
                double min_robot = 1000;
                double txx, tyy;
                for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
                    if (it->getId() == ti.target.getId()) {
                        //ToAmcl(it->getCurrentX(), it->getCurrentY(), txx, tyy);
                        txx = it->getAmclX();
                        tyy = it->getAmclY();
                        ti.targetID = it->getId();
                        ti.target = *it;
                        ti.tx = txx;
                        ti.ty = tyy;
                        Init_pos = it->init_position;
                        rs.second_state_ = ACTIVE_R;
                        pr_cnt = 0;
                        ROS_WARN("Stay searched targetID: %d", ti.targetID);
                        return;
                    }

                    //没找到target ID
                    // txx = it->getCurrentX();
                    // tyy = it->getCurrentY();
                    // ToAmcl(txx, tyy, txx, tyy);
                    txx = it->getAmclX();
                    tyy = it->getAmclY();
                    it->toRobot = sqrt(
                            (robotPoseX() - txx) * (robotPoseX() - txx) + (robotPoseY() - tyy) * (robotPoseY() - tyy));
                    if ((it->timer_ok) && (it->toRobot < 1.2)) {
                        if (min_robot > it->toRobot)
                            min_robot = it->toRobot;
                    }
                }
                {
                    if (min_robot != 1000) { //找target,换一个target，只要在target 1米范围内，离target最近的
                        for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
                            // txx = it->getCurrentX();
                            // tyy = it->getCurrentY();
                            // ToAmcl(txx, tyy, txx, tyy);
                            txx = it->getAmclX();
                            tyy = it->getAmclY();
                            if (it->toRobot == min_robot) {
                                ti.targetID = it->getId();
                                ti.target = *it;
                                ti.tx = txx;
                                ti.ty = tyy;
                                Init_pos = it->init_position;
                                rs.second_state_ = ACTIVE_R;
                                pr_cnt = 0;
                                ROS_INFO("Stay search targetID: %d", ti.targetID);
                            }
                        }
                    } else
                        ROS_WARN("Can not find target and substitute in Search, Lost.");
                }
            }
        }
            break;
    }
}

void Ltracker::second_state_decision(std::list <Human> &tracked_huamns) {
    if (greet_track_MODE()) {
        if (rs.second_state_ == ACTIVE_R) {
            //ROS_WARN("second is active!");
            if (PtInPolygon(std::make_tuple(ti.tx, ti.ty), Poly(), nCount)) {
                //要追踪的目标在小圈内才追踪
                if (this->stayMode(tracked_huamns)) {
                    rs.robot_state_ = STAY;
                    track_cnt = 0;
                } else {
                    if (stay_flag == false) {
                        rs.robot_state_ = TRACK;
                    } else {
                        track_cnt = track_cnt + 1;
                        if (track_cnt > 5)
                            rs.robot_state_ = TRACK;
                        else
                            rs.robot_state_ = STAY;
                    }
                }
            } else {
                rs.robot_state_ = LOST;
                ROS_INFO("Out of range!, ID: %d, dis: %f", ti.target.getId(), this->distToDetectRegion(ti.tx, ti.ty));
            }
        }
        if (rs.second_state_ == INACTIVE) {
            rs.robot_state_ = LOST;
        }
    } else //tracker
    {
        if (rs.second_state_ == ACTIVE_R) {
            if (this->stayMode_track(tracked_huamns)) {
                ROS_INFO("stayMode_track == true");
                //std::cout << "stayMode_track == true" << std::endl;
                rs.robot_state_ = STAY;
                track_cnt = 0;
            } else {
                if (stay_flag == false) {
                    rs.robot_state_ = TRACK;
                } else {
                    //ROS_INFO("stay_flag == true");
                    track_cnt = track_cnt + 1;
                    if (track_cnt > 5)
                        rs.robot_state_ = TRACK;
                    else
                        rs.robot_state_ = STAY;
                }
            }
        }
        if (rs.second_state_ == INACTIVE) {
            if (second_find_flag)
                rs.robot_state_ = STAY;
            else
                rs.robot_state_ = LOST;
        }
    }
}

//主要是second_state状态决定机器人运动状态
void Ltracker::moving_state(cv::Mat show, bool draw, int frz, int max_back, int &pr_cnt) {
    if (greet_track_MODE()) {
        //决定first state
        if (rs.robot_state_ == TRACK) {
            //判断目标点远近
            double dis_th = ti.target.dis_boundary;
            // double predict_x, predict_y;
            // ToAmcl(ti.target.getCurrentX(), ti.target.getCurrentY(), predict_x, predict_y);
            double predict_x = ti.target.getAmclX();
            double predict_y = ti.target.getAmclY();
            if (distToRobotPose(predict_x, predict_y) > dis_th)
                neighbour_flag = false;
            else
                neighbour_flag = true;
            // std::cout << "flag: " << neighbour_flag <<std::endl;
            //ROS_WARN("Before tracking");
            stay_flag = false;
            rs.first_state_ = ACTIVE_R;
            rs.second_state_ = INACTIVE;

            tp.pub_time_now = ros::Time::now();
            if (((tp.pub_time_now.toSec() - tp.pub_time_last.toSec()) > motion_.frz) && (motion_.lock_pub == false) &&
                (motion_.init_lock == true)) {
                motion_.tracking(tp, ti, rs, no_task, false);
                if (distToRobotPose(ti.tx, ti.ty) < 1.3) //摄像头看不到人的时候，给标记位清0
                    pr_cnt = 0;
                motion_.init_flag = false;
                ROS_INFO("pr_cnt: %d", pr_cnt);
                ROS_INFO("targetID: %d, isDiscard: %d", ti.targetID, ti.target.isDiscard);
                ROS_ERROR("Start tracking without human detection!");
            }
        }
        if (rs.robot_state_ == LOST)
            freeOperation(std::get<0>(Origin()), std::get<1>(Origin()));
        if (rs.robot_state_ == STAY) {
            rs.robot_show_state_ = TRACK_STAY;
            stay_flag = true;
            motion_.back_cnt = 0;
            ROS_ERROR("Enter stay mode!");
            rs.first_state_ = ACTIVE_R;
            rs.second_state_ = INACTIVE;
        }
    } else //tracker
    {
        if (rs.robot_state_ == TRACK) {
            //判断目标点远近
            first_found_cnt = 0;
            double dis_th = 1.5;
            double predict_x = ti.tx;
            double predict_y = ti.ty;
            if (distToRobotPose(predict_x, predict_y) > dis_th)
                neighbour_flag = false;
            else
                neighbour_flag = true;
            stay_flag = false;
            rs.first_state_ = ACTIVE_R;
            rs.second_state_ = INACTIVE;
            tp.pub_time_now = ros::Time::now();
            if (((tp.pub_time_now.toSec() - tp.pub_time_last.toSec()) > motion_.frz) && (motion_.lock_pub == false) &&
                (motion_.init_lock == true) && (ti.tx != 0 && ti.ty != 0)) {
                motion_.tracker_tracking(tp, ti, rs, no_task, false, path_tracking_flag);
                motion_.init_flag = false;
                ROS_INFO("targetID: %d, isDiscard: %d", ti.targetID, false);
                ROS_ERROR("Start tracking without human detection!");
            }
        }
        if (rs.robot_state_ == LOST) {
            rs.first_state_ = INACTIVE;
            rs.second_state_ = INACTIVE;
        }
        if (rs.robot_state_ == STAY) {
            //rs.robot_show_state_ = TRACK_STAY;
            first_found_cnt = 0;
            stay_flag = true;
            motion_.back_cnt = 0;
            ROS_ERROR("Enter stay mode!");
            rs.first_state_ = ACTIVE_R;
            if (!second_find_flag) {
                ROS_INFO("Real stay mode: STAY_SEARCH");
                rs.second_state_ = STAY_SEARCH;
            } else {
                ROS_INFO("Stay for second finding.");
                rs.second_state_ = ACTIVE_R;
            }
            //second_state_ = INACTIVE;
        }
    }
}

//在它正对面正负45度是否有人
bool Ltracker::stayMode(std::list <Human> tracked_huamns) {
    double x, y;
    if (fabs(robotVel()) != 0)
        return false;
    for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
        // double xx, yy;
        // ToAmcl(it->getCurrentX(), it->getCurrentY(), xx, yy);
        double xx = it->getAmclX();
        double yy = it->getAmclY();
        double robot_direction;
        if (robotYaw() < 0)
            robot_direction = (3.1416 - fabs(robotYaw())) + 3.1416;
        else
            robot_direction = robotYaw();
        robot_direction = (int) (robot_direction * 57.3); //*0.017453292519943
        int pub_oo = (int) cv::fastAtan2((yy - robotPoseY()), (xx - robotPoseX()));
        pub_oo = (int) (pub_oo - robot_direction + 360) % 360;
        if (pub_oo > 180)
            pub_oo = 360 - pub_oo;
        //ROS_ERROR("ID: %d, pub_oo: %d", it->getId(), pub_oo);
        if ((distToRobotPose(xx, yy) < 1.2) && (pub_oo < 30)) {
            return true;
        }
    }
    return false;
}

bool Ltracker::stayMode_track(std::list <Human> tracked_huamns) {

    double x, y;
    if (fabs(robotVel()) != 0)
        return false;

    bool f = false;
    for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
        if (it->getId() == ti.targetID)
            f = true;
    }
    if (!f) {
        ROS_INFO("Cannot find stay ID: %d", ti.targetID);
        return false;
    }

    double robot_direction;
    if (robotYaw() < 0)
        robot_direction = (3.1416 - fabs(robotYaw())) + 3.1416;
    else
        robot_direction = robotYaw();
    robot_direction = (int) (robot_direction * 57.3); //*0.017453292519943
    int pub_oo = (int) cv::fastAtan2((ti.ty - robotPoseY()), (ti.tx - robotPoseX()));
    pub_oo = (int) (pub_oo - robot_direction + 360) % 360;
    if (pub_oo > 180)
        pub_oo = 360 - pub_oo;
    if ((distToRobotPose(ti.tx, ti.ty) < 1.2) && (pub_oo < 30)) {
        return true;
    }
    return false;
}

void Ltracker::human_out(std::list <Human> tracked_huamns, double frz) {
    auto nearest = tracked_huamns.end();
    if (PtInPolygon(robotPose(), Poly(), nCount)) //机器人在小圈范围内
    {
        auto candidates = tracked_huamns;
        candidates.clear(); //静止或者方向ok的属于考虑目标
        //候选跟踪对象必须满足初始点在设定范围内，运动方向为设定方向
        for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
            // double x, y;
            // ToAmcl(it->getCurrentX(), it->getCurrentY(), x, y);
            double x = it->getAmclX();
            double y = it->getAmclY();
            if ((distToDoorPose(x, y) < 1.5) && it->is_close_to_door() && (it->isFar() || it->isfar)) {
                candidates.push_back(*it);
            }
        }
        double nearest_dis = 1000;
        for (auto it = candidates.begin(); it != candidates.end(); it++) {
            double disx = it->getAmclX();
            double disy = it->getAmclY();
            //ToAmcl(disx, disy, disx, disy);
            double dis = distToRobotPose(disx, disy);
            if (dis < nearest_dis) {
                nearest = it;
                nearest_dis = dis;
            }
        }
        if (nearest_dis != 1000) {

            tp.pub_time_now = ros::Time::now();
            //ROS_INFO("now: %f, last: %f", tp.pub_time_now.toSec(), tp.pub_time_last.toSec());
            if (((tp.pub_time_now.toSec() - tp.pub_time_last.toSec()) > motion_.frz) && (motion_.lock_pub == false) &&
                (motion_.init_lock == true)) {
                ti.target = *nearest;
                //ROS_INFO("Some one is going out.");
                motion_.tracking(tp, ti, rs, no_task, true);
                motion_.init_flag = false;
                ROS_ERROR("Thank you! See you next time!");
            }
        }
    }
}

//补上从门口到机器人范围掉ID情况
void Ltracker::fill_candidates(std::list <Human> &tracked_huamns) {
    auto candidates = tracked_huamns;
    double mini_dis = 1;
    candidates.clear(); //方向ok的，不在机器人圈内，但在大圈内的属于考虑目标
    for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
        // double x = it->getCurrentX();
        // double y = it->getCurrentY();
        // ToAmcl(x, y, x, y);
        double x = it->getAmclX();
        double y = it->getAmclY();
        std::tuple<double, double> p = std::make_tuple(x, y);
        //if ((PtInPolygon(p, Poly_wall(), Poly_wall().size())) && (it->getId() != 0) && (it->isDirection_ok() || it->isStatic()))
        {
            candidates.push_back(*it);
            //std::cout<< "candidates: " << candidates.size() <<std::endl;
        }
    }

    //cnt>10删
    for (auto it = last_candidates.begin(); it != last_candidates.end();) {
        if (it->cnt > 10) {
            // ROS_WARN("Delete ID: %d, cnt: %d",it->human.getId(),it->cnt);
            it = last_candidates.erase(it);
        } else
            it++;
    }
    // std::cout << "last candidates size: " << last_candidates.size() <<std::endl;
    for (auto it1 = last_candidates.begin(); it1 != last_candidates.end(); it1++) {
        bool found = false;
        for (auto it2 = candidates.begin(); it2 != candidates.end();) //找同ID的
        {
            if (it1->human.getId() == it2->getId()) {
                last_candidate candidate;
                candidate.cnt = 0;
                candidate.human = *it2;
                it1 = last_candidates.erase(it1);
                it1 = last_candidates.insert(it1, candidate);
                it2 = candidates.erase(it2);
                found = true;
                break;
            } else
                it2++;
        }
        if (found == false) { //找1米内的
            ROS_WARN("Cannot find ID: %d in last candidates", it1->human.getId());
            double min_dis = mini_dis;
            auto it_human = candidates.end();
            for (auto it2 = candidates.begin(); it2 != candidates.end(); it2++) {
                //candidates 和 last_candidates的human的位置都在icp-amcl平面上
                double dis = disTwoPoints(it1->human.getCurrentX(), it1->human.getCurrentY(), it2->getCurrentX(),
                                          it2->getCurrentY());
                //ROS_WARN("ID1: %d, ID: %d, dis is %f", it1->human.getId(),it2->getId(),dis);
                if ((dis < min_dis) && (it2->getId() > (it2->getNewestID() - 5))) {
                    it_human = it2;
                    min_dis = dis;
                }
            }
            if (min_dis != mini_dis) {
                //ROS_ERROR("it_human ID: %d", it_human->getId());
                it_human->init_position = it1->human.init_position;
                it_human->first_position_flag = it1->human.first_position_flag;
                it_human->from_door = it1->human.from_door;
                it_human->directionOK_cnt = it1->human.directionOK_cnt;
                it_human->direction_not_OK_cnt = it1->human.direction_not_OK_cnt;
                it_human->isDiscard = it1->human.isDiscard;
                last_candidate candidate;
                candidate.cnt = 0;
                candidate.human = *it_human;
                for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
                    if (it->getId() == it_human->getId()) {
                        it->IDoffset(it1->human);
                    }
                }
                ROS_WARN("Before track, change from %d to %d", it1->human.getId(), it_human->getId());
                ROS_WARN("ID: %d, isDiscard: %d", it1->human.getId(), it1->human.isDiscard);
                ROS_WARN("ID: %d, isDiscard: %d", it_human->getId(), it_human->isDiscard);
                it1 = last_candidates.erase(it1);
                it1 = last_candidates.insert(it1, candidate);
                candidates.erase(it_human);
                found = true;
            }
        }
        if (found == false)
            it1->cnt = it1->cnt + 1;
    }

    for (auto it = candidates.begin(); it != candidates.end(); it++) {
        last_candidate candidate;
        candidate.human = *it;
        candidate.cnt = 0;
        last_candidates.push_back(candidate);
    }
}

void Ltracker::human_predict_path(std::list <Human> &tracked_huamns) {
    //printf("--human_predict_path--in\n");
    //tracked_human_path_visual_test(tracked_huamns);
    bool tracker_find_flag = false;
    sensor_msgs::PointCloud2 msg_pointcloud;
    msg_pointcloud.width = 1000;
    msg_pointcloud.height = 1;
    msg_pointcloud.header.stamp = ros::Time::now();
    msg_pointcloud.is_dense = true;
    msg_pointcloud.is_bigendian = false;
    sensor_msgs::PointCloud2Modifier modifier(msg_pointcloud);
    modifier.setPointCloud2FieldsByString(1, "xyz");
    sensor_msgs::PointCloud2Iterator<float> iter_x(msg_pointcloud, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(msg_pointcloud, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(msg_pointcloud, "z");
    int count = 0;
    //printf("human_predict_path ros::Time::now=%lf\n",ros::Time::now().toSec());
    //人物路径
    for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
        // std::cout << "aaaaaaaaaa" << std::endl;
        std::deque <std::tuple<double, double, double>> s = it->getStates();
        bool hasPath = false;

        if (s.size() > 5) {
            int num = s.size();
            double startx = std::get<0>(s[0]);
            double starty = std::get<1>(s[0]);
            double endx = std::get<0>(s[num - 1]);
            double endy = std::get<1>(s[num - 1]);
            ToAmcl(startx, starty, startx, starty);
            ToAmcl(endx, endy, endx, endy);
            double distance = sqrt((startx - endx) * (startx - endx) + (starty - endy) * (starty - endy));
            if (distance > 1) {
                hasPath = true;
                //预测了2.5秒的路径
                std::vector <std::tuple<double, double, double, double, double>> predict_path = it->predict(2.5, true);
                //count += predict_path.size();
                //std::cout << "predict path size: " <<  predict_path.size() <<std::endl;
                for (int i = 0; i < predict_path.size(); i++) {
                    double x1 = std::get<0>(predict_path[i]);
                    double y1 = std::get<1>(predict_path[i]);
                    //std::cout << " x1: " << x1 << " y1:" << y1;
                    //std::cout << "count0: " << count << std::endl;
                    std::tuple<double, double> p = std::make_tuple(x1, y1);

                    // std::cout << "x1: " << x1 << " y1: " << y1 << std::endl;
                    // std::cout << "Poly_robot().size() "<< Poly_robot().size()<< std::endl;
                    // std::cout << "PtInPolygon: " << PtInPolygon(p, Poly_robot(), Poly_robot().size());
                    if (!PtInPolygon(p, Poly_robot())) {
                        //std::cout<< "22222" << std::endl;
                        *iter_x = x1;
                        *iter_y = y1;
                        *iter_z = 0;
                        ++iter_x, ++iter_y, ++iter_z;
                        count++;
                    }
                }
            }
        }

        if (!hasPath) {
            // double x = it->getCurrentX();
            // double y = it->getCurrentY();
            // ToAmcl(x, y, x, y);
            double x = it->getAmclX();
            double y = it->getAmclY();
            std::tuple<double, double> p0 = std::make_tuple(x, y);
            // if(!PtInPolygon(p0, Poly_robot(), Poly_robot().size()))
            // {
            //   std::cout<< "0000" << std::endl;
            //   *iter_x = x;
            //   *iter_y = y;
            //   *iter_z = 0;
            //   ++iter_x, ++iter_y, ++iter_z;
            //   count++;
            // }
            //std::cout << "laser size: " << laser_around.size() << std::endl;
            std::vector <std::pair<double, double>> laser_around = it->getLaser();
            for (auto it = laser_around.begin(); it != laser_around.end(); it++) {
                double x0 = it->first;
                double y0 = it->second;
                ToAmcl(x0, y0, x0, y0);
                //std::cout << "x0: " << x0 << " y0: " << y0 << std::endl;
                std::tuple<double, double> p = std::make_tuple(x0, y0);
                // std::cout << "x0: " << x0 << " y0: " << y0 << std::endl;
                // std::cout << "Poly_robot().size() "<< Poly_robot().size()<< std::endl;
                // std::cout << "PtInPolygon: " << PtInPolygon(p, Poly_robot(), Poly_robot().size()) << std::endl;

                if (!PtInPolygon(p, Poly_robot())) {
                    //std::cout<< "1111" << std::endl;
                    *iter_x = x0;
                    *iter_y = y0;
                    *iter_z = 0;
                    ++iter_x, ++iter_y, ++iter_z;
                    count++;
                }
            }
        }
        if (it->getId() == ti.targetID) {
            tracker_find_flag = true;
            std::deque <std::tuple<double, double, double>> s = it->getStates();
            std::vector <std::tuple<double, double, double, double, double>> predict_path;
            if (s.size() > 3) {
                //std::cout << "000" << std::endl;
                int num = s.size();
                double startx = std::get<0>(s[0]);
                double starty = std::get<1>(s[0]);
                double endx = std::get<0>(s[num - 1]);
                double endy = std::get<1>(s[num - 1]);
                ToAmcl(startx, starty, startx, starty);
                ToAmcl(endx, endy, endx, endy);
                double distance = sqrt((startx - endx) * (startx - endx) + (starty - endy) * (starty - endy));
                if (distance > 1) {
                    hasPath = true;
                    //预测了2.5秒的路径
                    predict_path = it->predict(2.5, true);
                }
            }
            if ((s.size() <= 3) || (predict_path.size() == 0)) {
                // std::cout << "111" << std::endl;
                // ROS_INFO("Use %d position as its predict path", it->getId());
                double vx = std::get<0>(it->getCurrent_speed());
                double vy = std::get<1>(it->getCurrent_speed());
                double v = sqrt(vx * vx + vy * vy);
                if (v > 0.4)
                    v = 0.4;
                double inte_x = it->getAmclX(); // getCurrentX();
                double inte_y = it->getAmclY(); // getCurrentY();
                double inte_t = it->getCurrentTime();
                predict_path.push_back(std::make_tuple(inte_x, inte_y, inte_t, vx, vy));
            }
            int num = predict_path.size() - 1;
            // std::cout << "num: " << num << std::endl;
            //ROS_INFO("Predict path ID: %d, x: %f, y: %f", it->getId(), std::get<0>(predict_path[num]), std::get<1>(predict_path[num]));
            ti.predict_path = predict_path;
        }
    }
    msg_pointcloud.header.frame_id = "world";
    //std::cout << "count: " << count << std::endl;
    modifier.resize(count);
    modifier.setPointCloud2FieldsByString(1, "xyz");

    predict_path_pointcloud_publisher_.publish(msg_pointcloud);

    // printf("--human_predict_path--out\n");
}

void Ltracker::rviz_visualization(bool use_greet, std::list <Human> &tracked_huamns) {
    visualization_msgs::Marker temp_id; //,temp_id;//
    visualization_msgs::Marker arrow;
    visualization_msgs::Marker polygon;
    visualization_msgs::Marker poly_wall;
    visualization_msgs::Marker poly_door;
    visualization_msgs::MarkerArray rviz_markers; //,rviz_markers_id; //
    visualization_msgs::Marker human_score;
    visualization_msgs::Marker direction_cnt;
    visualization_msgs::Marker project_human;
    visualization_msgs::Marker project_target;


    //---------------------------start draw rviz------------------------/
    temp_id.header.frame_id = "world";
    temp_id.header.stamp = ros::Time();
    temp_id.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    temp_id.action = visualization_msgs::Marker::ADD;
    temp_id.lifetime = ros::Duration(0.5);

    // temp_id.color.r = 0.7;
    // temp_id.color.g = 0.7;
    // temp_id.color.b = 0;
    // temp_id.color.a = 1;

    arrow.header.frame_id = "world";
    arrow.header.stamp = ros::Time();
    arrow.type = visualization_msgs::Marker::ARROW;
    arrow.action = visualization_msgs::Marker::ADD;
    arrow.lifetime = ros::Duration(0.5);

    human_score.header.frame_id = "world";
    human_score.header.stamp = ros::Time();
    human_score.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    human_score.action = visualization_msgs::Marker::ADD;
    human_score.lifetime = ros::Duration(0.5);

    human_score.color.r = 0.7;
    human_score.color.g = 0.7;
    human_score.color.b = 0.7;
    human_score.color.a = 1;

    direction_cnt.header.frame_id = "world";
    direction_cnt.header.stamp = ros::Time();
    direction_cnt.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    direction_cnt.action = visualization_msgs::Marker::ADD;
    direction_cnt.lifetime = ros::Duration(0.5);

    direction_cnt.color.r = 0;
    direction_cnt.color.g = 0.6;
    direction_cnt.color.b = 0.6;
    direction_cnt.color.a = 1;

    project_human.header.frame_id = "world";
    project_human.header.stamp = ros::Time();
    project_human.type = visualization_msgs::Marker::SPHERE;
    project_human.action = visualization_msgs::Marker::ADD;
    project_human.lifetime = ros::Duration(0.5);

    project_human.pose.orientation.w = 1.0;
    project_human.pose.orientation.x = 0.0;
    project_human.pose.orientation.y = 0.0;
    project_human.pose.orientation.z = 0.0;
    project_human.color.r = 0.4;
    project_human.color.g = 0.7;
    project_human.color.b = 0;
    project_human.color.a = 1;

    project_human.scale.x = 0.3;
    project_human.scale.y = 0.3;
    project_human.scale.z = 0.3;


    int ind = 0;

    for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
        double human_x, human_y;
        human_x = it->getAmclX();
        human_y = it->getAmclY();
        //ToAmcl(it->getCurrentX(), it->getCurrentY(), human_x, human_y);
        std::pair<double, double> s = Speed_amcl(std::make_tuple(it->getCurrentX(), it->getCurrentY()),
                                                 it->getCurrent_speed());
        double vx = s.first;
        double vy = s.second;
        double v_thr;

        temp_id.id = ind;
        temp_id.text = toString(it->getId());
        temp_id.pose.position.x = human_x;
        temp_id.pose.position.y = human_y;
        temp_id.pose.orientation.w = 1.0;
        temp_id.pose.orientation.x = 0.0;
        temp_id.pose.orientation.y = 0.0;
        temp_id.pose.orientation.z = 0.0;
        temp_id.scale.x = 0.3;
        temp_id.scale.y = 0.3;
        temp_id.scale.z = 0.3;
        //if (it->isfar || it->isFar())
        if (it->isHuman) {
            temp_id.color.r = 0.7;
            temp_id.color.g = 0;
            temp_id.color.b = 0.7;
            temp_id.color.a = 1;
        } else {
            temp_id.color.r = 0.7;
            temp_id.color.g = 0.7;
            temp_id.color.b = 0;
            temp_id.color.a = 1;
        }

        rviz_markers.markers.push_back(temp_id);
        ind = ind + 1;

        cv::Rect targetRect0 = targetRect;

        if (!missed && catched)
            //if(it->isProject)
        {
            project_human.id = ind;
            std::pair<double, double> reproject_point = ReProject(targetRect0, *it);
            //std::pair<double,double>  reproject_point = ReProject(it->getBoundingBox(),*it);
            project_human.pose.position.x = reproject_point.first;
            project_human.pose.position.y = reproject_point.second;
            // std::cout << "ID: " << it->getId() << "reproject dist: " << dist2Reproject(targetRect0, *it) << std::endl;

            //std::cout << "reproject x:" << reproject_point.first << " reproject y: " << reproject_point.second << std::endl;
            if (reproject_point.first != 0 && reproject_point.second != 0) {
                rviz_markers.markers.push_back(project_human);
                ind = ind + 1;
            }
        }


        if ((use_greet) && greet_track_MODE()) {
            human_score.id = ind;
            human_score.text = toString(
                    it->greet_score); // toString(it->get_amclStates().size());//toString(it->greet_score);
            human_score.pose.position.x = human_x + 0.3;
            human_score.pose.position.y = human_y;
            human_score.pose.orientation.w = 1.0;
            human_score.pose.orientation.x = 0.0;
            human_score.pose.orientation.y = 0.0;
            human_score.pose.orientation.z = 0.0;
            human_score.scale.x = 0.2;
            human_score.scale.y = 0.2;
            human_score.scale.z = 0.2;
            rviz_markers.markers.push_back(human_score);
            ind = ind + 1;

            direction_cnt.id = ind;
            direction_cnt.text = toString(sqrt(vx * vx + vy * vy)); //  toString(it->directionOK_cnt)
            direction_cnt.pose.position.x = human_x - 0.3;
            direction_cnt.pose.position.y = human_y;
            direction_cnt.pose.orientation.w = 1.0;
            direction_cnt.pose.orientation.x = 0.0;
            direction_cnt.pose.orientation.y = 0.0;
            direction_cnt.pose.orientation.z = 0.0;
            direction_cnt.scale.x = 0.2;
            direction_cnt.scale.y = 0.2;
            direction_cnt.scale.z = 0.2;
            rviz_markers.markers.push_back(direction_cnt);
            ind = ind + 1;
        }

        if (useGreet() && greet_track_MODE()) {
            if (neighbour_flag)
                v_thr = it->v_th_;
            else
                v_thr = it->v_th_ * 2;
        } else {
            v_thr = it->v_th_;
        }

        if (sqrt(vx * vx + vy * vy) > v_thr) //&& ()
        {
            // std::cout << "ID: " << it->getId() << "direction_ok: " << examiner_.isDirection_ok(it) << std::endl;
            if (it->isDirection_ok()) {
                double pub_oo = cv::fastAtan2(vy, vx) * 0.017453292519943;
                geometry_msgs::Quaternion dir = tf::createQuaternionMsgFromYaw(pub_oo);
                arrow.id = ind;
                arrow.pose.position.x = human_x;
                arrow.pose.position.y = human_y;
                arrow.pose.orientation.x = dir.x;
                arrow.pose.orientation.y = dir.y;
                arrow.pose.orientation.z = dir.z;
                arrow.pose.orientation.w = dir.w;
                arrow.scale.x = 0.6;
                arrow.scale.y = 0.05;
                arrow.scale.z = 0;
                if (it->from_door == true) {
                    arrow.color.r = 0.9;
                    arrow.color.g = 0.9;
                    arrow.color.b = 0;
                    arrow.color.a = 1;
                } else {
                    arrow.color.r = 0;
                    arrow.color.g = 0.9;
                    arrow.color.b = 0;
                    arrow.color.a = 1;
                }
                rviz_markers.markers.push_back(arrow);
                ind = ind + 1;
            } else {
                double pub_oo = cv::fastAtan2(vy, vx) * 0.017453292519943;
                geometry_msgs::Quaternion dir = tf::createQuaternionMsgFromYaw(pub_oo);
                arrow.id = ind;
                arrow.pose.position.x = human_x;
                arrow.pose.position.y = human_y;
                arrow.pose.orientation.x = dir.x;
                arrow.pose.orientation.y = dir.y;
                arrow.pose.orientation.z = dir.z;
                arrow.pose.orientation.w = dir.w;
                arrow.scale.x = 0.6;
                arrow.scale.y = 0.05;
                arrow.scale.z = 0;
                arrow.color.r = 0.9;
                arrow.color.g = 0;
                arrow.color.b = 0.9;
                arrow.color.a = 1;
                rviz_markers.markers.push_back(arrow);
                ind = ind + 1;
            }
        }
        //std::cout << "ind " << ind << std::endl;
    }

    if (!missed && catched) //红色点，根据框的宽度估计距离，投影出的点
    {
        cv::Rect targetRect0 = targetRect;
        project_target.header.frame_id = "world";
        project_target.header.stamp = ros::Time();
        project_target.type = visualization_msgs::Marker::SPHERE;
        project_target.action = visualization_msgs::Marker::ADD;
        project_target.lifetime = ros::Duration(0.5);

        project_target.pose.orientation.w = 1.0;
        project_target.pose.orientation.x = 0.0;
        project_target.pose.orientation.y = 0.0;
        project_target.pose.orientation.z = 0.0;
        project_target.color.r = 0.7;
        project_target.color.g = 0.;
        project_target.color.b = 0.4;
        project_target.color.a = 1;

        project_target.scale.x = 0.3;
        project_target.scale.y = 0.3;
        project_target.scale.z = 0.3;

        std::pair<double, double> repoint = ReProject(targetRect0);
        project_target.id = ind;
        project_target.pose.position.x = repoint.first;
        project_target.pose.position.y = repoint.second;
        rviz_markers.markers.push_back(project_target);
        ind = ind + 1;
    }


    if (rviz_markers.markers.size() > 0) {
        rviz_pub_.publish(rviz_markers);
    }
    if ((use_greet) && greet_track_MODE()) {
        //画多边形
        polygon.header.frame_id = "world";
        polygon.header.stamp = ros::Time();
        polygon.type = visualization_msgs::Marker::LINE_STRIP;
        polygon.action = visualization_msgs::Marker::ADD;
        polygon.lifetime = ros::Duration(0.5);
        polygon.scale.x = 0.1;
        polygon.color.r = 0.2;
        polygon.color.g = 0;
        polygon.color.b = 0.8;
        polygon.color.a = 1;

        for (int i = 0; i < Poly().size(); ++i) {
            geometry_msgs::Point p;
            p.x = std::get<0>(Poly()[i]);
            p.y = std::get<1>(Poly()[i]);
            p.z = 0;
            polygon.points.push_back(p);
        }
        geometry_msgs::Point p;
        p.x = std::get<0>(Poly()[0]);
        p.y = std::get<1>(Poly()[0]);
        p.z = 0;
        polygon.points.push_back(p);
        rviz_range_.publish(polygon);

        //多边形门
        poly_door.header.frame_id = "world";
        poly_door.header.stamp = ros::Time();
        poly_door.type = visualization_msgs::Marker::LINE_STRIP;
        poly_door.action = visualization_msgs::Marker::ADD;
        poly_door.lifetime = ros::Duration(0.5);
        poly_door.scale.x = 0.1;
        poly_door.color.r = 0;
        poly_door.color.g = 0.6;
        poly_door.color.b = 0.6;
        poly_door.color.a = 1;

        for (int i = 0; i < doorPoly().size(); ++i) {
            geometry_msgs::Point p;
            p.x = std::get<0>(doorPoly()[i]);
            p.y = std::get<1>(doorPoly()[i]);
            p.z = 0;
            poly_door.points.push_back(p);
        }

        p.x = std::get<0>(doorPoly()[0]);
        p.y = std::get<1>(doorPoly()[0]);
        p.z = 0;
        poly_door.points.push_back(p);
        rviz_poly_door_.publish(poly_door);

        //多边形墙
        poly_wall.header.frame_id = "world";
        poly_wall.header.stamp = ros::Time();
        poly_wall.type = visualization_msgs::Marker::LINE_STRIP;
        poly_wall.action = visualization_msgs::Marker::ADD;
        poly_wall.lifetime = ros::Duration(0.5);
        poly_wall.scale.x = 0.1;
        poly_wall.color.r = 0;
        poly_wall.color.g = 0.6;
        poly_wall.color.b = 0;
        poly_wall.color.a = 1;

        for (int i = 0; i < Poly_wall().size(); ++i) {
            geometry_msgs::Point p;
            p.x = std::get<0>(Poly_wall()[i]);
            p.y = std::get<1>(Poly_wall()[i]);
            p.z = 0;
            poly_wall.points.push_back(p);
        }

        p.x = std::get<0>(Poly_wall()[0]);
        p.y = std::get<1>(Poly_wall()[0]);
        p.z = 0;
        poly_wall.points.push_back(p);
        rviz_poly_wall_.publish(poly_wall);
    }

    //人物分数
}

double caculate_human_angle(double human_x, double human_y) {
    double x_, y_, degree_, theta_;
    x_ = human_x - robotPoseX();
    y_ = human_y - robotPoseY();
    theta_ = x_ > 0 ? atan(y_ / x_) - robotYaw() : atan(y_ / x_) + 3.1415926 - robotYaw();
    degree_ = theta_ / 3.1415926 * 180;
    while (degree_ < -180)
        degree_ += 360;
    while (degree_ > 180)
        degree_ -= 360;

    return degree_;
}

void Ltracker::human_publishment(std::list <Human> tracked_huamns) {
    std_msgs::Float32MultiArray humans_msg;
    int j = 0;
    double human_x, human_y;
    std::pair<double, double> s;
    double vx, vy, x_, y_, theta_, degree_;

    // std::cout << "size0: " << tracked_huamns.size() << std::endl;
    for (auto it = tracked_huamns.begin(); it != tracked_huamns.end();) {
        //ToAmcl(it->getCurrentX(), it->getCurrentY(), human_x, human_y);
        human_x = it->getAmclX();
        human_y = it->getAmclY();
        // if ((it->move_flag == false) || (fabs(caculate_human_angle(human_x, human_y)) > detectAngle()))
        //if ((sqrt(vx*vx+vy*vy)<0.1) || (fabs(caculate_human_angle(human_x, human_y)) > detectAngle()))
        if (fabs(caculate_human_angle(human_x, human_y)) > detectAngle())
            it = tracked_huamns.erase(it);
        else
            it++;
    }
    //std::cout << "size1: " << tracked_huamns.size() << std::endl;
    int msg_size = tracked_huamns.size();
    humans_msg.layout.data_offset = 0;
    humans_msg.data.resize(9 * msg_size);
    humans_msg.layout.dim.resize(2);
    humans_msg.layout.dim[0].label = "height";
    humans_msg.layout.dim[0].size = msg_size;
    humans_msg.layout.dim[0].stride = 9 * msg_size;

    humans_msg.layout.dim[1].label = "width";
    humans_msg.layout.dim[1].size = 9;
    humans_msg.layout.dim[1].stride = 9;

    //std::cout << "asdsd" << tracked_huamns.size() <<std::endl;
    //std::cout << "track human size: " << tracked_huamns.size() << std::endl;
    if (tracked_huamns.size() > 0) {
        //std::cout << tracked_huamns.size() << std::endl;
        for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++, j++) {
            s = Speed_amcl(std::make_tuple(it->getCurrentX(), it->getCurrentY()), it->getCurrent_speed());
            //ToAmcl(it->getCurrentX(), it->getCurrentY(), human_x, human_y);
            human_x = it->getAmclX();
            human_y = it->getAmclY();
            vx = s.first;
            vy = s.second;

            humans_msg.data[j * 9] = 0.9;

            humans_msg.data[j * 9 + 1] = human_x;
            humans_msg.data[j * 9 + 2] = human_y;
            humans_msg.data[j * 9 + 3] = sqrt(vx * vx + vy * vy);
            if (vx == 0 && vy == 0)
                degree_ = 0;
            else {
                theta_ = vx > 0 ? atan(vy / vx) : atan(vy / vx) + 3.1415926;
                degree_ = theta_ / 3.1415926 * 180;
                while (degree_ < 0)
                    degree_ += 360;
                while (degree_ > 360)
                    degree_ -= 360;
            }
            humans_msg.data[j * 9 + 8] = degree_;
            humans_msg.data[j * 9 + 5] = 1;
            humans_msg.data[j * 9 + 6] = it->getId();
            humans_msg.data[j * 9 + 7] = distToRobotPose(human_x, human_y);

            degree_ = caculate_human_angle(human_x, human_y);
            humans_msg.data[j * 9 + 4] = degree_;
        }
        humans_pub_.publish(humans_msg);
        // ROS_INFO("Normal output");
        // std::cout << "normal_output" <<std::endl;
    } else {
        // ROS_INFO("No valid human");
        //std::cout << "humana size shsould be 0" <<std::endl;
        msg_size = 0;
        humans_msg.layout.data_offset = 0;
        humans_msg.data.resize(9 * msg_size);
        humans_msg.layout.dim.resize(2);
        humans_msg.layout.dim[0].label = "height";
        humans_msg.layout.dim[0].size = msg_size;
        humans_msg.layout.dim[0].stride = 9 * msg_size;

        humans_msg.layout.dim[1].label = "width";
        humans_msg.layout.dim[1].size = 9;
        humans_msg.layout.dim[1].stride = 9;

        humans_pub_.publish(humans_msg);
    }
}

void Ltracker::humanPredictor() {
    std::list <Human> &tracked_huamns = trackedHuman();
    // frame = cameraFrame().clone();
    // cv::Mat show;
    // show = frame.clone();
    // yoloDetect(show);
    // yoloBoundingBox(show);

    int da;
    nh_.param("/strategy/greeter/detect_angle", da, 90);
    //std::cout << "detect Angle: " << da << std::endl;
    detectAngle() = da;
    //std::cout << "detectAngle() " << detectAngle() << std::endl;
    human_publishment(tracked_huamns);
    human_predict_path(tracked_huamns);
    rviz_visualization(use_greet, tracked_huamns);

    if (0) //调试用
    {
        cv::Mat show_map, show_map0;
        cv::Mat scoremap = globalScoreMapProto();
        // ROS_WARN("map");
        //std::cout << scoremap << std::endl;
        cv::Mat costmap = globalCostMapProto();
        cv::cvtColor(scoremap, show_map, CV_GRAY2RGB);
        cv::cvtColor(costmap, show_map0, CV_GRAY2RGB);

        for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
            // double x = it->getCurrentX();
            // double y = it->getCurrentY();
            // //std::cout<<x<<"\t"<<y<<std::endl;
            // ToAmcl(x, y, x, y);
            double x = it->getAmclX();
            double y = it->getAmclY();
            if (projectToMap(x, y, x, y)) {
                cv::circle(show_map, cv::Point((int) x, (int) y), 3, cv::Scalar(255, 0, 0), -1);
                putText(show_map, std::to_string(it->getId()), cv::Point(int(x), int(y)),
                        cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 0, 0), 1, CV_AA);
            }
        }

        double x = robotPoseX(); //直接用的amcl坐标，近似，不是标准
        double y = robotPoseY();
        ToAmcl(x, y, x, y);
        if (projectToMap(x, y, x, y)) {
            cv::circle(show_map, cv::Point((int) x, (int) y), 3, cv::Scalar(0, 0, 255), -1);
            putText(show_map, "robot", cv::Point(int(x), int(y)),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(0, 0, 255), 1, CV_AA);
        }

        cv::imshow("show map", show_map);
        cv::waitKey(10);
    }
}

bool Ltracker::choose_target_tracker(std::list <Human> tracked_huamns, double time_th) {
    ROS_WARN("Finding new target...");
    auto candidates = tracked_huamns;
    candidates.clear();
    path_tracking_flag = false;
    for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
        double x = it->getAmclX();
        double y = it->getAmclY();
        double t = it->getLastTime();
        if ((distToRobotPose(x, y) < 1) && (t > time_th))
            candidates.push_back(*it);
    }
    Human target0;
    if (candidates.size() == 0) {
        rs.robot_show_state_ == TRACKER_FIND;
        return false;
    }

    double nearest = 1000;
    //一米内选择和机器人夹角最近的
    for (auto it = candidates.begin(); it != candidates.end(); it++) {
        double x = it->getAmclX();
        double y = it->getAmclY();
        double d_angle = caculate_human_angle(x, y);

        if (fabs(d_angle) < nearest) {
            nearest = fabs(d_angle);
            target0 = *it;
        }
    }
    ti.target = target0;
    ti.targetID = target0.getId();
    ti.tx = target0.getAmclX();
    ti.ty = target0.getAmclY();
    // ti.predict_path = target0.
    Init_pos = target0.init_position;
    this->Tar_dis = false; //target discard
    this->isHUman = false;
    rs.second_state_ = ACTIVE_R;
    //pr_cnt = 0;
    motion_.back_cnt = 0;
    first_found_cnt = 0;
    second_find_cnt = 0;
    //_target_find_flag = 0;
    second_find_flag = false;
    tracker_loss_flag = false;
    tracker_loss_cnt = 0;
    //（这里需要推送状态，说搜寻到新的目标任务） todo_ziwei
    ROS_ERROR("New find target");
    ROS_WARN("Target ID: %d", ti.targetID);
    return true;
}

bool Ltracker::estimate_direction(double tx, double ty, double &pub_o) {
    double pub_oo = cv::fastAtan2((tx - robotPoseY()), (ty - robotPoseX())) * 0.017453292519943; //0-360
    std::vector <std::tuple<double, double>> points1, points2;
    std::vector<int> score_points1, score_points2;
    double adjust_x = cos(pub_oo) * 2 + tx;
    double adjust_y = sin(pub_oo) * 2 + ty;
    double map_x, map_y;
    if (!projectToMap(tx, ty, map_x, map_y))
        return false;
    if ((int) scoremap.at<unsigned char>((int) map_y, (int) map_x) < 200)
        return false;
    double pub_oo1 = pub_oo - 0.785;
    double pub_oo2 = pub_oo + 0.785;
    double xx1, yy1, xx2, yy2;
    int s;
    for (int i = 1; i < 11; i++) {
        xx1 = 0.2 * i * cos(pub_oo1) + tx;
        yy1 = 0.2 * i * sin(pub_oo1) + ty;
        points1.push_back(std::make_tuple(xx1, yy1));
        if (projectToMap(xx1, yy1, map_x, map_y))
            s = (int) scoremap.at<unsigned char>((int) map_y, (int) map_x);
        else
            s = 255;
        score_points1.push_back(s);

        xx2 = 0.2 * i * cos(pub_oo2) + tx;
        yy2 = 0.2 * i * sin(pub_oo2) + ty;
        points2.push_back(std::make_tuple(xx2, yy2));
        if (projectToMap(xx2, yy2, map_x, map_y))
            s = (int) scoremap.at<unsigned char>((int) map_y, (int) map_x);
        else
            s = 255;
        score_points2.push_back(s);
    }

    int num = score_points1.size();
    MatrixXd Y(num, 2);
    MatrixXd fx1(num, 1);
    MatrixXd fx2(num, 1);
    MatrixXd a1(2, 1);
    MatrixXd a2(2, 1);
    //   std::deque<std::tuple<double, double, double>> filtered_list;

    for (int i = 0; i < num; i++) {
        fx1(i, 0) = score_points1[i];
        fx2(i, 0) = score_points2[i];
        Y(i, 0) = i + 1;
        Y(i, 1) = 1;
    }
    a1 = (((Y.transpose()) * Y).inverse()) * (Y.transpose()) * fx1;
    a2 = (((Y.transpose()) * Y).inverse()) * (Y.transpose()) * fx2;
    if (a1(0, 0) > a2(0, 0))
        pub_o = (pub_oo2 + 0.785);
    else
        pub_o = (pub_oo1 - 0.785);
    return true;
}

//跟随模式
void Ltracker::tracker() {
    //ROS_INFO("IN TRACKER!");
    double dR = 1;
    frame = cameraFrame().clone();
    bool draw = true;
    double time_th = 0.1;
    cv::Mat show;
    if (draw)
        show = frame.clone();
    std::list <Human> &tracked_huamns = trackedHuman();
    auto candidates = tracked_huamns;
    candidates.clear();
    int pr_cnt = 0;
    if (tracked_huamns.size() > 0) {
        human_predict_path(tracked_huamns);
        human_publishment(tracked_huamns);
    }
    std_msgs::UInt32 show_state;
    if (rs.robot_show_state_ == TRACKER_TRACK)
        show_state.data = 7;
    if (rs.robot_show_state_ == TRACKER_FIND)
        show_state.data = 8;
    // if (rs.robot_show_state_ == TRACKER_LOST)
    //   show_state.data = 9;
    state_pub_.publish(show_state);

    //-----------逻辑开始------------------//
    if (tracked_huamns.size() > 0) {
        ROS_INFO("tracker_human size: %d", tracked_huamns.size());
        no_target_find_cnt = 0;
        first_state_decision(tracked_huamns, time_th);
        second_state_decision(tracked_huamns);
        // ROS_WARN("first state: %d, second state: %d, robot_state: %d", first_state_, second_state_, robot_state_);
        moving_state(show, draw, motion_.frz, motion_.max_back, pr_cnt);
    } else {
        ROS_INFO("tracker_human size==0");
        if (ti.predict_path.size() == 0) {
            ti.tx = 0;
            ti.ty = 0;
            path_tracking_flag = false;
            rs.robot_show_state_ = TRACKER_FIND;
        } else {
            rs.first_state_ = INACTIVE;
            rs.second_state_ = INACTIVE;
            //std::cout << "aaaa" << std::endl;
            int path_size = ti.predict_path.size() - 1;
            //std::cout << "path_size: " << path_size << std::endl;
            double txx = std::get<0>(ti.predict_path[path_size]);
            double tyy = std::get<1>(ti.predict_path[path_size]);
            if (distToRobotPose(txx, tyy) > 0.5) {
                double map_x, map_y;
                bool a = projectToMap(txx, txx, map_x, map_y);
                while ((((int) scoremap.at<unsigned char>((int) map_y, (int) map_x) > 240)) && (path_size > 0)) {
                    //std::cout << "cccc" << std::endl;
                    //std::cout << "size: " << path_size << std::endl;
                    path_size = path_size - 1;
                    double txx = std::get<0>(ti.predict_path[path_size]);
                    double tyy = std::get<1>(ti.predict_path[path_size]);
                    //std::cout << "txx: " << txx << " tyy: " << tyy << std::endl;
                    a = projectToMap(txx, txx, map_x, map_y);
                    //std::cout << "score: " << (int)scoremap.at<unsigned char>((int)map_y, (int)map_x) << std::endl;
                }
                if (path_size == 0) {
                    ROS_INFO("target predict path is in wall, tx=0, ty=0");
                    ti.tx = 0;
                    ti.ty = 0;
                    path_tracking_flag = false;
                    rs.robot_show_state_ = TRACKER_FIND;
                } else {
                    ti.tx = txx;
                    ti.ty = tyy;
                    //判断目标点远近
                    first_found_cnt = 0;
                    double dis_th = 1.5;
                    double predict_x = ti.tx;
                    double predict_y = ti.ty;
                    // if (distToRobotPose(predict_x, predict_y) > dis_th)
                    //   neighbour_flag = false;
                    // else
                    //   neighbour_flag = true;
                    stay_flag = false;
                    path_tracking_flag = true;
                    ROS_WARN("No target but Track the path end!");
                    tp.pub_time_now = ros::Time::now();
                    if (((tp.pub_time_now.toSec() - tp.pub_time_last.toSec()) > motion_.frz) &&
                        (motion_.lock_pub == false) &&
                        (ti.tx != 0 && ti.ty != 0)) {
                        motion_.tracker_tracking(tp, ti, rs, no_task, false, path_tracking_flag);
                        motion_.init_flag = false;
                        ROS_INFO("targetID: %d, isDiscard: %d", ti.targetID, false);
                        ROS_ERROR("Start tracking without human detection!");
                    }
                }
            } else {
                no_target_find_cnt = no_target_find_cnt + 1;
                ROS_INFO("NO target! %d", no_target_find_cnt); //一直没有human
                if (no_target_find_cnt > 15) {
                    ti.tx = 0;
                    ti.ty = 0;
                    rs.robot_show_state_ = TRACKER_FIND;
                }
                path_tracking_flag = false;
                rs.first_state_ = INACTIVE;
                rs.second_state_ = INACTIVE;
                //rs.robot_show_state_ = TRACKER_FIND;
                first_found_cnt = 0;
                second_find_cnt = 0;
                second_find_flag = false;
            }
        }
    }

    if (skipFlag()) //skipFlag()
    {
        ROS_WARN("Recieve Skip! Skip the tracking target!");
        // if(robotVel()!=0 || robot_Theta()!=0)
        {
            std_msgs::UInt32 stop;
            stop.data = 1;
            stop_track_pub_.publish(stop);
        }
        ti.tx = 0;
        ti.ty = 0;
        rs.first_state_ = INACTIVE;
        rs.second_state_ = INACTIVE;
        rs.robot_show_state_ = TRACKER_FIND;
        rs.robot_state_ = LOST;
        first_found_cnt = 0;
        second_find_cnt = 0;
        second_find_flag = false;
        skipFlag() = false;
    }

    rviz_visualization(use_greet, tracked_huamns);
}

// 迎宾模式：有物体进大圈，进入目标list，对每一个目标，直接出预测轨迹，保存轨迹进入轨迹List。选择离小圈最近的，发转向指令，判断是不是人。
//是人，初始化分类器，启动加摄像头的跟踪。不是人，remove from 目标list.永远都是 pop the top of the list.
//大圈里没有目标点时，human 的 ID 清零
void Ltracker::greeter() {
    std::vector <std::tuple<double, double>> points;
    cv::Mat show_map, show_map0;
    double time_th = 0.1;
    bool draw = true;
    double dR = 2; //第二个圈到第三个圈的距离
    static int pr_cnt = 0;
    double r_limit = 1; //人向门走的方向限制
    frame = cameraFrame().clone();
    cv::Mat show;
    if (draw)
        show = frame.clone();
    int height = frameHeight();
    int width = frameWidth();
    std::list <Human> &tracked_huamns = trackedHuman();
    // double frz = 1.2; //1秒1次

    //迎宾范围之外去目标
    for (auto it = tracked_huamns.begin(); it != tracked_huamns.end();) {
        // double x = it->getCurrentX();
        // double y = it->getCurrentY();
        // ToAmcl(x, y, x, y);
        double x = it->getAmclX();
        double y = it->getAmclY();
        if (!it->isfar)
            it->isfar = it->isFar();
        std::tuple<double, double> p = std::make_tuple(x, y);

        if ((!PtInPolygon(p, Poly(), nCount)) && (!PtInPolygon(p, Poly_wall(), Poly_wall().size()))) { //在机器人点目标去掉
            ROS_INFO("Out of range: %d", it->getId());
            it = tracked_huamns.erase(it);
        } else
            it++;
    }
    //开始画出人物
    human_predict_path(tracked_huamns);
    human_publishment(tracked_huamns);
    //判断人物是否在远离目标点push_back
    for (auto it = tracked_huamns.begin(); it != tracked_huamns.end();) {
        //ROS_WARN("erase *******");
        // double xx, yy;
        // ToAmcl(it->getCurrentX(), it->getCurrentY(), xx, yy);
        double xx = it->getAmclX();
        double yy = it->getAmclY();
        std::tuple<double, double> p = std::make_tuple(xx, yy);
        double orx = std::get<0>(Origin());
        double ory = std::get<1>(Origin());
        double dis = distToDoorPose(orx, ory) + 0.5;
        //人物列表确定，走远就删，此处跟人策略待改
        if (((it->isFar() || it->isfar) && (!it->timer_ok) && (init_range(p)))) //||((distToDetectRegion(xx,yy))>dis)
        {                                                                       //这里人只要走远就删掉
            //ROS_INFO("first_dis: %f, last_dis: %f", first_dis,last_dis);
            ROS_WARN("Delete: The ID: %d is walking out...", it->getId());
            it = tracked_huamns.erase(it);
        } else
            it++;
    }

    //ROS_INFO("Angle threshold: %f", Angle() / 3.14 * 180);
    //只有当robot不在工作状态时才实时更新潜在的目标点的位置
    if ((rs.robot_show_state_ == FREE) || (rs.robot_show_state_ == GOODBYE))
        fill_candidates(tracked_huamns);

    if (0) //debug
    {
        //FREE=0, GOODBYE, TRACK_FAR, TRACK_NEAR, TRACK_STAY, TRACK_BACK
        if (rs.robot_show_state_ == FREE)
            ROS_ERROR("ROBOT State: FREE");
        if (rs.robot_show_state_ == GOODBYE)
            ROS_ERROR("ROBOT State: GOODBYE");
        if (rs.robot_show_state_ == TRACK_FAR)
            ROS_ERROR("ROBOT State: TRACK_FAR");
        if (rs.robot_show_state_ == TRACK_NEAR)
            ROS_ERROR("ROBOT State: TRACK_NEAR");
        if (rs.robot_show_state_ == TRACK_STAY)
            ROS_ERROR("ROBOT State: TRACK_STAY");
        if (rs.robot_show_state_ == TRACK_BACK)
            ROS_ERROR("ROBOT State: TRACK_BACK");
    }

    std_msgs::UInt32 show_state;
    if (rs.robot_show_state_ == FREE)
        show_state.data = 0;
    if (rs.robot_show_state_ == GOODBYE)
        show_state.data = 5;
    if (rs.robot_show_state_ == TRACK_FAR)
        show_state.data = 1;
    if (rs.robot_show_state_ == TRACK_NEAR)
        show_state.data = 2;
    if (rs.robot_show_state_ == TRACK_STAY)
        show_state.data = 3;
    if (rs.robot_show_state_ == TRACK_BACK)
        show_state.data = 4;
    state_pub_.publish(show_state);

    //-----------------Only Mode 2:发传单模式 needs ------------------------//
    //保存上次状态，开启跟踪计时器功能，用于发传单模式

    if (MODE() == 2) {
        rs.robot_show_state_last = rs.robot_show_state_;
        if (rs.robot_show_state_ != TRACK_STAY) {
            timer_start = ros::Time::now();
            timer_end = ros::Time::now();
        } else if (rs.robot_show_state_last != TRACK_STAY && rs.robot_show_state_ == TRACK_STAY) {
            timer_start = ros::Time::now();
        } else if (rs.robot_show_state_last == TRACK_STAY && rs.robot_show_state_ == TRACK_STAY) {
            timer_end = ros::Time::now();
        }
    }

    //判断发传单跟踪时间是否到了
    if (MODE() == 2) {
        if ((timer_end.toSec() - timer_start.toSec()) > 2) {
            for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
                if (it->getId() == ti.targetID) {
                    ROS_WARN("ID: %d: timer is not ok", it->getId());
                    it->timer_ok = false;
                }
            }
        }
    }
    //------------------------------------------------------//

    //-------标志位处理----------------------------//
    //如果是从pause状态恢复，直接从INACTIVE开始重新找目标，
    if (ResumeFlag()) {
        rs.first_state_ = INACTIVE;
        rs.second_state_ = INACTIVE;
        // double disx, disy;
        // ToAmcl(ti.target.getCurrentX(), ti.target.getCurrentY(), disx, disy);
        double disx = ti.target.getAmclX();
        double disy = ti.target.getAmclY();
        if (distToRobotPose(disx, disy) < ti.target.dis_boundary) {
            ti.target.isDiscard = true;
            std::list<Human>::iterator it_find_id = find(tracked_huamns.begin(), tracked_huamns.end(), ti.target);
            if (it_find_id != tracked_huamns.end())
                it_find_id->isDiscard = true;
        }
        ResumeFlag() == false;
    }
    //接受到skip，放弃当前目标
    if (skipFlag()) {
        for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
            if (it->getId() == ti.targetID)
                it->isDiscard = true;
        }
        ti.target.isDiscard = true;
        ROS_WARN("Recieve Skip! targetId: %d, isDiscard: %d", ti.targetID, ti.target.isDiscard);
        skipFlag() = false;
    }
    //-----------------------------------------//
    if (tracked_huamns.size() > 0) {
        //Human ti.target = *(tracked_huamns.begin());
        first_state_decision(tracked_huamns, time_th, pr_cnt, show, draw);
        if (!(rs.first_state_ == INACTIVE && rs.second_state_ == INACTIVE))
            no_task = false;
        if (no_task) {
            human_out(tracked_huamns, motion_.frz);
        }
        second_state_decision(tracked_huamns);
        // ROS_WARN("first state: %d, second state: %d, robot_state: %d", first_state_, second_state_, robot_state_);
        moving_state(show, draw, motion_.frz, motion_.max_back, pr_cnt);
    } else {
        //ROS_WARN("NO target!");
        rs.robot_state_ = LOST;
        freeOperation(std::get<0>(Origin()), std::get<1>(Origin()));
    }

//  sensor_msgs::ImagePtr show_msg_ptr = cv_bridge::CvImage(std_msgs::Header(), "bgr8", show).toImageMsg();
//  sensor_msgs::Image show_msg = *show_msg_ptr;
//  //std::cout << show_msg << std::endl;
//  image_pub_.publish(show_msg);

    rviz_visualization(use_greet, tracked_huamns); //use_greet
    //显示
    if (0) //调试用
    {
        cv::Mat scoremap = globalScoreMapProto();
        ROS_WARN("map");
        // std::cout << scoremap << std::endl;
        cv::Mat costmap = globalCostMapProto();
        cv::cvtColor(scoremap, show_map, CV_GRAY2RGB);
        cv::cvtColor(costmap, show_map0, CV_GRAY2RGB);

        for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
            // double x = it->getCurrentX();
            // double y = it->getCurrentY();
            // //std::cout<<x<<"\t"<<y<<std::endl;
            // ToAmcl(x, y, x, y);
            double x = it->getAmclX();
            double y = it->getAmclY();
            if (projectToMap(x, y, x, y)) {
                cv::circle(show_map, cv::Point((int) x, (int) y), 3, cv::Scalar(255, 0, 0), -1);
                putText(show_map, std::to_string(it->getId()), cv::Point(int(x), int(y)),
                        cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 0, 0), 1, CV_AA);
            }
        }

        double x = robotPoseX(); //直接用的amcl坐标，近似，不是标准
        double y = robotPoseY();
        ToAmcl(x, y, x, y);
        if (projectToMap(x, y, x, y)) {
            cv::circle(show_map, cv::Point((int) x, (int) y), 3, cv::Scalar(0, 0, 255), -1);
            putText(show_map, "robot", cv::Point(int(x), int(y)),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(0, 0, 255), 1, CV_AA);
        }

        cv::imshow("show map", show_map);
        cv::waitKey(10);
    }
}

//laser没掉，摄像头框没了。
//laser没掉，摄像头框框错了。
void Ltracker::laserTrack_first_state_decision(std::list <Human> &tracked_huamns) {
    switch (rs.first_state_) {
        //first state进入INACTIVE的状态证明上一帧肯定是摄像头和激光都丢失了
        // first_found_cnt表示这暂时的状态
        case INACTIVE: {
            if (!missed) //这一帧摄像头找回来了，和first_state==ACTIVE_R处理方式不一样，没有targetID
            {
                first_found_cnt = 0;
                second_find_cnt = 0;
                second_find_flag = false;
                path_tracking_flag = false;
                cv::Rect targetRect0 = targetRect;
                {//没找到target id, 摄像头没掉。这里考虑目标是不是还在laser平面上存在的情况, 和没有track_humans在laser平面上相同
                    //1.单纯脚被前面人挡住掉id了，不仅自己的laser要和reproject距离足够小，还要和(ti.tx, ti.ty)距离比较小
                    double min_d = max_reproject_dist_small;
                    auto idx_human = tracked_huamns.end();
                    for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
                        double dd = dist2Reproject(targetRect0, *it);
                        if (dd < min_d
                            && distance_l2(it->getAmclX(), it->getAmclY(), ti.tx, ti.ty) < 1.5) //激光掉的范围也不大，可以马上补回来的类型
                        {
                            min_d = dd;
                            idx_human = it;
                        }
                    }
                    if (idx_human != tracked_huamns.end()) {
                        ROS_WARN("Got Camera Rect again, Change ID to %d, dist2Reproject: %f", idx_human->getId(),
                                 min_d);
                        ti.targetID = idx_human->getId();
                        ti.tx = idx_human->getAmclX();
                        ti.ty = idx_human->getAmclY();
                        idx_human->IDoffset(ti.target);
                        ti.target = *idx_human;
                        rs.second_state_ = ACTIVE_R;
                        buffer_cnt0 = 0;
                    } else {
                        //激光掉的范围太大了（被挡的时间太久了）or 目标激光已经消失了， move_base 发的点需要自己估计, 红点方向看附近有没有human
                        //有就发最近的human位置，没有对比和墙相交的点，在红色点之前就发和墙相交的点的位置，否则发红点位置

                        std::pair<double, double> repoint = ReProject(targetRect0);
                        std::pair<double, double> robotpoint = std::make_pair(robotPoseX(), robotPoseY());
                        auto choose_idx = tracked_huamns.end();
                        double distmin = max_reproject_dist_small;
                        for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
                            std::pair<double, double> humanpoint = std::make_pair(it->getAmclX(), it->getAmclY());
                            double d = dis2line(repoint, robotpoint, humanpoint);
                            if (d < distmin &&
                                distToRobotPose(it->getAmclX(), it->getAmclX()) <
                                distToRobotPose(repoint.first, repoint.second)) {
                                choose_idx = it;
                                distmin = d;
                            }
                        }
                        if (choose_idx !=
                            tracked_huamns.end()) //human点到线的距离<threshold && dist(human,robot)<dist(red,robot)
                        {
                            //发human的点
                            ROS_WARN("1.inactive, Human in red point line, Change ID to %d, dist2Reproject: %f",
                                     choose_idx->getId(), distmin);
                            ti.targetID = choose_idx->getId();
                            ti.tx = choose_idx->getAmclX();
                            ti.ty = choose_idx->getAmclY();
                            choose_idx->IDoffset(ti.target);
                            // correct_reproject = Reproject(targetRect0, *idx_human);
                            ti.target = *choose_idx;
                            rs.second_state_ = ACTIVE_R;
                            buffer_cnt0 = 0;
                        } else { //没有人可以依照了，ref_is_human = false
                            //ref_is_human = false;
                            double x = repoint.first;
                            double y = repoint.second;
                            bool out_wall = false;
                            if (projectToMap(x, y, x, y)) {
                                if ((int) scoremap.at<unsigned char>((int) y, (int) x) < 245) //红点在墙前，发红点
                                {
                                    ROS_WARN("1.inactive, send red point.");
                                    out_wall = true;
                                    ti.targetID = 888;
                                    ti.tx = repoint.first;
                                    ti.ty = repoint.second;
                                    rs.second_state_ = ACTIVE_R;
                                    buffer_cnt0 = 0;
                                }
                            }
                            if (!out_wall) //红点在墙内 或者 在地图外
                            {
                                double angle_oo =
                                        cv::fastAtan2((repoint.second - robotPoseY()), (repoint.first - robotPoseX())) *
                                        0.017453292519943; //0-360
                                double d = distToRobotPose(repoint.first, repoint.second);
                                int times = (int) d * 5 + 1;
                                for (int i = 1; i < times; i++) {
                                    double times_x = cos(angle_oo) * (d - i * 0.2);
                                    double times_y = sin(angle_oo) * (d - i * 0.2);
                                    double map_times_x, map_times_y;
                                    if (projectToMap(times_x, times_y, map_times_x, map_times_y)) {
                                        if ((int) scoremap.at<unsigned char>((int) map_times_y, (int) map_times_x) <
                                            245) {
                                            ROS_WARN("1.inactive, send wall point.");
                                            ti.targetID = 888;
                                            ti.tx = times_x;
                                            ti.ty = times_y;
                                            rs.second_state_ = ACTIVE_R;
                                            buffer_cnt0 = 0;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else { //这一帧摄像头还在掉，激光平面找
                ti.tx = 0;
                ti.ty = 0;
                first_found_cnt = first_found_cnt + 1;
                rs.second_state_ = INACTIVE;
                int max_cnt0;
                if (path_tracking_flag)
                    max_cnt0 = 25;//3.5s
                else
                    max_cnt0 = 15;//2s
                ROS_INFO("first_found_cnt: %d, path_tracking_flag: %d, tx: %f, ty: %f", first_found_cnt,
                         path_tracking_flag, ti.tx, ti.ty);
                //和纯激光不一样，时间没到，不发新目标点，继续上次追踪，相当于只是等待目标出现
                if ((first_found_cnt < max_cnt0) && (ti.tx != 0 && ti.ty != 0)) {
                    ROS_WARN("No Camera and no laser, do not send new target, KEEP Waitting...");
                    rs.second_state_ = ACTIVE_R;
                    buffer_cnt0 = 0;
                } else //时间到了，报丢失
                {
                    catched = false;
                    ROS_ERROR("TARGET LOST, NEED TO DETECT AGAIN");
                }
            }
        }
            break;
        case ACTIVE_R: //有目标状态
        {
            if (!missed) { //摄像头没掉,只要摄像头没掉，rs.second_state_一定是ACTIVE_R，只要有框不管是不是stay状态，一并处理
                second_find_cnt = 0;
                second_find_flag = false;
                path_tracking_flag = false;
                cv::Rect targetRect0 = targetRect;
                bool findId = false;
                auto find_it = tracked_huamns.end();
                for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
                    if (it->getId() == ti.targetID) {
                        findId = true;
                        find_it = it;
                        break;
                    }
                }
                if (findId) {
                    if (dist2Reproject(targetRect0, *find_it) < max_reproject_dist_small) {
                        ROS_WARN("Find camera rect, find laser ID");
                        ti.target = *find_it;
                        ti.tx = find_it->getAmclX();
                        ti.ty = find_it->getAmclY();
                        buffer_cnt0 = 0;
                        rs.second_state_ = ACTIVE_R;
                    } else //摄像头没掉，laser掉了； 1. 可能id错了，2 .可能摄像头暂时性错了 3.可能project错了，要给冗余缓冲时间buffer_cnt0
                    {
                        buffer_cnt0 = buffer_cnt0 + 1;
                        if (buffer_cnt0 > 4) //缓冲时间到了
                        {
                            //track_humans找离自己的reproject最近的，看他是不是原id，是就缓冲清零buffer_cnt0，更新(ti.tx, ti.ty)
                            //不是就更新id和(ti.tx, ti.ty)
                            //double min_dist = max_reproject_dist;
                            double min_d = max_reproject_dist_big;
                            auto idx_human = tracked_huamns.end();
                            for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
                                double dd = dist2Reproject(targetRect0, *it);
                                if (dd < min_d) {
                                    min_d = dd;
                                    idx_human = it;
                                }
                            }
                            if (idx_human->getId() == ti.targetID) { //是原id就缓冲清零buffer_cnt0，更新(ti.tx, ti.ty)
                                ROS_WARN("buffer time ok, Rect Change ID! from %d to %d, dist2Reproject: %f",
                                         ti.targetID, idx_human->getId(), min_d);
                                ti.target = *idx_human;
                                ti.tx = idx_human->getAmclX();
                                ti.ty = idx_human->getAmclY();
                                buffer_cnt0 = 0;
                                rs.second_state_ = ACTIVE_R;
                            } else if (idx_human != tracked_huamns.end()) { //不是就更新id和correct_reproject
                                ROS_WARN("buffer time ok, Rect Change ID! from %d to %d, dist2Reproject: %f",
                                         ti.targetID, idx_human->getId(), min_d);
                                ti.targetID = idx_human->getId();
                                ti.tx = idx_human->getAmclX();
                                ti.ty = idx_human->getAmclY();
                                idx_human->IDoffset(ti.target);
                                // correct_reproject = Reproject(targetRect0, *idx_human);
                                ti.target = *idx_human;
                                rs.second_state_ = ACTIVE_R;
                                buffer_cnt0 = 0;
                            } else { //投影的数值不对，依然按原始id跟踪，如果project问题不恢复，会导致失踪,
                                ROS_ERROR("!!!!!!!!Project Problems!!!!!!!");
                                ti.target = *find_it;
                                ti.tx = find_it->getAmclX();
                                ti.ty = find_it->getAmclY();
                                //correct_reproject = Reproject(targetRect0, *idx_human);
                                buffer_cnt0 = 0;
                                rs.second_state_ = ACTIVE_R;
                                break;
                            }
                        } else { //缓冲时间内 还是保持原id号继续跟踪, 如果id错了有可能越走距离越远
                            ROS_WARN("In buffer time, keep tracking original ID");
                            ti.target = *find_it;
                            ti.tx = find_it->getAmclX();
                            ti.ty = find_it->getAmclY();
                            rs.second_state_ = ACTIVE_R;
                            break;
                        }
                    }
                } else {//没找到target id, 摄像头没掉。这里考虑目标是不是还在laser平面上存在的情况, 和没有track_humans在laser平面上相同

                    //1.单纯脚被前面人挡住掉id了，不仅自己的laser要和reproject距离足够小，还要和(ti.tx, ti.ty)距离比较小
                    double min_d = max_reproject_dist_small;
                    auto idx_human = tracked_huamns.end();
                    for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
                        double dd = dist2Reproject(targetRect0, *it);
                        if (dd < min_d
                            && distance_l2(it->getAmclX(), it->getAmclY(), ti.tx, ti.ty) < 1.5) //激光掉的范围也不大，可以马上补回来的类型
                        {
                            min_d = dd;
                            idx_human = it;
                        }
                    }
                    if (idx_human != tracked_huamns.end()) {
                        ROS_WARN("Laser Drop, Change ID! from %d to %d, dist2Reproject: %f", ti.targetID,
                                 idx_human->getId(),
                                 min_d);
                        ti.targetID = idx_human->getId();
                        ti.tx = idx_human->getAmclX();
                        ti.ty = idx_human->getAmclY();
                        idx_human->IDoffset(ti.target);
                        // correct_reproject = Reproject(targetRect0, *idx_human);
                        ti.target = *idx_human;
                        rs.second_state_ = ACTIVE_R;
                        buffer_cnt0 = 0;
                    } else {
                        //激光掉的范围太大了（被挡的时间太久了）or 目标激光已经消失了， move_base 发的点需要自己估计, 红点方向看附近有没有human
                        //有就发最近的human位置，没有对比和墙相交的点，在红色点之前就发和墙相交的点的位置，否则发红点位置

                        std::pair<double, double> repoint = ReProject(targetRect0);
                        std::pair<double, double> robotpoint = std::make_pair(robotPoseX(), robotPoseY());
                        auto choose_idx = tracked_huamns.end();
                        double distmin = max_reproject_dist_small;
                        for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
                            std::pair<double, double> humanpoint = std::make_pair(it->getAmclX(), it->getAmclY());
                            double d = dis2line(repoint, robotpoint, humanpoint);
                            if (d < distmin &&
                                distToRobotPose(it->getAmclX(), it->getAmclX()) <
                                distToRobotPose(repoint.first, repoint.second)) {
                                choose_idx = it;
                                distmin = d;
                            }
                        }
                        if (choose_idx !=
                            tracked_huamns.end()) //human点到线的距离<threshold && dist(human,robot)<dist(red,robot)
                        {
                            //发human的点
                            ROS_WARN("Human in red point line, Change ID! from %d to %d, dist2Reproject: %f",
                                     ti.targetID,
                                     choose_idx->getId(), distmin);
                            ti.targetID = choose_idx->getId();
                            ti.tx = choose_idx->getAmclX();
                            ti.ty = choose_idx->getAmclY();
                            choose_idx->IDoffset(ti.target);
                            // correct_reproject = Reproject(targetRect0, *idx_human);
                            ti.target = *choose_idx;
                            rs.second_state_ = ACTIVE_R;
                            buffer_cnt0 = 0;
                        } else { //没有人可以依照了，ref_is_human = false
                            //ref_is_human = false;
                            double x = repoint.first;
                            double y = repoint.second;
                            bool out_wall = false;
                            if (projectToMap(x, y, x, y)) {
                                if ((int) scoremap.at<unsigned char>((int) y, (int) x) < 245) //红点在墙前，发红点
                                {
                                    ROS_WARN("1.active_r, send red point");
                                    out_wall = true;
                                    ti.targetID = 888;
                                    ti.tx = repoint.first;
                                    ti.ty = repoint.second;
                                    rs.second_state_ = ACTIVE_R;
                                    buffer_cnt0 = 0;
                                }
                            }
                            if (!out_wall) //红点在墙内 或者 在地图外
                            {
                                double angle_oo =
                                        cv::fastAtan2((repoint.second - robotPoseY()), (repoint.first - robotPoseX())) *
                                        0.017453292519943; //0-360
                                double d = distToRobotPose(repoint.first, repoint.second);
                                int times = (int) d * 5 + 1;
                                for (int i = 1; i < times; i++) {
                                    double times_x = cos(angle_oo) * (d - i * 0.2);
                                    double times_y = sin(angle_oo) * (d - i * 0.2);
                                    double map_times_x, map_times_y;
                                    if (projectToMap(times_x, times_y, map_times_x, map_times_y)) {
                                        if ((int) scoremap.at<unsigned char>((int) map_times_y, (int) map_times_x) <
                                            240) {
                                            ROS_WARN("1.active_r, send wall point");
                                            ti.targetID = 888;
                                            ti.tx = times_x;
                                            ti.ty = times_y;
                                            rs.second_state_ = ACTIVE_R;
                                            buffer_cnt0 = 0;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else { //摄像头掉了，laser那一套，但是不再使用找激光的方法了
                ROS_ERROR("First State Camera Rect Lost!");
                //std::cout << "first state active" << std::endl;
                double min_dis = 1000;
                double min_th;
                min_th = 0.8;

                if (rs.second_state_ != STAY_SEARCH) { //找ID号一样的
                    for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
                        if (it->getId() == ti.targetID) {
                            ROS_WARN("1.active_r, camera drop, find same laserID");
                            ti.target = *it;
                            ti.tx = it->getAmclX();
                            ti.ty = it->getAmclY();
                            second_find_cnt = 0;
                            second_find_flag = false;
                            rs.second_state_ = ACTIVE_R;
                            break;
                        } else {
                            //ID没找到,
                            it->toTarget = sqrt((ti.tx - it->getAmclX()) * (ti.tx - it->getAmclX()) +
                                                (ti.ty - it->getAmclY()) * (ti.ty - it->getAmclY())); //到目标ID的距离
                            if (it->toTarget < min_th) {
                                if ((min_dis > it->toTarget) && (it->getId() > (it->getNewestID() - 5)))
                                    min_dis = it->toTarget;
                            }
                        }
                    }
                    //这里考虑的是激光遮挡导致掉ID，摄像头完全掉了
                    if (rs.second_state_ == INACTIVE) {
                        second_find_cnt = second_find_cnt + 1;
                        ROS_INFO("second_find_cnt: %d", second_find_cnt);
                        //std::cout << "second_find_cnt: " << second_find_cnt << std::endl;
                        if (second_find_cnt < 7) //1秒左右,激光找人只能维持一秒
                        {
                            second_find_flag = true;
                            if ((min_dis != 1000)) //&& (!path_tracking_flag) //在上次目标消失位置的0.8米内有替代目标
                            {                      //找target,换一个target，只要在target 0.8米范围内，离target最近的
                                for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
                                    if (it->toTarget == min_dis) {
                                        ROS_WARN("camera and laser drop, Pure Laser Change2 ID! from %d to %d",
                                                 ti.targetID, it->getId());
                                        ti.targetID = it->getId();
                                        ti.tx = it->getAmclX();
                                        ti.ty = it->getAmclY();
                                        it->IDoffset(ti.target);
                                        ti.target = *it;
                                        rs.second_state_ = ACTIVE_R;
                                        second_find_cnt = 0;
                                        second_find_flag = false;
                                        path_tracking_flag = false;
                                        break;
                                    }
                                }
                            }
                            //rs.first_state_ = ACTIVE_R;
                        } else
                            //min_dis还是1000，证明上次目标0.8米内没有人；
                            //0.8米范围没有可能人拐弯了，这时候目标id不变，直接走到路径在墙外的最后一点，拐弯后关闭激光找人功能
                        {
                            //std::cout << "second: " << second_find_cnt << std::endl;
                            //todo_ziwei: 理论上路径最后一点不会在墙里，但仍要实验论证
                            second_find_flag = false;
                            int path_size = ti.predict_path.size() - 1;
                            double txx = std::get<0>(ti.predict_path[path_size]);
                            double tyy = std::get<1>(ti.predict_path[path_size]);
                            double map_x, map_y;
                            bool a = projectToMap(txx, txx, map_x, map_y);
                            while ((((int) scoremap.at<unsigned char>((int) map_y, (int) map_x) > 240)) &&
                                   (path_size > 0)) {
                                path_size = path_size - 1;
                                double txx = std::get<0>(ti.predict_path[path_size]);
                                double tyy = std::get<1>(ti.predict_path[path_size]);
                                a = projectToMap(txx, tyy, map_x, map_y);
                            }
                            if (path_size == 0) {
                                ROS_INFO("target predict path is in wall, tx=0, ty=0");
                                ti.tx = 0;
                                ti.ty = 0;
                                path_tracking_flag = false;
                            } else {
                                //path_tracking过程中激光不找人
                                if ((distToRobotPose(txx, tyy) > 0.5)) {
                                    ti.tx = txx;
                                    ti.ty = tyy;
                                    rs.second_state_ = ACTIVE_R;
                                    path_tracking_flag = true;
                                    ROS_WARN("Track the path end!");
                                } else {
                                    //这里path_tracking_flag还是true
                                    ROS_WARN("Still can not find target Id and the substitute, Lost.");
                                }
                            }
                        }
                    }
                }
                if (rs.second_state_ == STAY_SEARCH) //
                {
                    //std::cout << "STAY_SEARCH" << std::endl;
                    rs.second_state_ = INACTIVE;
                    for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
                        if (it->getId() == ti.target.getId()) {
                            ti.targetID = it->getId();
                            ti.target = *it;
                            ti.tx = it->getAmclX();
                            ti.ty = it->getAmclY();
                            rs.second_state_ = ACTIVE_R;
                            ROS_WARN("Camera drop, Stay searched targetID: %d", ti.targetID);
                            break;
                        }
                    }
                    //to_do推送丢失状态
                    if (rs.second_state_ != ACTIVE_R)
                        ROS_WARN("STAY Can not find target and substitute in Search, Lost.");
                }
            }
            break;
        }
    }
}

void Ltracker::laserTrack_second_state_decision(std::list <Human> &tracked_huamns) //和激光一样
{
    if (rs.second_state_ == ACTIVE_R) {
        if (this->stayMode_track(tracked_huamns)) {
            ROS_INFO("stayMode_track == true");
            //std::cout << "stayMode_track == true" << std::endl;
            rs.robot_state_ = STAY;
            track_cnt = 0;
        } else {
            if (stay_flag == false) {
                rs.robot_state_ = TRACK;
            } else {
                //ROS_INFO("stay_flag == true");
                track_cnt = track_cnt + 1;
                if (track_cnt > 5)
                    rs.robot_state_ = TRACK;
                else
                    rs.robot_state_ = STAY;
            }
        }
    }
    if (rs.second_state_ == INACTIVE) {
        if (second_find_flag)
            rs.robot_state_ = STAY;
        else
            rs.robot_state_ = LOST;
    }
}

void Ltracker::laserTrack_moving_state(int frz) {
    if (rs.robot_state_ == TRACK) {
        //判断目标点远近
        first_found_cnt = 0;
        double dis_th = 1.5;
        double predict_x = ti.tx;
        double predict_y = ti.ty;
        if (distToRobotPose(predict_x, predict_y) > dis_th)
            neighbour_flag = false;
        else
            neighbour_flag = true;
        stay_flag = false;
        rs.first_state_ = ACTIVE_R;
        rs.second_state_ = INACTIVE;
        tp.pub_time_now = ros::Time::now();
        if (((tp.pub_time_now.toSec() - tp.pub_time_last.toSec()) > motion_.frz) && (motion_.lock_pub == false) &&
            (motion_.init_lock == true) && (ti.tx != 0 && ti.ty != 0)) {
            motion_.tracker_tracking(tp, ti, rs, no_task, false, path_tracking_flag); //todo
            motion_.init_flag = false;
            ROS_ERROR("targetID: %d, Start tracking without human detection!", ti.targetID);
        }
    }
    if (rs.robot_state_ == LOST) {
        rs.first_state_ = INACTIVE;
        rs.second_state_ = INACTIVE;
    }
    if (rs.robot_state_ == STAY) {
        //rs.robot_show_state_ = TRACK_STAY;
        first_found_cnt = 0;
        stay_flag = true;
        motion_.back_cnt = 0;
        ROS_ERROR("Enter stay mode!");
        rs.first_state_ = ACTIVE_R;
        if (!second_find_flag) {
            ROS_INFO("Real stay mode: STAY_SEARCH");
            rs.second_state_ = STAY_SEARCH;
        } else {
            ROS_INFO("Stay for second finding.");
            rs.second_state_ = ACTIVE_R;
        }
    }
}


Ltracker::~Ltracker() {
    //close ncs, clear graphBuff
    ROS_INFO("Delete movidius facenet graph");
//  retCode = mvncDeallocateGraph(graphHandle);
//  graphHandle = NULL;
//
//  free(graphFileBuf);
//  retCode = mvncCloseDevice(deviceHandle);
//  deviceHandle = NULL;
}

void whiteImg_Mat(cv::Mat &Img) {
    double mean, stddev;
    cv::Mat temp_m, temp_sd;
    cv::meanStdDev(Img, temp_m, temp_sd);
    mean = temp_m.at<double>(0, 0) / 255.0;
    stddev = temp_sd.at<double>(0, 0) / 255.0;
    cv::Mat temp_image(Img.rows, Img.cols, CV_64F);
    //cv::imwrite( "/home/ziwei/human_track_dl/Image.jpg", Img);
    for (int i = 0; i < Img.rows; i++)
        for (int j = 0; j < Img.cols; j++) {
            double pixelVal = Img.at<uchar>(i, j) / 255.0;
            double temp = (pixelVal - mean) / stddev;
            temp_image.at<double>(i, j) = temp;
        }

    double max, min;
    cv::minMaxLoc(temp_image, &min, &max);
    for (int i = 0; i < Img.rows; i++)
        for (int j = 0; j < Img.cols; j++) {
            double pixelVal = temp_image.at<double>(i, j);
            Img.at<uchar>(i, j) = (uchar) round(255.0 * (pixelVal - min) / (max - min));
        }
    //cv::imwrite( "/home/ziwei/human_track_dl/grey_Image.jpg", Img);
}

float Ltracker::HueHistFeature(cv::Rect verifyRect) {
    cv::Mat frame = cameraFrame().clone();
    adjustOutBox(verifyRect, frameWidth(), frameHeight());
    cv::Mat verifyMat = frame(verifyRect);
    cv::resize(verifyMat, verifyMat, cv::Size(64 + 24, 128 + 24)); //keep in same size
    //上一有效图像的ref部分
//  std::cout << "before whiteImg" << std::endl;
    whiteImg_Mat(color_ref_Rect);
//  std::cout << "after whiteImg" << std::endl;
    IplImage copy = color_ref_Rect;
    IplImage *ref = &copy;
    // std::cout << " scn: " << color_ref_Rect.channels() << " depth: " << color_ref_Rect.depth() << std::endl;
    IplImage *ref_hsv = cvCreateImage(cvGetSize(ref), IPL_DEPTH_8U, 3);
    IplImage *ref_hue = cvCreateImage(cvGetSize(ref), IPL_DEPTH_8U, 1);
    // std::cout << "****" << std::endl;
    cvCvtColor(ref, ref_hsv, CV_BGR2HSV);       //转化到HSV空间
    // std::cout << "####" << std::endl;
    //std::cout << " ref_hsv: " << cvGetSize(ref)
    cvSplit(ref_hsv, ref_hue, NULL, NULL, NULL);    //获得H分量
    //计算H分量的直方图，即1D直方图
    //IplImage* h_plane=cvCreateImage( cvGetSize(ref_hsv),IPL_DEPTH_8U,1 );
    int hist_size = 50;          //将H分量的值量化到50个bins
    float ranges1[] = {0, 360}; //H分量的取值范围是[0,360)
    float *ranges = ranges1;
    float max_val = 0.f;
    CvHistogram *ref_hist = cvCreateHist(1, &hist_size, CV_HIST_ARRAY, &ranges, 1);
    cvCalcHist(&ref_hue, ref_hist, 0, NULL);
    cvGetMinMaxHistValue(ref_hist, 0, &max_val, 0, 0);
    //H分量的取值范围是[0,360),这个取值范围的值不能用一个byte来表示，为了能用一个byte表示，需要将H值做适当的量化处理,在这里我们将H分量的范围量化到[0,255
    cvConvertScale(ref_hist->bins, ref_hist->bins, max_val ? 255. / max_val : 0., 0);

    //这一帧recovery需要verify的部分

//  std::cout << "before whiteImg" << std::endl;
    whiteImg_Mat(verifyMat);
//  std::cout << "after whiteImg" << std::endl;
    IplImage copy0 = verifyMat;
    IplImage *verf = &copy0;


    IplImage *verf_hsv = cvCreateImage(cvGetSize(verf), IPL_DEPTH_8U, 3);
    IplImage *verf_hue = cvCreateImage(cvGetSize(verf), IPL_DEPTH_8U, 1);
    // std::cout << "****a" << std::endl;
    cvCvtColor(verf, verf_hsv, CV_BGR2HSV);       //转化到HSV空间
    // std::cout << "####a" << std::endl;
    cvSplit(verf_hsv, verf_hue, NULL, NULL, NULL);    //获得H分量
    //计算H分量的直方图，即1D直方图
    //IplImage* h_plane=cvCreateImage( cvGetSize(verf_hsv),IPL_DEPTH_8U,1);
    CvHistogram *verf_hist = cvCreateHist(1, &hist_size, CV_HIST_ARRAY, &ranges, 1);
    cvCalcHist(&verf_hue, verf_hist, 0, NULL);
    cvGetMinMaxHistValue(verf_hist, 0, &max_val, 0, 0);
    //H分量的取值范围是[0,360),这个取值范围的值不能用一个byte来表示，为了能用一个byte表示，需要将H值做适当的量化处理,在这里我们将H分量的范围量化到[0,255
    cvConvertScale(verf_hist->bins, verf_hist->bins, max_val ? 255. / max_val : 0., 0);

    //不需要计算Back Projection，因为框的位置可能不够准确，只需要计算直方图相似度
    cvNormalizeHist(ref_hist, 1);
    cvNormalizeHist(verf_hist, 1);

    float score = cvCompareHist(verf_hist, ref_hist, CV_COMP_INTERSECT);
    //cvCompareHist()比较的四个方法：相关系数，卡方，交集，常态分布BHATTACHARYYA距离
    //printf("CV_COMP_CORREL : %.4f\n",cvCompareHist(verf_hist,ref_hist,CV_COMP_CORREL)); //1
    //卡方效果最好
    // printf("CV_COMP_CHISQR : %.4f\n",cvCompareHist(verf_hist,ref_hist,CV_COMP_CHISQR)); //0
    //  printf("CV_COMP_INTERSECT : %.4f\n",cvCompareHist(verf_hist,ref_hist,CV_COMP_INTERSECT)); //1
//  printf("CV_COMP_BHATTACHARYYA : %.4f\n",cvCompareHist(verf_hist,ref_hist,CV_COMP_BHATTACHARYYA)); //0

    printf("CV_COMP_INTERSECT : %.4f\n", score); //1
    return score;
}

void Ltracker::laserTrack(std::list <Human> tracked_huamns) {
    //ref_is_human = true; //每一帧默认是以人为参照，只有摄像头没有人，会将ref_is_human设为false
    if (tracked_huamns.size() > 0) {
        no_target_find_cnt = 0;
        laserTrack_first_state_decision(tracked_huamns);
        laserTrack_second_state_decision(tracked_huamns);
        laserTrack_moving_state(motion_.frz);
    } else {
        if (!missed) //camera
        {
            cv::Rect targetRect0 = targetRect;
            //没有对比和墙相交的点，在红色点之前就发和墙相交的点的位置，否则发红点位置
            std::pair<double, double> repoint = ReProject(targetRect0);
            std::pair<double, double> robotpoint = std::make_pair(robotPoseX(), robotPoseY());
            //没有人可以依照，ref_is_human = false
            double x = repoint.first;
            double y = repoint.second;
            bool out_wall = false;
            if (projectToMap(x, y, x, y)) {
                if ((int) scoremap.at<unsigned char>((int) y, (int) x) < 240) //红点在墙前，发红点
                {
                    ROS_WARN("No track_humans but has Rect, Send red point.");
                    out_wall = true;
                    ti.targetID = 888;
                    ti.tx = repoint.first;
                    ti.ty = repoint.second;
                    rs.second_state_ = ACTIVE_R;
                    buffer_cnt0 = 0;
                }
            }
            if (!out_wall) //红点在墙内 或者 在地图外
            {
                double angle_oo = cv::fastAtan2((repoint.second - robotPoseY()), (repoint.first - robotPoseX())) *
                                  0.017453292519943; //0-360
                double d = distToRobotPose(repoint.first, repoint.second);
                int times = (int) d * 5 + 1;
                for (int i = 1; i < times; i++) {
                    double times_x = cos(angle_oo) * (d - i * 0.2);
                    double times_y = sin(angle_oo) * (d - i * 0.2);
                    double map_times_x, map_times_y;
                    if (projectToMap(times_x, times_y, map_times_x, map_times_y)) {
                        if ((int) scoremap.at<unsigned char>((int) map_times_y, (int) map_times_x) < 240) {
                            ROS_WARN("No track_humans but has Rect, Send wall point.");
                            ti.targetID = 888;
                            ti.tx = times_x;
                            ti.ty = times_y;
                            rs.second_state_ = ACTIVE_R;
                            buffer_cnt0 = 0;
                            break;
                        }
                    }
                }
            }
        } else //摄像头和激光都没有
        {
            ROS_INFO("No track_humans and No Rect");
            if (ti.predict_path.size() == 0) {
                ROS_INFO("target predict path=0, tx=0, ty=0");
                ti.tx = 0;
                ti.ty = 0;
                path_tracking_flag = false;
                rs.robot_show_state_ = TRACKER_FIND;
            } else {
                rs.first_state_ = INACTIVE;
                rs.second_state_ = INACTIVE;
                int path_size = ti.predict_path.size() - 1;
                //std::cout << "path_size: " << path_size << std::endl;
                double txx = std::get<0>(ti.predict_path[path_size]);
                double tyy = std::get<1>(ti.predict_path[path_size]);
                if (distToRobotPose(txx, tyy) > 0.5) {
                    double map_x, map_y;
                    bool a = projectToMap(txx, txx, map_x, map_y);
                    while ((((int) scoremap.at<unsigned char>((int) map_y, (int) map_x) > 240)) && (path_size > 0)) {
                        //std::cout << "size: " << path_size << std::endl;
                        path_size = path_size - 1;
                        double txx = std::get<0>(ti.predict_path[path_size]);
                        double tyy = std::get<1>(ti.predict_path[path_size]);
                        //std::cout << "txx: " << txx << " tyy: " << tyy << std::endl;
                        a = projectToMap(txx, txx, map_x, map_y);
                        // std::cout << "score: " << (int) scoremap.at<unsigned char>((int) map_y, (int) map_x) << std::endl;
                    }
                    if (path_size == 0) {
                        ROS_INFO("target predict path is in wall, tx=0, ty=0");
                        ti.tx = 0;
                        ti.ty = 0;
                        path_tracking_flag = false;
                        rs.robot_show_state_ = TRACKER_FIND;
                    } else {
                        ti.tx = txx;
                        ti.ty = tyy;
                        //判断目标点远近
                        first_found_cnt = 0;
                        double dis_th = 1.5;
                        double predict_x = ti.tx;
                        double predict_y = ti.ty;
                        // if (distToRobotPose(predict_x, predict_y) > dis_th)
                        //   neighbour_flag = false;
                        // else
                        //   neighbour_flag = true;
                        stay_flag = false;
                        path_tracking_flag = true;
                        ROS_WARN("No target but Track the path end!");
                        tp.pub_time_now = ros::Time::now();
                        if (((tp.pub_time_now.toSec() - tp.pub_time_last.toSec()) > motion_.frz) &&
                            (motion_.lock_pub == false) &&
                            (ti.tx != 0 && ti.ty != 0)) {
                            motion_.tracker_tracking(tp, ti, rs, no_task, false, path_tracking_flag);
                            motion_.init_flag = false;
                            ROS_ERROR("Start tracking without human detection!");
                        }
                    }
                } else {
                    no_target_find_cnt = no_target_find_cnt + 1;
                    ROS_INFO("NO laser and camera rect! %d", no_target_find_cnt); //一直没有human
                    if (no_target_find_cnt > 15) {
                        ROS_INFO("no_target_find_cnt>15, tx=0,ty=0");
                        ti.tx = 0;
                        ti.ty = 0;
                        rs.robot_show_state_ = TRACKER_FIND;
                    }
                    path_tracking_flag = false;
                    rs.first_state_ = INACTIVE;
                    rs.second_state_ = INACTIVE;
                    first_found_cnt = 0;
                    second_find_cnt = 0;
                    second_find_flag = false;
                }
            }
        }
    }
}

////Haar face
//void Ltracker::FaceDetection() {
//    cv::Mat frame = cameraFrame().clone();
//    if (frame.cols == 0)
//        return;
//    std::vector <cv::Rect> faces_0;
//    cv::Mat frame_gray;
//    cv::Mat crop;
//    cv::Mat res;
//    cv::Mat gray;
//    cv::cvtColor(frame, frame_gray, CV_BGR2GRAY);
//
//    whiteImg_Mat(frame);
//    cv::equalizeHist(frame_gray, frame_gray);
//
//    // Detect faces
//    face_cascade.detectMultiScale(frame_gray, faces_0, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
//    Faces = faces_0;
//}

//YOLO face
void Ltracker::FaceDetection() {
  mutexCameraRects.lock();
  auto tmpRects = Rects_face();
  mutexCameraRects.unlock();

  for(int i=0; i<tmpRects.size(); i++)
  {
   Faces.push_back(tmpRects[i].rect);
  }
  //std::cout << "Faces size: " << Faces.size() << std::endl;
}

bool Ltracker::FaceFindHuman(std::list <Human> tracked_huamns, std::vector <cameraRect> tmpRects, cv::Rect face, cv::Rect& HumanRect)
{
    //face的中心点
    float xo = face.x + 0.5 * face.width;
    float yo = face.y + 0.5 * face.height;
    int cor_index;
    double min_dist = 999;
    //找camera yolo的框
    for (int j = 0; j < tmpRects.size(); j++) {
        if (getOverlap(face, tmpRects[j].rect) == 0) //无交集的框跳过
            continue;
        float x_r = tmpRects[j].rect.x + tmpRects[j].rect.width + face.width;
        float x_l = tmpRects[j].rect.x - face.width;
        float y_d = tmpRects[j].rect.y + 0.33 * tmpRects[j].rect.height;
        float rio = (float) (face.width * face.height) /
                    (tmpRects[j].rect.width * tmpRects[j].rect.height);
        //std::cout << "xo: " << xo << " x_l:" << x_l << " x_r:" << x_r << " yo:" << yo << " y_d:" << y_d << " rio:" << rio << std::endl;
        //脸在人的上方，脸和身子的比例 大于八分之一
        if (xo < x_l || xo > x_r || yo > y_d || rio > 0.125)
            continue;
        //人框的中心点
        float x_o = tmpRects[j].rect.x + 0.5 * tmpRects[j].rect.width;
        float y_o = tmpRects[j].rect.y ; //+ 0.5 * tmpRects[j].rect.height;
        double temp_dist = disTwoPoints(x_o, y_o, xo, yo); //和人的框最上面一个中间点的距离
        if (temp_dist < min_dist) {
            min_dist = temp_dist;
            cor_index = j;
            HumanRect = tmpRects[j].rect;
        }
    }

    //找激光框,视觉框找到不找激光框.yolo已经找到就直接跳过
    if (min_dist != 999)
    {
        ROS_WARN("Find face in YOLO rects");
        return true;
    }
    else
    {
        for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++)
        {
            if(!it->isProject)
                continue;
            cv::Rect laserBox = it->getBoundingBox();
            if (getOverlap(face, laserBox) == 0) //无交集的框跳过
                continue;
            float x_r = laserBox.x + laserBox.width + face.width;
            float x_l = laserBox.x - face.width;
            float y_d = laserBox.y + 0.33 * laserBox.height;
            float rio = (float) (face.width * face.height) /
                        (laserBox.width * laserBox.height);
            //std::cout << "xo: " << xo << " x_l:" << x_l << " x_r:" << x_r << " yo:" << yo << " y_d:" << y_d << " rio:" << rio << std::endl;
            //脸在人的上方，脸和身子的比例 大于八分之一
            if (xo < x_l || xo > x_r || yo > y_d || rio > 0.125)
                continue;
            //人框的中心点
            float x_o = laserBox.x + 0.5 * laserBox.width;
            float y_o = laserBox.y ; //+ 0.5 * tmpRects[j].rect.height;
            double temp_dist = disTwoPoints(x_o, y_o, xo, yo); //和人的框最上面一个中间点的距离
            if (temp_dist < min_dist) {
                min_dist = temp_dist;
                HumanRect = laserBox;
            }
        }
    }

    if (min_dist != 999)
    {
        ROS_WARN("Find face in YOLO rects");
        return true;
    }
    else{
        ROS_WARN("Does not find matched rect");
        return false;
    }

}

//true: facefilter起作用了，需要改变faces数量。false: 没起作用，清空facesc
bool Ltracker::FaceFilter(std::vector <cv::Rect> &FacesRaw) {
    mutexCameraRects.lock();
    auto tmpRects = Rects_person();
    mutexCameraRects.unlock();

    if (!tmpRects.size()) {
        return false;
    }

    //画出人
    for (int i = 0; i < tmpRects.size(); i++) {
        adjustOutBox(tmpRects[i].rect, frameWidth(), frameHeight());
        cv::rectangle(show, tmpRects[i].rect, cv::Scalar(0, 255, 255), 2);
    }


    //std::cout << "FacesRaw size: " << FacesRaw.size() << std::endl;
    for (int i = 0; i < FacesRaw.size(); i++) {
        //画出全部脸
//      adjustOutBox(FacesRaw[i], frameWidth(),frameHeight());
//      cv::rectangle(show, FacesRaw[i], cv::Scalar(255,255,255), 2);

        float xo = FacesRaw[i].x + 0.5 * FacesRaw[i].width;
        float yo = FacesRaw[i].y + 0.5 * FacesRaw[i].height;
        int cor_index;
        double min_dist = 999;
        for (int j = 0; j < tmpRects.size(); j++) {
            if (getOverlap(FacesRaw[i], tmpRects[j].rect) == 0)
                continue;
            float x_r = tmpRects[j].rect.x + tmpRects[j].rect.width + FacesRaw[i].width;
            float x_l = tmpRects[j].rect.x - FacesRaw[i].width;
            float y_d = tmpRects[j].rect.y + 0.33 * tmpRects[j].rect.height;
            float rio = (float) (FacesRaw[i].width * FacesRaw[i].height) /
                        (tmpRects[j].rect.width * tmpRects[j].rect.height);
            //std::cout << "xo: " << xo << " x_l:" << x_l << " x_r:" << x_r << " yo:" << yo << " y_d:" << y_d << " rio:" << rio << std::endl;
            //脸在人的上方，脸和身子的比例
            if (xo < x_l || xo > x_r || yo > y_d || rio > 0.125)
                continue;

            float x_o = tmpRects[j].rect.x + 0.5 * tmpRects[j].rect.width;
            float y_o = tmpRects[j].rect.y; //+ 0.5 * tmpRects[j].rect.height;
            double temp_dist = disTwoPoints(x_o, y_o, xo, yo); //和人的框最上面一个中间点的距离
            if (temp_dist < min_dist) {
                min_dist = temp_dist;
                cor_index = j;
            }
        }
        if (min_dist != 999) {
            float x_r = tmpRects[cor_index].rect.x + tmpRects[cor_index].rect.width;
            float x_l = tmpRects[cor_index].rect.x;
            float y_d = tmpRects[cor_index].rect.y + 0.3 * tmpRects[cor_index].rect.height;
            if (xo > x_l && xo < x_r && yo < y_d) {
                tmpRects.erase(tmpRects.begin() + cor_index);
            }
        } else {
            //std::cout << "delete a face" << std::endl;
            FacesRaw.erase(FacesRaw.begin() + i);
            i--;
        }
    }

    if (FacesRaw.size() > 0)
        return true;
    else
        return false;
}

//对应每张脸找所对应的人
//默认返回分数越小，prob越大
float face2Person(cv::Rect Person, cv::Rect face) {
    float score = 10000;
    float xo = face.x + 0.5 * face.width;
    float yo = face.y + 0.5 * face.height;
    float dis_face = disTwoPoints(xo, yo, Person.x + Person.width, Person.y);

    score = dis_face;
    return score;
}

void Ltracker::visualStart(cv::Rect &target_rect, std::list <Human> tracked_huamns) {
    mutexCameraRects.lock();
    auto tmpRects = Rects_person();
    mutexCameraRects.unlock();

    if (!tmpRects.size()) {
        detectFlag() = false;
        return;
    }

    if (!detectFlag())
        return;

    ROS_INFO("Receive detect signal!");
    detectFlag() = false;

    auto candidates = tracked_huamns;
    candidates.clear();
    for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
        double x = it->getAmclX();
        double y = it->getAmclY();
        double d_angle = caculate_human_angle(x, y);
        // std::cout << "+++ x:" << it->getAmclX() << " y: " << it->getAmclY() << std::endl;
        //std::cout << "d_angle: " << d_angle << std::endl;
        //std::cout << "distToRobotPose(x, y): " << distToRobotPose(x, y) << std::endl;
        if (distToRobotPose(x, y) < 6 && fabs(d_angle) < 45)
            candidates.push_back(*it);
    }

    if (candidates.size() == 0) {
        ROS_ERROR("There is no laser target detect");
        return;
    }

    if (Faces.size() == 0) {
        ROS_ERROR("There is no face detect");
        return;
    }

    int max_area = 0;//0.25*frameHeight()*frameWidth();
    int idx = 1000;
    for (int i = 0; i < tmpRects.size(); i++) {
        int area = tmpRects[i].rect.width * tmpRects[i].rect.height;
        if (area > max_area) {
            max_area = area;
            idx = i;
        }
    }
//  std::cout << "0000" <<std::endl;
//  std::cout << "idx: " << idx << std::endl;
//  std::cout << "rect.x: " << tmpRects[idx].rect.x << std::endl;
//  std::cout << "rect.width: " << tmpRects[idx].rect.width << std::endl;
    float cx = tmpRects[idx].rect.x + 0.5 * tmpRects[idx].rect.width;
    // std::cout << "cx: " << cx << std::endl;
    float cy = tmpRects[idx].rect.y + 0.5 * tmpRects[idx].rect.height;
    // std::cout << "cy: " << cy << std::endl;
    float dx = cx - 0.5 * frameWidth();
    // std::cout << "dx: " << dx << std::endl;
    float dy = cy - 0.5 * frameHeight();
    //std::cout << "dy: " << dy << std::endl;
    float d = sqrt(dx * dx + dy * dy);
    //std::cout << "d: " << d << std::endl;
    //std::cout << "1111" <<std::endl;
    if (d > 0.3 * frameWidth()) {
        ROS_ERROR("The person is far away to the frame center");
        return;
    }

    cv::Rect M = tmpRects[idx].rect;
    adjustOutBox(M, frameWidth(), frameHeight());
    double min_d = max_reproject_dist_small;
    auto idx_human = candidates.end();
    for (auto it = candidates.begin(); it != candidates.end(); it++) {
        std::cout << "dist: " << dist2Reproject(M, *it) << std::endl;
        if (dist2Reproject(M, *it) < min_d) {
            min_d = dist2Reproject(M, *it);
            idx_human = it;
        }
    }
    if (idx_human == candidates.end()) {
        std::cout << "No satisfied laser candidate" << std::endl;
    } else {
        ti.target = *idx_human;
        ti.targetID = idx_human->getId();
        ti.tx = idx_human->getAmclX();
        ti.ty = idx_human->getAmclY();
        // correct_reproject = Reproject(tmpRects[idx].rect, *idx_human);
        // ti.predict_path = target0.
        Init_pos = idx_human->init_position;
        rs.second_state_ = ACTIVE_R;
        //pr_cnt = 0;
        motion_.back_cnt = 0;
        first_found_cnt = 0;
        second_find_cnt = 0;
        //_target_find_flag = 0;
        second_find_flag = false;
        tracker_loss_flag = false;
        tracker_loss_cnt = 0;
        //（这里需要推送状态，说搜寻到新的目标任务） todo_ziwei
        ROS_ERROR("New find target");
        ROS_WARN("Target ID: %d", ti.targetID);
    }

    //To the targetRect, calculate the score for each face,
    float min_dist = 999;
    int faceIdx;
    for (int i = 0; i < Faces.size(); i++) {
        if ((face2Person(M, Faces[i]) < min_dist) && getOverlap(M, Faces[i]) > 0) {
            min_dist = face2Person(M, Faces[i]);
            faceIdx = i;
        }
    }
    if(min_dist == 999)
    {
        ROS_ERROR("There is no matching face detect");
        return;
    }
    //choose reference face
    cv::Mat frame = cameraFrame().clone();
    if (min_dist != 999) {
        refFace = frame(Faces[faceIdx]);
        //用movidius计算出target的结果，保存在T_resultData32中
        T_resultData32 = compute(refFace, graphHandle_face);
        ROS_WARN("**Have gotten ref face** ");
    }

    //debug : save ref face
    IplImage copy = refFace;
    IplImage *pic_Ipl = &copy;
    image buff_;
    buff_ = ipl_to_image(pic_Ipl);
    rgbgr_image(buff_);
    //save_image_png(buff_, "/home/ziwei/human_track_dl/refface.png");

    //after find
    ROS_WARN("**Have gotten detection target** "); //todo_ziwei

    //kcf
    cv::Mat roi = frame(M);
    std::cout << " roi.x: " << M.x << " roi.y: " << M.y << " roi.width: " << M.width << " roi.height: " << M.height
              << std::endl;
    cv::resize(roi, roi, cv::Size(64, 128));
    target_rect = M;
    targetRect = target_rect;
    catched = true;
    missed = false;
    target_width_proto = M.width;
    target_height_proto = M.height;
    kcf.init(M, frame);//kcf初始化，提取目标特征，用来判断目标是否跟丢。

    //ferns
    fern.initParams();//初始化随机蕨，是个分类器，当跟丢时，通过该分类器判断新目标是否是原目标。
    std::vector <cv::Size> tmp_size = {cv::Size(64, 128)};
    fern.init(tmp_size);

//  fern_ref.initParams();//初始化随机蕨，是个分类器，当跟丢时，通过该分类器判断新目标是否是原目标。
//  fern_ref.init(tmp_size);

    std::vector < std::pair < std::vector < int > , int >> fern_data;
    cv::Mat patch = frame(M);//当跟踪目标的矩形框超过图像范围，有可能报错
    //std::cout << targetRect << "\t 2" << std::endl;

    cv::resize(patch, patch, cv::Size(64 + 24, 128 + 24)); //resize矩形框，变大后在里面取样本
    color_ref_Rect = patch;
    cv::cvtColor(patch, patch, CV_RGB2GRAY);
    std::vector<int> fern_result;
    fern_result.resize(fern.getNstructs()); //相当于取100个随机种子
    //采取正样本
    for (int x = 0; x <= 24; x += 4)
        for (int y = 0; y <= 24; y += 4) {
            cv::Mat tpatch = patch(cv::Rect(x, y, 64, 128));
            fern.calcFeatures(tpatch, 0, fern_result);
            fern_data.push_back(std::make_pair(fern_result, true));
        }
    //采取负样本
    for (int i = 0; i < 49; i++) {
        cv::Rect neg_box;
        getNegtiveBox(targetRect, neg_box); //随即取负样本
        cv::Mat tpatch = frame(neg_box);
        cv::resize(tpatch, tpatch, cv::Size(64, 128));
        cv::cvtColor(tpatch, tpatch, CV_RGB2GRAY);
        fern.calcFeatures(tpatch, 0, fern_result);
        fern_data.push_back(std::make_pair(fern_result, false));
    }
    //正负样本进行训练，目的为了找回跟丢目标
    fern.trainF(fern_data, 1);
    //fern_ref = fern;
    std::cout << "End visual start" << std::endl;
}

//控制给compute计算的脸的数量小于2保证实时性
void Ltracker::face_detect_num_control() {
    if (Faces.size() <= 2)
        return;
    else {
        //如果id没变找id对应的框和离画面中心最近的两个脸
        float cenx = frameWidth() * 0.5;
        float ceny = frameHeight() * 0.5;
        std::vector<float> vec;
        for (int i = 0; i < Faces.size(); i++) {
            float dis = disTwoPoints(Faces[i].x, Faces[i].y, cenx, ceny);
            vec.push_back(dis);
        }
        std::vector <cv::Rect> face_0;
        for (int i = 0; i < 2; i++) {
            std::vector<float>::iterator min = std::min_element(std::begin(vec), std::end(vec));
            int it = std::distance(std::begin(vec), min);
            for (int j = 0; j < Faces.size(); j++) {
                if (disTwoPoints(Faces[j].x, Faces[i].y, cenx, ceny) == *min)
                    face_0.push_back(Faces[j]);
            }
            vec.erase(min);
        }
        Faces = face_0;
    }
}

bool Ltracker::recovery_combination(std::list <Human> tracked_huamns, std::vector <cameraRect> tmpRects,
                                    cv::Rect &recovery_rect) {
    //std::cout << "Find in camera rects" << std::endl;
    cv::Mat frame = cameraFrame().clone();
    //先找摄像头的框
    float max_score = fern.getThreshFern() + 0.1;
    int idx = 1000;
    bool find_recovery_rect = false;
    for (int i = 0; i < tmpRects.size(); i++) {
        if (getOverlap(tmpRects[i].rect, targetRect) < 0.02 && area(tmpRects[i].rect, targetRect) < 0.3)
            continue;
        cv::Mat patch = frame(tmpRects[i].rect);
        cv::cvtColor(patch, patch, CV_RGB2GRAY);
        cv::resize(patch, patch, cv::Size(64, 128));
        std::vector<int> fern_result;
        fern_result.resize(fern.getNstructs());
        fern.calcFeatures(patch, 0, fern_result);
        float score = fern.measure_forest(fern_result) / fern.getNstructs();
        if (score > fern.getNstructs() * fern.getThreshFern())
            cv::putText(show, std::to_string(score),
                        cv::Point(tmpRects[i].rect.x, tmpRects[i].rect.y + tmpRects[i].rect.height),
                        cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255), 1, CV_AA);
        score = score + 0.3 * Kdist(tmpRects[i].rect, targetRect);
        //experiments:
//      std::cout << "left " << tmpRects[i].rect.x << std::endl;
//      HueHistFeature(tmpRects[i].rect);
        //std::cout << "camera score: " << score << " Kscore: " << 0.3*Kdist(tmpRects[i].rect, targetRect) << " original score: " << score - 0.3*Kdist(tmpRects[i].rect, targetRect) << " x: " << tmpRects[i].rect.x << std::endl;
        if (score > max_score) {
            max_score = score;
            idx = i;
        }
    }

    //找激光的框
    auto target = tracked_huamns.end();
    for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
        if (it->isProject) {
            //std::cout << it->getBoundingBox() << "/t 00" << std::endl;
            if (getOverlap(it->getBoundingBox(), targetRect) < 0.02 && area(it->getBoundingBox(), targetRect) < 0.3) {
                //std::cout << "***getover:" << getOverlap(it->getBoundingBox(),targetRect) << " area:" << area(it->getBoundingBox(),targetRect) << std::endl;
                continue;
            }
            cv::Mat patch = frame(it->getBoundingBox());
            cv::cvtColor(patch, patch, CV_RGB2GRAY);
            cv::resize(patch, patch, cv::Size(64, 128));
            std::vector<int> fern_result;
            fern_result.resize(fern.getNstructs());
            fern.calcFeatures(patch, 0, fern_result);
            float score = fern.measure_forest(fern_result) / fern.getNstructs();
            if (score > fern.getNstructs() * fern.getThreshFern())
                cv::putText(show, std::to_string(score),
                            cv::Point(it->getBoundingBox().x, it->getBoundingBox().y + it->getBoundingBox().height),
                            cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255), 1, CV_AA);
            score = score + 0.3 * Kdist(it->getBoundingBox(), targetRect);
            if (score > max_score) {
                max_score = score;
                target = it;
            }
        }
    }

    //看分数最大的框是视觉框还是激光框
    bool camera_find_flag = false;
    //recovery_rect_0是Ferns找出来的最优的框
    cv::Rect recovery_rect_0;
    if (target != tracked_huamns.end()) {
        camera_find_flag = true;
        recovery_rect_0 = target->getBoundingBox();
    } else {
        if (idx != 1000) {
            camera_find_flag = true;
            recovery_rect_0 = tmpRects[idx].rect;
        }
    }

    if (camera_find_flag) {
        //先找脸,找到直接return出去
        bool find_correct_face = false;
        if (Faces.size() > 0) { //看脸能否找到，找到就更新，否则报丢
            ROS_INFO("Has detect faces");
            float min_face_score = 1;
            cv::Rect recovery_face_rect;
            for (int i = 0; i < Faces.size(); i++) {
                V_resultData32 = compute(frame(Faces[i]), graphHandle_face);
                float total_diff = 0;
                float this_diff = 0;
                for (int j = 0; j < 128; j++) {
                    this_diff = pow((V_resultData32[j] - T_resultData32[j]), 2);
                    total_diff = total_diff + this_diff;
                    ROS_INFO("this_diff: %f, total_diff: %f", this_diff, total_diff);
                }
                ROS_INFO("total_diff: %f, min_face_score: %f", total_diff, min_face_score);
                if (total_diff < min_face_score) {
                    min_face_score = total_diff;
                    recovery_face_rect = Faces[i];
                    find_correct_face = true;
                }
            }
            if (find_correct_face) //找到脸，根据脸找框
            {
                cv::Rect face_person_rect;
                bool find_face_person = FaceFindHuman(tracked_huamns, tmpRects, recovery_face_rect, face_person_rect);
//                //---------------------------用FaceFindHuman替换--------------------------------------//
//                //暂定face的长宽中最远的距离为脸属于face2person的阈值.根据face2Person函数内容修改，todo_ziwei
//                float min_face_score =
//                        recovery_face_rect.width >= recovery_face_rect.height ? recovery_face_rect.width
//                                                                              : recovery_face_rect.height;
//                ROS_INFO("min_face_score: %f", min_face_score);
//                bool find_face_person = false;
//                cv::Rect face_person_rect;
//                for (int i = 0; i < tmpRects.size(); i++) {
//                    float face2person_score = face2Person(tmpRects[i].rect, recovery_face_rect);
//                    ROS_INFO("face2person_score0: %f", face2person_score);
//                    if (face2person_score < min_face_score) {
//                        min_face_score = face2person_score;
//                        find_face_person = true;
//                        face_person_rect = tmpRects[i].rect;
//                    }
//                }
//                for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
//                    if(!it->isProject)
//                        continue;
//                    float face2person_score = face2Person(it->getBoundingBox(), recovery_face_rect);
//                    ROS_INFO("face2person_score1: %f", face2person_score);
//                    if (face2person_score < min_face_score) {
//                        min_face_score = face2person_score;
//                        find_face_person = true;
//                        face_person_rect = it->getBoundingBox();
//                    }
//                }
//                //---------------------------------------------------------------------------//
                if (find_face_person) {
                    recovery_rect = face_person_rect; //按脸找到的结果更新person
                    find_recovery_rect = true;
                    ROS_WARN("Find face, choose face2person rect");
                }
            }
        }

        if (!find_recovery_rect) {
            //进入颜色判断
            //颜色判断也找视觉和激光的两种框里找,找出最对应的框
            float max_color_score = 0.3; //小于0.3,颜色匹配失败
            bool color_find = false;
            cv::Rect recovery_rect_color;
            for (int i = 0; i < tmpRects.size(); i++) {
                if (getOverlap(tmpRects[i].rect, targetRect) < 0.02 && area(tmpRects[i].rect, targetRect) < 0.3)
                    continue;
                float color_score = HueHistFeature(tmpRects[i].rect);
                if (color_score > max_color_score) {
                    max_color_score = color_score;
                    recovery_rect_color = tmpRects[i].rect;
                    color_find = true;
                }
            }
            for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
                if(!it->isProject)
                    continue;
                if (getOverlap(it->getBoundingBox(), targetRect) < 0.02 && area(it->getBoundingBox(), targetRect) < 0.3)
                    continue;
                float color_score = HueHistFeature(it->getBoundingBox());
                if (color_score > max_color_score) {
                    max_color_score = color_score;
                    recovery_rect_color = it->getBoundingBox();
                    color_find = true;
                }
            }
            if (color_find) { //找到颜色符合的框
                if (getOverlap(recovery_rect_color, recovery_rect_0) > 0.7) //color的框和Ferns选出来是同一个框
                {
                    //颜色和Ferns统一，跳出判断
                    recovery_rect = recovery_rect_0; //用Ferns选出来的框
                    find_recovery_rect = true;
                    ROS_INFO("Ferns and color choose the same Rect");
                } else {
                    bool find_correct_face = false;
                    //看是否有脸的存在，如果有，根据脸来判断；如果没有,根据color
                    if (Faces.size() > 0) {
                        float min_face_score = 1;
                        cv::Rect recovery_face_rect;
                        for (int i = 0; i < Faces.size(); i++) {
                            V_resultData32 = compute(frame(Faces[i]), graphHandle_face);
                            float total_diff = 0;
                            float this_diff = 0;
                            for (int j = 0; j < 128; j++) {
                                this_diff = pow((V_resultData32[j] - T_resultData32[j]), 2);
                                total_diff = total_diff + this_diff;
                            }
                            if (total_diff < min_face_score) {
                                min_face_score = total_diff;
                                recovery_face_rect = Faces[i];
                                find_correct_face = true;
                            }
                            ROS_WARN("Face %d score: %f", i, total_diff);
                            cv::putText(show, std::to_string(total_diff),
                                        cv::Point(Faces[i].x, Faces[i].y + Faces[i].height),
                                        cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 0), 1, CV_AA);
                        }

                        if (!find_correct_face) //没有脸
                        {
                            recovery_rect = recovery_rect_color; //按颜色的框找
                            find_recovery_rect = true;
                            ROS_INFO("NO face, find by color");
                        } else //检测到正确的脸
                        {
                            //选出face,看哪个Ferns和color的框哪个更符合
                            float color_face = face2Person(recovery_rect_color, recovery_face_rect);
                            float fern_face = face2Person(recovery_rect_0, recovery_face_rect);
                            recovery_rect = color_face > fern_face ? recovery_rect_0 : recovery_rect_color;
                            find_recovery_rect = true;
                            if (color_face > fern_face)
                                ROS_INFO("Find face, choose Fern rect");
                            else
                                ROS_INFO("Find face, choose color rect");
                        }
                    }
                }
            }
        }
    }

//        //---------只有Ferns----------------------//
//        if (camera_find_flag) {
//            recovery_rect = recovery_rect_0;
//            return true;
//        } else
//            return false;
//        //-------------------------------------//
        if(find_recovery_rect)
            return true;
        else
            return false;
    }


    void Ltracker::visualTrack(std::list <Human> tracked_huamns) {
        cv::Mat frame = cameraFrame().clone();
        int height = frameHeight();
        int width = frameWidth();
        mutexCameraRects.lock();
        auto tmpRects = Rects_person();
        mutexCameraRects.unlock();

//-------------------------------DEBUG part ----------------------------//

//  //show face score debug
//  if(Faces.size()>0)
//  {
//    for(int i = 0; i < Faces.size(); i++) {
//      V_resultData32 = compute(frame(Faces[i]), graphHandle_face);
//      float total_diff = 0;
//      float this_diff = 0;
//      for (int j = 0; j < 128; j++) {
//          this_diff = pow((V_resultData32[j] - T_resultData32[j]), 2);
//          total_diff = total_diff + this_diff;
//      }
//      ROS_WARN("Face %d score: %f", i, total_diff);
//      cv::putText(show, std::to_string(total_diff),
//              cv::Point(Faces[i].x, Faces[i].y + Faces[i].height),
//              cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255,255,0), 1, CV_AA);
////          std::cout << "face " << i << "score: " << total_diff << std::endl;
//    }
//  }

//    //color rect for debug
//    int idx_color;
//    for (int i = 0; i < tmpRects.size(); i++) {
//        float color_score = HueHistFeature(tmpRects[i].rect);
//    }

        //::cout << "visualTrack" << std::endl;
        //cv::Mat projectRect;
//---------------------------------------------------------------------------//

        if (!missed) {

            if (!tmpRects.size() && !tracked_huamns.size()) {
                ROS_INFO("tmpRects.size=0 and tracked_huamns.size=0");
                missed = true;
            } else {
                float max_ol = 0.2;
                int idx = 1000;
                for (int i = 0; i < tmpRects.size(); i++) {
                    float ol = getOverlap(tmpRects[i].rect, targetRect);
                    //std::cout << "targetRect: " << targetRect << " camera rect: " << tmpRects[i].rect << std::endl;
                    //std::cout << "camera ol: " << ol << std::endl;
                    if (ol > max_ol) {
                        max_ol = ol;
                        idx = i;
                    }
                }

                auto target0 = tracked_huamns.end();
                for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
                    //std::cout << "it->isHuman_tmp"  << it->isHuman_tmp << std::endl;
                    if (it->isProject) {
                        float ol = getOverlap(it->getBoundingBox(), targetRect);
                        //std::cout << "laser ol: " << ol << std::endl;
                        if (ol > max_ol) {
                            max_ol = ol;
                            target0 = it;
                        }
                    }
                }
                cv::Rect ref_rect;
                if (target0 != tracked_huamns.end()) {
                    ref_rect = target0->getBoundingBox();
                } else if (idx != 1000)
                    ref_rect = tmpRects[idx].rect;


                if ((idx == 1000) && (target0 == tracked_huamns.end())) {
                    ROS_INFO("NO enough overlap rect in this frame");
                    missed = true; //没有和上一次overlap大到一定程度的框，可能当前目标没检测出来
                } else {
                    kcf._roi = ref_rect;
                    float scale_addtion = sqrtf((float) target_width_proto * target_height_proto /
                                                (ref_rect.width * ref_rect.height));
                    double score;
                    bool success;
                    cv::Rect result_box = kcf.update(frame, scale_addtion, score,
                                                     success);//frame是整个大图像，目标检测区域是kcf._roi，score是得分，success是判断是不是跟丢
                    //std::cout << "result overlap: " << getOverlap(result_box, ref_rect) << std::endl;
//        std::cout << result_box << "\t 01" << std::endl;
//        std::cout << ref_rect << "\t 02" << std::endl;
                    //std::cout << "KCF SCORE: " << score << std::endl;
                    if (!success || getOverlap(result_box, ref_rect) < 0.7) {
                        ROS_INFO("KCF LOST");
                        missed = true;
                    } else {
                        // std::cout << "renew" << std::endl;
                        float max_ol_c = 0.2;
                        int idx_c = 1000;
                        for (int i = 0; i < tmpRects.size(); i++) {
                            float ol_c = getOverlap(tmpRects[i].rect, targetRect);
                            if (ol_c > max_ol_c) {
                                max_ol_c = ol_c;
                                idx_c = i;
                            }
                        }
                        // std::cout << "max_ol_c: " << max_ol_c << std::endl;
                        if (idx_c != 1000)
                            targetRect = tmpRects[idx_c].rect;
                        else
                            targetRect = result_box;

                        // projectRect = targetRect;
                        //targetRect = ref_rect;
                        adjustOutBox(targetRect, width, height);
                        cv::rectangle(show, targetRect, cv::Scalar(156, 102, 31), 2);
                        cv::putText(show, std::to_string(score),
                                    cv::Point(targetRect.x, targetRect.y + targetRect.height),
                                    cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(156, 102, 31), 1, CV_AA);


                        std::vector < std::pair < std::vector < int > , int >> fern_data;
                        //std::cout << targetRect << "\t 1" << std::endl;
                        cv::Mat patch = frame(targetRect);//当跟踪目标的矩形框超过图像范围，有可能报错
                        //std::cout << targetRect << "\t 2" << std::endl;

                        cv::resize(patch, patch, cv::Size(64 + 24, 128 + 24)); //resize矩形框，变大后在里面取样本
                        //更新 color_ref_Rect
                        color_cnt++;
                        if (color_cnt > 2) //KCF 2帧匹配上就更新color_rect
                        {
                            color_ref_Rect = patch;
                        }
                        cv::cvtColor(patch, patch, CV_RGB2GRAY);
                        std::vector<int> fern_result;
                        fern_result.resize(fern.getNstructs()); //相当于取100个随机种子
                        //采取正样本
                        for (int x = 0; x <= 24; x += 4)
                            for (int y = 0; y <= 24; y += 4) {
                                cv::Mat tpatch = patch(cv::Rect(x, y, 64, 128));
                                fern.calcFeatures(tpatch, 0, fern_result);
                                fern_data.push_back(std::make_pair(fern_result, true));
                            }
                        //采取负样本
                        for (int i = 0; i < 49; i++) {
                            cv::Rect neg_box;
                            getNegtiveBox(targetRect, neg_box); //随即取负样本
                            cv::Mat tpatch = frame(neg_box);
                            cv::resize(tpatch, tpatch, cv::Size(64, 128));
                            cv::cvtColor(tpatch, tpatch, CV_RGB2GRAY);
                            fern.calcFeatures(tpatch, 0, fern_result);
                            fern_data.push_back(std::make_pair(fern_result, false));
                        }
                        //正负样本进行训练，目的为了找回跟丢目标
                        fern.trainF(fern_data, 1);
                    }
                }
            }
        }

        //recovery部分
        if (missed)//如果跟丢，用fern分类器去检测视野中所有目标中是否有原目标
        {
            color_cnt = 0;
            cv::Rect recovery_rect;
            //std::cout << "before recovery combination" << std::endl;
            if (recovery_combination(tracked_huamns, tmpRects, recovery_rect)) {
                missed = false;
                kcf._roi = recovery_rect;
                targetRect = recovery_rect;
                adjustOutBox(targetRect, width, height);
                cv::rectangle(show, targetRect, cv::Scalar(255, 0, 255), 2);
            } else
                ROS_INFO("Does not find rect");
        }


        mutexCameraRects.lock();
        Rects_person().clear();
        mutexCameraRects.unlock();
    }

//判断前方有没有激光点,设置了一个速度阈值2米
    bool check_humans_in_view_angle(std::list <Human> tracked_huamns) {
        for (auto it = tracked_huamns.begin(); it != tracked_huamns.end(); it++) {
            double x = it->getAmclX();
            double y = it->getAmclY();
            std::cout << "x: " << x << " y: " << y << std::endl;
            std::cout << "angle: " << fabs(caculate_human_angle(x, y)) << " dist: " <<  distToRobotPose(x, y) << std::endl;
            if ((fabs(caculate_human_angle(x, y)) < 45) && distToRobotPose(x, y) < 2) {
                return true;
            }
        }
        return false;
    }

    void Ltracker::runWithVisual() {
        static int target_width_proto;
        static int target_height_proto;
        static KCFTracker kcf = KCFTracker(true/* hog*/, true/* fixed_window*/, true/* multiscale*/, true/* lab*/);
        static Ferns fern;
        // static Ferns fern_ref;

//  static bool draw = true;
        //clear faces

        std::list <Human> tracked_huamns(trackedHuman());
        cv::Mat frame = cameraFrame().clone();

        if (tracked_huamns.size() > 0) {
            human_predict_path(tracked_huamns);
            human_publishment(tracked_huamns);
            rviz_visualization(use_greet, tracked_huamns);
        }

        show = frame.clone();
        int height = frameHeight();
        int width = frameWidth();

        //face function
        Faces.clear();
        FaceDetection();

        //过滤检测结果中不正确的人脸
        if (FaceFilter(Faces)) {
            for (int ic = 0; ic < Faces.size(); ic++) // Iterate through all current elements (detected faces)
            {
                adjustOutBox(Faces[ic], width, height);
                cv::rectangle(show, Faces[ic], cv::Scalar(255, 255, 0), 2);
            }
        } else
            Faces.clear();

        face_detect_num_control(); //脸数控制

        //std::cout << "after faces" << std::endl;

        if (height < 10 || width < 10) return;
        //std::cout << " ******scn: " << color_ref_Rect.channels() << " depth: " << color_ref_Rect.depth() << std::endl;

        if (!catched) {
            //如果人无法站在镜头两米之前，机器人调整
            double angle_oo = robotYaw();
            bool adjust_flag = false;
            std::cout << "Rects_person().size(): " << Rects_person().size() << " check_humans_in_view_angle: " << check_humans_in_view_angle(tracked_huamns) << std::endl;
            if ((Rects_person().size() == 0) && (!check_humans_in_view_angle(tracked_huamns))) {
                double d = 1.5;
                int times = 5;
                float step = d / times;
                std::cout << "step: " << step << std::endl;
                for (int i = 1; i < times + 1; i++) {
                    double times_x = robotPoseX() + cos(angle_oo) * (i * step);
                    double times_y = robotPoseY() + sin(angle_oo) * (i * step);
                    double map_times_x, map_times_y;
                    if (projectToMap(times_x, times_y, map_times_x, map_times_y)) {
                        std::cout << "x: "<< times_x<< " y: " << times_y <<  " score0: " << (int) scoremap.at<unsigned char>((int) map_times_y, (int) map_times_x) << std::endl;
                        if ((int) scoremap.at<unsigned char>((int) map_times_y, (int) map_times_x) > 240) {
                            ROS_WARN("The robot face some barrier.");
                            adjust_flag = true;
                            break;
                        }
                    }
                }
                if (!adjust_flag) {
                    angle_oo = robotYaw() + 0.34; //20度
                    std::cout << "angle_oo0: " << angle_oo << std::endl;
                    for (int i = 1; i < times + 1; i++) {
                        double times_x = robotPoseX() + cos(angle_oo) * (i * step);
                        double times_y = robotPoseY() + sin(angle_oo) * (i * step);
                        double map_times_x, map_times_y;
                        if (projectToMap(times_x, times_y, map_times_x, map_times_y)) {
                            std::cout << "x: "<< times_x<< " y: " << times_y << "score1: " << (int) scoremap.at<unsigned char>((int) map_times_y, (int) map_times_x) << std::endl;
                            if ((int) scoremap.at<unsigned char>((int) map_times_y, (int) map_times_x) > 240) {
                                ROS_WARN("The robot face some barrier.");
                                adjust_flag = true;
                                break;
                            }
                        }
                    }
                }
                if (!adjust_flag) {
                    angle_oo = robotYaw() - 0.34; //20度
                    std::cout << "angle_oo1: " << angle_oo << std::endl;
                    for (int i = 1; i < times; i++) {
                        double times_x = robotPoseX() + cos(angle_oo) * (i * step);
                        double times_y = robotPoseY() + sin(angle_oo) * (i * step);
                        double map_times_x, map_times_y;
                        if (projectToMap(times_x, times_y, map_times_x, map_times_y)) {
                            std::cout <<"x: "<< times_x<< " y: " << times_y<< "score2: " << (int) scoremap.at<unsigned char>((int) map_times_y, (int) map_times_x) << std::endl;
                            if ((int) scoremap.at<unsigned char>((int) map_times_y, (int) map_times_x) > 240) {
                                ROS_WARN("The robot face some barrier.");
                                adjust_flag = true;
                                break;
                            }
                        }
                    }
                }
            }

            if (adjust_flag) {
                double angle_pub = robotYaw() + 3.14;
                ti.tx = robotPoseX();
                ti.ty = robotPoseY();
                ROS_WARN("Robot Position needed to be adjust");
                tp.pub_time_now = ros::Time::now();
                if (((tp.pub_time_now.toSec() - tp.pub_time_last.toSec()) > motion_.frz) &&
                    (motion_.lock_pub == false)) {
                    motion_.tracker_tracking(tp, ti, rs, no_task, false, path_tracking_flag);
                    motion_.init_flag = false;
                    ROS_ERROR("Start tracking without human detection!");
                }

            } else {
                cv::Rect target_rect;
                ROS_ERROR("NO TARGET, NEED TO DETECT TARTGET");
                visualStart(target_rect, tracked_huamns);
                //std::cout << " roi.x: " << target_rect.x << " roi.y: " << target_rect.y << std::endl;
                if (target_rect.x != 0 && target_rect.y != 0) {
                    adjustOutBox(target_rect, width, height);
                    cv::rectangle(show, target_rect, cv::Scalar(0, 255, 255), 2);
                }
            }
        } else {
            visualTrack(tracked_huamns);
            laserTrack(tracked_huamns);

            if (skipFlag()) //skipFlag()
            {
                ROS_WARN("Recieve Skip! Skip the tracking target!");
                // if(robotVel()!=0 || robot_Theta()!=0)
                {
                    std_msgs::UInt32 stop;
                    stop.data = 1;
                    stop_track_pub_.publish(stop);
                }
                ti.tx = 0;
                ti.ty = 0;
                rs.first_state_ = INACTIVE;
                rs.second_state_ = INACTIVE;
                rs.robot_show_state_ = TRACKER_FIND;
                rs.robot_state_ = LOST;
                first_found_cnt = 0;
                second_find_cnt = 0;
                second_find_flag = false;
                skipFlag() = false;
            }

        }

        sensor_msgs::ImagePtr show_msg_ptr = cv_bridge::CvImage(std_msgs::Header(), "bgr8", show).toImageMsg();
        sensor_msgs::Image show_msg = *show_msg_ptr;
        //std::cout << show_msg << std::endl;
        image_pub_.publish(show_msg);

    }


    void Ltracker::run() {
//  if (use_greet)
//  {
//    if (greet_track_MODE())
//    {
//      greeter();
//    }
//    else
//    {
//      if(use_camera())
//        runWithVisual();
//      else
//        tracker();
//    }
//  }
//
//  else
//  {
//    humanPredictor();
//  }
        runWithVisual();
    }