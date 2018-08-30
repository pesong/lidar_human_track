#include "connect.h"
using namespace Eigen;

//extern std::mutex mutexLaserSet;
//从camera frame回来通过了Human detection 的human

inline float getOverlap(const cv::Rect &b1, const cv::Rect &b2) //b1是激光的框，b2是摄像头识别的人
{
#define min___(a, b) (a > b ? b : a)
#define max___(a, b) (a < b ? b : a)
    int ws1 = min___(b1.x + b1.width, b2.x + b2.width) - max___(b1.x, b2.x);
    int hs1 = min___(b1.y + b1.height, b2.y + b2.height) - max___(b1.y, b2.y);
    float o = max___(0, ws1) * max___(0, hs1);
    o = o / (b2.width * b2.height);
    //o = o / (b1.width * b1.height + b2.width * b2.height - o);
    return o;
}


inline float distance_l2(float x1, float y1, float x2, float y2)
{
    float dx = x1-x2;
    float dy = y1-y2;
    return sqrt(dx*dx+dy*dy);
};


//返回面积比
inline float area(cv::Rect A, cv::Rect B) {
    cv::Rect max = (A.width*A.height > B.width*B.height ? A:B);
    cv::Rect min = (A.width*A.height < B.width*B.height ? A:B);
    float o =  (float)(min.width*min.height)/(max.width*max.height); //0.3一下肯定不是对应
    return o;
}


std::vector<Human> &cameraHumans()
{
    static std::vector<Human> v;
    return v;
}


std::vector<cameraRect> &Rects_person()
{
    static std::vector<cameraRect> v;
    return v;
}


std::vector<cameraRect> &Rects_face()
{
    static std::vector<cameraRect> v;
    return v;
}

std::vector<int> &selectIds()
{
    static std::vector<int> v;
    return v;
}

std::deque<laserSet> &Laser_sets()
{
    static std::deque<laserSet> v;
    return v;
}



std::vector<std::vector<Camera_rect_Info>::iterator>::iterator
max_overlap_element(std::vector<std::vector<Camera_rect_Info>::iterator> group)
{
    auto first = group.begin();
    auto last = group.end();
    if (first==last) return last;
    auto largest = first;

    while (++first!=last)
        if ((*largest)->max_overlap < (*first)->max_overlap)    // or: if (comp(*largest,*first)) for version (2)
            largest=first;
    return largest;
}


void change_max_idx (std::vector<int> &repeat_tmp, int max_idx_before , int max_idx_now)
{
    for(int i = 0; i < repeat_tmp.size(); i++)
    {
        if(repeat_tmp[i] ==  max_idx_before)
        {
            repeat_tmp[i] = max_idx_now;
            break;
        }

    }
}

////中心点距离，面积比例, overlap
//float Kdist(cv::Rect A, cv::Rect B)
//{
//    //中心点距离，归一化到1
//    float x1 = A.x+0.5*A.width;
//    float y1 = A.y+0.5*A.height;
//    float x2 = B.x+0.5*B.width;
//    float y2 = B.y+0.5*B.height;
//    float score1 = distance_l2(x1,y1,x2,y2)/frameDiag();
//
//    //面积比例
//
//    float score2 = 1 - area(A,B);
//    if(score2 > 0.7)
//        score2 = 100;
//
//    //overlap
//    float score3 = 1 - getOverlap(A,B);
//    //std::cout << "score1: " << score1 << "score2: " <<  score2 << "score3: " <<  score3 << std::endl;
//
//    return score1+score2+score3;
//
//}

//中心点距离，面积比例, overlap
float Kdist(cv::Rect A, cv::Rect B)
{
    //中心点距离，归一化到1
    float x1 = A.x+0.5*A.width;
    float y1 = A.y+0.5*A.height;
    float x2 = B.x+0.5*B.width;
    float y2 = B.y+0.5*B.height;
    float score1 = 1 - distance_l2(x1,y1,x2,y2)/frameDiag();

    //面积比例

    float score2 = area(A,B);

    //overlap
    float score3 = getOverlap(A,B);
    //std::cout << "score1: " << score1 << "score2: " <<  score2 << "score3: " <<  score3 << std::endl;

    float score = (score1+score2+score3)/3;

    return score;

}



float laser_min(int laser_num, MatrixXd X)
{
    float min = 1000;
    for (int i=0; i < X.cols(); i++)
    {
        if(X(laser_num,i) < min)
            min = X(laser_num,i);
    }
    return min;
}



//////laser的框和camera检测人的框匹配
//void laser_camera_match(std::list<Human>& humans) {
//    mutexCameraRects.lock();
//    std::vector <cameraRect> Cam_rects = Rects_person();
//    mutexCameraRects.unlock();
//
//    mutexSelectId.lock();
//    std::vector<int> tmp = selectIds();
//    mutexSelectId.unlock();
//    if ((tmp.size() == 0) || (Cam_rects.size() == 0))
//        return;
//    auto pos = unique(tmp.begin(), tmp.end());
//    std::vector<int> usefulIds(tmp.begin(), pos);
//
//
//    std::vector<laserRect> lasers;
//    laserRect a;
//   // std::cout << "usefulIds.size()" << usefulIds.size() << std::endl;
//    for (auto it = humans.begin(); it != humans.end(); it++) {
//        it->isHuman_tmp = false; //所有目标默认不是人
//        for (int i = 0; i < usefulIds.size(); i++) {
//            if(it->getId() == usefulIds[i])
//            {
//                //std::cout << usefulIds[i] << " ";
//                a.id = it->getId();
//                a.rect = it->getBoundingBox();
//                lasers.push_back(a);
//            }
//        }
//    } //这里都是和camera框有交集的overlap
//    //std::cout << std::endl;
//   // MatrixXd fx1(num, 1);F
//    MatrixXd X(lasers.size(),Cam_rects.size()); //laser / camera score matrix
//
//    MatrixXd Y(Cam_rects.size(),lasers.size()); //camera/laser idx
//
//    //记录laser/camera score matrix
//    //std::cout << "cam: " << Cam_rects.size() << " " << "lasers: " << lasers.size() << std::endl;
//    std::vector<float> tmp1,tmp2;
//    for(int j = 0; j < Cam_rects.size(); j++)
//    {
//       for(int i = 0; i < lasers.size(); i++)
//       {
//           //std::cout << "laser id: " << lasers[i].id << std::endl;
//           X(i, j) = Kdist(lasers[i].rect, Cam_rects[j].rect);
//           //std::cout << "X(" << i << "," << j << ") " << X(i, j) ;
//           tmp1.push_back(X(i, j));
//       }
//       //std::cout<<std::endl;
//       tmp2 = tmp1;
//       sort(tmp2.begin(), tmp2.end());
//      // std::cout << "tmp2: " << tmp2.size() <<std::endl;
//       for(int i0 = 0; i0 < tmp2.size(); i0++){
//           for(int j0 = 0; j0 < tmp1.size(); j0++){
//               if(tmp2[i0] == tmp1[j0]){
//                    Y(j,i0) = j0;
//                   // std::cout << "Y(" << j << "," << i0 << ") "<<Y(j,i0) ;
//               }
//           }
//       }
//       tmp1.clear();
//       tmp2.clear();
//    }
//
//    bool flag;
//    for(int i = 0; i < Cam_rects.size(); i++)
//    {
//        if(X(Y(i,0),i) > 100)
//            continue;
//        flag = false;
//        for(int j = i+1; j < Cam_rects.size(); j++) {
//            if (Y(i, 0) == Y(j, 0)) {
//                flag = true;
//                if(X(Y(i,0),i) == laser_min(Y(i,0), X)) //选中
//                {
//                    flag = false;
//                    if(lasers.size()<2)
//                    {
//                        for(int a = i+1; a < Cam_rects.size(); a++)
//                           X(Y(a,0),a) =  X(Y(a,0),a)+ 100;
//                    }
//                    else
//                    {
//                        if((Y(j, 0)) != Y(j, 1))
//                            Y(j,0) = Y(j,1);
//                    }
//
//                }
//                else
//                {
//                    if(lasers.size()<2)
//                        break;
//                    else
//                        if(Y(i, 0) == Y(i, 1)) //只比较前两个，如果不对就放弃
//                            break;
//                        else
//                        {
//                           Y(i, 0) = Y(i, 1);
//                        }
//                }
//            }
//            else
//               flag = false;
//        }
//
//
//        if(!flag && X(Y(i,0),i)<100)
//        {
//            //更新laser
//            for(auto it = humans.begin(); it != humans.end(); it++)
//            {
//              if(lasers[Y(i,0)].id == it->getId())
//                it->updateCameraBoundingBox(Cam_rects[i].rect);
//                it->isHuman_tmp = true;
//
//            }
//        }
//    }
//}

////laser的框和camera检测人的框匹配
//void laser_camera_match(std::list<Human>& humans) {
//
//    mutexCameraRects.lock();
//    std::vector<cameraRect> Cam_rects = Rects_person();
//    mutexCameraRects.unlock();
//
//    mutexSelectId.lock();
//    std::vector<int>usefulIds =  selectIds();
//    mutexSelectId.unlock();
//
//    std::vector <Camera_rect_Info> laser_set; //用来记录对应laser_id对应一组框的信息
//    std::vector<int> repeat_tmp; //用来记录所有laser_id对应cameraRect中，最大的overlap在Rects_person()的idx
//    //std::vector<int> find_ids;
//    for (auto it = humans.begin(); it != humans.end(); it++) {
//        it->isHuman_tmp = false; //所有目标默认不是人
//        for (int i = 0; i < usefulIds.size(); i++) {
//            if (it->getId() == usefulIds[i]) { //找到存在和camera rects overlap超过一定threshold的所有laser_id
//                //find_ids.push_back(it->getId());
//                float max_overlap = 0;
//                int max_overlap_idx = 100;
//                Camera_rect_Info this_rect;
//                //对于每个存在的laer_id都和每个camera rects计算过overlap
//                for (int j = 0; j < Cam_rects.size(); j++) {
//                    float overf = getOverlap(it->getBoundingBox(), Cam_rects[j].rect);
//                    if (overf > 0) { //只要有overlap都push进idx_overlap里
//                        this_rect.idx_overlap.push_back(std::make_pair(j, overf));
//                        if (overf > max_overlap) {
//                            max_overlap = overf;
//                            max_overlap_idx = j;
//                        }
//                    }
//                }
//                this_rect.laser_id = it->getId();
//                this_rect.max_overlap_idx = max_overlap_idx;
//                this_rect.max_overlap = max_overlap;
//                laser_set.push_back(this_rect);
//                //repeat_tmp是存每个laser id 对应的最大的框的idx
//                repeat_tmp.push_back(max_overlap_idx);
//            }
//        }
//    }
//
//    if (laser_set.size() == 0) //这一帧没有找到匹配的laer_id, 按照default都不是人，return
//        return;
//
//    //std::vector<int> repeat_tmp_proto = repeat_tmp; //reapt_tmp_proto是原始的按照laser_id顺序排列的max_overlap_idx
//    //找所有laser_id对应overlap最大的框有没有重复
//    sort(repeat_tmp.begin(), repeat_tmp.end());
//    auto pos = unique(repeat_tmp.begin(), repeat_tmp.end());
//    std::vector<int> repeat_idx; //default repeat_idx.size()== 0
//
//    //laser_id对应的最大max overlap有重复
//    if (pos != repeat_tmp.end()) {
//        std::vector<int> tmp(pos, repeat_tmp.end());
//        repeat_idx = tmp;
//    }
//
//    while (repeat_idx.size() > 0) {
//        auto start = laser_set.begin(); //laser_set是所有id的对应的camera rects的集合
//        std::vector <std::vector<Camera_rect_Info>::iterator> same_max_idx_iter_group;
//
//        //从所有id组对应的框中找到所有使用camera max idx=repeat_idx[0]的id组,push进same_max_idx_iter_group
//        //每次只考虑repeat的第一个数，然后repeat_tmp会重新排列
//        while (find_if(start, laser_set.end(), max_idx_finder(repeat_idx[0])) != laser_set.end()) {
//            start = find_if(start, laser_set.end(), max_idx_finder(repeat_idx[0]));
//            same_max_idx_iter_group.push_back(start); //用（*start）可以提取任意一个laser_set里的元素
//        }
//
//        //在一堆max_overlap_idx相同的的laser_id框里选出overlap最大的那个
//        auto it_max = max_overlap_element(same_max_idx_iter_group);
//        same_max_idx_iter_group.erase(it_max); //去除最大的这个
//
//        //去除最大的以后，剩下每个id组在自己组里找overlap第二大的框
//        for (auto it = same_max_idx_iter_group.begin(); it != same_max_idx_iter_group.end();) {
//            if ((*it)->idx_overlap.size() <= 1) //该laser_id对应的camera rect只有一个，并且不是overlap最大的
//                laser_set.erase(*it);
//            else {
//                float max_tmp = 0;
//                int max_idx;
//                for (int i = 0; i < (*it)->idx_overlap.size(); i++) {
//                    if ((*it)->idx_overlap[i].first == (*it)->max_overlap_idx)
//                        continue;
//                    if ((*it)->idx_overlap[i].second > max_tmp) {
//                        max_tmp = (*it)->idx_overlap[i].second;
//                        max_idx = (*it)->idx_overlap[i].first;
//                    }
//                }
//                (*it)->max_overlap_idx = max_idx;
//                (*it)->max_overlap = max_tmp;
//                change_max_idx(repeat_tmp, (*it)->max_overlap_idx, max_idx);
//            }
//        }
//
//        sort(repeat_tmp.begin(), repeat_tmp.end());
//        pos = unique(repeat_tmp.begin(), repeat_tmp.end());
//        if (pos != repeat_tmp.end()) {
//            std::vector<int> tmp(pos, repeat_tmp.end());
//            repeat_idx = tmp;
//        } else {
//            repeat_idx.clear();
//            break;
//        }
////        //去除所和repeat_idx[0]有重复的数
////        int var = repeat_idx[0];
////        for(auto it = repeat_idx.begin(); it != repeat_idx.end();)
////        {
////            if(*it == var)
////                repeat_idx.erase(it);
////            else
////                it++;
////        }
//    }
//
//    for (auto it = humans.begin(); it != humans.end(); it++) {
//        for (int i = 0; i < laser_set.size(); i++) {
//            if (it->getId() == laser_set[i].laser_id) {
//                std::cout << "select id: " << it->getId() << std::endl;
//                it->isHuman_tmp = true;
//                int J = laser_set[i].max_overlap_idx;
//                cv::Rect rect0 = Cam_rects[J].rect;
//                it->updateCameraBoundingBox(rect0);
//            }
//        }
//    }
//
//}

//每帧激光进来投影（有可投影上的，有不能的),更新laser投到camera frame上的框
void laser_human_extraction(std::list<Human> &humans)
{
    std::vector<laserRect> human_rects;
    std::vector<Human> eligible_humans;
    std::list<Human> return_humans;
    laserSet eligible_set;
    //当前帧的人如果能投影到camera frame
    for (auto it = humans.begin(); it != humans.end(); it++)
    {
        laserRect lr;
        it->isProject = false;
        if(projectBoxInCamera(*it))
        {
            it->isProject = true;
            lr.rect = it->getBoundingBox();

            lr.id = it->getId();
        }
        human_rects.push_back(lr);
        eligible_humans.push_back(*it);
    }

    eligible_set.laser_rects = human_rects;
    eligible_set.laser_humans = eligible_humans;
    eligible_set.timestamp = ros::Time::now();

    //写Laser_sets()要锁住 to_do ziwei
    //boost::unique_lock <boost::shared_mutex> lockLaserSet_write(mutexLaserSet);
    mutexLaserSet.lock();
    Laser_sets().push_back(eligible_set);
    if(Laser_sets().size()>Buff_size)
        Laser_sets().pop_front();
    mutexLaserSet.unlock();

    //laser_camera_match(humans);


    mutexSelectId.lock();
    //std::cout << "before select human" << std::endl;
    if(human_rects.size() > 0){
        for (auto it = humans.begin(); it != humans.end(); it++) {
            it->isHuman_tmp = false;
            for (int i = 0; i < selectIds().size(); i++) {
                if (it->getId() == selectIds()[i]) {
                   // std::cout << "select id: " << selectIds()[i] << std::endl;

                    it->isHuman_tmp = true;

                    //std::cout << "isHuman_tmp = true" << std::endl;
                }
            }
        }
    }
    mutexSelectId.unlock();
    //return return_humans;
}