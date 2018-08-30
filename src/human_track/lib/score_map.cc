#include "score_map.h"

//costmap & scoremap



//把世界坐标转换成图像的坐标
bool projectToMap(double xi, double yi, double &xo, double &yo)
{
    //std::cout << "xi: " << xi << " yi: " << yi   << std::endl; 
    // xi = 50;
    // yi = 60;
    double map_resolution = mapResolution();
    double map_origin_x = mapOriginX();
    double map_origin_y = mapOriginY();
    int map_width = mapWidth();
    int map_height = mapHeight();
    xo = (xi - map_origin_x)/map_resolution;
    yo = (yi - map_origin_y)/map_resolution;
   // ROS_INFO("map_resolution: %f, map_origin_x: %f, map_origin_y: %f, map_width: %d, map_height: %d ", map_resolution, map_origin_x, map_origin_y, map_width, map_height);
   // ROS_INFO("xi: %f, yi: %f, xo: %f, yo: %f", xi, yi, xo, yo);
    //std::cout << "xo: " << xo << " yo: " << yo << std::endl;
    return xo>0 && yo>0 && xo<map_width && yo<map_height;
}



//把crop赋值到full上
void pasteRoi(cv::Mat &full, cv::Mat &crop, cv::Rect &roi)
{
    //std::cout<<crop.cols<<'\t'<<crop.rows<<'\t'<<roi<<std::endl;
    assert(roi.width==crop.cols && roi.height==crop.rows);
    unsigned char *src_ptr, *dst_ptr;
    for(int y=1; y<roi.height-1; y++)
    {
        src_ptr = crop.ptr<unsigned char>(y)+1;
        dst_ptr = full.ptr<unsigned char>(y+roi.y) + roi.x + 1;
        for(int i=roi.width-2; i; i--)
            *dst_ptr++ = *src_ptr++;
    }

}

cv::Mat transCostMap(cv::Mat &signed_map)
{
 // probabilities are in the range [0,100].  Unknown is -1.
 // ban walking in unknwon region
     char* src_ptr=(char*)signed_map.data;
     cv::Mat ret = cv::Mat(signed_map.rows,signed_map.cols,CV_8UC1);
     unsigned char* dst_ptr=ret.data;

     for(int i=signed_map.cols*signed_map.rows; i; i--,src_ptr++,dst_ptr++)
     {
         if(*src_ptr>99 || *src_ptr<0) *dst_ptr=255; //障碍物255
         else *dst_ptr=0; //非障碍物0
     }

     return ret;
}

cv::Mat transScoreMap(cv::Mat &costmap)
{
//get the score map form costmap
//using a region-grow-like method
    const static int step = 4;
    int w = costmap.cols;
    int h = costmap.rows;

    unsigned char* swap1 = (unsigned char*)malloc(sizeof(unsigned char)*w*h);
    unsigned char* swap2 = (unsigned char*)malloc(sizeof(unsigned char)*w*h);
    bool swap_flag = true;
    unsigned char *dst_ptr, *src_ptr, *src_ptr_l, *src_ptr_r, *src_ptr_t, *src_ptr_b;
    memcpy(swap2,costmap.data,sizeof(unsigned char)*w*h);
    memcpy(swap1,costmap.data,sizeof(unsigned char)*w*h);
    //memset(swap1,0,sizeof(unsigned char)*w*h);

    //迭代的思想，障碍物是大于0的，离墙越近数值越大，每次迭代相当于向障碍物反方向增加一个障碍物的点
    for(unsigned char value = 255; value > step; value -= step, swap_flag = !swap_flag) //赋一整个swap
    {
        for(int y=1; y<h-1; y++) //处理一列
        {
            if(swap_flag)
            {
                dst_ptr = swap1+y*w+1;
                src_ptr = swap2+y*w+1;
            }
            else
            {
                dst_ptr = swap2+y*w+1;
                src_ptr = swap1+y*w+1;
            }
            src_ptr_l = src_ptr-1;
            src_ptr_r = src_ptr+1;
            src_ptr_t = src_ptr+w;
            src_ptr_b = src_ptr-w;
            for(int cnt_w=w-2;
                    cnt_w;
                    cnt_w--, //处理一行
                    //src_ptr++,
                    src_ptr_l++,
                    src_ptr_r++,
                    src_ptr_t++,
                    src_ptr_b++,
                    dst_ptr++)
            {
                if(*dst_ptr) continue;
                if(*src_ptr_l || *src_ptr_r || *src_ptr_t || *src_ptr_b) //这是保证src周围有一个障碍物。或者离障碍物近的
                    *dst_ptr = value; //就给dst赋值
            }
        }
    }
    cv::Mat ret(h,w,CV_8UC1);
    //ROS_WARN("h: %d, w: %d",h,w);
    memcpy(ret.data, swap_flag?swap1:swap2, sizeof(unsigned char)*h*w);
    free(swap1);
    free(swap2);
    
    return ret; 
}

bool& isMapInited()
{
    static bool v = false;
    return v;
}

void protoMapCallback(const nav_msgs::OccupancyGrid & map)
{
     //std::cout << "mapcallback" << std::endl;
     //if(isMapInited()) return;
     //std::cout<<"Got proto occupancy grid map"<<std::endl;
     ROS_WARN("Got proto occupancy grid map");

//update the map information from costmap occupanygrid message
     mapResolution() = map.info.resolution;
     mapWidth() = map.info.width;
     mapHeight() = map.info.height;
     mapOriginX() = map.info.origin.position.x;
     mapOriginY() = map.info.origin.position.y;
     mapOriginZ() = map.info.origin.position.z;
     mapOrientX() = map.info.origin.orientation.x;
     mapOrientY() = map.info.origin.orientation.y;
     mapOrientZ() = map.info.origin.orientation.z;
     mapOrientW() = map.info.origin.orientation.w;
     std::cout << "mapResolution(): " << mapResolution() << " mapWidth(): "<< mapWidth() << std::endl;

//hao xiang nei cun xie lou le ?!
     int size_buf[2];
     size_buf[0] = mapHeight();
     size_buf[1] = mapWidth();
     cv::Mat signed_map = cv::Mat(2,size_buf,CV_8SC1,(char*)&map.data.front());

     globalCostMap() = transCostMap(signed_map);
     globalCostMapProto() = globalCostMap().clone();
     //std::cout<<"global costmap genetated "<<std::endl;
     ROS_INFO("global costmap genetated");
     globalScoreMap() = transScoreMap(globalCostMap());
     globalScoreMapProto() = globalScoreMap().clone();
//
	  
     ROS_INFO("global scoremap genetated");
     //std::cout<<"global scoremap genetated "<<std::endl;

	//计算梯度方向
     cv::Mat grad_x(mapHeight(),mapWidth(),CV_32FC1); //gradient x
     cv::Mat grad_y(mapHeight(),mapWidth(),CV_32FC1); // gradient y
     cv::Mat mag(mapHeight(),mapWidth(),CV_32FC1); // gradient mag
     cv::Mat ori(mapHeight(),mapWidth(),CV_32FC1); // gradient orient
     cv::Sobel(globalScoreMap(), grad_x, CV_32FC1, 1, 0, CV_SCHARR, 1, 0, cv::BORDER_DEFAULT);   
     cv::Sobel(globalScoreMap(), grad_y, CV_32FC1, 0, 1, CV_SCHARR, 1, 0, cv::BORDER_DEFAULT);  
     cv::cartToPolar(grad_x, grad_y, mag, ori, false);  
     //cv::GaussianBlur(ori, ori, cv::Size(7,7), 0 );

     globalOrientMap() = ori;

     isMapInited() = true;
     /*
     cv::imshow("1", global_costmap);
     cv::imshow("2", global_costmap_proto);
     cv::imshow("3", global_scoremap);
     cv::imshow("4", global_scoremap_proto);
     cv::waitKey(-1);
     */
     //std::cout << "protoMapCallback finished" << std::endl;
 }

void updateMapCallback(const map_msgs::OccupancyGridUpdate & update)
{   
     if(!isMapInited())
     {
         //std::cout<<"proto costmap has not received yet/updateMapCallback"<<std::endl;
         ROS_INFO("proto costmap has not received yet/updateMapCallback");
         return;
     }
     ROS_INFO("receiving costmap update/updateMapCallback");
     //std::cout<<"receiving costmap update/updateMapCallback"<<std::endl;
     int size_buf[2];
     size_buf[0] = update.height;
     size_buf[1] = update.width;
     cv::Mat signed_map = cv::Mat(2,size_buf,CV_8SC1,(char*)&update.data.front());

     cv::Mat local_costmap = transCostMap(signed_map);
     cv::Mat local_scoremap = transScoreMap(local_costmap);

     cv::Rect roi_rect(update.x,update.y,update.width,update.height);

     pasteRoi(globalCostMap(),local_costmap,roi_rect);
     pasteRoi(globalScoreMap(),local_scoremap,roi_rect);

     //cv::Mat show = global_scoremap.clone();
     //cv::rectangle(show, roi_rect,0);
     //cv::imshow("0", show);
     //cv::waitKey(10);
}
