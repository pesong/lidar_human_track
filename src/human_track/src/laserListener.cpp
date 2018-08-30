#include <laserListener.h>
#include <angles/angles.h>

inline void rotate(double r_x, double r_y, double *x, double *y, double yaw)
{
    *x = -r_y * sin(yaw) + r_x * cos(yaw);
    *y = r_x * sin(yaw) + r_y * cos(yaw);
}

tf::Matrix3x3 &Odom_Amcl()
{
    static tf::Matrix3x3 v;
    return v;
}

inline double getAngleWithViewpoint(float r1, float r2, float included_angle)
{
    return atan2(r2 * sin(included_angle), r1 - r2 * cos(included_angle));
}
float laserListener::getTimestamp()
{
    timeval t;
    gettimeofday(&t, NULL);

    static int constSec = t.tv_sec;

    return t.tv_sec - constSec + (float)t.tv_usec / 1000000;
}

void laserListener::Laser2Amcl(tf::Matrix3x3 odom_amcl, double xi, double yi, double &xo, double &yo)
{
    double pose_x, pose_y, amcl_pos_sin, amcl_pos_cos;
    pose_x = odom_amcl.getColumn(2).getX();       //x
    pose_y = odom_amcl.getColumn(2).getY();       //y
    amcl_pos_sin = odom_amcl.getColumn(0).getY(); //sin
    amcl_pos_cos = odom_amcl.getColumn(0).getX(); //cos
    //先转到world坐标系
    xo = amcl_pos_cos * xi - amcl_pos_sin * yi + pose_x;
    yo = amcl_pos_cos * yi + amcl_pos_sin * xi + pose_y;
}

bool laserListener::ToMap(tf::Matrix3x3 odom_amcl, double xi, double yi, double &xoo, double &yoo)
{
    if (fabs(mapResolution()-0)<0.0001)
        return false;
    scoremap = globalScoreMapProto();
    //std::cout << "mapResolution(): " << mapResolution() << std::endl;
    double xo, yo;
    Laser2Amcl(odom_amcl, xi, yi, xo, yo);
    //std::cout << "asdsds" << std::endl;
    //再转到scoremap上
    xoo = (xo - mapOriginX()) / mapResolution();
    yoo = (yo - mapOriginY()) / mapResolution();
   // ROS_INFO("human_track xi: %f, yi: %f, xoo: %f, yoo: %f", xi, yi, xo, yo);

    return xoo > 0 && yoo > 0 && xoo < mapWidth() && yoo < mapHeight();
}

// void laserListener::mapCallback(const sensor_msgs::Image::ConstPtr &scoremap_)
// {
//     // std::cout << "mapCallback" << std::endl;
//     // std::cout << "**#######################" << std::endl;
//     //
//     //ROS_INFO("isMapInit");
//     this->scoremap = cv::Mat(scoremap_->height, scoremap_->width, CV_8UC1, cv::Scalar(0));
//     //this->scoremap =  cv_bridge::toCvShare(scoremap_, "mono8")->image;
//     this->scoremap = cv_bridge::toCvCopy(scoremap_, sensor_msgs::image_encodings::TYPE_8UC1)->image;
//     // this->scoremap = cv::Mat(scoremap_->height,scoremap_->width,CV_16UC1,cv::Scalar(0));
//     // temp.convertTo(this->scoremap,CV_16UC1);
//     // cv::imshow("ssss",scoremap);
//     // cv::waitKey(3);
//     isMapInit = true;
//     //std::cout << "scoremap: " << scoremap << std::endl;
// }
void laserListener::syncCallback(const sensor_msgs::LaserScan::ConstPtr &message, const geometry_msgs::PoseWithCovarianceStamped::ConstPtr &pose)
{
    //ROS_INFO(" In syncCallback ");
    
    last_pose = pose;
    last_message = message;
    laserMessage() = message;
    
    amclCallback(pose);
    
    laserCallback_(message); 
    
    LaserShow(last_message,last_pose);
    //ROS_INFO(" syncCallback Done");
    
}

// void laserListener::mapInfoCallback(const std_msgs::Float32MultiArray &map_msg)
// {
//     //std::cout << "**************************" << std::endl;
//     //ROS_INFO("MapInfo");
//     mapResolution = map_msg.data[0];
//     mapX = map_msg.data[1];
//     mapY = map_msg.data[2];
//     map_w = map_msg.data[3];
//     map_h = map_msg.data[4];
//     //std::cout << mapY << std::endl;
//     //ROS_INFO("mapX: %d, mapY: %d, map_w: %d, map_h: %d", mapX, mapY, map_w, map_h);
//     isMapInfo = true;
// }

// void laserListener::laserXCallback(const std_msgs::Float32 &laserX_msg)
// {
//     laserX() = laserX_msg.data;
//     //std::cout << "LASERX: " << laserX() << std::endl;
// }

laserListener::laserListener(tracker *trk, std::string frame_name) : tracker_(trk), laser_frame_id(frame_name),
                                                                     frame_id(0), pose_y(0), sin_(0),
                                                                     cos_(1), max_angle_(170), min_angle_(10),
                                                                     window_(1), neighbors_(3)
{
    ros::NodeHandle nh("~"), nh_;
    isMapInit = false;
    isMapInfo = false;
    LaserInit = false;
    plicp_init = false;
    // std::cout << "1111111laserx: " << laserX() << std::endl;
    //nh.param("laser_pose", laser_x, 0.0);
    
    nh.param("min_seg_size", min_seg_size, 1);
    nh.param("max_seg_size", max_seg_size, 120); //120
    nh.param("min_clu_size", min_clu_size, 3);
    nh.param("max_clu_size", max_clu_size, 350);
    nh.param("segment_dist_thres", segment_dist_thres, 0.2);
    //nh.param("cluster_dist_thres", cluster_dist_thres, 0.65);
    nh.param("cluster_dist_thres", cluster_dist_thres, 0.5);
    nh.param("min_model_dist", min_model_dist, 0.95f);
    nh.param("max_model_dist", max_model_dist, 1.5f);
    nh.param("max_seg_length", max_seg_length, 1.2);
    nh.param("max_density", max_density, 300.0); //300
    nh.param("min_dist", min_dist, 0.0);
    //nh.param("max_dist", max_dist, 12.0);
    nh.param("max_dist", max_dist, 20.0);
    nh.param("use_image", use_image_, false);
    nh.param("tb_filter", tb_filter, false);
    nh.param("st_filter", st_filter, true);
    rviz_points_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("segments", 1);
    //icp_pose_2d_pub_ = nh_.advertise<std_msgs::Float32MultiArray>("icp_pose_2d", 3);
    laser_filt_pub_ = nh_.advertise<sensor_msgs::LaserScan>("filtered_scan", 3);
    if (st_filter)
        st_laser_pub_ = nh_.advertise<sensor_msgs::LaserScan>("scan_st", 3);
    if (tb_filter)
        tb_laser_pub_ = nh_.advertise<sensor_msgs::LaserScan>("scan_tb", 3);
    laser_sync = nh_.advertise<sensor_msgs::LaserScan>("laser_sync", 3);
    //amcl_sync = nh_.advertise<geometry_msgs::PoseWithCovarianceStamped>("amcl_sync", 3);
    //ROS_INFO("laserListener created");
    curTimestamp = getTimestamp();
    preTimestamp = getTimestamp();
    //laserScan_sub_ = new message_filters::Subscriber<sensor_msgs::LaserScan>(nh, "iri_hokuyo_lase/scan", 5);
    //laserScan_sub_->registerCallback(boost::bind(&laserListener::laserCallback, this, _1));
    //laserScan_sub_ = new ros::Subscriber(nh.subscribe("/iri_hokuyo_lase/scan", 3, &laserListener::laserCallback, this));
    // string boost_xml;
    // nh.getParam("boost_xml", boost_xml);
    // boost.load(boost_xml.c_str(), "boost");
    lengine_params libEngineParams;
    libEngineParams.jumpdist = 0;
    libEngineParams.feature_set = 0; // WE SHOULD TRY CHANGING THAT TO 2
    libEngineParams.laser_range = 0;
    libEngineParams.sanity = 1;
    libEngineParams.segonly = 0;
    v_x = 0;
    v_y = 0;
    v_theta = 0;
    libEngine = new lengine(libEngineParams);
    plicp_min_reading = 0.1;
    plicp_max_reading = 10;
    plicp_angular_correction = 0.4;
    plicp_linear_correction = 0.2;
    trk->setLaser(0); //laserX()
    odom_received = false;
}

void laserListener::laserScanToLDP(const sensor_msgs::LaserScan &scan, LDP &ldp)
{

    unsigned int n = scan.ranges.size() - 5;

    ldp = ld_alloc_new(n);

    for (unsigned int i = 0; i < n; i++)
    {

        // calculate position in laser frame

        double r = scan.ranges[i];

        if (r > scan.range_min && r < scan.range_max)
        {

            // fill in laser scan data

            ldp->valid[i] = 1;

            ldp->readings[i] = r;
        }
        else
        {

            ldp->valid[i] = 0;

            ldp->readings[i] = -1; // for invalid range
        }

        ldp->theta[i] = scan.angle_min + i * scan.angle_increment;

        ldp->cluster[i] = -1;
    }

    ldp->min_theta = ldp->theta[0];

    ldp->max_theta = ldp->theta[n - 1];

    ldp->odometry[0] = 0.0;

    ldp->odometry[1] = 0.0;

    ldp->odometry[2] = 0.0;

    ldp->true_pose[0] = 0.0;

    ldp->true_pose[1] = 0.0;

    ldp->true_pose[2] = 0.0;
}

void laserListener::plicpInit()
{
    // plicp initialization
    m_input_.laser[0] = 0.0;
    m_input_.laser[1] = 0.0;
    m_input_.laser[2] = 0.0;

    m_input_.min_reading = plicp_min_reading;
    m_input_.max_reading = plicp_max_reading;

    // **** CSM parameters - comments copied from algos.h (by Andrea Censi)
    // Maximum angular displacement between scans
    m_input_.max_angular_correction_deg = plicp_angular_correction;
    // Maximum translation between scans (m)
    m_input_.max_linear_correction = plicp_linear_correction;

    // Maximum ICP cycle iterations
    m_input_.max_iterations = 10;
    // A threshold for stopping (m)
    m_input_.epsilon_xy = 0.000001;
    // A threshold for stopping (rad)
    m_input_.epsilon_theta = 0.000001;

    // Maximum distance for a correspondence to be valid
    m_input_.max_correspondence_dist = 0.1;
    // Noise in the scan (m)
    m_input_.sigma = 0.010;

    // Use smart tricks for finding correspondences.
    m_input_.use_corr_tricks = 1;
    // Restart: Restart if error is over threshold
    m_input_.restart = 0;
    // Restart: Threshold for restarting
    m_input_.restart_threshold_mean_error = 0.01;
    // Restart: displacement for restarting. (m)
    m_input_.restart_dt = 1.0;
    // Restart: displacement for restarting. (rad)
    m_input_.restart_dtheta = 0.1;
    // Max distance for staying in the same clustering
    m_input_.clustering_threshold = 0.25;
    // Number of neighbour rays used to estimate the orientation
    m_input_.orientation_neighbourhood = 20;
    // If 0, it's vanilla ICP
    m_input_.use_point_to_line_distance = 1;
    // Discard correspondences based on the angles
    m_input_.do_alpha_test = 0;
    // Discard correspondences based on the angles - threshold angle, in degrees
    m_input_.do_alpha_test_thresholdDeg = 20.0;
    // Percentage of correspondences to consider: if 0.9,
    // always discard the top 10% of correspondences with more error
    m_input_.outliers_maxPerc = 0.90;
    // Parameters describing a simple adaptive algorithm for discarding.
    //  1) Order the errors.
    //  2) Choose the percentile according to outliers_adaptive_order.
    //     (if it is 0.7, get the 70% percentile)
    //  3) Define an adaptive threshold multiplying outliers_adaptive_mult
    //     with the value of the error at the chosen percentile.
    //  4) Discard correspondences over the threshold.
    //  This is useful to be conservative; yet remove the biggest errors.
    m_input_.outliers_adaptive_order = 0.7;
    m_input_.outliers_adaptive_mult = 2.0;
    // If you already have a guess of the solution, you can compute the polar angle
    // of the points of one scan in the new position. If the polar angle is not a monotone
    // function of the readings index, it means that the surface is not visible in the
    // next position. If it is not visible, then we don't use it for matching.
    m_input_.do_visibility_test = 1;
    // no two points in laser_sens can have the same corr.
    m_input_.outliers_remove_doubles = 1;
    // If 1, computes the covariance of ICP using the method http://purl.org/censi/2006/icpcov
    m_input_.do_compute_covariance = 0;
    // Checks that find_correspondences_tricks gives the right answer
    m_input_.debug_verify_tricks = 0;
    // If 1, the field 'true_alpha' (or 'alpha') in the first scan is used to compute the
    // incidence beta, and the factor (1/cos^2(beta)) used to weight the correspondence.");
    m_input_.use_ml_weights = 0;
    // If 1, the field 'readings_sigma' in the second scan is used to weight the
    // correspondence by 1/sigma^2
    m_input_.use_sigma_weights = 0;
} // End of plicpInit

void laserListener::poseCallback(const nav_msgs::Odometry::ConstPtr &pose)
{
    //ROS_INFO("AMCL got");

    // double roll, pitch, yaw;
    // tf::Quaternion q((pose->pose.pose.orientation.x), (pose->pose.pose.orientation.y), (pose->pose.pose.orientation.z),
    //                  (pose->pose.pose.orientation.w));
    // tf::Matrix3x3 m(q);
    // m.getRPY(roll, pitch, yaw);
    // pose_x = pose->pose.pose.position.x;
    // pose_y = pose->pose.pose.position.y;
    v_x = pose->twist.twist.linear.x;
    v_y = pose->twist.twist.linear.y;
    v_theta = pose->twist.twist.angular.z;
    // ROS_WARN("v_x: %f, v_y: %f, v_theta: %f", v_x,v_y,v_theta);
    // pose_theta = yaw;
    // //ROS_INFO("X: %lf, Y: %lf, theta: %lf", pose_x, pose_y, pose_theta);
    // sin_ = sin(yaw);
    // cos_ = cos(yaw);

    // odom_received = true;
}

void laserListener::amclCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr &pose) 
//void laserListener::amclInvoke(const geometry_msgs::PoseWithCovarianceStamped &pose)
{
    //ROS_INFO("AMCL got");
    double roll, pitch, yaw;
    tf::Quaternion q((pose->pose.pose.orientation.x), (pose->pose.pose.orientation.y), (pose->pose.pose.orientation.z),
                     (pose->pose.pose.orientation.w));
    tf::Matrix3x3 m(q);
    m.getRPY(roll, pitch, yaw);
    pose_x = pose->pose.pose.position.x;
    pose_y = pose->pose.pose.position.y;
    //v_x = pose.twist.twist.linear.x;
    //v_y = pose.twist.twist.linear.y;
    //v_theta = pose.twist.twist.angular.z;
    pose_theta = yaw;
    //ROS_INFO("X: %lf, Y: %lf, theta: %lf", pose_x, pose_y, pose_theta);
    sin_ = sin(yaw);
    cos_ = cos(yaw);
    //std::cout << "sin_: " << sin_ << " cos_: " << cos_ << std::endl;
    //tracker_->setTheta(yaw);
    odom_received = true;
}

double laserListener::wrongDistribution(std::vector<Point3D_str> &pts, Point3D_str &center)
{
    double cov_x = 0, cov_y = 0, cov_xy = 0;
    for (std::vector<Point3D_str>::iterator it = pts.begin(); it != pts.end(); ++it)
    {
        cov_x += (it->x - center.x) * (it->x - center.x);
        cov_y += (it->y - center.y) * (it->y - center.y);
        cov_xy += (it->y - center.y) * (it->x - center.x);
    }
    NEWMAT::SymmetricMatrix cov(2);
    NEWMAT::DiagonalMatrix eigen(2);
    // ROS_INFO("eigen: %lf,  %lf", cov_x, cov_y);
    cov(1, 1) = cov_x;
    cov(2, 2) = cov_y;
    cov(2, 1) = cov_xy;
    //ROS_INFO("eigen: %lf,  %lf", cov(1,1), cov(2,2));
    NEWMAT::EigenValues(cov, eigen);
    //ROS_INFO("eigen: %lf,  %lf", eigen(1, 1), eigen(2, 2));
    return eigen(2, 2) / eigen(1, 1);
}

void laserListener::clusterFeature(std::vector<Point3D_container> &clusters)
{
    //ROS_INFO("extracting");
    //Frame = new clusterFrame(laser_frame_id);
    Frame = boost::shared_ptr<clusterFrame>(new clusterFrame(laser_frame_id));
    //boost::shared_ptr<clusterFrame> Frame(new clusterFrame(laser_frame_id));
    geometry_msgs::Point32 pt32;
    Point3D_str pt_cog;
    double length;
    int id = 0;
    float probs;
    //ROS_INFO("cluster size: %d", clusters.size());

    lfeatures_class lfeatures = lfeatures_class(0);
    if (use_image_)
        lfeatures.compute_descriptor(clusters, descriptor);

    //libEngine->computeFeatures(clusters, descriptor);
    for (std::vector<Point3D_container>::iterator it = clusters.begin(); it != clusters.end(); ++it, id++)
    {

        length = distance_L2_XY(&(it->pts.front()), &(it->pts.at(it->pts.size() / 2))) +
                 distance_L2_XY(&(it->pts.at(it->pts.size() / 2)), &(it->pts.back()));
        it->compute_cog(&pt_cog);
        double dist = sqrt(pt_cog.x * pt_cog.x + pt_cog.y * pt_cog.y);
        double close_dist = 1.2;
        //        if (it->occ_l || it->occ_r)
        //            ROS_ERROR("OCCLUDED");
        //        ROS_INFO("cog  ----  X: %lf, Y: %lf, Length: %lf, size: %d, model: %d, label: %d", pt_cog.x, pt_cog.y, length,
        //                 it->pts.size(), it->model_label, it->label);
        double clust_size = it->pts.size();
        if ((clust_size < min_clu_size) && (dist >= 0))
        {
            //ROS_ERROR("clust_size < min_clu_size");
            continue;
        }

        double density = clust_size / length;
        if ((((it->occ_l || it->occ_r) &&
              (length > 1.2 && it->pts.size() > 30)) ||
             (it->occ_l && it->occ_r) || length > 2 || density * dist > 340 || density < 20 || clust_size * dist > 480))
        {
            // ROS_ERROR("ignore the cluster, points density: %lf", density);
            continue;
        }

        if ((dist < min_dist || dist > max_dist) && (dist >= close_dist))
        {
            // ROS_ERROR("observation out of range: %f", dist);
            continue;
        }
        double pca = wrongDistribution(it->pts, pt_cog);
        if ((pca > std::max(250.0, 200 * dist) || pca < 3.5) && (dist >= close_dist))
        {
            // ROS_ERROR("wrong PCA");
            continue;
        }

        if (use_image_)
        {
            lFeatures = cv::Mat::zeros(1, 18, CV_32FC1);
            for (int i = 0; i < 17; i++)
            {
                lFeatures.at<float>(0, i + 1) = descriptor[id][i];
            }
            probs = boost->predict(lFeatures, cv::Mat(), true);
            probs = 1 / (1 + exp(-2 * probs));
            //ROS_INFO("human probability: %lf", probs);
            if (probs < 0.01)
            {
                // continue;
            }
        }
        else
        {
            probs = 0.3;
        }

        // double amcl_x,amcl_y;
        // Laser2Amcl(Odom_Amcl(), pt_cog.x, pt_cog.y, amcl_x, amcl_y);
        // std::cout << " observation.x: " << pt_cog.x << " observation.y: " << pt_cog.y  <<" num: " << it->pts.size() << std::endl;
        ///std::cout << "laserX(): " << laserX() << std::endl;
        //pt_cog.x += laserX(); //base_laser 到 base_link
        pt32.x = (icp_cos_ * pt_cog.x - icp_sin_ * pt_cog.y) + icp_pose_x;
        pt32.y = (icp_cos_ * pt_cog.y + icp_sin_ * pt_cog.x) + icp_pose_y;
        //        pt32.x = (cos_ * pt_cog.x - sin_ * pt_cog.y) + pose_x;
        //        pt32.y = (cos_ * pt_cog.y + sin_ * pt_cog.x) + pose_y;
        pt32.z = 0;
        Frame->observations.push_back(
            Observation(pt_cog.x, pt_cog.y, id, probs, true, pt32, dist, clust_size, density, pca, length));
    }
    //    ROS_INFO("Cog size: %d", Frame->observations.size());
//    ROS_INFO("extracted");
    frame_id++;
    Frame->frame_id = frame_id;

    tracker_->update(Frame);
    tracker_->extractModels(feed_back, clust_id_tb, clust_id_st);

    if (st_filter)
        laserFilter_st(clusters);
    if (tb_filter)
        laserFilter_tb(clusters);
    clusters.clear();
}

//  merge the segments into clusters
void laserListener::segmentMerge(std::vector<Point3D_container> &segments, int &label)
{
    //ROS_INFO("merging");
    std::vector<Point3D_container> clusters;
    clusters.resize(label);
    if (label == 0)
    {
        if (tb_filter)
        {
            cluster_list.push_front(clusters);
            assert(cluster_list.size() == message_list.size());
            if (message_list.size() > 3)
            {
                message_list.pop_back();
                cluster_list.pop_back();
            }
        }
        return;
    }
    std::vector<bool> occ_l;
    occ_l.resize(label);
    std::vector<bool> occ_r;
    occ_r.resize(label);
    std::vector<bool> occ_;
    occ_.resize(label);

    //ROS_INFO("label size: %d", label);
    int j = 1;
    for (std::vector<Point3D_container>::iterator it = clusters.begin(); it != clusters.end(); ++it)
    {
        it->label = j;
        j++;
        occ_[j - 1] = false;
        occ_l[j - 1] = false;
        occ_r[j - 1] = false;
    }
    for (int i = 0; i < segments.size(); i++)
    {
        if (segments[i].label == 0)
        {
            //ROS_ERROR("0000 %d,  %d", segments[i].pts.size(), i);
        }
        if (!occ_[segments[i].label - 1])
        {
            clusters[segments[i].label - 1].model_label = segments[i].model_label;
            occ_[segments[i].label - 1] = true;
            occ_r[segments[i].label - 1] = segments[i].occ_r;
        }
        //        ROS_INFO("1111 %d", segments[i].label);
        clusters[segments[i].label - 1].pts.insert(clusters[segments[i].label - 1].pts.end(),
                                                   segments[i].pts.begin(),
                                                   segments[i].pts.end());
        // ROS_INFO("2222 %d", segments[i].occ_l);
        //ROS_ERROR("cluster label ( %d )points after insertion: %d", segments[i].label,
        //           clusters[segments[i].label - 1].pts.size());

        occ_l[segments[i].label - 1] = segments[i].occ_l;

        //ROS_INFO("3333");
    }
    //ROS_INFO("jjjj");
    j = 0;
    for (std::vector<Point3D_container>::iterator it = clusters.begin(); it != clusters.end(); ++it)
    {
        it->occ_l = occ_l[it->label - 1];
        it->occ_r = occ_r[it->label - 1];
        //ROS_ERROR("cluster points size: %d", it->pts.size());

        /*for (std::vector<Point3D_str>::iterator it_ = it->pts.begin(); it_ != it->pts.end(); ++it_) {
            //ROS_ERROR("cluster points error: %d", );
            ROS_INFO("point: %lf, %lf", it_->x, it_->y);
        }*/
    }
    j++;

    segments.clear();
    // ROS_INFO("merged");
    clusterFeature(clusters);
}

//cluster the segments recrusively
int laserListener::segmentClustering(std::vector<Point3D_container> &segments)
{
    int label = 0;
    int indexI, indexJ;
    std::vector<std::pair<int, int>> paired, select_proto; // pair1: index, pair2: label;
                                                           
    for (int i = 0; i < segments.size(); i++)
    {
        if (i == segments.size() - 1) //搜寻是否出现在paired里，否则加入paired中
        {
            auto result = find_if(paired.begin(), paired.end(), index_finder(i));
            if (result == paired.end())
            {
                label++;
                paired.push_back(std::make_pair(i, label));
                // std::cout << " paired9 size: " << paired.size() << std::endl;
                // std::cout << " index9: " << i << std::endl;
                segments[i].label = label;
                continue;
            }
        }
        double min_clust_dis = cluster_dist_thres;
        for (int j = i + 1; j < segments.size(); j++)
        {
            //std::cout << " i: " << i << " j: " << j << std::endl;
            //距离和model_label都要考虑
            if ((segments[i].model_label == segments[j].model_label) && (distance_L2_XY(segments[i].cog, segments[j].cog) < min_clust_dis))
            {
                //std::cout << " dis0: " << min_clust_dis << std::endl;
                min_clust_dis = distance_L2_XY(segments[i].cog, segments[j].cog);
                //std::cout << " dis1: " << min_clust_dis << std::endl;
                // std::cout << " i: " << i << " j: " << j << " min_clust_dis: " << min_clust_dis << std::endl;
                indexI = i;
                indexJ = j;
                // std::cout << " indexII: " << indexI << " indexJJ: " << indexJ << std::endl;
            }
        }
        if (min_clust_dis != cluster_dist_thres) //在cluster_dist_thres范围里有匹配对象
        {
            //std::cout << " indexI: " << indexI << " indexJ: " << indexJ << std::endl;
            //std::cout << " dis: " << min_clust_dis << std::endl;
            auto resultI = find_if(paired.begin(), paired.end(), index_finder(indexI));
            auto resultJ = find_if(paired.begin(), paired.end(), index_finder(indexJ));
            // std::cout << " paired0 size: " << paired.size() << std::endl;
            if (resultI != paired.end() && resultJ == paired.end()) //paired里找到了I,J标记为和I一样
            {
                segments[indexJ].label = segments[indexI].label;
                paired.push_back(std::make_pair(indexJ, segments[indexI].label));
                // std::cout << " paired5 size: " << paired.size() << std::endl;
                // std::cout << " index5: " << indexJ << std::endl;
            }
            else if (resultJ != paired.end() && resultI == paired.end()) //paired里找到了J,I标记为和J一样
            {
                segments[indexI].label = segments[indexJ].label;
                paired.push_back(std::make_pair(indexI, segments[indexI].label));
                // std::cout << " paired6 size: " << paired.size() << std::endl;
                // std::cout << " index6: " << indexI << std::endl;
            }
            else if (resultJ != paired.end() && resultI != paired.end()) //I,J都在pair里找到了，若满足条件，合并成一类
            {
                //判断两类中其它的segments之间的距离是否也在cluster_dist_thres内
                auto result_a = paired.begin();
                auto result_b = paired.begin();
                bool dist_ok_flag = true; //一旦有超距离现象，dist_ok_flag置false;
                while (find_if(result_a, paired.end(), label_finder(segments[indexI].label)) != paired.end())
                {
                    while (find_if(result_b, paired.end(), label_finder(segments[indexJ].label)) != paired.end())
                    {
                        if (distance_L2_XY(segments[result_a->first].cog, segments[result_b->first].cog) > cluster_dist_thres)
                        {
                            dist_ok_flag = false;
                            break;
                        }
                        else
                            result_b = find_if(result_b, paired.end(), label_finder(segments[indexJ].label))+1;  
                    }
                    if (dist_ok_flag)
                        result_a = find_if(result_a, paired.end(), label_finder(segments[indexI].label))+1;       
                    else
                        break;
                }

                if (dist_ok_flag)
                {
                    //合并两类
                    // std::cout << " paired1 size: " << paired.size() << std::endl;
                    select_proto.clear();
                    label++;
                    int label_a = segments[indexI].label; //label_a,label_b分别代表两类点的label
                    auto result_a = find_if(paired.begin(), paired.end(), label_finder(label_a));
                    while (result_a != paired.end())
                    {
                        select_proto.push_back(*result_a);
                        paired.erase(result_a);
                        // std::cout << " paired2 size: " << paired.size() << std::endl;
                        result_a = find_if(paired.begin(), paired.end(), label_finder(label_a));
                    }
                    int label_b = segments[indexJ].label;
                    auto result_b = find_if(paired.begin(), paired.end(), label_finder(label_b));
                    while (result_b != paired.end())
                    {
                        select_proto.push_back(*result_b);
                        paired.erase(result_b);
                        // std::cout << " paired3 size: " << paired.size() << std::endl;
                        result_b = find_if(paired.begin(), paired.end(), label_finder(label_b));
                    }
                    for (auto it = select_proto.begin(); it != select_proto.end(); it++)
                    {
                        //paired[it->first].second = label;
                        segments[it->first].label = label;
                        paired.push_back(std::make_pair(it->first, label));
                        // std::cout << " paired4 size: " << paired.size() << std::endl;
                        // std::cout << " index4: " << it->first << std::endl;
                    }
                }
            }
            else if (resultJ == paired.end() && resultI == paired.end()) //I,J都不在pair里，新增一类
            {
                label++;
                segments[indexI].label = label;
                segments[indexJ].label = label;
                paired.push_back(std::make_pair(indexI, segments[indexI].label));
                paired.push_back(std::make_pair(indexJ, segments[indexJ].label));
                //std::cout << " paired7 size: " << paired.size() << std::endl;
                //std::cout << " index7: " << indexI << std::endl;
                //std::cout << " index8: " << indexJ << std::endl;
            }
        }
        else //在cluster_dist_thres范围里无匹配对象
        {
            auto result = find_if(paired.begin(), paired.end(), index_finder(i));
            if (result == paired.end())
            {
                label++;
                segments[i].label = label;
                paired.push_back(std::make_pair(i, segments[i].label));
                //std::cout << " paired8 size: " << paired.size() << std::endl;
                //std::cout << " index8: " << i << std::endl;
                //std::cout << " ii " << i << std::endl;
            }
        }
    }

    //根据 renew_label 重新标label
    int renew_label = 0;
    //std::cout << " paired size: " << paired.size() << std::endl;
    for (int i = 0; i <= label; i++)
    {
        //std::cout << "find index: " << i+1 <<std::endl;
        auto result = find_if(paired.begin(), paired.end(), label_finder(i + 1));
        if (result != paired.end())
            renew_label++;
        // std::cout << "renew label:" << renew_label << std::endl;
        while (result != paired.end())
        {
            segments[result->first].label = renew_label;
            //std::cout << " index: " << result->first << " label: " << segments[result->first].label;
            paired.erase(result);
            result = find_if(paired.begin(), paired.end(), label_finder(i + 1));
        }
        //std::cout << std::endl;
    }
    return renew_label;
}

void laserListener::segmentFilter(std::vector<Point3D_container> &segments)
{
    //ROS_INFO("segmenting: %d", segments.size());
    int label = 0;
    //serch the segments nearby:
    //1. the model_label should be the same, 2. the segments cannot more than the clust_dis_threshould
    label = laserListener::segmentClustering(segments);
    //std::cout << " label: " << label <<std::endl;
    segmentMerge(segments, label);
}

void laserListener::laserFilter_st(std::vector<Point3D_container> &clusters)
{
    //message = *message_;
    //ROS_INFO("start filtering1");

    //ROS_ERROR("erase cluster size: %ld", clust_id_st.size());
    if (clust_id_st.size() > 0)
    {
        for (vector<int>::iterator it = clust_id_st.begin(); it != clust_id_st.end(); ++it)
        {
            if (*it >= clusters.size() || *it < 0)
            {
                //ROS_ERROR("INVALID CLUSTER ID");
                continue;
            }
            for (std::vector<Point3D_str>::iterator pt_it = clusters[*it].pts.begin();
                 pt_it != clusters[*it].pts.end(); ++pt_it)
            {
                message.ranges[pt_it->id] = 0;
            }
        }
    }
    //message.header.stamp = ros::Time();

    st_laser_pub_.publish(message);
}

void laserListener::laserFilter_tb(std::vector<Point3D_container> &clusters)
{
    //message = *message_;
    //ROS_INFO("start filtering1");

    cluster_list.push_front(clusters);
    assert(cluster_list.size() == message_list.size());

    if (message_list.size() > 3)
    {
        //ROS_ERROR("erase cluster size: %ld", clust_id_tb.size());
        if (clust_id_tb.size() > 0)
        {
            for (vector<int>::iterator it = clust_id_tb.begin(); it != clust_id_tb.end(); ++it)
            {
                if (*it >= cluster_list.back().size() || *it < 0)
                {
                    //ROS_ERROR("INVALID CLUSTER ID");
                    continue;
                }
                for (std::vector<Point3D_str>::iterator pt_it = cluster_list.back()[*it].pts.begin();
                     pt_it != cluster_list.back()[*it].pts.end(); ++pt_it)
                {
                    message_list.back().ranges[pt_it->id] = 0;
                }
            }
        }
        //message.header.stamp = ros::Time();

        tb_laser_pub_.publish(message_list.back());
        message_list.pop_back();
        cluster_list.pop_back();
    }

    /*std::set<int> indices_to_delete;
    //ROS_INFO("start filtering");
    // For each point in the current line scan
    for (unsigned int i = 0; i < message_->ranges.size(); i++) {
        for (int y = -window_; y < window_ + 1; y++) {
            int j = i + y;
            if (j < 0 || j >= (int) message_->ranges.size() || (int) i == j) { // Out of scan bounds or itself
                continue;
            }

            double angle = abs(angles::to_degrees(
                    getAngleWithViewpoint(message_->ranges[i], message_->ranges[j], y * message_->angle_increment)));
            if (angle < min_angle_ || angle > max_angle_) {
                for (int index = std::max<int>(i - neighbors_, 0);
                     index <= std::min<int>(i + neighbors_, (int) message_->ranges.size() - 1); index++) {
                    if (message_->ranges[i] <
                        message_->ranges[index]) // delete neighbor if they are farther away (note not self)
                        indices_to_delete.insert(index);
                }
            }

        }
    }
    //ROS_INFO("filtering");
    ROS_DEBUG(
            "ScanShadowsFilter removing %d Points from scan with min angle: %.2f, max angle: %.2f, neighbors: %d, and window: %d",
            (int) indices_to_delete.size(), min_angle_, max_angle_, neighbors_, window_);
    for (std::set<int>::iterator it = indices_to_delete.begin(); it != indices_to_delete.end(); ++it) {
        message.ranges[*it] = 0.0;  // Failed test to set the ranges to invalid value
    }*/
}


//  split the laserscan into segments
//void laserListener::laserInvoke(const sensor_msgs::LaserScan::ConstPtr &message_)
void laserListener::laserCallback_(const sensor_msgs::LaserScan::ConstPtr &message_)
{
    //std::cout << "In laserCallback_" <<std::endl;
    //std::cout << "laser(): " << laserX() << std::endl;
    //std::cout << "asdsdsd: " << message_->header.seq << std::endl;
   // laserX() = laserX();
    //ROS_INFO("PROCESSING LASER");
    //if (message_->header.seq % 3 != 0)
    //return;
    //ROS_ERROR("msize: %ld, csize: %ld", message_list.size(), cluster_list.size());
    //LaserInit = false;
    static double previous_cogx, previous_cogy;
    static double init_time = message_->header.stamp.toSec();
    sensor_msgs::LaserScan current_message, filtered_message;
    static sensor_msgs::LaserScan previous_message;
    static int rviz_ind = 0;
    visualization_msgs::Marker rviz_segment;
    visualization_msgs::MarkerArray rviz_segments;
    //rviz
    rviz_segment.header.frame_id = "world";
    rviz_segment.header.stamp = ros::Time();
    rviz_segment.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    rviz_segment.action = visualization_msgs::Marker::ADD;
    rviz_segment.lifetime = ros::Duration(0.2);
    rviz_segment.scale.x = 0.2;
    rviz_segment.scale.y = 0.2;
    rviz_segment.scale.z = 0.2;
    rviz_segment.color.r = 0.5;
    rviz_segment.color.g = 0.0;
    rviz_segment.color.b = 0.5;
    rviz_segment.color.a = 1;
    //pose_sub = nh.subscribe("odom", 5, &laserListener::poseCallback, &listener_);

    //_message = message_;

    preTimestamp = curTimestamp;
    curTimestamp = message_->header.stamp.toSec() - init_time;

    if (!odom_received)
    {
       // ROS_ERROR("Receive no odometry");
        return;
    }
    //    else {
    //        ROS_ERROR("Receive no odometry");
    //    }

    bool newseg = true;
    bool occlude_L, occlude_R;
    laser_frame_id = message_->header.frame_id;
    std::vector<Point3D_container> segments;
    segments.clear();
    Point3D_container segment;
    segment.model_label = -1;
    segment.label = 0;
    uint seg_size = 0;
    message = *message_;

    tf::Matrix3x3 odom_(cos_, -sin_, pose_x,
                        sin_, cos_, pose_y,
                        0, 0, 1);
    tf::Matrix3x3 odom_amcl(cos_, -sin_, pose_x,
                            sin_, cos_, pose_y,
                            0, 0, 1);

//    std::cout << "LaserInit: " <<  LaserInit << std::endl;
   // std::cout << "plicp_init: " <<  plicp_init << std::endl;
    
    if (!plicp_init)
    {
        plicpInit();
        laserScanToLDP(message, m_previous_ldp_scan_);
        icp_base = odom_;
        icp_odom = odom_ * icp_base.inverse();
        plicp_init = true;
        icp_pose_theta = pose_theta;
    }
    else
    {

        float deltaT = curTimestamp - preTimestamp;
        m_input_.laser_ref = m_previous_ldp_scan_;
        laserScanToLDP(message, m_previous_ldp_scan_);
        m_input_.laser_sens = m_previous_ldp_scan_;
         //       m_input_.first_guess[0] = deltaT * v_x;
         //       m_input_.first_guess[1] = deltaT * v_y;
                m_input_.first_guess[2] = 0;//deltaT * v_theta;

        if (true)
        {

            sm_icp(&m_input_, &m_output_);

            if (m_output_.valid)
            {
                icp_delta_x = m_output_.x[0];
                icp_delta_y = m_output_.x[1];
                icp_delta_theta = m_output_.x[2];
                //icp_delta_theta = m_output_.x[2];
            }
            else
            {
                icp_delta_x = deltaT * v_x;
                icp_delta_y = deltaT * v_y;
                icp_delta_theta = deltaT * v_theta;
            }
            icp_pose_theta += icp_delta_theta;
            icp_delta_cos_ = cos(icp_delta_theta);
            icp_delta_sin_ = sin(icp_delta_theta);
        }
        else
        {
            icp_delta_cos_ = 1;
            icp_delta_sin_ = 0;
            icp_delta_x = 0;
            icp_delta_y = 0;
        }
        ld_free(m_input_.laser_ref);
        
       
        //std::cout << m_output_.x[0] << std::endl;
        tf::Matrix3x3 icp_delta(icp_delta_cos_, -icp_delta_sin_, icp_delta_x, icp_delta_sin_, icp_delta_cos_,
                                icp_delta_y, 0, 0, 1);
        icp_base *= icp_delta;
        icp_pose_x = icp_base.getColumn(2).getX();
        icp_pose_y = icp_base.getColumn(2).getY();
        icp_sin_ = icp_base.getColumn(0).getY();
        icp_cos_ = icp_base.getColumn(0).getX();
        icp_odom = odom_ * icp_base.inverse();

        //tf::Matrix3x3 icp_inverse = icp_base.inverse();
        std_msgs::Float32MultiArray icp2d_msg;
        icp2d_msg.data.resize(8);
        icp2d_msg.data[0] = odom_amcl.getColumn(2).getX(); //x
        icp2d_msg.data[1] = odom_amcl.getColumn(2).getY(); //y
        icp2d_msg.data[2] = odom_amcl.getColumn(0).getY(); //sin
        icp2d_msg.data[3] = odom_amcl.getColumn(0).getX(); //cos
        //icp has the true orient & odom amcl has the true offset
        icp2d_msg.data[4] = icp_base.getColumn(2).getX(); //x
        icp2d_msg.data[5] = icp_base.getColumn(2).getY(); //y
        icp2d_msg.data[6] = icp_base.getColumn(0).getY(); //sin
        icp2d_msg.data[7] = icp_base.getColumn(0).getX(); //cos
        //icp2d_msg.data[8] = message.header.seq; //cos
       // std::cout << "human_track laser time: " << message.header.stamp << std::endl;
       // std::cout << "human_track amclPose: " << icp2d_msg.data[0] << std::endl;
       // std::cout << "last amclPose: " << last_pose->pose.pose.position.x << std::endl;
        //std::cout << "laser_sync and amcl_sync" <<std::endl;
        laserCallback(laserMessage());
        //laser_sync.publish(last_message);
        //amcl_sync.publish(last_pose);
      //
      
        //std::cout << "human_track last_laser_sync: " << last_message->header.stamp << std::endl;
       // std::cout << "human_track last_amcl_sync: " << last_pose->header.stamp << std::endl;
        
        //icp_pose_2d_pub_.publish(icp2d_msg);
        //icp2dMsg()=icp2d_msg;
        htrack_poseCallback(icp2d_msg);

        tracker_->setTheta(icp_pose_theta);
        tracker_->setMatrix(icp_odom, icp_base.inverse());
        //ROS_ERROR("odom and icp comparison: x: %lf, %lf, y: %lf %lf, theta:%lf %lf %lf %lf", pose_x, icp_pose_x,
        //pose_y, icp_pose_y, pose_theta, icp_pose_theta, icp_sin_, icp_cos_);
        if (feed_back.size() > 0)
        {
            processModels(deltaT);
        }
    }
    int valid_num = 0;
    
    filtered_message = message;
    //std::cout << "message size: " << message.ranges.size() << std::endl;
    for (int i = 10; i < message.ranges.size() - 10; i++)
    {
        if (message.ranges[i] > message_->range_max ||
            message.ranges[i] < message_->range_min)
        {
            //ROS_ERROR("deadlock check");
            filtered_message.ranges[i] = 0;
            continue;
        }

        //??
        int offset = 0;
        while (true)
        {
            if (message.ranges[++offset + i] < message_->range_max &&
                message.ranges[++offset + i] > message_->range_min)
                break;
        }
        valid_num++;

       // ROS_INFO("laser data:%lf", message.ranges[i]);
        if (newseg)
        {
            segment.pts.clear();
            if (message.ranges[i - 1] < message_->range_max)
                occlude_R = (message.ranges[i] - message.ranges[i - 1]) > 0.25 ? true : false;
            else
                occlude_R = false;
            newseg = false;
        }
        double angle = message_->angle_min + i * (message_->angle_increment);
        double v_x = message.ranges[i];
        double v_y = 0;
        rotate(v_x, v_y, &v_x, &v_y, angle);
       // std::cout << "v_x: " << v_x << " v_y: " << v_y << std::endl;
        //std::cout << isMapInit <<std::endl;

        // //激光点转到world再转到scoremap,在墙里的去掉
        // double map_x,map_y;
        // if(isMapInit && isMapInfo){
        //     if(ToMap(odom_amcl,v_x,v_y,map_x,map_y)){
        //         if ((int)scoremap.at<unsigned char>((int)map_y,(int)map_x) > 230)
        //             continue;
        //    }
        // }
        
        Point3D_str pts;
        pts.x = v_x;
        pts.y = v_y;
        pts.z = 0;
        pts.id = i;
        segment.pts.push_back(pts);
        //ROS_INFO("point: %lf, %lf", pts.x, pts.y);
        //ROS_INFO("segment size:%d", segment.pts.size());
        double shadow_angle = abs(angles::to_degrees(atan2(message.ranges[i] * sin(message_->angle_increment),
                                                           message.ranges[i + offset] -
                                                               message.ranges[i] * cos(message_->angle_increment))));
        //??
        if (abs(message.ranges[i + offset] - message.ranges[i]) > segment_dist_thres || shadow_angle > 175 ||
            shadow_angle < 5) //shadow_angle用来分界,角度过大或过小证明是两个segments
        {                     //            shadow_angle = abs(angles::to_degrees(atan2(message.ranges[i + 1] * sin(message_->angle_increment),
                              //                                                        message.ranges[i + 2] -
                              //                                                        message.ranges[i + 1] * cos(message_->angle_increment))));
                              //            if (shadow_angle > 175||shadow_angle < 5)
                              //                i++;
            newseg = true;
            if (message.ranges[i + offset] > 0.5)
                occlude_L = (message.ranges[i] - message.ranges[i + offset]) > 0.25 ? true : false;
            else
                occlude_L = false;

            segment.occ_l = occlude_L;
            segment.occ_r = occlude_R;
            int seg_size = segment.pts.size();
            //seg_length: segment里第一个点和最后一个点的距离
            double seg_length = distance_L2_XY(&segment.pts.front(), &segment.pts.back());
            //dis:segment中间那个点距laser的距离
            double dist = sqrt(segment.pts[segment.pts.size() / 2].x * segment.pts[segment.pts.size() / 2].x +
                               segment.pts[segment.pts.size() / 2].y * segment.pts[segment.pts.size() / 2].y);
            double seg_density = seg_size / seg_length;
            //ROS_INFO("segment size:%d, %ld", segments.size(), seg_length);
            //ROS_ERROR("NEW SEGMENT");

            //左或右没被挡住，距离laser小于一定阈值，segment内激光点数量大于一定数量，
            //std::cout << "max_dist: " << max_dist <<std::endl;
            if ((!occlude_R || !occlude_L) && dist < max_dist && seg_size > min_seg_size &&
                seg_size * dist < max_seg_size && seg_length < max_seg_length &&
                seg_density * dist < max_density)
            { //?? seg_size * dist < max_seg_size, seg_density * dist < max_density
                if (seg_size > 6)
                {
                    int count = 0;
                    for (std::vector<Point3D_str>::iterator it = segment.pts.begin() + 1;
                         it != segment.pts.end() - 1; ++it)
                    {
                        double tan_1 = atan(((it + 1)->x - it->x) / ((it + 1)->y - it->y));
                        double tan_2 = atan((it->x - (it - 1)->x) / (it->y - (it - 1)->y));
                        if (tan_2 == 0)
                            tan_2 = 0.00001;
                        if (fabs(tan_1 / tan_2 - 1) < 0.1)
                            count++;
                    }
                    if (count >= seg_size * 0.5)
                    {
                        filtered_message.ranges[i] = 0;
                        continue;
                    }
                }
                //ROS_ERROR("PUSH SEGMENTS with Size: %ld", segment.pts.size());

                /* for (std::vector<Point3D_str>::iterator it_ = segment.pts.begin(); it_ != segment.pts.end(); ++it_) {
                     //ROS_ERROR("cluster points error: %d", );
                     ROS_INFO("point: %lf, %lf", it_->x, it_->y);
                 }*/

                Point3D_str cogL;
                segment.compute_cog(&cogL);
                segment.cog.x = cogL.x;
                segment.cog.y = cogL.y;
                
                double map_x, map_y;
                rviz_segment.id = rviz_ind;
                double amcl_x, amcl_y;
                // amcl_x = cogL.x;
                // amcl_y = cogL.y;
                Laser2Amcl(odom_amcl, cogL.x, cogL.y, amcl_x, amcl_y);
                //std::cout << " cogL.x: " << cogL.x << " cogL.y: " << cogL.y << std::endl;
                //ROS_INFO("map_width: %d, map_height: %d ", mapWidth(),mapHeight());
                if (ToMap(odom_amcl, cogL.x, cogL.y, map_x, map_y))
                    rviz_segment.text = toString((int)scoremap.at<unsigned char>((int)map_y, (int)map_x));
                rviz_segment.pose.position.x = amcl_x;
                rviz_segment.pose.position.y = amcl_y;
                rviz_segment.pose.orientation.w = 1.0;
                rviz_segment.pose.orientation.x = 0.0;
                rviz_segment.pose.orientation.y = 0.0;
                rviz_segment.pose.orientation.z = 0.0;
                rviz_segments.markers.push_back(rviz_segment);
                rviz_ind = rviz_ind + 1;
                Odom_Amcl() = odom_amcl;
                double score_th;
                //map的去除点要和距离有关。
                //阈值score_th和距离相关，当距离大于激光最大距离的0.6后，score_th在230至250间变化
                if (fabs(mapResolution()-0)>0.0001)
                {
                    if (ToMap(odom_amcl, cogL.x, cogL.y, map_x, map_y))
                    {
                        //std::cout << " map_x: " << (int)map_x << " map_y: " << (int)map_y << " amcl_x: " << amcl_x << " amcl_y: " <<amcl_y << " score: "<< (int)scoremap.at<unsigned char>((int)map_y,(int)map_x);
                        if (dist < 0.6 * message_->range_max)
                        {
                            score_th = 230;
                        }
                        else
                        {
                            double pro = (dist / message_->range_max) * (250 - 230);
                            score_th = (int)(230 + pro);
                        }
                        if ((int)scoremap.at<unsigned char>((int)map_y, (int)map_x) > score_th)
                        {
                            continue;
                        }
                        // else
                        //  ROS_ERROR("IN MAP: %d", (int)scoremap.at<unsigned char>((int)map_y,(int)map_x));
                    }
                }
                /////draw rviz
                //rviz_segment
                if (feed_back.size() > 0)
                {
                    /////
                    float dist_ = 10;
                    for (std::deque<fb_model>::iterator it = feed_back.begin(); it != feed_back.end(); it++)
                    {
                        float new_dist_ = sqrt(
                            (cogL.x - it->x) * (cogL.x - it->x) + (cogL.y - it->y) * (cogL.y - it->y));
                        if (new_dist_ < std::min(std::max(it->length, min_model_dist), max_model_dist) &&
                            new_dist_ < dist_)
                        {
                            dist_ = new_dist_;
                            segment.model_label = it->id;
                        }
                    }
                    //rviz_segment.text = toString(segment.pts.size());
                    // rviz_segment.text = toString((int)scoremap.at<unsigned char>((int)map_y,(int)map_x));
                    // rviz_segment.text = toString(score_th);
                    // rviz_segment.id = rviz_ind;
                    // double amcl_x,amcl_y;
                    // Laser2Amcl(odom_amcl, cogL.x, cogL.y, amcl_x, amcl_y);
                    // rviz_segment.pose.position.x = amcl_x;
                    // rviz_segment.pose.position.y = amcl_y;
                    // rviz_segment.pose.orientation.w = 1.0;
                    // rviz_segment.pose.orientation.x = 0.0;
                    // rviz_segment.pose.orientation.y = 0.0;
                    // rviz_segment.pose.orientation.z = 0.0;
                    // rviz_segments.markers.push_back(rviz_segment);
                    // rviz_ind = rviz_ind+1;
                }
                segments.push_back(segment);
            }
            else if (seg_size == 2 && message.ranges[i] < 2)
            { //(shadow_angle > 175 || shadow_angle < 10)) {
                message.ranges[i] = 0;
                message.ranges[i - 1] = 0;
                filtered_message.ranges[i] = 0;
                filtered_message.ranges[i - 1] = 0;
            }
            else if (seg_size == 1 && message.ranges[i] < 2)
            { // (shadow_angle > 170 || shadow_angle < 10)) {
                message.ranges[i] = 0;

            } //ignore the last segment
        }
        //std::cout << "aaaaaaaaaa" << std::endl;
        //ROS_INFO("%lf,%lf", laser_x[i], laser_y[i]);
    }
    //ROS_INFO("Valid points number:%d", valid_num);
    //ROS_INFO("segments size:%d", segments.size());
    //ROS_INFO("FINISH LASER");
    //st_laser_pub_.publish(message);
    if (tb_filter)
    {
        message_list.push_front(message);
    }
    segmentFilter(segments);
    laser_filt_pub_.publish(filtered_message);
    rviz_points_pub_.publish(rviz_segments);
}

laserListener::~laserListener()
{
}

void laserListener::processModels(float deltaT)
{

    for (std::deque<fb_model>::iterator it = feed_back.begin(); it != feed_back.end(); ++it)
    {
        float x = it->x + it->v_x * deltaT - icp_pose_x;
        float y = it->y + it->v_y * deltaT - icp_pose_y;
        //        it->x = (cos_ * x + sin_ * y) - laser_x;
        //        it->y = (cos_ * y - sin_ * x);
        //std::cout << "laser_xxxx: " << laser_x << std::endl;
        it->x = (icp_cos_ * x + icp_sin_ * y) ; //- laserX();
        it->y = (icp_cos_ * y - icp_sin_ * x);
        it->length *= (0.8 + abs(it->v_x) + abs(it->v_y)); //?? 从原本length基础上改
        //ROS_ERROR("before model id: %ld, x:%lf, y:%lf", it->id, it->x, it->y);
        //        float x = it->x + std::min(it->v_x, 1.0f) * deltaT;
        //        float y = it->y + std::min(it->v_y, 1.0f) * deltaT;
        //        it->x = (cos_ * x + sin_ * y)-pose_x;
        //        it->x = (cos_ * y - sin_ * x)-pose_y;
        //ROS_INFO("model id: %ld, x:%lf, y:%lf", it->id, it->x, it->y);
        //
    }
}

void laserListener::LaserShow(sensor_msgs::LaserScan::ConstPtr message_, geometry_msgs::PoseWithCovarianceStamped::ConstPtr pose)
{
    std::vector<std::pair<double, double>> laser_amcl;
    for (int i = 10; i < message_->ranges.size() - 10; i++)
    {
        //std::cout << message_->ranges.size() << std::endl;
        if (message_->ranges[i] > message_->range_max ||
            message_->ranges[i] < message_->range_min)
            continue;
        double angle = message_->angle_min + i * (message_->angle_increment);
        double x = message_->ranges[i];
        double y = 0;
        rotate(x, y, &x, &y, angle);
        //x = x + laserX();
        //
        double roll, pitch, yaw;
        tf::Quaternion q((pose->pose.pose.orientation.x), (pose->pose.pose.orientation.y), (pose->pose.pose.orientation.z),
                         (pose->pose.pose.orientation.w));
        tf::Matrix3x3 m(q);
        m.getRPY(roll, pitch, yaw);
        double pose_cos, pose_sin, pose_x, pose_y, xo, yo;
        pose_x = pose->pose.pose.position.x;
        pose_y = pose->pose.pose.position.y;
        pose_theta = yaw;
        //ROS_INFO("X: %lf, Y: %lf, theta: %lf", pose_x, pose_y, pose_theta);
        pose_sin = sin(yaw);
        pose_cos = cos(yaw);
        //
        xo = pose_cos * x - pose_sin * y + pose_x;
        yo = pose_cos * y + pose_sin * x + pose_y;
        laser_amcl.push_back(std::make_pair(xo, yo));
    }

    //std::cout << "human_track_lasertime: " << message_->header.stamp << std::endl;
    //std::cout << "human_track_amcltime: " << pose->header.stamp << std::endl;  
    // std::cout << "human_track_amclpose: " << pose_x << " corresponding laser: " << message_->header.stamp << std::endl; 
    
    // std::cout << "human_track_amclx" << " ";
    // for (int i = 0; i < laser_amcl.size(); i++)
    // {
    //     std::cout << laser_amcl[i].first << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "human_track_amcly" << " ";
    // for (int i = 0; i < laser_amcl.size(); i++)
    // {
    //     std::cout << laser_amcl[i].second << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "laserScan" << " ";

    // for (int i = 0; i < message_->ranges.size(); i++)
    // {
    //     std::cout << message.ranges[i] << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "time" << " ";

    // for (int i = 0; i < message_->ranges.size(); i++)
    // {
    //     std::cout << message_->header.stamp << " ";
    // }
    // std::cout << std::endl;
}

//
// Created by song on 16-4-9.
// Modified by ziwei on 17-5-10.
//
