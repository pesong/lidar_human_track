#include <detector.hpp>

//using namespace std;
using std::string;
using std::vector;
//using namespace cv;
using cv::Size;
using cv::Mat;
using cv::Rect;
using cv::HOGDescriptor;
using cv::Range;

//using namespace sensor_msgs;
using sensor_msgs::Image;
using sensor_msgs::LaserScan;
using namespace Header;

detector::detector() : nh("~") {
    //string s = nh.getNamespace();
    //ROS_INFO("[DETECTOR] my namespace %s ", s.c_str());

    if (nh.hasParam("camera_yaml") && nh.hasParam("camera_name") && nh.hasParam("boost_xml")) {
        string camera_yaml;
        string cname;
        string boost_xml;

        nh.getParam("camera_yaml", camera_yaml);

        nh.getParam("camera_name", cname);

        nh.getParam("boost_xml", boost_xml);
        if (!camera_calibration_parsers::readCalibrationYml(camera_yaml, cname, params.cInfo)) {
            ROS_ERROR("[DETECTOR] Failure reading camera calibration parameters.");
            return;
        }
	    boost = cv::ml::Boost::create();
        // Load the boost classifier
        boost->load(boost_xml.c_str(), "boost");

        ROS_INFO("[DETECTOR] Camera calibration & Boost Classifier Loaded");
    } else {
        ROS_ERROR(
                "[DETECTOR] Need to set the parameters in order to load the camera calibration and the boost classifier.");
        return;
    }

    if (nh.hasParam("laser_window_width") && nh.hasParam("laser_window_height") &&
        nh.hasParam("rect") && nh.hasParam("hog_hit_threshold") && nh.hasParam("hog_group_threshold") &&
        nh.hasParam("hog_meanshift") && nh.hasParam("tf_timeout") &&
        //      nh.hasParam("cameraA") && nh.hasParam("cameraB") &&
        nh.hasParam("laserA") && nh.hasParam("laserB") &&
        nh.hasParam("m_to_pixels") && nh.hasParam("body_ratio") && nh.hasParam("jumpdist") &&
        nh.hasParam("feature_set") && nh.hasParam("laser_range") && nh.hasParam("fusion_prob") &&
        nh.hasParam("min_camera_prob") && nh.hasParam("min_laser_prob")) {
        nh.getParam("laser_window_width", params.laser_window_width);
        nh.getParam("laser_window_height", params.laser_window_height);
        nh.getParam("rect", params.rect);

        nh.getParam("hog_hit_threshold", params.hog_hit_threshold);
        nh.getParam("hog_group_threshold", params.hog_group_threshold);
        nh.getParam("hog_meanshift", params.hog_meanshift);
        nh.getParam("tf_timeout", params.tf_timeout);
        //		nh.getParam("cameraA", params.cameraA);
        //		nh.getParam("cameraB", params.cameraB);
        nh.getParam("laserA", params.laserA);
        nh.getParam("laserB", params.laserB);
        nh.getParam("m_to_pixels", params.m_to_pixels);
        nh.getParam("body_ratio", params.body_ratio);
        nh.getParam("jumpdist", params.jumpdist);
        nh.getParam("feature_set", params.feature_set);
        nh.getParam("laser_range", params.laser_range);
        nh.getParam("fusion_prob", params.fusion_prob);
        nh.getParam("min_camera_prob", params.min_camera_prob);
        nh.getParam("min_laser_prob", params.min_laser_prob);
        ROS_INFO("[DETECTOR] Parameters loaded.");
    } else {
        ROS_ERROR("[DETECTOR] Wrong parameters loaded.");
    }

    // Initiates the laserLib with the parameters read from the server
    laserProcessor = new laserLib(params.jumpdist, params.feature_set, params.laser_range);

    window_size = Size(params.laser_window_width, params.laser_window_height);

    // Set the default detector
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    switch (params.feature_set) {
        case 0:
            params.no_features = FEATURE_SET_0;
            break;

        case 1:
            params.no_features = FEATURE_SET_1;
            break;

        default:
            params.no_features = FEATURE_SET_1;
            break;
    }

    // Convert camera info (cInfo) to actual needed camera matrix K and distortion coefficient D
    // ready to be used by opencv functions
    CameraInfo2CV(params.cInfo, K, D, params.rect);

    // Initializing the detector publisher
    detectionPublisher = nh.advertise<human_track::ClusterClass>("humanDetections", 1);
    ROS_INFO("[DETECTOR] Detector running OK with %d features.", params.no_features);
}

detector::~detector() {
    delete laserProcessor;
}

/// TODO NOT SURE IF NEEDED

/*
void detector::setClusters(human_track::ClusteredScan sd)
{
        clusterData=sd;
}

human_track::ClusteredScan detector::getClusters()
{
        return clusterData;
}
 */

void detector::getTF(const sensor_msgs::Image::ConstPtr &image, const sensor_msgs::LaserScan::ConstPtr &lScan) {
    // Read the transform between the laser and the camera
    // Essential for syncronising
    try {
        //        acquisition_time = image->header.stamp; /// Maybe need to change to ros::Time::now()
        acquisition_time = ros::Time::now();
        ros::Duration timeout(1.0 / params.tf_timeout);

        tf_listener_.waitForTransform(lScan->header.frame_id, "/logitech_camera_link"/*origin image->header.frame_id*/,
                                      acquisition_time, timeout);
        tf_listener_.lookupTransform(lScan->header.frame_id, "/logitech_camera_link"/*image->header.frame_id*/,
                                     acquisition_time, transform);
    } catch (tf::TransformException &ex) {
        ROS_WARN("[DETECTOR] TF exception:\n%s", ex.what());
    }
}

void detector::processLaser(const sensor_msgs::LaserScan::ConstPtr &lScan,
                            vector <human_track::ClusteredScan> &clusterData) {
    // Load the scan to the processor
    laserProcessor->loadScan(*lScan);

    // Segment the laser scan
    laserProcessor->getClusters(clusterData);

    // Only process scans with valid data
    if (clusterData.size() > 0) {
        laserProcessor->getFeatures(clusterData);
    } else {
        ROS_WARN("[DETECTOR] No valid clusters");
    }

    initClusterData(clusterData);
    findProjectedClusters(clusterData);
}

void detector::getImage(const sensor_msgs::Image::ConstPtr &image) {
    // Copy img to openCV format
    try {
        cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::MONO8);
    } catch (cv_bridge::Exception &e) {
        ROS_ERROR("[DETECTOR] cv_bridge exception: %s", e.what());
        return;
    }
}

void detector::initClusterData(vector <human_track::ClusteredScan> &clusterData) {
    for (uint i = 0; i < clusterData.size(); i++) {
        clusterData[i].cog_projected = false;
        clusterData[i].crop_projected = false;

        clusterData[i].detection_label = NOT_HUMAN;
        clusterData[i].detection_laser_prob = 0.0;
        clusterData[i].detection_camera_prob = 0.0;
        clusterData[i].detection_fusion_prob = 0.0;

        clusterData[i].label = NOT_HUMAN;
    }
}

void detector::findProjectedClusters(vector <human_track::ClusteredScan> &clusterData) {
    // Iterate through every cog of the scanClusters
    for (uint i = 0; i < clusterData.size(); i++) {
        if (clusterData[i].cog.x > 6)
            continue;
        // Convert the cog to image coordinates
        projectPoint(clusterData[i].cog, prPixel, K, D, transform);

        //printf("pt2D: \n x= %lf y= %lf \n z= %lf \n", clusterData[i].cog.x,  clusterData[i].cog.y, clusterData[i].cog.z);
        // If the pixel is projected within the image limits
        if (prPixel.x >= 0 && prPixel.x < cv_ptr->image.cols &&
            prPixel.y >= 0 && prPixel.y < cv_ptr->image.rows) {

            clusterData[i].cog_projected = true;
            // Get the box size and corners
            getBox(clusterData[i].cog, prPixel, rect, params.m_to_pixels, params.body_ratio);
            printf("pt2D: \n x= %lf y= %lf \n", clusterData[i].cog.x, clusterData[i].cog.y);
            printf("x= %lf \n y= %lf \n width= %d \n height= %d \n", prPixel.x, prPixel.y, rect.width, rect.height);
            // Check if the whole box lies inside the image
            if (checkBox(params.cInfo, rect)) {
                // Flag the cluster as 'fusable'. Meaning they appear with a valid box on the image.
                printf("x= %d \n y= %d \n width= %d \n height= %d \n", rect.x, rect.y, rect.width, rect.height);
                printf("pt2D: \n x= %lf y= %lf \n z= %lf \n", clusterData[i].cog.x, clusterData[i].cog.y,
                       clusterData[i].cog.z);
                clusterData[i].crop_projected = true;
            }
        }
    }
}

void detector::classifyLaser(std_msgs::Float32MultiArray &features, float &probs) {
    lFeatures = Mat::zeros(1, params.no_features + 1, CV_32FC1);

    //    fprintf(stderr, "%d\n", features.data.size());
    //    fprintf(stderr, "%d %d\n", boost.get_data()->var_all, boost.get_active_vars()->cols);

    for (int i = 0; i < (int) features.data.size(); i++) {
        lFeatures.at<float>(0, i + 1) = features.data[i];
    }

    if ((int) features.data.size() != params.no_features) {
        ROS_ERROR("[DETECTOR] Wrong number of computed features.");
    }

    // Find the prediction
    probs = boost->predict(lFeatures, Mat(), true);

    // Convert prediction to probabilty
    probs = 1 / (1 + exp(params.laserA * probs + params.laserB));
    //laserProb.push_back(pred);
}

void detector::classifyCamera(geometry_msgs::Point32 &cog, float &prob) {
    // Convert the cog to image coordinates
    projectPoint(cog, prPixel, K, D, transform);

    // Get the box size and corners
    getBox(cog, prPixel, rect, params.m_to_pixels, params.body_ratio);

    // Extract the crop from the image
    getCrop(crop, cv_ptr->image, rect);
    //printf("x= %d \n y= %d \n width= %d \n height= %d \n", rect.x, rect.y, rect.width, rect.height);
    //printf("pt2D: \n x= %lf y= %lf \n z= %lf \n", cog.x,  cog.y, cog.z);
    //cv::cvtColor(src_img, mono_img, CV_RGB2GRAY);
    vector <Rect> foundM;
    vector<double> weightM;

    // We don't really care about the class so we put the threshold really low to even negative predictions
    //    hog.detectMultiScale(crop, foundM, weightM);

    //  double t = (double)cv::getTickCount();
    //  // run the detector with default parameters. to get a higher hit-rate
    //  // (and more false alarms, respectively), decrease the hitThreshold and
    //  // groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
    //  hog.detectMultiScale(crop, foundM, weightM, 0, Size(8,8), Size(0,0), 1.1, 0, false);
    //  t = (double)cv::getTickCount() - t;
    //  ROS_INFO("tdetection time = %gms", t*1000./cv::getTickFrequency());
    hog.detectMultiScale(crop, foundM, weightM, params.hog_hit_threshold,
                         cv::Size(8, 8), cv::Size(0, 0), 1.05,
                         params.hog_group_threshold, params.hog_meanshift);

    // ROS_INFO("[DETECTOR] after hog OK, weightM size %d ", (int)(foundM.size()));

    /// TODO HOW TO ACCEPT A POSITIVE (mean shift)
    if (weightM.size() > 0) {
        printf("x= %d \n y= %d \n width= %d \n height= %d \n", rect.x, rect.y, rect.width, rect.height);
        printf("pt2D: \n x= %lf y= %lf \n z= %lf \n", cog.x, cog.y, cog.z);
        ROS_INFO("[DETECTOR] after hog OK, weightM size %d , %d", (int) (weightM.size()), (int) (foundM.size()));
        prob = *max_element(weightM.begin(), weightM.end());
        ROS_INFO("[DETECTOR] max element %f", prob);
    }
}

void detector::detectFusion(vector <human_track::ClusteredScan> &clusterData, human_track::ClusterClass &detections) {
    //ROS_INFO("[DETECTOR] clusters size %d", clusterData->clusters.size());
    // Variables where the probabilities of the detectors are stored
    // If there is no fusion the probability of the laser is taken

    float laserProb;
    float cameraProb;
    float fusionProb;

    for (uint i = 0; i < clusterData.size(); i++) {
        // put the cog into the ClusterClass
        //ROS_INFO("i %d",i);

        // ONLY with camera
        laserProb = 0; //MIN_PROB;
        cameraProb = 0; //MIN_PROB;

        classifyLaser(clusterData[i].features, laserProb);

        if (laserProb < params.min_laser_prob) {
            clusterData[i].crop_projected = false;
        }

        if (clusterData[i].crop_projected == true) {
            classifyCamera(clusterData[i].cog, cameraProb);
            if (cameraProb == 0) {
                clusterData[i].crop_projected = false;
                // Take Down probability (Not possible to fusion with zero)
                fusionProb = laserProb * 0.25;
            } else {
                //Bayesian fusion
                fusionProb = (laserProb * cameraProb) /
                             ((laserProb * cameraProb) + (1.0 - laserProb) * (1.0 - cameraProb));
            }
            // fusionProb = cameraProb;
        } else {
            //Keep just the laser Prob.
            fusionProb = laserProb;
        }

        clusterData[i].detection_laser_prob = laserProb;
        clusterData[i].detection_camera_prob = cameraProb;
        clusterData[i].detection_fusion_prob = fusionProb;

        if (fusionProb > params.fusion_prob) {
            if (clusterData[i].crop_projected == true) {
                if (cameraProb > params.min_camera_prob)
                    clusterData[i].detection_label = FUSION_HUMAN;
            } else {
                clusterData[i].detection_label = LASER_HUMAN;
            }
        }
        //        ROS_INFO("[DETECTOR] cluster %d: projection %d fusion %d prob %3.3f label %d",i+1,
        //                clusterData->projected[i], clusterData->fusion[i], clusterData->detection_probs[i], clusterData->detection_labels[i]);
    }
    //    ROS_INFO("[DETECTOR] Publishing detections");
    laserProcessor->getHeader(detections.header);
    detections.clusterData = clusterData;
    detectionPublisher.publish(detections);
}

void detector::detectHumans(const Image::ConstPtr &image, const LaserScan::ConstPtr &lScan) {
    clusterData.clear();
    detections = human_track::ClusterClass();

    getTF(image, lScan);
    getImage(image);
    processLaser(lScan, clusterData);

    detectFusion(clusterData, detections);
}
