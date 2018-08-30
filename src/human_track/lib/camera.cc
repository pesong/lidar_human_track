#include "camera.h"

cv::Mat& cameraFrame(){static cv::Mat v; return v;}
int& frameHeight(){static int v;return v;}
int& frameWidth(){static int v;return v;}
int& frameDiag(){static int v;return v;}


Eigen::MatrixXd ypr2R(Eigen::MatrixXd ypr)
{
	//std::cout << "in ypr2R" << std::endl;
	//typedef typename Derived::Scalar Scalar_t;

	double y = ypr(0,0) / 180.0 * M_PI;
	double p = ypr(0,1) / 180.0 * M_PI;
	double r = ypr(0,2) / 180.0 * M_PI;
	//std::cout << "y: " << y << std::endl;

	Eigen::MatrixXd Rz(3,3);
	//Eigen::Matrix<Scalar_t, 3, 3> Rz;
	Rz << cos(y), -sin(y), 0,
			sin(y), cos(y), 0,
			0, 0, 1;
	//std::cout << "Rz: " << Rz << std::endl;
	Eigen::MatrixXd Ry(3,3);
	//Eigen::Matrix<Scalar_t, 3, 3> Ry;
	Ry << cos(p), 0., sin(p),
			0., 1., 0.,
			-sin(p), 0., cos(p);

	Eigen::MatrixXd Rx(3,3);
	//Eigen::Matrix<Scalar_t, 3, 3> Rx;
	Rx << 1., 0., 0.,
			0., cos(r), -sin(r),
			0., sin(r), cos(r);

	return Rz * Ry * Rx;
}

Eigen::MatrixXd Mat2Matrix(cv::Mat M)
{
	Eigen::MatrixXd R(M.rows, M.cols);
	for(int i=0; i<M.rows; i++)
		for(int j=0; j<M.rows; j++)
		{
			R(i,j) = M.at<double>(i, j);
		}
	return R;
}

//将ros发过来的摄像头数据转换成opencv里的mat,并存储到全局变量里
void cameraCallback(const sensor_msgs::Image::ConstPtr &image)
{
	cv_bridge::CvImagePtr cv_ptr;
	cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::RGB8);
	//cv::Mat temp=cv_ptr->image.clone();
	cv::Mat temp;
	cv::GaussianBlur(cv_ptr->image, temp, cv::Size(3,3), 0.1);
	cvtColor(temp,temp, CV_BGR2RGB);
    cv::Mat image0 = temp;
    // IplImage* iplimg;
    // *iplimg = IplImage(image0);
    IplImage copy = image0;
    IplImage *frame = &copy;
    //cvFlip(frame, NULL, 0); //翻转
    temp = cv::cvarrToMat(frame, true);

	cameraFrame() = temp.clone();
	frameWidth() = temp.cols;
	frameHeight() = temp.rows;
	frameDiag() = (int)sqrt(frameWidth()*frameWidth()+frameHeight()*frameHeight());
	//cv::imshow("wwww",temp);
	//cv::waitKey(10);
}

bool adjustOutBox(cv::Rect &rect, int cols, int rows)
{
	bool ret = true; //if rect is not out of bound , return true
	int x = rect.x;
	int y = rect.y;
	if(rect.width <0)
	{
		ret = false;
		rect.width = 2;
	}
	if(rect.height <0)
	{
		ret = false;
		rect.height = 2;
	}

	int x_r = x + rect.width - 1;
	int y_b = y + rect.height - 1;
	if (x < 2)
	{
		ret = false;
		x = 2;
	}

	if (y < 2)
	{
		ret = false;
		y = 2;
	}
	if (x_r >= cols - 1)
	{
		ret = false;
		x_r = cols - 2;
	}
	if (y_b >= rows - 1)
	{
		ret = false;
		y_b = rows - 2;
	}
	if (ret)
	{
		return ret;
	}
	else
	{
		int w,h;
		if(x_r+1-x < 0)
			w = 1;
		else
			w = x_r+1-x;
		if(y_b+1-y < 0)
			h = 1;
		else
			h = y_b+1-y;
		rect = cv::Rect(x,y,w,h);
		//std::cout << rect << "/t adjust" << std::endl;
		return ret;
	}
}

//将真实世界的坐标转换成摄像头程像平面的坐标，并将人物在摄像头上框出来
bool projectBoxInCamera(Human &human) {

	//std::cout << "projectInCamera" << std::endl;
	double pose_cos = icpPoseCos();
	double pose_sin = icpPoseSin();
	double pose_x = amclPoseX();
	double pose_y = amclPoseY();
	int cols = frameWidth();
	int rows = frameHeight();
//
	cv::Rect rect;

	//转到base_link坐标系
	auto theworld = [&](double xi, double yi, double &xo, double &yo) {
		xo = pose_cos * (xi - pose_x) + pose_sin * (yi - pose_y);
		yo = pose_cos * (yi - pose_y) - pose_sin * (xi - pose_x);
	};

//	auto theworld = [&](double xi, double yi, double &xo, double &yo) {
//		xo = cos(robotYaw())*xi-sin(robotYaw())*yi+robotPoseX();
//		yo = cos(robotYaw())*yi+sin(robotYaw())*xi+robotPoseY();
//	};
	cv::Mat in(1, 3, CV_64FC1);
	cv::Mat out(1, 3, CV_64FC1);
	cv::Mat R = cv::Mat::zeros(3, 1, CV_64FC1); //rotation raw,pitch,yaw
	cv::Mat T = cv::Mat::zeros(3, 1, CV_64FC1); //transform
	cv::Mat K = cv::Mat::zeros(3, 3, CV_64FC1); //camera intristic matrix
	cv::Mat D = cv::Mat::zeros(1, 5, CV_64FC1); //distortion

	double x_l, y_l;
	//theworld(human.getCurrentX(), human.getCurrentY(), x_l, y_l);
	std::vector<int> projectxs, projectys;

//	std::cout << "getLaser: " << std::endl;
//	std::cout << human.getLaser().size() << std::endl;
//	for (int i =0; i < human.getLaser().size(); i++ )
//	{
//		std::cout << human.getLaser()[i].first << " ";
//	}
//	std::cout << std::endl;

	ros::NodeHandle nhh_("~");
	double cam_pos, fx, fy, cx, cy, roll, pitch, yaw, T1, T2, T3;
	nhh_.param("/human_track/camera/camera_position", cam_pos, 0.0);
	nhh_.param("/human_track/camera/fx", fx, 0.0);
	nhh_.param("/human_track/camera/fy", fy, 0.0);
	nhh_.param("/human_track/camera/cx", cx, 0.0);
	nhh_.param("/human_track/camera/cy", cy, 0.0);
	nhh_.param("/human_track/camera/roll", roll, 0.0);
	nhh_.param("/human_track/camera/pitch", pitch, 0.0);
	nhh_.param("/human_track/camera/yaw", yaw, 0.0);
	nhh_.param("/human_track/camera/T1", T1, 0.0);
	nhh_.param("/human_track/camera/T2", T2, 0.0);
	nhh_.param("/human_track/camera/T3", T3, 0.0);


	T.at<double>(0, 0) = T1;
	T.at<double>(0, 1) = T2;
	T.at<double>(0, 2) = T3; //0.47
	R.at<double>(2) = roll;  //0.1 //roll
	R.at<double>(1) = yaw;  //0.05 //yaw
	R.at<double>(0) = pitch;  //0.1 //pitch
	K.at<double>(0, 0) = fx;
	K.at<double>(1, 1) = fy;
	K.at<double>(0, 2) = cx;//320.279; 160.13
	K.at<double>(1, 2) = cy;//245.871; 122.93
	K.at<double>(2, 2) = 1;

	//std::cout << "start" << std::endl;
	//std::cout << human.getLaser().size() << std::endl;



	double x, y;
	for (int i = 0; i < human.getLaser().size(); i++) {
		x = human.getLaser()[i].first;
		y = human.getLaser()[i].second;
		//std::cout << human.getLaser()[i].second << " ";
		theworld(x, y, x_l, y_l);
		//脚的点
		in.at<double>(0) = -y_l;
		in.at<double>(1) = cam_pos;
		in.at<double>(2) = x_l;

		//将真实世界的坐标转换成摄像头程像平面的坐标 //投影到画面上这部还可优化
		cv::projectPoints(in, R, T, K, D, out);
		int projectx, projecty;
		projectx = (int) out.at<double>(0, 0);
		projecty = (int) out.at<double>(0, 1);
		//std::cout << projecty << " ";
		//std::cout << " " << projectx ;
		projectxs.push_back(projectx);
		projectys.push_back(projecty);
	}
	//std::cout << std::endl;

	//std::cout << "projectxs.size: " << projectxs.size() << std::endl;
//	std::cout << "before project" << std::endl;


	/*--------------------------调参时打开--------------------------------------------------*

double x,y;
for (int i =0; i < human.getLaser().size(); i++ ) {
    x = human.getLaser()[i].first;
    y = human.getLaser()[i].second;
    std::cout << human.getLaser()[i].second << " ";
    theworld(x, y, x_l, y_l);
    //脚的点
    in.at<double>(0) = -y_l; //Z， z=-y
    in.at<double>(1) = 0.6; //Y y=-z
    in.at<double>(2) = x_l; //X

    //将真实世界的坐标转换成摄像头程像平面的坐标 //投影到画面上这部还可优化
    cv::projectPoints(in, R, T, K, D, out);
    int projectx, projecty;
    projectx = (int) out.at<double>(0, 0);
    projecty = (int) out.at<double>(0, 1);
    std::cout << projecty << " ";
    //std::cout << " " << projectx ;
    projectxs.push_back(projectx);
    projectys.push_back(projecty);
}
std::cout << std::endl;

//std::cout << "projectxs.size: " << projectxs.size() << std::endl;
//	std::cout << "before project" << std::endl;
if (projectxs.size()>0)
{
    int it_max_x = *max_element(projectxs.begin(), projectxs.end());
    //std::cout << "it_max_x: " << it_max_x << std::endl;
    int it_min_x = *min_element(projectxs.begin(), projectxs.end());
    int it_max_y = *max_element(projectys.begin(), projectys.end());
    int it_min_y = *min_element(projectys.begin(), projectys.end());
    rect.x = it_min_x;
    rect.y = (it_min_y+it_max_y)*0.5;
    std::cout << " it_min_y: " << it_min_y << " it_max_y: " << it_max_y << std::endl;
    std::cout << " rect.y:" << rect.y << std::endl;
    rect.width = it_max_x - it_min_x;
    rect.height = 30;
}
else
{

    theworld(human.getCurrentX(), human.getCurrentY(), x_l, y_l);
    in.at<double>(0) = -y_l;
    in.at<double>(1) = 0.6;
    in.at<double>(2) = x_l;

    //将真实世界的坐标转换成摄像头程像平面的坐标 //投影到画面上这部还可优化
    cv::projectPoints(in, R, T, K, D, out);
    int projectx, projecty;
    int centerx = (int) out.at<double>(0, 0);
    int centery = (int) out.at<double>(0, 1);
    std::cout << "current x: " << human.getCurrentX() << std::endl;
    std::cout << "current y: " << human.getCurrentY() << std::endl;
    std::cout << "correct x: " << centerx << std::endl;
    std::cout << "correct y: " << centery << std::endl;


    int width = 30;
    int height = 30;
    rect.x = centerx - width / 2;
    rect.y = centery - height / 2;
    rect.width = width;
    rect.height = height;
}


/*----------------------------------------------------------------------------*/





	if (projectxs.size() > 0) {
		int it_max_x = *max_element(projectxs.begin(), projectxs.end());
		//std::cout << "it_max_x: " << it_max_x << std::endl;
		int it_min_x = *min_element(projectxs.begin(), projectxs.end());
		int it_max_y = *max_element(projectys.begin(), projectys.end());
		int it_min_y = *min_element(projectys.begin(), projectys.end());



//	rect.x = it_min_x;
//	rect.y = it_min_y;
//	rect.width = it_max_x - it_min_x;
//	//std::cout << " it_min_x: " << rect.x << " it_min_y: " << rect.y << " width: " << rect.width << std::endl;
//	rect.height = 30;



		//auto it = max_element(std::begin(cloud), std::end(cloud));

//    if (x > 32 && x < cols - 32 && y > rows - 12) y = rows - 15;
//	if (!(x > 32 && x < cols - 32 && y > 0 && y < rows - 12))
//	{
//		return false;
//	}
		double distance = sqrt(x_l * x_l + y_l * y_l);
		//ROS_INFO("x: %d, y: %d, cols: %d, rows: %d, distance: %f", x, y,cols - 32,rows -12, distance);
		int width, height;
		//fixed



		height = 1080 / distance;
		//std::cout << "height: " << height << std::endl;
		rect.x = it_min_x;
		rect.y = 0.5*(it_min_y+it_max_y) - height;
		//rect.width = it_max_x - it_min_x;
		rect.width = 1080 / (2.4 * distance);
		//rect.width = width;
		rect.height = height;


		if(adjustOutBox(rect, rows, cols))
		{
			if (rect.y < 0) {
				rect.y = 10;
				rect.height = rows - 21;
			}
			//std::cout << rect << "/t update Box" << std::endl;
			human.updateBoundingBox(rect);
		}
		else
			return false;


	}
	// minimum HOG size: 64*128
//	if (distance < 1)
//    {
//        std::cout << "**** Reason Two ****" << std::endl;
//        return false;
//    }

	if (rect.x < 10 || rect.y < 10 || (rect.x + rect.width) > (cols-10) || (rect.y + rect.height) > (rows-10)) //|| rect.width < 75 || rect.height < 150
	{
//        std::cout << "**** Reason Three ****" << std::endl;
//        std::cout << "rect.x " << rect.x << " rect.y: " << rect.y << " rect.width "  << rect.width << " rect.height "  << rect.height << std::endl;
		return false;
	}

	return true;
}

std::pair<double,double> ReProject(cv::Rect A, Human hum) {

//	float z_h = 1080/A.height;
//	float z_w = (1080/2.4)/A.width;
//	float z;
//	z = z_w;
	double pose_x = amclPoseX();
	double pose_y = amclPoseY();
	double amcl_pos_sin = amclPoseSin();
	double amcl_pos_cos = amclPoseCos();
	//base_link转到amcl坐标系
	auto theAmcl = [&](double xi, double yi, double &xo, double &yo) {
		xo = amcl_pos_cos * xi - amcl_pos_sin * yi + pose_x;
		yo = amcl_pos_cos * yi + amcl_pos_sin * xi + pose_y;
	};

	auto theBaseLink = [&](double xi, double yi, double &xo, double &yo) {
		xo = amcl_pos_cos * (xi - pose_x) + amcl_pos_sin * (yi - pose_y);
		yo = amcl_pos_cos * (yi - pose_y) - amcl_pos_sin * (xi - pose_x);
	};

	double xoo,yoo;
	theBaseLink(hum.getAmclX(), hum.getAmclY(),xoo,yoo);
	//std::cout << " xoo: " << xoo << " yoo: " << yoo << std::endl;
	//float z = sqrt(xoo*xoo+yoo*yoo);
	float z = xoo;
	//std::cout << " z: " << z << std::endl;

	ros::NodeHandle nhh_("~");
	double cam_pos, fx, fy, cx, cy, roll, pitch, yaw, T1, T2, T3;
	nhh_.param("/human_track/camera/camera_position", cam_pos, 0.0);
	nhh_.param("/human_track/camera/fx", fx, 0.0);
	nhh_.param("/human_track/camera/fy", fy, 0.0);
	nhh_.param("/human_track/camera/cx", cx, 0.0);
	nhh_.param("/human_track/camera/cy", cy, 0.0);
	nhh_.param("/human_track/camera/roll", roll, 0.0);
	nhh_.param("/human_track/camera/pitch", pitch, 0.0);
	nhh_.param("/human_track/camera/yaw", yaw, 0.0);
	nhh_.param("/human_track/camera/T1", T1, 0.0);
	nhh_.param("/human_track/camera/T2", T2, 0.0);
	nhh_.param("/human_track/camera/T3", T3, 0.0);


	Eigen::MatrixXd T(3, 1); //transform
	Eigen::MatrixXd K(3, 3); //camera intristic matrix
	//std::cout << "0001" <<std::endl;
	T(0, 0) = T1;
	T(1, 0) = T2;
	T(2, 0) = T3;

	K(0, 0) = fx;
	K(1, 1) = fy;
	K(0, 2) = cx;//320.279; 160.13
	K(1, 2) = cy;//245.871; 122.93
	K(2, 2) = 1;

	float u = A.x+0.5*A.width;
	float v = A.y+A.height;
	float x1 = (u-K(0,2))/K(0,0);
	float y1 = (v-K(1,2))/K(1,1);
	float x = x1*z;
	float y = y1*z;

	//std::cout << "2222" <<std::endl;


	Eigen::MatrixXd p(3, 1); //point in camera frame
	Eigen::MatrixXd P(3, 1); //point in camera frame

	p(0, 0) = x;
	p(1, 0) = y;
	p(2, 0) = z;



	cv::Mat R = cv::Mat::zeros(3, 1, CV_64FC1); //rotation raw,pitch,yaw
	R.at<double>(2) = roll;  //0.1 //roll
	R.at<double>(1) = yaw;  //0.05 //yaw
	R.at<double>(0) = pitch;  //0.1 //pitch
	cv::Mat RR_M = cv::Mat::zeros(3, 3, CV_64FC1);
	cv::Rodrigues(R, RR_M);

	Eigen::MatrixXd RR = Mat2Matrix(RR_M);

	//RR = ypr2R(ypr);

	P = RR.inverse()*(p-T);


	double xi,yi,xo,yo;
	xi = P(2, 0);
	yi = P(0, 0)*(-1);
	//std::cout << "xi: " << xi << " yi: " << yi << std::endl;
	theAmcl(xi,yi,xo,yo);

	std::pair<double,double> point = std::make_pair(xo,yo);

	return point;
	// std::cout << "6666" << std::endl;
}


std::pair<double,double> ReProject(cv::Rect A)
{

//	float z_h = 1080/A.height;
	float z = (1080/2.4)/A.width;
//	float z;
//	z = z_w;
    double pose_x = amclPoseX();
    double pose_y = amclPoseY();
    double amcl_pos_sin = amclPoseSin();
    double amcl_pos_cos = amclPoseCos();
    //base_link转到amcl坐标系
    auto theAmcl = [&](double xi, double yi, double &xo, double &yo) {
        xo = amcl_pos_cos * xi - amcl_pos_sin * yi + pose_x;
        yo = amcl_pos_cos * yi + amcl_pos_sin * xi + pose_y;
    };

	//std::cout << " z: " << z << std::endl;

	Eigen::MatrixXd T(3, 1); //transform
	Eigen::MatrixXd K(3, 3); //camera intristic matrix
	//std::cout << "0001" <<std::endl;
	T(0, 0) = 0;
	T(1, 0) = 0;
	T(2, 0) = 0;

	K(0, 0) = 476.398;
	K(1, 1) = 476.397;
	K(0, 2) = 320.279;//320.279; 160.13
	K(1, 2) = 245.871;//245.871; 122.93
	K(2, 2) = 1;

	float u = A.x+0.5*A.width;
	float v = A.y+A.height;
	float x1 = (u-K(0,2))/K(0,0);
	float y1 = (v-K(1,2))/K(1,1);
	float x = x1*z;
	float y = y1*z;

	//std::cout << "2222" <<std::endl;


	Eigen::MatrixXd p(3, 1); //point in camera frame
	Eigen::MatrixXd P(3, 1); //point in camera frame

	p(0, 0) = x;
	p(1, 0) = y;
	p(2, 0) = z;



	cv::Mat R = cv::Mat::zeros(3, 1, CV_64FC1); //rotation raw,pitch,yaw
	R.at<double>(2) = 0.07;  //0.1 //roll
	R.at<double>(1) = -0.22;  //0.05 //yaw
	R.at<double>(0) = -0.05;  //0.1 //pitch
	cv::Mat RR_M = cv::Mat::zeros(3, 3, CV_64FC1);
	cv::Rodrigues(R, RR_M);

	Eigen::MatrixXd RR = Mat2Matrix(RR_M);
	//std::cout << "3333" << std::endl;

	//RR = ypr2R(ypr);

	P = RR.inverse()*(p-T);



//	in.at<double>(0) = -y_l; //Z， z=-y
//	in.at<double>(1) = 0.8; //Y y=-z
//	in.at<double>(2) = x_l; //X

	double xi,yi,xo,yo;
	xi = P(2, 0);
	yi = P(0, 0)*(-1);
	//std::cout << "xi: " << xi << " yi: " << yi << std::endl;
	theAmcl(xi,yi,xo,yo);

	std::pair<double,double> point = std::make_pair(xo,yo);

	return point;
};

