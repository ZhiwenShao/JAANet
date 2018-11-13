#include <iostream>
#include <fstream>

//#include <time.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"
#include <vector>


const double pi = CV_PI;
const int landmark_size = 49;
const int img_size = 200;
//control the size of cropped bounding box
const double box_enlarge = 2.9;

using namespace std;


int main()
{
	ifstream input_path, input_land;
	ofstream output_path, output_land;
	string path;

	//without the postfix like .jpg, .png
	input_path.open("BP4D_all_path.txt");
	input_land.open("BP4D_all_land.txt");

	output_path.open("BP4D_output_path.txt");
	output_land.open("BP4D_output_land.txt");

	double land_x, land_y;

	std::vector<float> landmarks;

	double xMin, xMax, yMin, yMax, cx, cy, halfSize, scale;
	float deltaX, deltaY, l, sinVal,cosVal;

	const double half_img_size = (img_size - 1) / 2.0;

	cv::Mat img, aligned, ptsMat_3d;

	int count = 0;

	while (getline(input_path, path))//read each line
	{
		landmarks.clear();

		for (int j = 0; j < landmark_size; j++)
		{
			input_land >> land_x >> land_y;
			landmarks.push_back(land_x-1);
			landmarks.push_back(land_y-1);					
		}

		count ++;

		img = cv::imread(("BP4D/" + path + ".jpg").data(), CV_LOAD_IMAGE_COLOR);

		cv::Mat ptsMat(landmarks.size() / 2, 2, CV_32F, landmarks.data());

		cv::Mat leftEyeSet = ptsMat.rowRange(19, 25);
		cv::Mat rightEyeSet = ptsMat.rowRange(25, 31);
		float leftEye[2]={(cv::mean(leftEyeSet.colRange(0,1))).val[0], (cv::mean(leftEyeSet.colRange(1,2))).val[0]};
		float rightEye[2]={(cv::mean(rightEyeSet.colRange(0,1))).val[0], (cv::mean(rightEyeSet.colRange(1,2))).val[0]};

		deltaX = (rightEye[0] - leftEye[0]);
		deltaY = (rightEye[1] - leftEye[1]);
		l = sqrtf(deltaX * deltaX
			+ deltaY * deltaY);
		sinVal = deltaY / l;
		cosVal = deltaX / l;
		float mdataVeri1[9] = {
			cosVal, sinVal, 0, //-(float)cx * cosVal - (float)cy * sinVal,
			-sinVal, cosVal, 0, //(float)cx * sinVal - (float)cy * cosVal,
			0, 0, 1
		};
		cv::Mat trans1(3, 3, CV_32F, mdataVeri1);
		cv::Mat tranPts = cv::Mat::ones(5, 3, CV_32F);

		float landmarks_5p[10] = {
			leftEye[0], leftEye[1], rightEye[0], rightEye[1], landmarks[2*14-2], landmarks[2*14-1], landmarks[2*32-2], landmarks[2*32-1], landmarks[2*38-2], landmarks[2*38-1]
		};
		cv::Mat ptsMat_5p(5, 2, CV_32F, landmarks_5p);

		ptsMat_5p.copyTo(tranPts(cv::Rect(0, 0, 2, 5)));
		tranPts = (trans1 * tranPts.t()).t();

		cv::minMaxIdx(tranPts.col(0), &xMin, &xMax);
		cv::minMaxIdx(tranPts.col(1), &yMin, &yMax);

		cx = (xMin + xMax) * 0.5f;
		cy = (yMin + yMax) * 0.5f;
		halfSize = 0.5 * box_enlarge
			* ((xMax - xMin > yMax - yMin) ? (xMax - xMin) : (yMax - yMin));
		scale = half_img_size / halfSize;
		float mdataVeri2[9] = {
			scale, 0, scale * (halfSize - cx),
			0, scale, scale * (halfSize - cy),
			0, 0, 1
		};
		cv::Mat trans2(3, 3, CV_32F, mdataVeri2);
		cv::Mat trans = trans2 * trans1;
		cv::warpAffine(img, aligned, trans.rowRange(0, 2),
			cv::Size(img_size, img_size),
			cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(128,128,128));

		ptsMat_3d = cv::Mat::ones(landmarks.size() / 2, 3, CV_32F);
		ptsMat.copyTo(ptsMat_3d(cv::Rect(0, 0, 2, landmarks.size() / 2)));

		ptsMat_3d = (trans * ptsMat_3d.t()).t();			


		for (int r = 0; r < ptsMat_3d.rows; r++)
		{
			for (int c = 0; c < 2; c++)
			{
				output_land << ptsMat_3d.at<float>(r, c) << '\t';
			}
		}
		output_land << "\n";

		string newpath("BP4D_aligned/" + path + ".jpg");
		output_path << newpath << endl;
		cv::imwrite(newpath.data(), aligned);//jpg, default quality: 95

		std::cout << count << '\n';
	}

	input_path.close();
	input_land.close();
	output_path.close();
	output_land.close();

	return 0;
}
