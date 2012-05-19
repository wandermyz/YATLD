#include "PatchVariance.h"
#include <math.h>

using namespace cv;

void PatchVariance::init(const Mat& frame, const BoundingBox& initBoundingBox)
{
	//compute initial variance
	integral(frame, integralImg, sqIntegralImg, CV_64F);
	initVariance = computeVariance(initBoundingBox);
}

double PatchVariance::computeVariance(const BoundingBox& patch)
{
	double sum = integralImg.at<double>(patch.br()) 
				- integralImg.at<double>(patch.br().y, patch.x) 
				- integralImg.at<double>(patch.y, patch.br().x) 
				+ integralImg.at<double>(patch.tl());
	double sumSq = sqIntegralImg.at<double>(patch.br()) 
					- sqIntegralImg.at<double>(patch.br().y, patch.x) 
					- sqIntegralImg.at<double>(patch.y, patch.br().x) 
					+ sqIntegralImg.at<double>(patch.tl());
	return sumSq / patch.area() - pow(sum / patch.area(), 2); 
}