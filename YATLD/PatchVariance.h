#ifndef YATLD_PATCH_VARIANCE_H_
#define YATLD_PATCH_VARIANCE_H_

#include <opencv.hpp>
#include "BoundingBox.h"

class PatchVariance
{
private:
	cv::Mat integralImg;
	cv::Mat sqIntegralImg;	
	double initVariance;

	double computeVariance(const BoundingBox& patch);

public:
	void init(const cv::Mat& frame, const BoundingBox& initBoundingBox);
	
	inline void update(const cv::Mat& frame)
	{
		integral(frame, integralImg, sqIntegralImg, CV_64F);
	}

	inline bool acceptPatch(const BoundingBox patch)
	{
		return computeVariance(patch) > 0.5 * initVariance;
	}
};

#endif