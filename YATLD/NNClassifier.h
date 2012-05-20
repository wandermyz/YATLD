#ifndef YATLD_NN_CLASSIFIER_H
#define YATLD_NN_CLASSIFIER_H

#include <vector>
#include <opencv.hpp>
#include "Settings.h"

class NNClassifier
{
private:
	std::vector<cv::Mat> positiveSamples;
	std::vector<cv::Mat> negativeSamples;

public:
	void train(const cv::Mat& patchImg, bool isPositive);
	float getRelativeSimilarity(const cv::Mat& patchImg) const;
	
	inline float getSimilarity(const cv::Mat& patch, const cv::Mat ref) const
	{
#ifdef DEBUG
		assert(patch.size() == ref.size());
#endif

		cv::Mat res;
		cv::matchTemplate(patch, ref, res, CV_TM_CCOEFF_NORMED);

#ifdef DEBUG
		assert(res.rows == 1 && res.cols == 1);
#endif

		return 0.5f * (res.at<float>(0) + 1);
	}

	inline bool acceptPatch(const cv::Mat& patch) const
	{
		return getRelativeSimilarity(patch) > NN_THRESHOLD;
	}
};

#endif