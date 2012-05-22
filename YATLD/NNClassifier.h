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
	NNClassifier();
	void train(const cv::Mat& patchImg, bool isPositive);
	void getSimilarity(const cv::Mat& patchImg, float* relative, float* conservative) const;
	void forgetPositive(int count);
	void forgetNegative(int count);

	inline float getPairSimilarity(const cv::Mat& patch, const cv::Mat ref) const
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

	inline int getPositiveNum() const
	{
		return positiveSamples.size();
	}

	inline int getNegativeNum() const
	{
		return negativeSamples.size();
	}
};

#endif