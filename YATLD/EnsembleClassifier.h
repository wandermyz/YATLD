#ifndef YATLD_ENSEMBLE_CLASSIFIER_H
#define YATLD_ENSEMBLE_CLASSIFIER_H

#include <opencv.hpp>
#include "Settings.h"
#include "PixelComparator.h"
#include "BoundingBox.h"

class EnsembleClassifier
{
private:
	PixelComparator* comparators;
	cv::Mat frame;
	cv::Mat frameBlurred;

public:
	EnsembleClassifier();
	virtual ~EnsembleClassifier();

	void init(const cv::Mat& frame);

	inline void update(const cv::Mat& frame)
	{
		this->frame = frame;
		GaussianBlur(frame, frameBlurred, cv::Size(0, 0), FERN_GAUSSIAN_SIGMA);
	}

	void train(const cv::Mat& patchImg, bool isPositive);	//patchImg should be blurred outside
	inline float getPosterior(const BoundingBox& patch) const
	{
		return getPosterior(frameBlurred(patch));
	}
	float getPosterior(const cv::Mat& patchImg) const;

	inline bool acceptPatch(const BoundingBox& patch) const
	{
		return getPosterior(patch) >= 0.5;
	}

	inline const cv::Mat& getFrameBlurred()
	{
		return frameBlurred;
	}



};



#endif