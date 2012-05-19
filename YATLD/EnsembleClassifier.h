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

	void init();

	inline void update(const cv::Mat& frame)
	{
		this->frame = frame;
		GaussianBlur(frame, frameBlurred, cv::Size(0, 0), FERN_GAUSSIAN_SIGMA);
	}

	void train(const BoundingBox& patch, bool isPositive);
	bool accept(const BoundingBox& patch) const;
};



#endif