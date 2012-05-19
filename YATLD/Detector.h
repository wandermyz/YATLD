#ifndef YATLD_DETECTOR_H_
#define YATLD_DETECTOR_H_

#include <opencv.hpp>
#include <vector>
#include "Settings.h"
#include "PatchVariance.h"
#include "EnsembleClassifier.h"
#include "BoundingBox.h"

class Detector
{
private:
	cv::Mat frame;
	cv::Mat outputFrame;
	std::vector<cv::Size> gridSizes;
	
	PatchVariance patchVariance;
	EnsembleClassifier ensembleClassifier;

	void generateGridSizes(const BoundingBox& initBoundingBox);

	int stepH, stepV;

public:
	Detector();

	void init(const cv::Mat& frame, const BoundingBox& boundingBox, cv::Mat& outputFrame);
	void update(const cv::Mat& frame, cv::Mat& outputFrame);

	inline EnsembleClassifier& getEnsembleClassifier()
	{
		return ensembleClassifier;
	}
};

#endif