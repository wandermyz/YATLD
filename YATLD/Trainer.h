#ifndef YATLD_TRAINER_H_
#define YATLD_TRAINER_H_

#include <opencv.hpp>
#include "Detector.h"
#include "BoundingBox.h"
#include "Tracker.h"

class Trainer
{
private:
	Detector& detector;
	Tracker& tracker;

public:
	Trainer(Detector& detector, Tracker& tracker);

	void init(const cv::Mat& frame, const BoundingBox& boundingBox);
	void update(const cv::Mat& frame);
};

#endif