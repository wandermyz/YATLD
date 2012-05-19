#ifndef YATLD_TRAINER_H_
#define YATLD_TRAINER_H_

#include <opencv.hpp>
#include "Detector.h"
#include "BoundingBox.h"

class Trainer
{
private:
	Detector& detector;

public:
	Trainer(Detector& detector);

	void init(const cv::Mat& frame, const BoundingBox& boundingBox);

};

#endif