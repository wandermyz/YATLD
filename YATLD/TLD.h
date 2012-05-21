#ifndef YATLD_TLD_H_
#define YATLD_TLD_H_

#include <opencv.hpp>
#include <string>
#include "Detector.h"
#include "Trainer.h"
#include "BoundingBox.h"
#include "Tracker.h"

class TLD
{
private:
	cv::Mat frame;
	cv::Mat outputFrame;
	BoundingBox boundingBox;

	Detector detector;
	Tracker tracker;
	Trainer trainer;

public:
	TLD();

	void init(const cv::Mat& frame, const BoundingBox& boundingBox, cv::Mat& outputFrame);
	void update(const cv::Mat& frame, cv::Mat& outputFrame);

	inline BoundingBox getBoundingBox() const
	{
		return boundingBox;
	}
};

#endif
