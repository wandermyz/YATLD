#ifndef YATLD_TRACKER_H
#define YATLD_TRACKER_H

#include <opencv.hpp>
#include "BoundingBox.h"
#include <vector>
#include <algorithm>
#include "Detector.h"

class Tracker
{
private:
	Detector& detector;

	cv::Mat prevFrame;	//deep copy
	BoundingBox prevBoundingBox;
	bool tracked;

	std::vector<cv::Point2f> prevPoints;
	std::vector<cv::Point2f> nextPoints;
	std::vector<cv::Point2f> backwardPoints;
	std::vector<uchar> status;
	std::vector<float> errors;
	std::vector<uchar> backwardStatus;
	std::vector<float> backwardErrors;
	std::vector<float> ncc;
	std::vector<float> fbErrors;
	std::vector<float> xOffsets, yOffsets, sqScales, displacement, residual;
	int nGoodPoints;	//good points will be moved to the first nGoodPoints elements in prevPoints and nextPoints

	cv::Size lkWindowSize;
	cv::TermCriteria lkTermCreteria;

	inline float median(std::vector<float> elements)
	{
		return median(elements, elements.size());
	}

	inline float median(std::vector<float> elements, int num)	//deep copy for n_elements
	{
#ifdef DEBUG
		assert(num <= elements.size());
#endif

		int n = (int)(num / 2);
		std::nth_element(elements.begin(), elements.begin() + n, elements.begin() + num);
		return elements[n];
	}

public:
	Tracker(Detector& detector);

	void init(const cv::Mat& frame, const BoundingBox& initBoundingBox);
	void update(const cv::Mat& frame, cv::Mat& outputFrame);

	inline const BoundingBox getBoundingBox() const
	{
		return prevBoundingBox;
	}

	inline bool isTracked() const
	{
		return tracked;
	}
};

#endif