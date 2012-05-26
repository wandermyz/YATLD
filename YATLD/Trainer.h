#ifndef YATLD_TRAINER_H_
#define YATLD_TRAINER_H_

#include <opencv.hpp>
#include "Detector.h"
#include "BoundingBox.h"
#include "Tracker.h"
#include <vector>	
#include <queue>

class Trainer
{
private:
	Detector& detector;
	Tracker& tracker;
	BoundingBox result;
	cv::RNG rng;
	cv::Mat frame;

	cv::PatchGenerator generator;

	bool reliable;
	bool resultFound;

	//parameters
	float shiftChange, scaleChange, rotationChange, gaussianSigma;
	int synthesisNum, positiveNum, negativeNum;

	//containers
	std::vector<const BoundingBox*> positivePatches;
	std::vector<const BoundingBox*> negativePatches;

	void combine();
	void generatePatches();
	//void learnByPExpert(const cv::Mat& frame, bool init);
	//void learnByNExpert(const cv::Mat& frame, bool init);
	void trainEnsemble(bool init);
	void trainNN(bool init);
	static bool compareOverlap(const BoundingBox* bb1, const BoundingBox* bb2);

public:
	Trainer(Detector& detector, Tracker& tracker);

	void init(const cv::Mat& frame, const BoundingBox& boundingBox);
	void update(const cv::Mat& frame);

	inline const BoundingBox& getResult() const
	{
		return result;
	}

	inline bool isResultFound() const
	{
		return resultFound;
	}
};

#endif