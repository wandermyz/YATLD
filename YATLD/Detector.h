#ifndef YATLD_DETECTOR_H_
#define YATLD_DETECTOR_H_

#include <opencv.hpp>
#include <vector>
#include "Settings.h"
#include "PatchVariance.h"
#include "EnsembleClassifier.h"
#include "NNClassifier.h"
#include "BoundingBox.h"

class Detector
{
private:
	cv::Mat frame;
	cv::Mat outputFrame;
	std::vector<BoundingBox> scanGrids;
	
	PatchVariance patchVariance;
	EnsembleClassifier ensembleClassifier;
	NNClassifier nnClassifier;

	const BoundingBox* finalBoundingBox;	//TODO: clustering

	void generateScanGrids(const BoundingBox& initBoundingBox);

public:
	Detector();

	void init(const cv::Mat& frame, const BoundingBox& boundingBox, cv::Mat& outputFrame);
	void update(const cv::Mat& frame, cv::Mat& outputFrame);

	inline EnsembleClassifier& getEnsembleClassifier()
	{
		return ensembleClassifier;
	}

	inline const std::vector<BoundingBox>& getScanGrids()
	{
		return scanGrids;
	}

	inline NNClassifier& getNNClassifier()
	{
		return nnClassifier;
	}

	inline const BoundingBox* getBoundingBox()
	{
		return finalBoundingBox;
	}

	inline void refreshGridOverlap(const BoundingBox& ref)
	{
		for (std::vector<BoundingBox>::iterator it = scanGrids.begin(); it != scanGrids.end(); ++it)
		{
			it->refreshOverlap(ref);
		}
	}

	inline const PatchVariance& getPatchVariance() const
	{
		return patchVariance;
	}
};

#endif