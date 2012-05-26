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

	//const BoundingBox* finalBoundingBox;	
	std::vector<const BoundingBox*> detectedBoundingBoxes;
	std::vector<BoundingBox> clusteredBoundingBoxes;	//clusterred bounding boxes

	void generateScanGrids(const BoundingBox& initBoundingBox);
	void cluster();
	static bool isIdenticalBoundingBox(const BoundingBox* bb1, const BoundingBox* bb2);

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

	/*inline const BoundingBox* getBoundingBox()
	{
		return finalBoundingBox;
	}*/

	inline const std::vector<const BoundingBox*>& getDetectedBoundingBoxes() const
	{
		return detectedBoundingBoxes;
	}

	inline const std::vector<BoundingBox>& getClusteredBoundingBoxes() const
	{
		return clusteredBoundingBoxes;
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

	inline bool isDetected() const
	{
		return !detectedBoundingBoxes.empty();
	}
};

#endif