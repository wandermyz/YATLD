#include "Detector.h"
#include <iostream>

using namespace cv;
using namespace std;

Detector::Detector()
{
}

void Detector::init(const cv::Mat& frame, const BoundingBox& boundingBox, Mat& outputFrame)
{
	this->frame = frame;
	this->outputFrame = outputFrame;
	this->finalBoundingBox = &boundingBox;

	generateScanGrids(boundingBox);
	patchVariance.init(frame, boundingBox);

	ensembleClassifier.init(frame);
}

void Detector::update(const cv::Mat& frame, cv::Mat& outputFrame)
{
#ifdef DEBUG
	cout << "[Detector]" << endl;
#endif

	this->frame = frame;
	this->outputFrame = outputFrame;

#ifdef DEBUG
	for (std::vector<BoundingBox>::iterator it = scanGrids.begin(); it != scanGrids.end(); ++it)
	{
		it->resetStates();
	}
	int nPatches = 0, nEnsemblePatches = 0, nNNPatches = 0, nFinalPatches = 0;
#endif

	float maxNNSimilarity = 0;
	vector<BoundingBox>::const_iterator maxNNpatchIt;

	patchVariance.update(frame);
	ensembleClassifier.update(frame);

#ifdef MEASURE_TIME
	double totalTime = 0;
	double varianceTotalTime = 0;
	double ensembleTotalTime = 0;
	double nnTotalTime = 0;
#endif

	for (vector<BoundingBox>::iterator patchIt = scanGrids.begin(); patchIt != scanGrids.end(); ++patchIt)
	{
#ifdef MEASURE_TIME
		double time = 0, varianceTime = 0, ensembleTime = 0, nnTime = 0;
#endif

		do
		{
#ifdef MEASURE_TIME
			time = getTickCount();
#endif

			patchIt->state = UnknownState;

#ifdef DEBUG
			nPatches++;
			//rectangle(outputFrame, patch, Scalar(255,255,255));
#endif

#ifdef MEASURE_TIME
			varianceTime = getTickCount();
#endif
			if (!patchVariance.acceptPatch(*patchIt))
			{
				patchIt->state = RejectedByVariance;
				break;
			}

			//Ensemble classifier
#ifdef DEBUG
			nEnsemblePatches++;
			//rectangle(outputFrame, patch, Scalar(255,0,0), 2);
#endif

#ifdef MEASURE_TIME
			ensembleTime = getTickCount();
#endif
			if (!ensembleClassifier.acceptPatch(*patchIt))
			{
				patchIt->state = RejectedByEnsemble;
				break;
			}

			//NN classifier
#ifdef DEBUG
			nNNPatches++;
			//rectangle(outputFrame, *patchIt, Scalar(255,0,0), 1);
#endif

#ifdef MEASURE_TIME
			nnTime = getTickCount();
#endif
			nnClassifier.getSimilarity(frame(*patchIt), &patchIt->relativeSimilarity, &patchIt->conservativeSimilarity);
			if (patchIt->relativeSimilarity > maxNNSimilarity)
			{
				maxNNSimilarity = patchIt->relativeSimilarity;
				maxNNpatchIt = patchIt;
			}

			if (patchIt->relativeSimilarity <= NN_THRESHOLD)
			{
				patchIt->state = RejectedByNN;
				break;
			}	

			//final
			patchIt->state = DetectedAcceptedByNN;

#ifdef DEBUG
			nFinalPatches++;
#endif
		} while (false);

#ifdef MEASURE_TIME
		double endTime = getTickCount();
		varianceTotalTime += ((ensembleTime == 0 ? endTime : ensembleTime) - varianceTime) / getTickFrequency();
		ensembleTotalTime += (ensembleTime == 0 ? 0 : ((nnTime == 0 ? endTime : nnTime) - ensembleTime)) / getTickFrequency();
		nnTotalTime += (nnTime == 0 ? 0 : (endTime - varianceTime)) / getTickFrequency();
		totalTime += ((double)getTickCount() - time) / getTickFrequency();
#endif

		//rectangle(outputFrame, *patchIt, Scalar(255,0,0), 2);
	}

#ifdef MEASURE_TIME
	printf("Detector Time: %f, %f, %f, %f\n", varianceTotalTime, ensembleTotalTime, nnTotalTime, totalTime);
#endif

#ifdef DEBUG
	cout << "Total patches: " << nPatches << endl;
	cout << "After Patch variance: " << nEnsemblePatches << endl;
	cout << "After Ensemble classifier: " << nNNPatches << endl;
	cout << "After NN classifier: " << nFinalPatches << endl;
	cout << "Maximum NN Similairty: " << maxNNSimilarity << endl;
	//cout << endl;
	//rectangle(outputFrame, *maxNNpatchIt, Scalar(0, 0, 255), 2);
#endif

	finalBoundingBox = maxNNSimilarity <= NN_THRESHOLD ? NULL : &(*maxNNpatchIt);
}

void Detector::generateScanGrids(const BoundingBox& initBoundingBox)
{
	vector<Size> gridSizes;
	Size bbSize = initBoundingBox.size();
	double scale = 1.0;
	while (bbSize.width <= frame.cols && bbSize.height <= frame.rows)
	{
		gridSizes.push_back(bbSize);

		scale *= 1.2;
		bbSize = Size((int)(initBoundingBox.width * scale + 0.5), (int)(initBoundingBox.height * scale + 0.5));
	}

	scale = 1.0 / STEP_S;
	bbSize = Size((int)(initBoundingBox.width * scale + 0.5), (int)(initBoundingBox.height * scale + 0.5));
	//while (bbSize.width * bbSize.height >= MIN_BB_AREA)
	while (bbSize.width >= MIN_BB_SIDE && bbSize.height >= MIN_BB_SIDE)
	{
		gridSizes.push_back(bbSize);
		scale /= 1.2;
		bbSize = Size((int)(initBoundingBox.width * scale + 0.5), (int)(initBoundingBox.height * scale + 0.5));
	}

	//int stepH = (int)(STEP_H * frame.cols + 0.5);
	//int stepV = (int)(STEP_V * frame.rows + 0.5);

	BoundingBox patch;
	for (vector<Size>::const_iterator scaleIt = gridSizes.begin(); scaleIt != gridSizes.end(); ++scaleIt)
	{
		patch.width = scaleIt->width;
		patch.height = scaleIt->height;

		int minSide = min(scaleIt->width, scaleIt->height);
		//int stepH = (int)(STEP_H * scaleIt->width + 0.5);
		//int stepV = (int)(STEP_V * scaleIt->height + 0.5);
		int stepH = (int)(STEP_H * minSide + 0.5);
		int stepV = (int)(STEP_V * minSide + 0.5);

#ifdef DEBUG
		assert(stepH >= 1 && stepV >= 1);
#endif

		for (patch.y = 0; patch.br().y < frame.rows; patch.y += stepV)
		{
			for (patch.x = 0; patch.br().x < frame.cols; patch.x += stepH)
			{
				patch.refreshOverlap(initBoundingBox);
				scanGrids.push_back(patch);
			}
		}
	}
}



