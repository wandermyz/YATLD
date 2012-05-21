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

	ensembleClassifier.init();
}

void Detector::update(const cv::Mat& frame, cv::Mat& outputFrame)
{
	this->frame = frame;
	this->outputFrame = outputFrame;

#ifdef DEBUG
	int nPatches = 0, nEnsemblePatches = 0, nNNPatches = 0, nFinalPatches = 0;
#endif

	float maxNNSimilarity = 0;
	vector<BoundingBox>::const_iterator maxNNpatchIt;


	patchVariance.update(frame);
	ensembleClassifier.update(frame);

	for (vector<BoundingBox>::iterator patchIt = scanGrids.begin(); patchIt != scanGrids.end(); ++patchIt)
	{
		patchIt->positive = false;
		//refersh overlap
		//TODO

#ifdef DEBUG
		nPatches++;
		//rectangle(outputFrame, patch, Scalar(255,255,255));
#endif
		if (!patchVariance.acceptPatch(*patchIt))
		{
			continue;
		}

		//Ensemble classifier
#ifdef DEBUG
		nEnsemblePatches++;
		//rectangle(outputFrame, patch, Scalar(255,0,0), 2);
#endif
		if (!ensembleClassifier.acceptPatch(*patchIt))
		{
			continue;
		}

		//NN classifier
#ifdef DEBUG
		nNNPatches++;
		//rectangle(outputFrame, *patchIt, Scalar(255,0,0), 1);
#endif

		float relativeSim, conservativeSim;
		nnClassifier.getSimilarity(frame(*patchIt), &relativeSim, &conservativeSim);
		if (relativeSim > maxNNSimilarity)
		{
			maxNNSimilarity = relativeSim;
			maxNNpatchIt = patchIt;
		}
		if (relativeSim <= NN_THRESHOLD)
		{
			continue;
		}

		//final
		patchIt->positive = true;
		patchIt->confidence = conservativeSim;

#ifdef DEBUG
		nFinalPatches++;
#endif

		//rectangle(outputFrame, *patchIt, Scalar(255,0,0), 2);
	}

#ifdef DEBUG
	cout << "Total patches: " << nPatches << endl;
	cout << "After Patch variance: " << nEnsemblePatches << endl;
	cout << "After Ensemble classifier: " << nNNPatches << endl;
	cout << "After NN classifier: " << nFinalPatches << endl;
	cout << "Maximum NN Similairty: " << maxNNSimilarity << endl;
	cout << endl;
	//rectangle(outputFrame, *maxNNpatchIt, Scalar(0, 0, 255), 2);
#endif

	finalBoundingBox = &(*maxNNpatchIt);
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
		int stepH = (int)(STEP_H * scaleIt->width + 0.5);
		int stepV = (int)(STEP_V * scaleIt->height + 0.5);

#ifdef DEBUG
		assert(stepH >= 1 && stepV >= 1);
#endif

		for (patch.y = 0; patch.br().y <= frame.rows; patch.y += stepV)
		{
			for (patch.x = 0; patch.br().x <= frame.cols; patch.x += stepH)
			{
				patch.refreshOverlap(initBoundingBox);
				scanGrids.push_back(patch);
			}
		}
	}
}



