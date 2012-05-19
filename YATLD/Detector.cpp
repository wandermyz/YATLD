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

	stepH = (int)(STEP_H * frame.cols + 0.5);
	stepV = (int)(STEP_V * frame.rows + 0.5);

	generateGridSizes(boundingBox);
	patchVariance.init(frame, boundingBox);

	ensembleClassifier.init();
}

void Detector::update(const cv::Mat& frame, cv::Mat& outputFrame)
{
	this->frame = frame;
	this->outputFrame = outputFrame;

#ifdef DEBUG
	int nPatches = 0, nEnsemblePatches = 0, nNNPatches = 0;
#endif

	patchVariance.update(frame);
	ensembleClassifier.update(frame);

	BoundingBox patch;

	for (vector<Size>::const_iterator scaleIt = gridSizes.begin(); scaleIt != gridSizes.end(); ++scaleIt)
	{
		patch.width = scaleIt->width;
		patch.height = scaleIt->height;

		for (patch.y = 0; patch.br().y <= frame.rows; patch.y += stepV)
		{
			for (patch.x = 0; patch.br().x <= frame.cols; patch.x += stepH)
			{
#ifdef DEBUG
				nPatches++;
				//rectangle(outputFrame, patch, Scalar(255,255,255));
#endif
				if (!patchVariance.acceptPatch(patch))
				{
					continue;
				}

				//Ensemble classifier
#ifdef DEBUG
				nEnsemblePatches++;
				//rectangle(outputFrame, patch, Scalar(255,0,0), 2);
#endif
				if (!ensembleClassifier.accept(patch))
				{
					continue;
				}

				//NN classifier
#ifdef DEBUG
				nNNPatches++;
				rectangle(outputFrame, patch, Scalar(255,0,0), 2);
#endif
			}
		}
	}

#ifdef DEBUG
	cout << "Total patches: " << nPatches << endl;
	cout << "After Patch variance: " << nEnsemblePatches << endl;
	cout << "After Ensemble classifier: " << nNNPatches << endl;
#endif
}

void Detector::generateGridSizes(const BoundingBox& initBoundingBox)
{
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
	while (bbSize.width * bbSize.height >= MIN_BB_AREA)
	{
		gridSizes.push_back(bbSize);
		scale /= 1.2;
		bbSize = Size((int)(initBoundingBox.width * scale + 0.5), (int)(initBoundingBox.height * scale + 0.5));
	}
}

