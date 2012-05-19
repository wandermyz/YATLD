#include "Trainer.h"
#include "Settings.h"
#include <stdio.h>
#include <iostream>
#include <queue>
#include <algorithm>
using namespace cv;
using namespace std;

Trainer::Trainer(Detector& detector) : detector(detector)
{
	
}

void Trainer::init(const cv::Mat& frame, const BoundingBox& boundingBox)
{
	int positiveCount = 0, negativeCount = 0;

	priority_queue<const BoundingBox*, vector<const BoundingBox*>, CompareOverlap> positivePatches;		//use heap to keep sorted by overlapping
	vector<const BoundingBox*> negativePatches;

	for (vector<BoundingBox>::const_iterator patchIt = detector.getScanGrids().begin(); patchIt != detector.getScanGrids().end(); ++patchIt)
	{
		if (patchIt->getOverlap() >= INIT_POSITIVE_OVERLAP)
		{
			positivePatches.push(&(*patchIt));
		}
		else if (patchIt->getOverlap() < INIT_NEGATIVE_OVERLAP && (float)patchIt->width / (float)boundingBox.width >= INIT_NEGATIVE_MIN_SCALE)
		{
			negativePatches.push_back(&(*patchIt));
		}
	}

	RNG rng;
	//synthesize positive patches
	for (int i = 0; i < INIT_POSITIVE_NUM && !positivePatches.empty(); ++i)
	{
		const BoundingBox& patch = *(positivePatches.top());
		positivePatches.pop();

		/*cout << patch.getOverlap() << endl;
		Mat tmp = frame.clone();
		rectangle(tmp, boundingBox, Scalar(0,255,255), 2);
		rectangle(tmp, patch, Scalar(255,0,0), 1);
		imshow("init", tmp);
		waitKey();*/

		for (int j = 0; j < INIT_WARP_NUM; j++)
		{
			Mat patchImg = frame(patch);
			Mat warpedImg;

			int shiftH = (int)(rng.uniform(-INIT_SHIFT_CHANGE * frame.cols, INIT_SHIFT_CHANGE * frame.cols) + 0.5);
			int shiftV = (int)(rng.uniform(-INIT_SHIFT_CHANGE * frame.rows, INIT_SHIFT_CHANGE * frame.rows) + 0.5);
			double scale = 1 + rng.uniform(-INIT_SCALE_CHANGE, INIT_SCALE_CHANGE);
			double rotation = rng.uniform(-INIT_ROTATION_CHANGE, INIT_ROTATION_CHANGE);

			//printf("%f, %f, %f, %f\n", shiftH, shiftV, scale, rotation);

			//TODO: check boundary
			patchImg.adjustROI(-shiftV, shiftV, -shiftH, shiftH);

			Mat rotMat = getRotationMatrix2D(Point(patchImg.cols / 2, patchImg.rows / 2), rotation, scale);
			warpAffine(patchImg, warpedImg, rotMat, patchImg.size());

			//Add gaussian noise
			Mat gaussianNoise(warpedImg.size(), CV_8S);
			rng.fill(gaussianNoise, RNG::NORMAL, 0, INIT_GAUSSIAN_SIGMA);

			add(warpedImg, gaussianNoise, warpedImg, noArray(), CV_8U);

			//imshow("init", warpedImg);
			//waitKey();  

			detector.getEnsembleClassifier().update(warpedImg);
			detector.getEnsembleClassifier().train(Rect(0, 0, warpedImg.cols, warpedImg.rows), true);
		}

		positiveCount++;
	}

	//permutating negative patches
	detector.getEnsembleClassifier().update(frame);
	random_shuffle(negativePatches.begin(), negativePatches.end());
	for (negativeCount = 0; negativeCount < negativePatches.size() && negativeCount < INIT_NEGATIVE_NUM; negativeCount++)
	{
		const BoundingBox& patch = *negativePatches[negativeCount];
		detector.getEnsembleClassifier().train(patch, false);

		/*cout << patch.getOverlap() << endl;
		Mat tmp = frame.clone();
		rectangle(tmp, boundingBox, Scalar(0,255,255), 2);
		rectangle(tmp, patch, Scalar(255,0,0), 1);
		imshow("init", tmp);
		waitKey();*/
	}

#ifdef DEBUG
	printf("Init learning: Positive: %d * %d = %d; Negative: %d\n", positiveCount, INIT_WARP_NUM, positiveCount * INIT_WARP_NUM, negativeCount);
#endif
}

