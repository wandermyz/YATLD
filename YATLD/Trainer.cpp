#include "Trainer.h"
#include "Settings.h"
#include <stdio.h>
#include <iostream>
using namespace cv;
using namespace std;

Trainer::Trainer(Detector& detector) : detector(detector)
{
	
}

void Trainer::init(const cv::Mat& frame, const BoundingBox& boundingBox)
{
	//namedWindow("init");

	int positiveCount = 0, negativeCount = 0;

	Rect patch;
	patch.width = boundingBox.width;
	patch.height = boundingBox.height;

	//int hOffset = (int)(STEP_H * frame.cols + 0.5);
	//int vOffset = (int)(STEP_V * frame.rows + 0.5);

	int hOffset = (int)(STEP_H * patch.width + 0.5);
	int vOffset = (int)(STEP_V * patch.height + 0.5);

	//generate positive patches
	RNG rng;
	//TODO: use overlap to find closet bounding boxes
	for (patch.y = boundingBox.y - vOffset; patch.y <= boundingBox.y + vOffset; patch.y += vOffset)
	{
		if (patch.y < 0 || patch.br().y > frame.rows)
		{
			continue;
		}

		for (patch.x = boundingBox.x - hOffset; patch.x <= boundingBox.x + hOffset; patch.x += hOffset)
		{
			if (patch.x < 0 || patch.br().x > frame.cols)
			{
				continue;
			}

			positiveCount++;

			for (int i = 0; i < INIT_WARP_NUM; i++)
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
		}
	}

	//generate negative patches
	detector.getEnsembleClassifier().update(frame);
	for (negativeCount = 0; negativeCount < positiveCount * INIT_WARP_NUM; negativeCount++)
	{
		patch.x = rng.uniform(0, frame.cols - patch.width);
		patch.y = rng.uniform(0, frame.rows - patch.height);

		if (patch.br().y >= boundingBox.y /*- vOffset*/ && patch.y <= boundingBox.br().y /*+ vOffset*/
				&& patch.br().x >= boundingBox.x /*- hOffset*/ && patch.x <= boundingBox.br().x /*+ hOffset*/)
		{
			continue;
		}

		detector.getEnsembleClassifier().train(patch, false);
	}

#ifdef DEBUG
	printf("Init learning: Positive: %d * %d = %d; Negative: %d\n", positiveCount, INIT_WARP_NUM, positiveCount * INIT_WARP_NUM, negativeCount);
#endif
}