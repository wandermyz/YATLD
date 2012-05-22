#include "Trainer.h"
#include "Settings.h"
#include <stdio.h>
#include <iostream>
#include <queue>
#include <algorithm>
#include <assert.h>
using namespace cv;
using namespace std;

Trainer::Trainer(Detector& detector, Tracker& tracker) : detector(detector), tracker(tracker)
{
	
}

void Trainer::init(const cv::Mat& frame, const BoundingBox& boundingBox)
{
	//set parameters
	shiftChange = INIT_SHIFT_CHANGE;
	scaleChange = INIT_SCALE_CHANGE;
	rotationChange = INIT_ROTATION_CHANGE;
	gaussianSigma = INIT_GAUSSIAN_SIGMA;
	synthesisNum = INIT_SYNTHESIS_NUM;
	positiveNum = INIT_POSITIVE_NUM;
	negativeNum = INIT_NEGATIVE_NUM;
 
	result = &boundingBox;
	generatePatches();
	learnByPExpert(frame, true);
	learnByNExpert(frame, true);

	reliable = true;

	//set parameters back to update
	shiftChange = UPDATE_SHIFT_CHANGE;
	scaleChange = UPDATE_SCALE_CHANGE;
	rotationChange = UPDATE_ROTATION_CHANGE;
	gaussianSigma = UPDATE_GAUSSIAN_SIGMA;
	synthesisNum = UPDATE_SYNTHESIS_NUM;
	positiveNum = UPDATE_POSITIVE_NUM;
	negativeNum = UPDATE_NEGATIVE_NUM;
}

void Trainer::update(const Mat& frame)
{
#ifdef DEBUG
	cout << "[Trainer]" << endl;
#endif
	combine();

#ifdef DEBUG	//make sure everything is correctly assigned
	if (tracker.getBoundingBox() != NULL)
	{
		assert(tracker.getBoundingBox()->state != UnknownState);
		assert(tracker.getBoundingBox()->conservativeSimilarity >= 0);
		assert(tracker.getBoundingBox()->relativeSimilarity >= 0);
	}

	if (detector.getBoundingBox() != NULL)
	{
		assert(tracker.getBoundingBox()->state != UnknownState);
		assert(detector.getBoundingBox()->conservativeSimilarity >= 0);
		assert(detector.getBoundingBox()->relativeSimilarity >= 0);
	}

	if (result != NULL)
	{
		assert(result->state != UnknownState);
		assert(result->conservativeSimilarity >= 0);
		assert(result->relativeSimilarity >= 0);
	}
#endif

	bool reInit = detector.getBoundingBox() != NULL
		&& (tracker.getBoundingBox() == NULL || detector.getBoundingBox()->conservativeSimilarity > tracker.getBoundingBox()->conservativeSimilarity);

	//update reliable
	if (result == NULL)
	{
		reliable = false;
	}
	else if (!reliable || reInit)
	{
		reliable = result->conservativeSimilarity > TRAINER_CORE_THRESHOLD;
	}

	if (reInit)
	{
		tracker.reset(*detector.getBoundingBox());
#ifdef DEBUG
		cout << "Tracker reset." << endl;
#endif
	}

#ifdef DEBUG
	cout << "Reliable: " << reliable << endl;
#endif

	//P-N learning
	if (reliable)
	{
		detector.refreshGridOverlap(*result);
		generatePatches();
		learnByPExpert(frame, false);
		learnByNExpert(frame, false);

/*#ifdef DEBUG
		printf("Learning: Positive: %d * %d = %d; Negative: %d\n", positiveCount, synthesisNum, positiveCount * synthesisNum, negativeCount);
#endif*/
	}
}

void Trainer::combine()
{
	result = NULL;
	
	if (detector.getBoundingBox() != NULL)
	{
		result = detector.getBoundingBox();
	}

	if (tracker.getBoundingBox() != NULL)
	{
		if (result == NULL || tracker.getBoundingBox()->conservativeSimilarity > result->conservativeSimilarity)
		{
			result = tracker.getBoundingBox();
		}
	}
}

void Trainer::generatePatches()
{
	positivePatches.clear();
	negativePatches.clear();

	//generate patches
	for (vector<BoundingBox>::const_iterator patchIt = detector.getScanGrids().begin(); patchIt != detector.getScanGrids().end(); ++patchIt)
	{
		if (patchIt->getOverlap() >= PN_POSITIVE_OVERLAP)
		{
			positivePatches.push_back(&(*patchIt));
		}
		else if (patchIt->getOverlap() < PN_NEGATIVE_OVERLAP && (float)patchIt->width / (float)result->width >= PN_NEGATIVE_MIN_SCALE)
		{
			negativePatches.push_back(&(*patchIt));
		}
	}
}

bool Trainer::compareOverlap(const BoundingBox* bb1, const BoundingBox* bb2)
{
	return bb1->getOverlap() > bb2->getOverlap();	
}

void Trainer::learnByPExpert(const Mat& frame, bool init)
{
	int nnCount = 0, ensembleCount = 0;

	//train NN
#ifdef DEBUG
	assert(init || result->relativeSimilarity > 0);
#endif

	if (init || result->relativeSimilarity - NN_THRESHOLD <= TRAINER_MARGIN_THRESHOLD)
	{
		Rect intersect = *result & Rect(0, 0, frame.cols, frame.rows);
		detector.getNNClassifier().train(frame(intersect), true);
		nnCount++;
	}

	//synthesize positive patches
	int positiveCount = 0;
	//sort(positivePatches.begin(), positivePatches.end(), compareOverlap);
	nth_element(positivePatches.begin(), positivePatches.begin() + positiveNum, positivePatches.end(), compareOverlap);

	for (int i = 0; i < positiveNum && !positivePatches.empty(); ++i)
	{
		//const BoundingBox& patch = *(positivePatches[positivePatches.size() - 1]);
		//positivePatches.pop_back();
		const BoundingBox& patch = *positivePatches[i];

		/*cout << patch.getOverlap() << endl;
		Mat tmp = frame.clone();
		rectangle(tmp, *result, Scalar(0,255,255), 2);
		rectangle(tmp, patch, Scalar(255,0,0), 1);
		imshow("init", tmp);
		waitKey();*/


		//all patches are from grid
		/*Cases for learning:
			1. rejected by NN
			or 2. margin too small
			or 3. rejected by Essemble
			3 implies 1
		*/

#ifdef DEBUG
		assert(patch.state != TrackedAcceptedByNN && patch.state != TrackedRejectedByNN && patch.state != UnknownState);
#endif
		if (patch.state == RejectedByVariance)
		{
			cout << "Warning: Rejected by variance happened in P-Expert!" << endl;
		}

		bool trainEnsemble;
		//bool trainNN;

		if (init)
		{
			trainEnsemble = true;
			//trainNN = true;
		}
		else
		{
			trainEnsemble = patch.state == RejectedByEnsemble;
			/*if (trainEnsemble)
			{
				//Will NN also be trained?
				float relativeSim;
				detector.getNNClassifier().getSimilarity(frame(patch), &relativeSim, NULL);
				trainNN = relativeSim - NN_THRESHOLD <= TRAINER_MARGIN_THRESHOLD; 
#ifdef DEBUG
				if (trainNN)
				{
					cout << "Warning: Rejected by Ensemble but NN is trained! Relative Similarity = " << relativeSim << endl;
				}
#endif
			}
			else
			{
				trainNN = patch.relativeSimilarity - NN_THRESHOLD <= TRAINER_MARGIN_THRESHOLD;
#ifdef DEBUG
				if (trainNN)
				{
					cout << "NN is trained. Relative Similarity = " << patch.relativeSimilarity << endl;
				}
#endif
			}*/
		}

		if (trainEnsemble)
		{
			for (int j = 0; j < synthesisNum; j++)
			{
				Mat patchImg = frame(patch);
				Mat warpedImg;

				int shiftH = (int)(rng.uniform(-shiftChange * frame.cols, shiftChange * frame.cols) + 0.5);
				int shiftV = (int)(rng.uniform(-shiftChange * frame.rows, shiftChange * frame.rows) + 0.5);
				double scale = 1 + rng.uniform(-scaleChange, scaleChange);
				double rotation = rng.uniform(-rotationChange, rotationChange);

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

				if (trainEnsemble)
				{
					detector.getEnsembleClassifier().update(warpedImg);
					detector.getEnsembleClassifier().train(Rect(0, 0, warpedImg.cols, warpedImg.rows), true);
				}

				/*if (trainNN)
				{
					detector.getNNClassifier().train(warpedImg, true);
				}*/

				ensembleCount++;
			}

			//positiveCount++;
		}	//if(trainEnsemble)

		/*if (trainNN)
		{
			detector.getNNClassifier().train(frame(patch), true);
		}*/
	}

	cout << "Positive: " << "Ensemble " << ensembleCount << ", NN " << nnCount << endl;

	if (detector.getNNClassifier().getPositiveNum() > NN_MAX_POSITIVE)
	{
		detector.getNNClassifier().forgetPositive(detector.getNNClassifier().getPositiveNum() - NN_MIN_POSITIVE);
#ifdef DEBUG
		cout << "NN forgets " << detector.getNNClassifier().getPositiveNum() - NN_MIN_POSITIVE << " positive samples" << endl;
#endif
	}
}

void Trainer::learnByNExpert(const Mat& frame, bool init)
{
	int ensembleCount = 0, nnCount = 0;
	
	//permutating negative patches
 	random_shuffle(negativePatches.begin(), negativePatches.end());		
	detector.getEnsembleClassifier().update(frame);

	for (int i = 0; i < negativePatches.size() && nnCount < negativeNum; i++)
	{
		const BoundingBox& patch = *negativePatches[i]; 

		bool trainEnsemble, trainNN;
		if (init)
		{
			trainEnsemble = true;
			trainNN = true;
		}
		else if (patch.state == RejectedByVariance || patch.state == RejectedByEnsemble)
		{
			trainEnsemble = false;
			trainNN = false;
		}
		else
		{
			trainEnsemble = true;

#ifdef DEBUG
			assert(patch.relativeSimilarity >= 0);
#endif
			trainNN = NN_THRESHOLD - patch.relativeSimilarity <= TRAINER_MARGIN_THRESHOLD;
		}

		if (trainEnsemble)
		{
			detector.getEnsembleClassifier().train(patch, false);
			ensembleCount++;
		}

		if (trainNN)
		{
			detector.getNNClassifier().train(frame(patch), false);
			nnCount++;
		}

		/*cout << patch.getOverlap() << endl;
		Mat tmp = frame.clone();
		rectangle(tmp, boundingBox, Scalar(0,255,255), 2);
		rectangle(tmp, patch, Scalar(255,0,0), 1);
		imshow("init", tmp);
		waitKey();*/
	}

	cout << "Negative: " << "Ensemble " << ensembleCount << ", NN " << nnCount << endl;

	if (detector.getNNClassifier().getNegativeNum() > NN_MAX_NEGATIVE)
	{
		detector.getNNClassifier().forgetNegative(detector.getNNClassifier().getNegativeNum() - NN_MIN_NEGATIVE);
#ifdef DEBUG
		cout << "NN forgets " << detector.getNNClassifier().getNegativeNum() - NN_MIN_NEGATIVE << " negative samples" << endl;
#endif
	}
}

