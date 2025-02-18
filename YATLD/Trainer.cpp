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
 
	this->frame = frame;
	result = boundingBox;
	resultFound = true;
	generatePatches();
	//learnByPExpert(frame, true);
	//learnByNExpert(frame, true);
	trainEnsemble(true);
	trainNN(true);

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
	this->frame = frame;
	combine();

#ifdef DEBUG	//make sure everything is correctly assigned
	if (tracker.getBoundingBox() != NULL)
	{
		assert(tracker.getBoundingBox()->state != UnknownState);
		assert(tracker.getBoundingBox()->conservativeSimilarity >= 0);
		assert(tracker.getBoundingBox()->relativeSimilarity >= 0);
	}

	/*if (detector.getBoundingBox() != NULL)
	{
		assert(tracker.getBoundingBox()->state != UnknownState);
		assert(detector.getBoundingBox()->conservativeSimilarity >= 0);
		assert(detector.getBoundingBox()->relativeSimilarity >= 0);
	}*/

	if (resultFound)
	{
		assert(result.state != UnknownState);
		assert(result.conservativeSimilarity >= 0);
		assert(result.relativeSimilarity >= 0);
	}
#endif

	/*bool reInit = detector.getBoundingBox() != NULL
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
	}*/

#ifdef DEBUG
	cout << "Reliable: " << reliable << endl;
#endif

	//P-N learning
	if (reliable)
	{
		detector.refreshGridOverlap(result);
		generatePatches();
		trainEnsemble(false);
		trainNN(false);
		//learnByPExpert(frame, false);
		//learnByNExpert(frame, false);


/*#ifdef DEBUG
		printf("Learning: Positive: %d * %d = %d; Negative: %d\n", positiveCount, synthesisNum, positiveCount * synthesisNum, negativeCount);
#endif*/
	}
}

void Trainer::combine()
{
	if (tracker.isTracked())
	{
		result = *tracker.getBoundingBox();
		resultFound = true;
		if (result.conservativeSimilarity > TRAINER_CORE_THRESHOLD)
		{
			reliable = true;
		}

#ifdef DEBUG
		assert(result.conservativeSimilarity > 0);
#endif

		vector<const BoundingBox*> confidentClusters;
		if (detector.isDetected())
		{
			for (vector<BoundingBox>::const_iterator it = detector.getClusteredBoundingBoxes().begin(); it != detector.getClusteredBoundingBoxes().end(); ++it)
			{
				if (it->getOverlap(result) < DETECTOR_CLUSTER_MIN_OVERLAP) //far away
				{
					float conservativeSim;
					if (it->state == DetectedAcceptedByNN)
					{
#ifdef DEBUG
						assert(it->conservativeSimilarity > 0);
#endif
						conservativeSim = it->conservativeSimilarity;
					}
					else if (it->state == DetectedCluster)
					{
						detector.getNNClassifier().getSimilarity(frame(*it), NULL, &conservativeSim);
					}
#ifdef DEBUG
					else
					{
						assert(0);
					}
#endif

#ifdef DEBUG
					cout << "Tracker " << result.conservativeSimilarity << " vs. Detector " << conservativeSim << endl;
#endif
					if (conservativeSim > result.conservativeSimilarity)
					{
						confidentClusters.push_back(&(*it));
					}
				}
			}

#ifdef DEBUG
			cout << "Confident Clusters: " << confidentClusters.size() << endl;
#endif

			if (confidentClusters.size() == 1)	//only 1 cluster, re-init tracker
			{
				//tracker.reset(*confidentClusters[0]);
				reliable = false;
				result = *confidentClusters[0];
#ifdef DEBUG
				cout << "Tracker reset." << endl;
#endif
			}
			else	//no cluster, or more than one cluster, average	//TODO: why?
			{
				BoundingBox avgBb(0,0,0,0);
				int nClosed = 0;

				for (vector<const BoundingBox*>::const_iterator it = detector.getDetectedBoundingBoxes().begin(); it != detector.getDetectedBoundingBoxes().end(); ++it)
				{
					const BoundingBox& bb = **it;
					if (bb.getOverlap(result) > 0.7)	//TODO: threshold
					{
						avgBb.x += bb.x;
						avgBb.y += bb.y;
						avgBb.width += bb.width;
						avgBb.height += bb.height;
						nClosed++;
					}
				}

				if (nClosed > 0)
				{
					int weight = 10;	//TODO: threshold
					result.x = cvRound((float)(weight * result.x + avgBb.x) / (float)(weight + nClosed));
					result.y = cvRound((float)(weight * result.y + avgBb.y) / (float)(weight + nClosed));
					result.width = cvRound((float)(weight * result.width + avgBb.width) / (float)(weight + nClosed));
					result.height = cvRound((float)(weight * result.height + avgBb.height) / (float)(weight + nClosed));
					result.state = TrackedWeighted;
#ifdef DEBUG
					result.relativeSimilarity = -1;
					result.conservativeSimilarity = -1;
					cout << "Result weighted by " << nClosed << " closed bounding boxes." << endl;
#endif
				}
			}

		}
	}	
	else //if not tracked
	{
		reliable = false;
		resultFound = false;

		if (detector.isDetected())
		{
			if (detector.getClusteredBoundingBoxes().size() == 1)
			{
				result = detector.getClusteredBoundingBoxes()[0];
				resultFound = true;
#ifdef DEBUG
				cout << "Tracker reset." << endl;
#endif
			}
		}
	}

	if (resultFound && (result.state == DetectedCluster || result.state == TrackedWeighted))
	{
		Rect intersect = result & Rect(0, 0, frame.cols, frame.rows);
		detector.getNNClassifier().getSimilarity(frame(intersect), &result.relativeSimilarity, &result.conservativeSimilarity);
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
		//else if (patchIt->getOverlap() < PN_NEGATIVE_OVERLAP && patchIt->state != RejectedByVariance)
		else if (patchIt->getOverlap() < PN_NEGATIVE_OVERLAP && detector.getPatchVariance().acceptPatch(*patchIt, 0.25))	//TODO: why?
		//else if (patchIt->getOverlap() < PN_NEGATIVE_OVERLAP)
		{
			negativePatches.push_back(&(*patchIt));
		}
	}

	//sort positive, shuffle negative

	nth_element(positivePatches.begin(), positivePatches.begin() + min(positiveNum, (int)positivePatches.size()), positivePatches.end(), compareOverlap);
	random_shuffle(negativePatches.begin(), negativePatches.end());
}

bool Trainer::compareOverlap(const BoundingBox* bb1, const BoundingBox* bb2)
{
	return bb1->getOverlap() > bb2->getOverlap();	
}

void Trainer::trainEnsemble(bool init)
{
	vector<pair<Mat, bool>> ensembleSamples;
	//const Mat& blurred = detector.getEnsembleClassifier().getFrameBlurred();

	Mat blurred;
	GaussianBlur(frame, blurred, Size(9, 9), 1.5);	//TODO: threshold

	//synthesize positive
	/*for (int i = 0; i < min(positiveNum, (int)positivePatches.size());  i++)
	{
		const BoundingBox& patch = *positivePatches[i];
		Mat patchImg = blurred(patch);

		for (int j = 0; j < synthesisNum; j++)
		{
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

			ensembleSamples.push_back(make_pair<Mat, bool>(warpedImg, true));
		}
	}*/

	//find positive hull
	int x1 = frame.cols, x2 = 0;
	int y1 = frame.rows, y2 = 0;

	for (int i = 0; i < min(positiveNum, (int)positivePatches.size()); i++)
	{
		x1 = min(positivePatches[i]->x, x1);
		y1 = min(positivePatches[i]->y, y1);
		x2 = max(positivePatches[i]->br().x, x2);
		y2 = max(positivePatches[i]->br().y, y2);
	}

	BoundingBox positiveHull(x1, y1, x2 - x1, y2 - y1);

	for (int j = 0; j < synthesisNum; j++)
	{
		int shiftH, shiftV;
		Mat warpedImg;

		if (j > 0)
		{
			shiftH = (int)(rng.uniform(-shiftChange * frame.cols, shiftChange * frame.cols) + 0.5);
			shiftV = (int)(rng.uniform(-shiftChange * frame.rows, shiftChange * frame.rows) + 0.5);
			double scale = 1 + rng.uniform(-scaleChange, scaleChange);
			double rotation = rng.uniform(-rotationChange, rotationChange);

			Mat rotMat = getRotationMatrix2D(Point(positiveHull.x + positiveHull.width / 2, positiveHull.y + positiveHull.height / 2), rotation, scale);
			warpAffine(blurred, warpedImg, rotMat, blurred.size());

			//Add gaussian noise
			Mat gaussianNoise(warpedImg.size(), CV_8S);
			rng.fill(gaussianNoise, RNG::NORMAL, 0, INIT_GAUSSIAN_SIGMA);

			add(warpedImg, gaussianNoise, warpedImg, noArray(), CV_8U);
		}
		else
		{
			warpedImg = blurred;
			shiftH = 0;
			shiftV = 0;
		}

		/*Mat tmp = warpedImg.clone();
		rectangle(tmp, positiveHull, Scalar(255,0,0), 1);

		imshow("test", tmp);
		waitKey();*/

		for (int i = 0; i < min(positiveNum, (int)positivePatches.size()); i++)
		{
			Mat warpedPatch = warpedImg(*positivePatches[i]);
			warpedPatch.adjustROI(-shiftV, shiftV, -shiftH, shiftH);
			ensembleSamples.push_back(make_pair<Mat,bool>(warpedPatch, true));
			/*if (i == 0)
			{
				imshow("test", warpedPatch);
				waitKey();
			}*/
		}
	}
	
	//add negative samples
	for (int i = 0; i < negativePatches.size() / 2; i++)	//TODO: for comparison, just take half
	{
		const BoundingBox& patch = *negativePatches[i];
		ensembleSamples.push_back(make_pair<Mat, bool>(blurred(patch), false));
		//ensembleSamples.push_back(make_pair<Mat, bool>(frame(patch), false));
	}

	//shuffle
	random_shuffle(ensembleSamples.begin(), ensembleSamples.end());

	int posCount = 0, negCount = 0;
	for (int i = 0; i < ensembleSamples.size(); i++)
	{
		double posterior = detector.getEnsembleClassifier().getPosterior(ensembleSamples[i].first);
		if (ensembleSamples[i].second && posterior <= 0.5)		//TODO: evaluate threshold
		{
			detector.getEnsembleClassifier().train(ensembleSamples[i].first, true);
			posCount++;
		}
		else if (!ensembleSamples[i].second && posterior >= 0.5)
		{
			detector.getEnsembleClassifier().train(ensembleSamples[i].first, false);
			negCount++;
		}
	}

	cout << "Train Ensemble: Positive " << posCount << ", Negative " << negCount << endl;
}

void Trainer::trainNN(bool init)
{
	int posCount = 0, negCount = 0;

	//train positive
	#ifdef DEBUG
		assert(init || result.relativeSimilarity > 0);
	#endif

	if (init || result.relativeSimilarity < 0.65)		//TODO: evaluate threshold
	{	
		Rect intersect = result & Rect(0, 0, frame.cols, frame.rows);
		detector.getNNClassifier().train(frame(intersect), true);
		posCount++;
	}

	//train negative
	for (int i = 0; i < min(negativeNum, (int)negativePatches.size()); i++)
	{
		const BoundingBox& patch = *negativePatches[i];
		
#ifdef DEBUG
		assert(init || patch.state != UnknownState);
#endif

		float relativeSimilarity;
		if (!init && (patch.state == DetectedAcceptedByNN || patch.state == RejectedByNN))
		{
#ifdef DEBUG
			assert(patch.relativeSimilarity > 0);
#endif
			relativeSimilarity = patch.relativeSimilarity;
		}
		else
		{
			detector.getNNClassifier().getSimilarity(frame(patch), &relativeSimilarity, NULL);
		}

		if (relativeSimilarity > 0.5)	//TODO: threshold?
		{
			detector.getNNClassifier().train(frame(patch), false);
			negCount++;
		}
	}

	cout << "Train NN: Positive " << posCount << ", Negative " << negCount << endl;

	if (detector.getNNClassifier().getPositiveNum() > NN_MAX_POSITIVE)
	{
		detector.getNNClassifier().forgetPositive(detector.getNNClassifier().getPositiveNum() - NN_MIN_POSITIVE);
#ifdef DEBUG
		cout << "NN forgets " << detector.getNNClassifier().getPositiveNum() - NN_MIN_POSITIVE << " positive samples" << endl;
#endif
	}

	if (detector.getNNClassifier().getNegativeNum() > NN_MAX_NEGATIVE)
	{
		detector.getNNClassifier().forgetNegative(detector.getNNClassifier().getNegativeNum() - NN_MIN_NEGATIVE);
#ifdef DEBUG
		cout << "NN forgets " << detector.getNNClassifier().getNegativeNum() - NN_MIN_NEGATIVE << " negative samples" << endl;
#endif
	}
}

//void Trainer::learnByPExpert(const Mat& frame, bool init)
//{
//	int nnCount = 0, ensembleCount = 0;
//
//	//train NN
//#ifdef DEBUG
//	assert(init || result->relativeSimilarity > 0);
//#endif
//
//	if (init || result->relativeSimilarity - NN_THRESHOLD <= TRAINER_MARGIN_THRESHOLD)
//	{
//		Rect intersect = *result & Rect(0, 0, frame.cols, frame.rows);
//		detector.getNNClassifier().train(frame(intersect), true);
//		nnCount++;
//	}
//
//	int positiveCount = 0;
//	nth_element(positivePatches.begin(), positivePatches.begin() + positiveNum, positivePatches.end(), compareOverlap);
//
//	for (int i = 0; i < positiveNum && !positivePatches.empty(); ++i)
//	{
//		const BoundingBox& patch = *positivePatches[i];
//
//#ifdef DEBUG
//		assert(patch.state != TrackedAcceptedByNN && patch.state != TrackedRejectedByNN && patch.state != UnknownState);
//#endif
//		if (patch.state == RejectedByVariance)
//		{
//			cout << "Warning: Rejected by variance happened in P-Expert!" << endl;
//		}
//
//		bool trainEnsemble;
//
//		if (init)
//		{
//			trainEnsemble = true;
//		}
//		else
//		{
//			trainEnsemble = patch.state == RejectedByEnsemble;
//		}
//
//		if (trainEnsemble)
//		{
//			for (int j = 0; j < synthesisNum; j++)
//			{
//				Mat patchImg = frame(patch);
//				Mat warpedImg;
//
//				int shiftH = (int)(rng.uniform(-shiftChange * frame.cols, shiftChange * frame.cols) + 0.5);
//				int shiftV = (int)(rng.uniform(-shiftChange * frame.rows, shiftChange * frame.rows) + 0.5);
//				double scale = 1 + rng.uniform(-scaleChange, scaleChange);
//				double rotation = rng.uniform(-rotationChange, rotationChange);
//
//				//printf("%f, %f, %f, %f\n", shiftH, shiftV, scale, rotation);
//
//				//TODO: check boundary
//				patchImg.adjustROI(-shiftV, shiftV, -shiftH, shiftH);
//
//				Mat rotMat = getRotationMatrix2D(Point(patchImg.cols / 2, patchImg.rows / 2), rotation, scale);
//				warpAffine(patchImg, warpedImg, rotMat, patchImg.size());
//
//				//Add gaussian noise
//				Mat gaussianNoise(warpedImg.size(), CV_8S);
//				rng.fill(gaussianNoise, RNG::NORMAL, 0, INIT_GAUSSIAN_SIGMA);
//
//				add(warpedImg, gaussianNoise, warpedImg, noArray(), CV_8U);
//
//				//imshow("init", warpedImg);
//				//waitKey();  
//
//				if (trainEnsemble)
//				{
//					detector.getEnsembleClassifier().update(warpedImg);
//					detector.getEnsembleClassifier().train(Rect(0, 0, warpedImg.cols, warpedImg.rows), true);
//				}
//
//				ensembleCount++;
//			}
//		}	//if(trainEnsemble)
//	}
//
//	cout << "Positive: " << "Ensemble " << ensembleCount << ", NN " << nnCount << endl;
//
//	if (detector.getNNClassifier().getPositiveNum() > NN_MAX_POSITIVE)
//	{
//		detector.getNNClassifier().forgetPositive(detector.getNNClassifier().getPositiveNum() - NN_MIN_POSITIVE);
//#ifdef DEBUG
//		cout << "NN forgets " << detector.getNNClassifier().getPositiveNum() - NN_MIN_POSITIVE << " positive samples" << endl;
//#endif
//	}
//}
//
//void Trainer::learnByNExpert(const Mat& frame, bool init)
//{
//	int ensembleCount = 0, nnCount = 0;
//	
//	//permutating negative patches
// 	random_shuffle(negativePatches.begin(), negativePatches.end());		
//	detector.getEnsembleClassifier().update(frame);
//
//	for (int i = 0; i < negativePatches.size(); i++)
//	{
//		const BoundingBox& patch = *negativePatches[i]; 
//
//		bool trainEnsemble, trainNN;
//		if (init)
//		{
//			trainEnsemble = true;
//			trainNN = true;
//		}
//		else if (patch.state == RejectedByVariance || patch.state == RejectedByEnsemble)
//		{
//			trainEnsemble = false;
//			trainNN = false;
//		}
//		else
//		{
//			trainEnsemble = true;
//
//#ifdef DEBUG
//			assert(patch.relativeSimilarity >= 0);
//#endif
//			trainNN = NN_THRESHOLD - patch.relativeSimilarity <= TRAINER_MARGIN_THRESHOLD;
//		}
//
//		if (trainEnsemble)
//		{
//			detector.getEnsembleClassifier().train(patch, false);
//			ensembleCount++;
//		}
//
//		if (i < negativeNum && trainNN)
//		{
//			detector.getNNClassifier().train(frame(patch), false);
//			nnCount++;
//		}
//
//		/*cout << patch.getOverlap() << endl;
//		Mat tmp = frame.clone();
//		rectangle(tmp, boundingBox, Scalar(0,255,255), 2);
//		rectangle(tmp, patch, Scalar(255,0,0), 1);
//		imshow("init", tmp);
//		waitKey();*/
//	}
//
//	cout << "Negative: " << "Ensemble " << ensembleCount << ", NN " << nnCount << endl;
//
//	if (detector.getNNClassifier().getNegativeNum() > NN_MAX_NEGATIVE)
//	{
//		detector.getNNClassifier().forgetNegative(detector.getNNClassifier().getNegativeNum() - NN_MIN_NEGATIVE);
//#ifdef DEBUG
//		cout << "NN forgets " << detector.getNNClassifier().getNegativeNum() - NN_MIN_NEGATIVE << " negative samples" << endl;
//#endif
//	}
//}
//
