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
	//this->finalBoundingBox = &boundingBox;
	detectedBoundingBoxes.push_back(&boundingBox);
	clusteredBoundingBoxes.push_back(boundingBox);

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

	detectedBoundingBoxes.clear();
	clusteredBoundingBoxes.clear();

	float maxNNSimilarity = 0;
	vector<BoundingBox>::const_iterator maxNNpatchIt;

	patchVariance.update(frame);
	ensembleClassifier.update(frame);

	for (vector<BoundingBox>::iterator patchIt = scanGrids.begin(); patchIt != scanGrids.end(); ++patchIt)
	{
		do
		{
			patchIt->state = UnknownState;

#ifdef DEBUG
			nPatches++;
			//rectangle(outputFrame, patch, Scalar(255,255,255));
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
			detectedBoundingBoxes.push_back(&(*patchIt));

#ifdef DEBUG
			nFinalPatches++;
			rectangle(outputFrame, *patchIt, Scalar(0,255,255), 1);
#endif
		} while (false);

		//rectangle(outputFrame, *patchIt, Scalar(255,0,0), 2);
	}

#ifdef DEBUG
	cout << "Total patches: " << nPatches << endl;
	cout << "After Patch variance: " << nEnsemblePatches << endl;
	cout << "After Ensemble classifier: " << nNNPatches << endl;
	cout << "After NN classifier: " << nFinalPatches << endl;
	cout << "Maximum NN Similairty: " << maxNNSimilarity << endl;
	//cout << endl;
	//rectangle(outputFrame, *maxNNpatchIt, Scalar(0, 0, 255), 2);
#endif

	//finalBoundingBox = maxNNSimilarity <= NN_THRESHOLD ? NULL : &(*maxNNpatchIt);
	if (!detectedBoundingBoxes.empty())
	{
		cluster();
	}
}

void Detector::cluster()
{
	vector<int> clusterInd;
	int nClusters;

	if (detectedBoundingBoxes.size() == 1)
	{
		clusteredBoundingBoxes.push_back(*detectedBoundingBoxes[0]);
		nClusters = 1;
		return;
	}
	else if (detectedBoundingBoxes.size() == 2)
	{
		clusterInd.push_back(0);
		clusterInd.push_back(detectedBoundingBoxes[0]->getOverlap(*detectedBoundingBoxes[1]) < DETECTOR_CLUSTER_MIN_OVERLAP ? 1 : 0);
		nClusters = clusterInd[1] == 1 ? 2 : 1;
	}
	else
	{
		nClusters = partition(detectedBoundingBoxes, clusterInd, isIdenticalBoundingBox);
	}

	int* counters = new int[nClusters];
	memset(counters, 0, nClusters * sizeof(int));

	//create clusters
	for (int i = 0; i < nClusters; i++)	
	{
		clusteredBoundingBoxes.push_back(BoundingBox(0, 0, 0, 0));
		//boundingBoxes[i].conservativeSimilarity = 0;
		//boundingBoxes[i].relativeSimilarity = 0;
		clusteredBoundingBoxes[i].state = DetectedCluster;
	}

	//find average
	for (int i = 0; i < detectedBoundingBoxes.size(); i++)
	{
		clusteredBoundingBoxes[clusterInd[i]].x += detectedBoundingBoxes[i]->x;
		clusteredBoundingBoxes[clusterInd[i]].y += detectedBoundingBoxes[i]->y;
		clusteredBoundingBoxes[clusterInd[i]].width += detectedBoundingBoxes[i]->width;
		clusteredBoundingBoxes[clusterInd[i]].height += detectedBoundingBoxes[i]->height;
		counters[clusterInd[i]]++;
	}

	for (int i = 0; i < nClusters; i++)
	{
#ifdef DEBUG
		assert(counters[i] > 0);
#endif

		clusteredBoundingBoxes[i].x = cvRound(clusteredBoundingBoxes[i].x / (double)counters[i]);
		clusteredBoundingBoxes[i].y = cvRound(clusteredBoundingBoxes[i].y / (double)counters[i]);
		clusteredBoundingBoxes[i].width = cvRound(clusteredBoundingBoxes[i].width / (double)counters[i]);
		clusteredBoundingBoxes[i].height = cvRound(clusteredBoundingBoxes[i].height / (double)counters[i]);
		rectangle(outputFrame, clusteredBoundingBoxes[i], Scalar(0, 128, 255), 2);
	}
	#ifdef DEBUG
		cout << "Clusters: " << nClusters << endl;
	#endif
	
	delete [] counters;
}

bool Detector::isIdenticalBoundingBox(const BoundingBox* bb1, const BoundingBox* bb2)
{
	return (bb1->getOverlap(*bb2) >= DETECTOR_CLUSTER_MIN_OVERLAP);
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



