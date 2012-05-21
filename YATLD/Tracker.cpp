#include "Tracker.h"
#include "Settings.h"
#include <algorithm>
using namespace cv;
using namespace std;

Tracker::Tracker(Detector& detector) : detector(detector), lkWindowSize(LK_WINDOW_SIZE, LK_WINDOW_SIZE), 
	lkTermCreteria(TermCriteria::COUNT+TermCriteria::EPS, LK_TERM_COUNT, LK_TERM_EPSILON)
{

}

void Tracker::init(const Mat& frame, const BoundingBox& initBoundingBox)
{
	this->prevFrame = frame.clone();
	this->prevBoundingBox = initBoundingBox;
}

void Tracker::update(const Mat& frame, Mat& outputFrame)
{
	//generate grid points
	prevPoints.clear();
	Point p;
	
	int stepH = max(1, (int)((double)prevBoundingBox.width / (double)(TRACKER_GRID_POINT_NUM_H + 1) + 0.5));
	int stepV = max(1, (int)((double)prevBoundingBox.height / (double)(TRACKER_GRID_POINT_NUM_V + 1) + 0.5));

	for (p.y = prevBoundingBox.y + stepV; p.y <= prevBoundingBox.br().y - stepV; p.y += stepV)
	{
		for (p.x = prevBoundingBox.x + stepH; p.x <= prevBoundingBox.br().x - stepV; p.x += stepH)
		{
			prevPoints.push_back(p);
		}
	}

	//draw grid
	for (int i = 0; i < prevPoints.size(); i++)
	{
		circle(outputFrame, prevPoints[i], 2, Scalar(255, 255, 255), 1);
	}
	rectangle(outputFrame, prevBoundingBox, Scalar(255, 0, 0));

	//L-K optical flow
	calcOpticalFlowPyrLK(prevFrame, frame, prevPoints, nextPoints, status, errors, lkWindowSize, LK_LEVEL, lkTermCreteria, LK_DERIV_LAMBDA, 0);
	calcOpticalFlowPyrLK(frame, prevFrame, nextPoints, backwardPoints, backwardStatus, backwardErrors, lkWindowSize, LK_LEVEL, lkTermCreteria, LK_DERIV_LAMBDA, 0);

	int k;

	//filter by status
	//int k = 0;
	//for (int i = 0; i < nextPoints.size(); i++)
	//{
	//	if (status[i] == 1 /*&& backwardStatus[i] == 1*/)
	//	{
	//		prevPoints[k] = prevPoints[i];
	//		nextPoints[k] = nextPoints[i];
	//		//backwardPoints[k] = backwardPoints[i];
	//		//errors[k] = errors[i];
	//		k++;
	//	}
	//}
	//nGoodPoints = k;

//#ifdef DEBUG
//	cout << "Filtered by status: " << nGoodPoints << endl;
//#endif

	nGoodPoints = nextPoints.size();

	//compute FB errors
	fbErrors.clear();
	for (int i = 0; i < nGoodPoints; i++)
	{
		fbErrors.push_back(pow((float)(backwardPoints[i].x - prevPoints[i].x), 2) + pow((float)(backwardPoints[i].y - prevPoints[i].y), 2));
		//fbErrors.push_back(norm(backwardPoints[i] - prevPoints[i]));
	}

	float medianFbErr = median(fbErrors, nGoodPoints);

	//compute NCC
	ncc.clear();
	Mat prevSub, nextSub, res;
	for (int i = 0; i < nGoodPoints; i++)
	{
		getRectSubPix( prevFrame, Size(TRACKER_GRID_PATCH_SIZE,TRACKER_GRID_PATCH_SIZE), prevPoints[i], prevSub);
		getRectSubPix( frame, Size(TRACKER_GRID_PATCH_SIZE,TRACKER_GRID_PATCH_SIZE), nextPoints[i], nextSub);
		matchTemplate( prevSub, nextSub, res, CV_TM_CCOEFF_NORMED);
		ncc.push_back(res.at<float>(0));
	}
	float medianNcc = median(ncc, nGoodPoints);
	
	//filter by NCC
	k = 0;
	for (int i = 0; i < nGoodPoints; i++)
	{
		if (status[i] == 1 && ncc[i] > medianNcc)
		{
			prevPoints[k] = prevPoints[i];
			nextPoints[k] = nextPoints[i];
			backwardPoints[k] = backwardPoints[i];
			backwardStatus[k] = backwardStatus[i];
			k++;
		}
	}
	nGoodPoints = k;
	

#ifdef DEBUG
	cout << "Filtered by NCC: " << nGoodPoints << endl;
#endif

	//Backward L-K optical flow
	//nextPoints.erase(nextPoints.begin() + nGoodPoints, nextPoints.end());
	//calcOpticalFlowPyrLK(frame, prevFrame, nextPoints, backwardPoints, backwardStatus, backwardErrors, lkWindowSize, LK_LEVEL, lkTermCreteria);
	
	//filter by backward status
	//k = 0;
	//for (int i = 0; i < nGoodPoints; i++)
	//{
	//	if (backwardStatus[i] == 1)
	//	{
	//		prevPoints[k] = prevPoints[i];
	//		nextPoints[k] = nextPoints[i];
	//		k++;
	//	}
	//}
	//nGoodPoints = k;

	/*vector<float> tmp(fbErrors);
	sort(tmp.begin(), tmp.end());
	for (int i = 0; i < tmp.size(); i++)
	{
		cout << tmp[i] << endl;
	}
	cout << "median = " << medianFbErr << endl;*/

	//filter by FB errors
 	k = 0;
	for (int i = 0; i < nGoodPoints; i++)
	{
		if (backwardStatus[i] == 1 && fbErrors[i] <= medianFbErr)
		{
			nextPoints[k] = nextPoints[i];
			prevPoints[k] = prevPoints[i];
			k++;
		}
	}
	nGoodPoints = k;

#ifdef DEBUG
	cout << "Filtered by FB: " << nGoodPoints << endl;
#endif

	tracked = nGoodPoints > 0;

	if (tracked)
	{
		//find x, y offsets
		xOffsets.clear();
		yOffsets.clear();
		displacement.clear();
		for (int i = 0; i < nGoodPoints; i++)
		{
			xOffsets.push_back(nextPoints[i].x - prevPoints[i].x);
			yOffsets.push_back(nextPoints[i].y - prevPoints[i].y);
			displacement.push_back(sqrt(pow(xOffsets[i], 2) + pow(yOffsets[i], 2)));
		}

		float xOffset = median(xOffsets);
		float yOffset = median(yOffsets);
		float displacementMedian = median(displacement);
		cout << displacementMedian << endl;

		//find residual
		residual.clear();
		for (int i = 0; i < nGoodPoints; i++)
		{
			residual.push_back(abs(displacement[i] - displacementMedian));
		}
		float residualMedian = median(residual);
		cout << residualMedian << endl;

		tracked = residualMedian <= LK_FAILURE_RESIDUAL;
		if (tracked)
		{
			//find scale
			sqScales.clear();
			for (int i = 0; i < nGoodPoints; i++)
			{
				xOffsets.push_back(nextPoints[i].x - prevPoints[i].x);
				yOffsets.push_back(nextPoints[i].y - prevPoints[i].y);
				for (int j = i + 1; j < nGoodPoints; j++)
				{
					sqScales.push_back(
						(pow(nextPoints[i].x - nextPoints[j].x, 2) + pow(nextPoints[i].y - nextPoints[j].y, 2))
						/ (pow(prevPoints[i].x - prevPoints[j].x, 2) + pow(prevPoints[i].y - prevPoints[j].y, 2))
						);
				}
			}
			float scale = sqScales.size() > 0 ? sqrt(median(sqScales)) : 1.0f;

			//printf("%f %f\n", xOffset, yOffset);

			//update bounding box
			float s1 = 0.5f * (scale-1) * prevBoundingBox.width;		//TODO: try scale with center of the median
			float s2 = 0.5f * (scale-1) * prevBoundingBox.height;
			prevBoundingBox.x += (int)(xOffset - s1 + 0.5);
			prevBoundingBox.y += (int)(yOffset - s2 + 0.5);
			prevBoundingBox.width = (int)(prevBoundingBox.width * scale + 0.5);
			prevBoundingBox.height = (int)(prevBoundingBox.height * scale + 0.5);

			//find conservative similarity
			BoundingBox trimedBoundingBox = prevBoundingBox & BoundingBox(0, 0, frame.cols, frame.rows);
			detector.getNNClassifier().getSimilarity(frame(trimedBoundingBox), NULL, &prevBoundingBox.confidence);

			//draw output
			for (int i = 0; i < nGoodPoints; i++) 
			{
				//circle(outputFrame, prevPoints[i], 2, Scalar(255, 255, 255), 1);
				//circle(outputFrame, nextPoints[i], 2, Scalar(0, 255, 0), 1);
				line(outputFrame, prevPoints[i], nextPoints[i], Scalar(0, 255, 0), 1);
			}
			//rectangle(outputFrame, prevBoundingBox, Scalar(0, 255, 255), 2);
		}
	}
	
	prevFrame = frame.clone();
}

