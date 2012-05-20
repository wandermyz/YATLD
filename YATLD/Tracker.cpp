#include "Tracker.h"
#include "Settings.h"
#include <algorithm>
using namespace cv;
using namespace std;

Tracker::Tracker() : lkWindowSize(LK_WINDOW_SIZE, LK_WINDOW_SIZE), 
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

	//L-K optical flow
	calcOpticalFlowPyrLK(prevFrame, frame, prevPoints, nextPoints, status, errors, lkWindowSize, LK_LEVEL, lkTermCreteria);
	calcOpticalFlowPyrLK(frame, prevFrame, nextPoints, backwardPoints, backwardStatus, backwardErrors, lkWindowSize, LK_LEVEL, lkTermCreteria);

	//filter by status
	int k = 0;
	for (int i = 0; i < nextPoints.size(); i++)
	{
		if (status[i] == 1 && backwardStatus[i] == 1)
		{
			prevPoints[k] = prevPoints[i];
			nextPoints[k] = nextPoints[i];
			backwardPoints[k] = backwardPoints[i];
			errors[k] = errors[i];
			k++;
		}
	}
	nGoodPoints = k;

#ifdef DEBUG
	cout << "Filtered by status: " << nGoodPoints << endl;
#endif

	//filter by errors
	float medianErr = median(errors, nGoodPoints);
	k = 0;
	for (int i = 0; i < nGoodPoints; i++)
	{
		if (errors[i] <= medianErr)
		{
			prevPoints[k] = prevPoints[i];
			nextPoints[k] = nextPoints[i];
			backwardPoints[k] = backwardPoints[i];
			k++;
		}
	}
	nGoodPoints = k;

#ifdef DEBUG
	cout << "Filtered by errors: " << nGoodPoints << endl;
#endif

	//Backward L-K optical flow
	//nextPoints.erase(nextPoints.begin() + nGoodPoints, nextPoints.end());
	//calcOpticalFlowPyrLK(frame, prevFrame, nextPoints, backwardPoints, backwardStatus, backwardErrors, lkWindowSize, LK_LEVEL, lkTermCreteria);

	//compute FB errors
	fbErrors.clear();
	for (int i = 0; i < nGoodPoints; i++)
	{
		fbErrors.push_back(pow((float)(backwardPoints[i].x - prevPoints[i].x), 2) + pow((float)(backwardPoints[i].y - prevPoints[i].y), 2));
	}

	float medianFbErr = median(fbErrors, nGoodPoints);

	//filter by FB errors
 	k = 0;
	for (int i = 0; i < nGoodPoints; i++)
	{
		if (fbErrors[i] <= medianFbErr)
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

	//find boundingbox
	xOffsets.clear();
	yOffsets.clear();
	scales.clear();
	for (int i = 0; i < nGoodPoints; i++)
	{
		xOffsets.push_back(nextPoints[i].x - prevPoints[i].x);
		yOffsets.push_back(nextPoints[i].y - prevPoints[i].y);
		//TODO: scale
	}
	float xOffset = median(xOffsets);
	float yOffset = median(yOffsets);

	//printf("%f %f\n", xOffset, yOffset);

	prevBoundingBox.x += (int)(xOffset + 0.5);
	prevBoundingBox.y += (int)(yOffset + 0.5);

	//draw output
	for (int i = 0; i < nGoodPoints; i++) 
	{
		//circle(outputFrame, prevPoints[i], 2, Scalar(255, 255, 255), 1);
		//circle(outputFrame, nextPoints[i], 2, Scalar(0, 255, 0), 1);
		line(outputFrame, prevPoints[i], nextPoints[i], Scalar(0, 255, 0), 1);
	}
	rectangle(outputFrame, prevBoundingBox, (0, 255, 255), 2);
}

