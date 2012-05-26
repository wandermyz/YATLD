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
#ifdef DEBUG
	cout << "[Tracker]" << endl;
#endif

	tracked = false;
	
	//generate grid points
	prevPoints.clear();
	Point p;
	
	int stepH = ceil((double)prevBoundingBox.width / (double)TRACKER_GRID_POINT_NUM_H);
	int stepV = ceil((double)prevBoundingBox.height / (double)TRACKER_GRID_POINT_NUM_V);

	for (p.y = prevBoundingBox.y; p.y < prevBoundingBox.br().y; p.y += stepV)
	{
		for (p.x = prevBoundingBox.x; p.x < prevBoundingBox.br().x; p.x += stepH)
		{
			prevPoints.push_back(p);
		}
	}

	//draw grid
#ifdef DEBUG
	cout << "Optical flow points: " << prevPoints.size() << endl;
#endif

	for (int i = 0; i < prevPoints.size(); i++)
	{
		circle(outputFrame, prevPoints[i], 2, Scalar(255, 255, 255), 1);
	}

	do
	{
		//L-K optical flow
		calcOpticalFlowPyrLK(prevFrame, frame, prevPoints, nextPoints, status, errors, lkWindowSize, LK_LEVEL, lkTermCreteria);
		calcOpticalFlowPyrLK(frame, prevFrame, nextPoints, backwardPoints, backwardStatus, backwardErrors, lkWindowSize, LK_LEVEL, lkTermCreteria);

		//compute FB errors
		int nGoodPoints = nextPoints.size();
		fbErrors.clear();
		for (int i = 0; i < nGoodPoints; i++)
		{
			fbErrors.push_back(norm(backwardPoints[i] - prevPoints[i]));
		}

		//compute NCC
		ncc.clear();
		Mat prevSub, nextSub, res;
		for (int i = 0; i < nGoodPoints; i++)
		{
			if (status[i] == 1)
			{
				getRectSubPix( prevFrame, Size(TRACKER_GRID_PATCH_SIZE,TRACKER_GRID_PATCH_SIZE), prevPoints[i], prevSub);                
				getRectSubPix( frame, Size(TRACKER_GRID_PATCH_SIZE,TRACKER_GRID_PATCH_SIZE), nextPoints[i], nextSub);
				matchTemplate( prevSub, nextSub, res, CV_TM_CCOEFF_NORMED);
				ncc.push_back(res.at<float>(0));
			}
			else
			{
				ncc.push_back(0);
			}
		}

	
		//filter by NCC
		float medianNcc = median(ncc, nGoodPoints);
		int k = 0;
		for (int i = 0; i < nGoodPoints; i++)
		{
			if (status[i] == 1 && ncc[i] > medianNcc)
			{
				prevPoints[k] = prevPoints[i];
				nextPoints[k] = nextPoints[i];
				fbErrors[k] = fbErrors[i];
				backwardStatus[k] = backwardStatus[i];
				k++;
			}
		}
		if (k == 0)
		{
			break;
		}
		nGoodPoints = k;
	

	#ifdef DEBUG
		cout << "Filtered by errors: " << nGoodPoints << endl;
	#endif

		//filter by FB errors
		float medianFbErr = median(fbErrors, nGoodPoints);
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
		if (k == 0)
		{
			break;
		}
		nGoodPoints = k;

	#ifdef DEBUG
		cout << "Filtered by FB: " << nGoodPoints << endl;
	#endif

		//find x, y offsets
		xOffsets.clear();
		yOffsets.clear();
		displacement.clear();
		for (int i = 0; i < nGoodPoints; i++)
		{
			xOffsets.push_back(nextPoints[i].x - prevPoints[i].x);
			yOffsets.push_back(nextPoints[i].y - prevPoints[i].y);
			//displacement.push_back(sqrt(pow(xOffsets[i], 2) + pow(yOffsets[i], 2)));
		}

		float xOffset = median(xOffsets);
		float yOffset = median(yOffsets);
		//float displacementMedian = median(displacement);
		//cout << displacementMedian << endl;

		//find residual
		/*residual.clear();
		for (int i = 0; i < nGoodPoints; i++)
		{
			residual.push_back(abs(displacement[i] - displacementMedian));
		}
		float residualMedian = median(residual);
		//cout << residualMedian << endl;

		if (residualMedian > LK_FAILURE_RESIDUAL)
		{
			break;
		}*/

		//find scale
		scales.clear();
		for (int i = 0; i < nGoodPoints; i++)
		{
			for (int j = i + 1; j < nGoodPoints; j++)
			{
				scales.push_back(norm(nextPoints[i]-nextPoints[j]) / norm(prevPoints[i] - prevPoints[j]));
			}
		}
		float scale = scales.size() > 0 ? median(scales) : 1.0f;

		//printf("%f %f\n", xOffset, yOffset);

		//update bounding box
		float s1 = 0.5f * (scale-1) * prevBoundingBox.width;		//TODO: try scale with center of the median
		float s2 = 0.5f * (scale-1) * prevBoundingBox.height;
		
		//important!!! (int)(x + 0.5) not work if x < 0!!!
		prevBoundingBox.x = (int)(prevBoundingBox.x + xOffset - s1 + 0.5);
		prevBoundingBox.y = (int)(prevBoundingBox.y + yOffset - s2 + 0.5);

		prevBoundingBox.width = (int)(prevBoundingBox.width * scale + 0.5);
		prevBoundingBox.height = (int)(prevBoundingBox.height * scale + 0.5);

		//find conservative similarity
		Rect trimmed = prevBoundingBox & Rect(0, 0, frame.cols, frame.rows);

		if (trimmed.area() == 0)
		{
			break;
		}
		detector.getNNClassifier().getSimilarity(frame(trimmed), &prevBoundingBox.relativeSimilarity, &prevBoundingBox.conservativeSimilarity);
		prevBoundingBox.state = (prevBoundingBox.relativeSimilarity > NN_THRESHOLD) ? TrackedAcceptedByNN : TrackedRejectedByNN;

		//draw output
		for (int i = 0; i < nGoodPoints; i++) 
		{
			//circle(outputFrame, prevPoints[i], 2, Scalar(255, 255, 255), 1);
			//circle(outputFrame, nextPoints[i], 2, Scalar(0, 255, 0), 1);
			line(outputFrame, prevPoints[i], nextPoints[i], Scalar(0, 255, 0), 1);
		}
		//rectangle(outputFrame, prevBoundingBox, Scalar(0, 255, 255), 2);

		tracked = true;
	} while(0);
	
	prevFrame = frame.clone();
}

