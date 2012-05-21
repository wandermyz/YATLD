#include "TLD.h"
using namespace cv;
using namespace std;

TLD::TLD() : tracker(detector), trainer(detector, tracker)
{
}

void TLD::init(const Mat& frame, const BoundingBox& boundingBox, Mat& outputFrame)
{
	this->frame = frame;
	this->boundingBox = boundingBox;
	this->outputFrame = outputFrame;

	detector.init(frame, boundingBox, outputFrame);
	tracker.init(frame, boundingBox);
	trainer.init(frame, boundingBox);
	rectangle(this->outputFrame, boundingBox.tl(), boundingBox.br(), Scalar(0, 255, 255), 2);
}

void TLD::update(const Mat& frame, Mat& outputFrame)
{
	this->frame = frame;
	this->outputFrame = outputFrame;

	detector.update(frame, outputFrame);
	tracker.update(frame, outputFrame);	
	
	const BoundingBox* result = NULL;
	
	if (detector.getBoundingBox() == NULL && tracker.getBoundingBox() == NULL)
	{
		result = NULL;
	}
	else if (detector.getBoundingBox() == NULL && tracker.getBoundingBox() != NULL)
	{
		result = tracker.getBoundingBox();
	}
	else if (detector.getBoundingBox() != NULL && tracker.getBoundingBox() == NULL)
	{
		result = detector.getBoundingBox();
	}
	else if (detector.getBoundingBox()->confidence > tracker.getBoundingBox()->confidence)
	{
		result = detector.getBoundingBox();
	}
	else
	{
		result = tracker.getBoundingBox();
	}

	if (detector.getBoundingBox() != NULL)
	{
		result = detector.getBoundingBox();
		rectangle(this->outputFrame, *detector.getBoundingBox(), Scalar(0, 0, 255), 1);	//red
	}

	if (tracker.getBoundingBox() != NULL)
	{
		if (result == NULL || tracker.getBoundingBox()->confidence > result->confidence)
		{
			result = tracker.getBoundingBox();
		}
		rectangle(this->outputFrame, *tracker.getBoundingBox(), Scalar(0, 255, 255), 1);	//yellow
	}

	if (result != NULL)
	{
		rectangle(this->outputFrame, *result, Scalar(255,0,0), 2);
	}

	trainer.update(frame);

//#ifdef DEBUG
	cout << "Detector (R) Confidence: " << (detector.getBoundingBox() == NULL ? -1 : detector.getBoundingBox()->confidence) << endl;
	cout << "Tracker (Y) Confidence: " << (tracker.getBoundingBox() == NULL ? -1 : tracker.getBoundingBox()->confidence) << endl;
	cout << endl;
//#endif
}