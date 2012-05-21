#include "TLD.h"
using namespace cv;
using namespace std;

TLD::TLD() : tracker(detector), trainer(detector)
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

	//detector.update(frame, outputFrame);
	tracker.update(frame, outputFrame);	

	//rectangle(this->outputFrame, detector.getBoudingBox(), Scalar(0, 0, 255), 2);	//red
	rectangle(this->outputFrame, tracker.getBoundingBox(), Scalar(0, 255, 255), 2);	//yellow

#ifdef DEBUG
	//cout << "Detector (R) Confidence: " << detector.getBoudingBox().confidence << endl;
	cout << "Tracker (Y) Confidence: " << tracker.getBoundingBox().confidence << endl;
	cout << endl;
#endif
}