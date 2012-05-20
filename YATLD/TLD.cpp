#include "TLD.h"
using namespace cv;
using namespace std;

TLD::TLD() : trainer(detector)
{
}

void TLD::init(const Mat& frame, const BoundingBox& boundingBox, Mat& outputFrame)
{
	this->frame = frame;
	this->boundingBox = boundingBox;
	this->outputFrame = outputFrame;

	tracker.init(frame, boundingBox);
	detector.init(frame, boundingBox, outputFrame);
	trainer.init(frame, boundingBox);
	rectangle(this->outputFrame, boundingBox.tl(), boundingBox.br(), Scalar(0, 255, 255), 2);
}

void TLD::update(const Mat& frame, Mat& outputFrame)
{
	this->frame = frame;
	this->outputFrame = outputFrame;

	tracker.update(frame, outputFrame);
	//detector.update(frame, outputFrame);
	//rectangle(this->outputFrame, boundingBox.tl(), boundingBox.br(), Scalar(0, 255, 255), 2);
}