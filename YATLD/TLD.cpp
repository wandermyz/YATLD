#include "TLD.h"
using namespace cv;
using namespace std;

TLD::TLD() : tracker(detector), trainer(detector, tracker)
{
}

void TLD::init(const Mat& frame, const BoundingBox& boundingBox, Mat& outputFrame)
{
	this->frame = frame;
	this->outputFrame = outputFrame;

	detector.init(frame, boundingBox, outputFrame);
	tracker.init(frame, boundingBox);
	trainer.init(frame, boundingBox);
	rectangle(this->outputFrame, boundingBox.tl(), boundingBox.br(), Scalar(0, 255, 255), 2);

#ifdef DEBUG
	cout << endl;
#endif
}

void TLD::update(const Mat& frame, Mat& outputFrame)
{
	this->frame = frame;
	this->outputFrame = outputFrame;

	detector.update(frame, outputFrame);
	tracker.update(frame, outputFrame);	
	
	float detectorConf = -1;
	float trackerConf = -1;
	if (detector.getBoundingBox() != NULL)
	{
		rectangle(this->outputFrame, *detector.getBoundingBox(), Scalar(0, 0, 255), 1);	//red
		detectorConf = detector.getBoundingBox()->conservativeSimilarity;
	}

	if (tracker.getBoundingBox() != NULL)
	{
		rectangle(this->outputFrame, *tracker.getBoundingBox(), Scalar(0, 255, 255), 1);	//yellow
		trackerConf = tracker.getBoundingBox()->conservativeSimilarity;
	}

	trainer.update(frame);
	if (trainer.getResult() != NULL)
	{
		rectangle(this->outputFrame, *trainer.getResult(), Scalar(255,0,0), 2);
	}

	
//#ifdef DEBUG
	cout << "[Result]" << endl;
	cout << "Detector (R) Confidence: " << detectorConf << endl;
	cout << "Tracker (Y) Confidence: " << trackerConf << endl;
	cout << "NN Samples: " << detector.getNNClassifier().getPositiveNum() << "+, " << detector.getNNClassifier().getNegativeNum() << "-"<< endl;
	cout << endl;
//#endif
}