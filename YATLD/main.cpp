#include <opencv.hpp>
#include <fstream>
#include "TLD.h"
using namespace cv;
using namespace std;

BoundingBox readBoundingBox(char* file){
  ifstream bb_file (file);
  string line;
  getline(bb_file,line);
  istringstream linestream(line);
  string x1,y1,x2,y2;
  getline (linestream,x1, ',');
  getline (linestream,y1, ',');
  getline (linestream,x2, ',');
  getline (linestream,y2, ',');
  int x = atoi(x1.c_str());// = (int)file["bb_x"];
  int y = atoi(y1.c_str());// = (int)file["bb_y"];
  int w = atoi(x2.c_str())-x;// = (int)file["bb_w"];
  int h = atoi(y2.c_str())-y;// = (int)file["bb_h"];
  return BoundingBox(x,y,w,h);
}

int main(int argc, char* argv[])
{
	TLD tld;

	VideoCapture cap(argv[1]);
	
	if (!cap.isOpened())
	{
		return -1;
	}

	BoundingBox boundingBox = readBoundingBox(argv[2]);

	namedWindow("video", 1);
	
	Mat rgbFrame, frame, outputFrame;

	cap >> rgbFrame;
	outputFrame = rgbFrame.clone();
	cvtColor(rgbFrame, frame, CV_RGB2GRAY);
	 
	tld.init(frame, boundingBox, outputFrame);
	imshow("video", outputFrame);

	while(waitKey() != 27)
	{	
		cap >> rgbFrame;
		outputFrame = rgbFrame.clone();
		cvtColor(rgbFrame, frame, CV_RGB2GRAY);

		tld.update(frame, outputFrame);

		boundingBox = tld.getBoundingBox();
		imshow("video", outputFrame);
	}


	return 0;
}