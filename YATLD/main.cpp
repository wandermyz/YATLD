#include <opencv.hpp>
#include <fstream>
#include "TLD.h"
using namespace cv;
using namespace std;

BoundingBox readBoundingBox(const string& file){
  ifstream bb_file (file);
  if (!bb_file)
  {
	  cerr << "Error opening file: " << file << endl;
  }
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

	string videoPath, initPath;
	if (argc == 4 && string(argv[1]) == "-s")
	{
		videoPath = string("..\\..\\datasets\\") + argv[2] + "_" + argv[3] + "\\" + argv[3] + ".mpg";
		initPath = string("..\\..\\datasets\\") + argv[2] + "_" + argv[3] + "\\" + "init.txt";
	}
	else if (argc == 3)
	{
		videoPath = argv[1];
		initPath = argv[2];
	}
	else
	{
		return -1;
	}

	VideoCapture cap(videoPath);
	
	if (!cap.isOpened())
	{
		cerr << "Error opening file: " << videoPath << endl;
		return -1;
	}

	BoundingBox boundingBox = readBoundingBox(initPath);

	namedWindow("video", 1);
	
	Mat rgbFrame, frame, outputFrame;

	cout << "Frame 1" << endl;
	cap >> rgbFrame;
	outputFrame = rgbFrame.clone();
	cvtColor(rgbFrame, frame, CV_RGB2GRAY);
	 
	tld.init(frame, boundingBox, outputFrame);
	imshow("video", outputFrame);

	int frameCount = 2;
#ifdef DEBUG
	while(waitKey() != 27)
#else
	while(waitKey(1) != 27)
#endif
	{	
		cout << "Frame " << frameCount << endl;
		cap >> rgbFrame;
		outputFrame = rgbFrame.clone();
		cvtColor(rgbFrame, frame, CV_RGB2GRAY);

		tld.update(frame, outputFrame);

		boundingBox = tld.getBoundingBox();
		imshow("video", outputFrame);

		frameCount++;

		if (frameCount == cap.get(CV_CAP_PROP_FRAME_COUNT))
		{
			break;
		}
	}


	return 0;
}