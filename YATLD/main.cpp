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

	string videoPath, initPath, outputPath;
	if (argc == 4)
	{
		string dir = string(argv[1]) + "\\";
		videoPath = dir + argv[2] + "_" + argv[3] + "\\" + argv[3] + ".mpg";
		initPath = dir + argv[2] + "_" + argv[3] + "\\" + "init.txt";
		outputPath = dir + argv[2] + "_" + argv[3] + "\\" + "YATLD.txt";
	}
	else if (argc == 3)
	{
		videoPath = argv[1];
		initPath = argv[2];
		outputPath = ".\\YATLD.txt";
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

	ofstream fout(outputPath);

	if (!fout)
	{
		cerr << "Cannot create file: " << outputPath << endl;
		return -1;
	}

	BoundingBox boundingBox = readBoundingBox(initPath);
	fout << boundingBox << endl;

	namedWindow("video", 1);
	
	Mat rgbFrame, frame, outputFrame;
	
	int frameCount = 1;
	cout << "Frame " << frameCount << endl;
	cap >> rgbFrame;
	outputFrame = rgbFrame.clone();
	cvtColor(rgbFrame, frame, CV_RGB2GRAY);
	 
	tld.init(frame, boundingBox, outputFrame);
	imshow("video", outputFrame);

	int nFrames = cap.get(CV_CAP_PROP_FRAME_COUNT);

#ifdef DEBUG
	while(waitKey(frameCount < 23 ? 1 : 0) != 27)
#else
	while(waitKey(1) != 27)
#endif
	{	
		frameCount++;
		cout << "Frame " << frameCount << endl;
		cap >> rgbFrame;
		outputFrame = rgbFrame.clone();
		cvtColor(rgbFrame, frame, CV_RGB2GRAY);

		tld.update(frame, outputFrame);
		imshow("video", outputFrame);

		if (tld.getBoundingBox() != NULL)
		{
			fout << *tld.getBoundingBox() << endl;
		}
		else
		{
			fout << "NaN,NaN,NaN,NaN" << endl;
		}

		if (frameCount == cap.get(CV_CAP_PROP_FRAME_COUNT))
		{
			break;
		}
	}
	fout.close();

	return 0;
}