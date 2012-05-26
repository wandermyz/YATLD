#ifndef YATLD_PIXEL_COMPARATOR
#define YATLD_PIXEL_COMPARATOR

#include <opencv.hpp>
#include "Settings.h"

class PixelComparator
{
private:
	cv::Point2f srcPoints[NUM_FERN_FEATURES];
	cv::Point2f dstPoints[NUM_FERN_FEATURES];
	int nPositive[FERN_INDEX_SIZE];
	int nTotal[FERN_INDEX_SIZE];
	float distribution[FERN_INDEX_SIZE];

public:
	PixelComparator();
	virtual ~PixelComparator();

	void init(cv::RNG& rng);

	inline int encode(const cv::Mat& patch) const
	{
		int ind = 0;

		for (int j = 0; j < NUM_FERN_FEATURES; j++)
		{
			ind <<= 1;
			bool cmp = patch.at<uchar>((int)(srcPoints[j].y * patch.rows), (int)(srcPoints[j].x * patch.cols)) 
					  > patch.at<uchar>((int)(dstPoints[j].y * patch.rows), (int)(dstPoints[j].x * patch.cols));
			ind += cmp ? 1 : 0;
		}

#ifdef DEBUG
		assert(ind < FERN_INDEX_SIZE);
#endif

		return ind;
	}

	inline void train(const cv::Mat& patch, bool isPositive)
	{
		int ind = encode(patch);

		nTotal[ind]++;
		if (isPositive)
		{
			nPositive[ind]++;
		}
		
		distribution[ind] = (float)nPositive[ind] / (float)nTotal[ind];	//total won't be zero
	}

	inline float getPosterior(const cv::Mat& patch) const
	{
		return distribution[encode(patch)];
	}

};

#endif