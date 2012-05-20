#include "NNClassifier.h"
using namespace cv;
using namespace std;

void NNClassifier::train(const cv::Mat& patchImg, bool isPositive)
{
	Mat normPatch;
	resize(patchImg, normPatch, Size(NORMALIZED_PATCH_SIZE, NORMALIZED_PATCH_SIZE));
	
	if (isPositive)
	{
		positiveSamples.push_back(normPatch);
	}
	else
	{
		negativeSamples.push_back(normPatch);
	}
}

float NNClassifier::getRelativeSimilarity(const cv::Mat& patchImg) const	//TODO: Is k*O(log(n)) possible?
{
	//TODO: No Gaussian blur?

	Mat normPatch;
	resize(patchImg, normPatch, Size(NORMALIZED_PATCH_SIZE, NORMALIZED_PATCH_SIZE));

	float maxPos = 0;
	for (vector<Mat>::const_iterator it = positiveSamples.begin(); it != positiveSamples.end(); ++it)
	{
		float pos = getSimilarity(normPatch, *it);
		if (pos > maxPos)
		{
			maxPos = pos;
		}
	}

	float maxNeg = 0;
	for (vector<Mat>::const_iterator it = negativeSamples.begin(); it != negativeSamples.end(); ++it)
	{
		float neg = getSimilarity(normPatch, *it);
		if (neg > maxNeg)
		{
			maxNeg = neg;
		}
	}

#ifdef DEBUG
	assert(maxPos + maxNeg > 0);
#endif

	return maxPos / (maxPos + maxNeg);
}