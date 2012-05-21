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

void NNClassifier::getSimilarity(const cv::Mat& patchImg, float* relative, float* conservative) const	//TODO: Is k*O(log(n)) possible?
{
	//TODO: No Gaussian blur?
	assert(relative != NULL || conservative != NULL);

	Mat normPatch;
	resize(patchImg, normPatch, Size(NORMALIZED_PATCH_SIZE, NORMALIZED_PATCH_SIZE));

	float maxPos = 0;
	float maxHalfPos;

	for (int i = 0; i < positiveSamples.size(); i++)
	{
		float pos = getPairSimilarity(normPatch, positiveSamples[i]);
		if (pos > maxPos)
		{
			maxPos = pos;
		}
		if (i == positiveSamples.size() / 2)
		{
			maxHalfPos = maxPos;
			if (relative == NULL)
			{
				break;
			}
		}
	}

	float maxNeg = 0;
	for (vector<Mat>::const_iterator it = negativeSamples.begin(); it != negativeSamples.end(); ++it)
	{
		float neg = getPairSimilarity(normPatch, *it);
		if (neg > maxNeg)
		{
			maxNeg = neg;
		}
	}

#ifdef DEBUG
	assert(maxPos + maxNeg > 0);
#endif

	if (relative != NULL)
	{
		*relative = maxPos / (maxPos + maxNeg);
	}

	if (conservative != NULL)
	{
		*conservative = maxHalfPos / (maxHalfPos + maxNeg);
	}
}
