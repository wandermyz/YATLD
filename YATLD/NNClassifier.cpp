#include "NNClassifier.h"
//#include <omp.h>
using namespace cv;
using namespace std;


NNClassifier::NNClassifier()
{
	//omp_set_num_threads(8);
}

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
	float maxHalfPos = 0;
	float maxNeg = 0;

	int posNum = positiveSamples.size();
	if (relative == NULL)
	{
		posNum /= 2;
	}
	int negNum = negativeSamples.size();

	float* pos = new float[posNum];
	float* neg = new float[negNum];

	//double t = (double)getTickCount();
//#pragma omp parallel for
	for (int i = 0; i < posNum; i++)
	{
		pos[i] = getPairSimilarity(normPatch, positiveSamples[i]);

		//#pragma omp critical
		//cout << i << " ";
	}
	//cout << endl << endl;
	//t = ((double)getTickCount() - t)/getTickFrequency();
	//cout << t << endl;

//#pragma omp parallel for
	for (int i = 0; i < negNum; i++)
	{
		neg[i] = getPairSimilarity(normPatch, negativeSamples[i]);
	}

	for (int i = 0; i < posNum; i++)
	{
		if (pos[i] > maxPos)
		{
			maxPos = pos[i];
		}

		if (i == positiveSamples.size() / 2)
		{
			maxHalfPos = maxPos;
		}
	}

	for (int i = 0; i < negNum; i++)
	{
		if (neg[i] > maxNeg)
		{
			maxNeg = neg[i];
		}
	}

#ifdef DEBUG
	assert(maxPos + maxNeg > 0);
#endif

	if (relative != NULL)
	{
		//*relative = maxPos / (maxPos + maxNeg);
		*relative = (1 - maxNeg) / (2 - maxPos - maxNeg);		//?
	}

	if (conservative != NULL)
	{
		//*conservative = maxHalfPos / (maxHalfPos + maxNeg);
		*conservative = (1 - maxNeg) / (2 - maxHalfPos - maxNeg);	//?
	}

	delete [] pos;
	delete [] neg;
}

void NNClassifier::forgetPositive(int count)
{
	//TODO: more efficient and randomized
	RNG rng;
	for (int i = 0; i < count; i++)	
	{
		int ind = rng.uniform(0, positiveSamples.size());
		positiveSamples.erase(positiveSamples.begin() + ind);
	}
}

void NNClassifier::forgetNegative(int count)
{
	//TODO: more efficient and randomized
	RNG rng;
	for (int i = 0; i < count; i++)	
	{
		int ind = rng.uniform(0, negativeSamples.size());
		negativeSamples.erase(negativeSamples.begin() + ind);
	}
}
