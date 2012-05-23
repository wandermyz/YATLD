#include "EnsembleClassifier.h"
#include <stdio.h>
using namespace cv;

EnsembleClassifier::EnsembleClassifier()
{
	comparators = new PixelComparator[NUM_FERNS];
}

EnsembleClassifier::~EnsembleClassifier()
{
	delete [] comparators;
	comparators = NULL;
}

void EnsembleClassifier::init(const Mat& frame)
{
	RNG rng;

	//generate base classifiers
	for (int i = 0; i < NUM_FERNS; i++)
	{
		comparators[i].init(rng);
	}

	this->frame = frame;
	GaussianBlur(frame, frameBlurred, cv::Size(0, 0), FERN_GAUSSIAN_SIGMA);
}

void EnsembleClassifier::train(const Mat& patchImg, bool isPositive)
{
	for (int i = 0; i < NUM_FERNS; i++)
	{
		comparators[i].train(patchImg, isPositive);
	}
}

float EnsembleClassifier::getPosterior(const Mat& patchImg) const
{
	double posterior = 0;
	for (int i = 0; i < NUM_FERNS; i++)
	{
		posterior += comparators[i].getPosterior(patchImg);
	}

	//printf("%f\n", posterior / NUM_FERNS);
	return (float)(posterior / NUM_FERNS);
}

