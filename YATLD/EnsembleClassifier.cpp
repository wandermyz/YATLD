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

void EnsembleClassifier::init()
{
	RNG rng;

	//generate base classifiers
	for (int i = 0; i < NUM_FERNS; i++)
	{
		comparators[i].init(rng);
	}
}

void EnsembleClassifier::train(const BoundingBox& patch, bool isPositive)
{
	for (int i = 0; i < NUM_FERNS; i++)
	{
		comparators[i].train(frameBlurred(patch), isPositive);
	}
}

bool EnsembleClassifier::accept(const BoundingBox& patch) const
{
	double posterior = 0;
	for (int i = 0; i < NUM_FERNS; i++)
	{
		posterior += comparators[i].getPosterior(frameBlurred(patch));
	}

	//printf("%f\n", posterior / NUM_FERNS);
	return posterior / NUM_FERNS >= 0.5;
}
