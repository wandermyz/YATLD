#include "PixelComparator.h"
using namespace cv;

PixelComparator::PixelComparator()
{
	/*srcPoints = new Point2f[NUM_FERN_FEATURES];
	dstPoints = new Point2f[NUM_FERN_FEATURES];
	
	nPositive = new int[FERN_INDEX_SIZE];
	nTotal = new int[FERN_INDEX_SIZE];
	distribution = new float[FERN_INDEX_SIZE];*/

	memset(nPositive, 0, FERN_INDEX_SIZE * sizeof(int));
	memset(nTotal, 0, FERN_INDEX_SIZE * sizeof(int));
	memset(distribution, 0, FERN_INDEX_SIZE * sizeof(float));
}

PixelComparator::~PixelComparator()
{
	/*delete [] srcPoints;
	srcPoints = NULL;

	delete [] dstPoints;
	dstPoints = NULL;

	delete [] nPositive;
	nPositive = NULL;

	delete [] nTotal;
	nTotal = NULL;

	delete [] distribution;
	distribution = NULL;*/
}

void PixelComparator::init(RNG& rng)
{
	for (int j = 0; j < NUM_FERN_FEATURES; j++)
	{
		srcPoints[j].x = rng.uniform(0.0f, 1.0f);
		srcPoints[j].y = rng.uniform(0.0f, 1.0f);
		dstPoints[j].x = rng.uniform(0.0f, 1.0f);
		dstPoints[j].y = rng.uniform(0.0f, 1.0f);
	}
}
