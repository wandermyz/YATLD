#ifndef YATLD_BOUNDING_BOX_H
#define YATLD_BOUNDING_BOX_H

#include <opencv.hpp>
#include <iostream>

typedef enum
{
	UnknownState,	//0
	RejectedByVariance,	//1
	RejectedByEnsemble,	//2
	RejectedByNN,		//3
	DetectedAcceptedByNN,	//4
	DetectedCluster,		//5
	TrackedRejectedByNN,	//6
	TrackedAcceptedByNN,	//7
	TrackedWeighted,		//8
} PatchState;

class BoundingBox : public cv::Rect
{
private:
	float overlap;	//will be refreshed in Detector::init()  !important: currently, detector won't refresh it in update()!!

public:
	float relativeSimilarity;
	float conservativeSimilarity;
	PatchState state;

#ifdef DEBUG
	BoundingBox() : overlap(-1), relativeSimilarity(-1), conservativeSimilarity(-1), state(UnknownState) { }
	BoundingBox(const cv::Rect& rect) : cv::Rect(rect), overlap(-1), relativeSimilarity(-1), conservativeSimilarity(-1), state(UnknownState) { }
	BoundingBox(int x, int y, int w, int h) : cv::Rect(x, y, w, h), overlap(-1), relativeSimilarity(-1), conservativeSimilarity(-1), state(UnknownState) { }
#else
	BoundingBox() { }
	BoundingBox(const cv::Rect& rect) : cv::Rect(rect) { }
	BoundingBox(int x, int y, int w, int h) : cv::Rect(x, y, w, h) { }
#endif

	inline cv::Point tr() const { return cv::Point(x + width, y); }
	inline cv::Point bl() const { return cv::Point(x, y + height); }

	inline float refreshOverlap(const cv::Rect& ref)
	{
		overlap = getOverlap(ref);
		return overlap;
	}

	float getOverlap(const cv::Rect& ref) const
	{
		int w = std::max(0, std::min(x + width, ref.x + ref.width) - std::max(x, ref.x));
		int h = std::max(0, std::min(y + height, ref.y + ref.height) - std::max(y, ref.y));
		int unionArea = area() + ref.area() - w * h;
		float tmpOverlap = (float)(w * h) / (float)unionArea;
		return tmpOverlap;
	}

	inline float getOverlap() const
	{
#ifdef DEBUG
		assert(overlap >= 0);
#endif
		return overlap;
	}

	inline bool hasOverlap(const cv::Rect& ref) const
	{
		return (x + width >= ref.x) && (x <= ref.x + ref.width) && (y + height >= ref.y) && (y <= ref.y + ref.height);
	}

	inline bool isPositive() const
	{
		return state == DetectedAcceptedByNN || state == TrackedAcceptedByNN || state == TrackedRejectedByNN;
	}

#ifdef DEBUG
	inline void resetStates()
	{
		overlap = -1;
		relativeSimilarity = -1;
		conservativeSimilarity = -1;
		state = UnknownState;
	}
#endif
};

/*
class CompareOverlap
{
public:
	inline bool operator() (const BoundingBox* bb1, const BoundingBox* bb2)
	{
		return bb1->getOverlap() < bb2->getOverlap();	
	}
};
*/

std::ostream& operator<<(std::ostream& os, const BoundingBox& boundingBox);

#endif