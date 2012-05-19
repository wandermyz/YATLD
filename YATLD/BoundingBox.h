#ifndef YATLD_BOUNDING_BOX_H
#define YATLD_BOUNDING_BOX_H

#include <opencv.hpp>

class BoundingBox : public cv::Rect
{
private:
	float overlap;	//will be refreshed in Detector

public:
	BoundingBox() : overlap(0) { }
	BoundingBox(const cv::Rect& rect) : cv::Rect(rect), overlap(0) { }
	BoundingBox(int x, int y, int w, int h) : cv::Rect(x, y, w, h), overlap(0) { }

	inline cv::Point tr() const { return cv::Point(x + width, y); }
	inline cv::Point bl() const { return cv::Point(x, y + height); }

	float refreshOverlap(const cv::Rect& ref)
	{
		int w = std::max(0, std::min(x + width, ref.x + ref.width) - std::max(x, ref.x));
		int h = std::max(0, std::min(y + height, ref.y + ref.height) - std::max(y, ref.y));
		int unionArea = area() + ref.area() - w * h;
		overlap = (float)(w * h) / (float)unionArea;
		return overlap;
	}

	inline float getOverlap() const
	{
		return overlap;
	}

	inline bool hasOverlap(const cv::Rect& ref) const
	{
		return (x + width >= ref.x) && (x <= ref.x + ref.width) && (y + height >= ref.y) && (y <= ref.y + ref.height);
	}
};

class CompareOverlap
{
public:
	inline bool operator() (const BoundingBox* bb1, const BoundingBox* bb2)
	{
		return bb1->getOverlap() < bb2->getOverlap();
	}
};

#endif