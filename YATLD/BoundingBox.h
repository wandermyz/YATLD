#ifndef BOUNDING_BOX_H
#define BOUNDING_BOX_H

#include <opencv.hpp>

class BoundingBox : public cv::Rect
{
private:

public:
	BoundingBox() { }
	BoundingBox(const cv::Rect& rect) : cv::Rect(rect) { }
	BoundingBox(int x, int y, int w, int h) : cv::Rect(x, y, w, h) { }

	inline cv::Point tr() const { return cv::Point(x + width, y); }
	inline cv::Point bl() const { return cv::Point(x, y + height); }
	
	float overlap(const cv::Rect& ref) const
	{
		int w = std::max(0, std::min(x + width, ref.x + ref.width) - std::max(x, ref.x));
		int h = std::max(0, std::min(x + width, ref.x + ref.width) - std::max(x, ref.x));
		int unionArea = area() + ref.area() - w * h;
		return (float)(w * h) / (float)unionArea;
	}

	inline bool hasOverlap(const cv::Rect& ref) const
	{
		return (x + width >= ref.x) && (x <= ref.x + ref.width) && (y + height >= ref.y) && (y <= ref.y + ref.height);
	}
};

#endif