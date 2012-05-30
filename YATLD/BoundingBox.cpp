#include "BoundingBox.h"

std::ostream& operator<<(std::ostream& os, const BoundingBox& boundingBox)
{
	os << boundingBox.x << "," << boundingBox.y << "," << boundingBox.br().x << "," << boundingBox.br().y << "," << boundingBox.state;
	return os;
}