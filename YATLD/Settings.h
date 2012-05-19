#ifndef YATLD_SETTINGS_H_
#define YATLD_SETTINGS_H_

#include <opencv.hpp>

#define MIN_BB_AREA 50
#define STEP_H 0.1
#define STEP_V 0.1
#define STEP_S 1.2

#define INIT_SHIFT_CHANGE 0.01
#define INIT_SCALE_CHANGE 0.01
#define INIT_ROTATION_CHANGE 0 //10.0 //* CV_PI / 180.0
#define INIT_WARP_NUM 20
#define INIT_POSITIVE_NUM 10
#define INIT_NEGATIVE_NUM 20	//?
#define INIT_GAUSSIAN_SIGMA 5
#define INIT_POSITIVE_OVERLAP 0.3	//too small?
#define INIT_NEGATIVE_OVERLAP 0.2
#define INIT_NEGATIVE_MIN_SCALE 0.5 //relative to init bounding box **

#define NUM_FERNS 10
#define NUM_FERN_FEATURES 13
#define FERN_INDEX_SIZE 8192	//2^13 = 8192
#define FERN_GAUSSIAN_SIGMA 3


#endif
