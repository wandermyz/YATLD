#ifndef YATLD_SETTINGS_H_
#define YATLD_SETTINGS_H_

#include <opencv.hpp>

//#define MIN_BB_AREA 50
#define MIN_BB_SIDE 20
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

#define NORMALIZED_PATCH_SIZE 15
#define NN_THRESHOLD 0.6

#define TRACKER_GRID_POINT_NUM_H 10
#define TRACKER_GRID_POINT_NUM_V 10
#define TRACKER_GRID_PATCH_SIZE 10
#define LK_WINDOW_SIZE 4
#define LK_LEVEL 5
#define LK_TERM_COUNT 20
#define LK_TERM_EPSILON 0.03

#define TRACKER_GRID_POINT_NUM TRACKER_GRID_POINT_NUM_H * TRACKER_GRID_POINT_NUM_V

#endif
