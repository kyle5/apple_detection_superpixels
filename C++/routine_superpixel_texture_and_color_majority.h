#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include <algorithm>
#include <vector>
#include <stdexcept>
#include <iostream>

#include <stdio.h>

extern "C" {
    #include "vl/generic.h"
    #include "vl/slic.h"
}

#include "kdtree.h"

enum orchard_classification { APPLE, LEAF, TREE_TRUNK, TOO_DARK };
using namespace std;
using namespace cv;

// this gets the pixels that are valid in the training mask
Mat compute_valid_pixels( Mat input_img_rgb_uchar_train_mask );
// this gets the grid points from a superpixels image
vector<vector<long> > compute_grid_points_over_superpixels( Mat input_img );
// this gets the features at specified indices from an image
Mat compute_features_over_image( Mat input_img_rgb_uchar, vector<vector<long> > grid_keypoints, string feature_type );
// this creates the machine learning structure used to classify the features that were computed from the image
// KD_tree_kyle create_kd_tree( vector<Mat> features_over_image );
// this computes the superpixels throughout the image
Mat compute_superpixels( Mat input_img_rgb_uchar );
vector<orchard_classification> query_superpixel_features( flann::Index * kdtrees, vector<orchard_classification> combined_labels, Mat cur_features );
Mat multiply_matrix_by_255( Mat valid_img_8U );
Mat find_equal_16U( Mat superpixel_img, int compare_number_input );
