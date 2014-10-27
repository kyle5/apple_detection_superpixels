#include "routine_superpixel_texture_and_color_majority.h"
#include<dirent.h>

using namespace std;
using namespace cv;

Mat create_red_img_from_segmentation_mat( Mat segmentations_cv_16U ) {
	fprintf( stderr, "start of create_red_img_from_segmentation_mat\n" );
	int max_value = -1;
	unsigned short * ptr_to_data = (unsigned short *) segmentations_cv_16U.ptr();
	for ( int i = 0; i < segmentations_cv_16U.rows * segmentations_cv_16U.cols; i++ ) {
		unsigned short cur = ptr_to_data[i];
		int cur_int = (int) cur;
		if ( cur_int > max_value) max_value = cur_int;
	}
	int number_per_shade = (max_value+1) / 255;
	double shade_multiplier = 255 / (double) (max_value+1);
	Mat segmentation_in_red = Mat::zeros( segmentations_cv_16U.rows, segmentations_cv_16U.cols, CV_8U );
	for ( int i = 0; i < segmentations_cv_16U.rows * segmentations_cv_16U.cols; i++ ) {
		unsigned short cur = ptr_to_data[i];
		int r = i%segmentations_cv_16U.rows;
		int c = i/segmentations_cv_16U.rows;
		segmentation_in_red.at<uchar>(r,c) = ( uchar ) (((double) cur ) * shade_multiplier);
	}
	fprintf( stderr, "end of create_red_img_from_segmentation_mat\n" );
	return segmentation_in_red;
}

Mat create_mat_from_array( const Mat input_img, vl_uint32* segmentation ) {
	fprintf( stderr, "start of create_mat_from_array\n" );
	Mat segmentations_cv_16U = Mat::zeros( input_img.rows, input_img.cols, CV_16U );
  for (int i = 0; i < input_img.rows; ++i) {
    for (int j = 0; j < input_img.cols; ++j) {
      segmentations_cv_16U.at<unsigned short>(i, j) = ( unsigned short ) segmentation[j + input_img.cols*i];
    }
  }
	fprintf( stderr, "end of create_mat_from_array\n" );
  return segmentations_cv_16U;
}

Mat compute_superpixels( Mat mat ) {
  float* image = new float[mat.rows*mat.cols*mat.channels()];
  for (int i = 0; i < mat.rows; ++i) {
		for (int j = 0; j < mat.cols; ++j) {
			image[j + mat.cols*i + mat.cols*mat.rows*0] = mat.at<cv::Vec3b>(i, j)[0];
			image[j + mat.cols*i + mat.cols*mat.rows*1] = mat.at<cv::Vec3b>(i, j)[1];
			image[j + mat.cols*i + mat.cols*mat.rows*2] = mat.at<cv::Vec3b>(i, j)[2];
		}
  }
  vl_uint32* segmentation = new vl_uint32[mat.rows*mat.cols];
  vl_size height = mat.rows;
  vl_size width = mat.cols;
  vl_size channels = mat.channels();
  vl_size region = 30;        
  float regularization = 1000.;
  vl_size minRegion = 10;
  vl_slic_segment(segmentation, image, width, height, channels, region, regularization, minRegion);
	// convert the 1D segmentation array into a Mat to be used by the following routines 
 	//Mat classifications = Mat::zeros( mat.rows, mat.cols, CV_16U );
	Mat segmentations_cv_16U = create_mat_from_array( mat, segmentation );
	return segmentations_cv_16U;
}

vector<vector<long> > compute_grid_points_over_superpixels( Mat input_img_16U ) {
	fprintf(stderr, "aaa\n");
	int max_value = -1;
	unsigned short * ptr_to_data = (unsigned short *) input_img_16U.ptr();
	for ( int i = 0; i < (input_img_16U.rows * input_img_16U.cols); i++ ) {
		unsigned short cur = ptr_to_data[i];
		int cur_int = (int) cur;
		if ( cur_int > max_value){
			max_value = cur_int;
		}
	}
	
	bool print_statements = false;
	if ( print_statements ) fprintf( stderr, "max_value: %d\n", max_value );
	if ( print_statements ) fprintf(stderr, "aab\n");
	
	// create max number length vector
	vector< vector<long > > grid_points_over_superpixels( max_value+1, vector<long>() );
	
	int edge_buffer = 40;
	int step_size = 10;
	for( int i = edge_buffer; i < input_img_16U.rows - edge_buffer; i += step_size ) {
		for( int j = edge_buffer; j < input_img_16U.cols - edge_buffer; j += step_size ) {
			long pixel_index = j * ((long) input_img_16U.rows) + i;
			if ( print_statements ) fprintf(stderr, "aac\n");
			int superpixel_image_point_value = (int) input_img_16U.at<ushort>(i, j);
			int superpixel_idx_real;
			bool add_point = false;
			if( max_value > 1 ) {
				if ( print_statements ) fprintf(stderr, "here 1\n");
				superpixel_idx_real = superpixel_image_point_value;
				add_point = true;
			} else {
				if ( print_statements ) fprintf(stderr, "here 2\n");
				superpixel_idx_real = 0;
				if ( input_img_16U.at<ushort>(i, j) > 0 ) {
					add_point = true;
				}
			}
			if ( add_point ) {
				if (superpixel_idx_real > (((int) grid_points_over_superpixels.size())-1) ) throw runtime_error( "error in superpixel vector logic!" );
				grid_points_over_superpixels[superpixel_idx_real].push_back( pixel_index );
			}
		}
	}
	fprintf(stderr, "aah\n");
	return grid_points_over_superpixels;
}

vector<KeyPoint> convert_vector_of_indices_to_keypoints( vector<vector<long>> all_keypoints_indices, const Mat mat ) {
	int r = mat.rows;
	vector<KeyPoint> vector_keypoints;
	for ( long i = 0; i < (long) all_keypoints_indices.size(); i++ ) {
		for ( long j = 0; j < (long) all_keypoints_indices[i].size(); j++) {
			long ci = all_keypoints_indices[i][j];
			KeyPoint cur_keypoint( ci / r, ci % r, 10 );
			vector_keypoints.push_back( cur_keypoint );
		}
	}
	return vector_keypoints;
}

// function to compute the features at each selected grid point
Mat compute_features_over_image( const Mat input_img_rgb_uchar, vector<KeyPoint> points_for_feature_computation, string feature_type ) {
//	SurfDescriptorExtractor extractor;
	FREAK extractor;
	Mat input_img_gray_uchar;
	cvtColor( input_img_rgb_uchar, input_img_gray_uchar, CV_RGB2GRAY );
	vector<Mat> all_superpixel_features;
	
	// 
	fprintf( stderr, "feature points to compute:  %d\n", (int) points_for_feature_computation.size() );
  Mat descriptors_object_8U;
	extractor.compute( input_img_gray_uchar, points_for_feature_computation, descriptors_object_8U );
	if ( descriptors_object_8U.rows != (int) points_for_feature_computation.size() ) {
		throw runtime_error("not as many descriptors as points");
	}
	return descriptors_object_8U;
}

Mat compute_valid_pixels( Mat input_img_rgb_uchar_train_mask ) {
	Mat dst;
	int threshold_value = 250, max_BINARY_value = 255;
	Mat pixels_at_255;
	Mat src_gray;
	cvtColor(input_img_rgb_uchar_train_mask, src_gray, CV_BGR2GRAY);
	threshold( src_gray, pixels_at_255, threshold_value, max_BINARY_value, THRESH_BINARY );
	imwrite("valid_image_before.pgm", pixels_at_255);
	uchar * ptr_temp = (uchar *) pixels_at_255.ptr();
	for ( int i = 0; i < pixels_at_255.rows*pixels_at_255.cols; i++ ) {
		ptr_temp[i] = ptr_temp[i] == 255 ? 1 : 0;
	}
	pixels_at_255.convertTo(pixels_at_255, CV_16U);
	return pixels_at_255;
}

Mat draw_grid_points_on_image( Mat valid_img_16U, vector<vector<long> > grid_keypoints_train ) {
	fprintf(stderr, "aaaa\n");
	Mat grid_keypoints_8U = Mat::zeros( valid_img_16U.rows, valid_img_16U.cols, CV_8U );
	for ( int i = 0; i < (int) grid_keypoints_train.size(); i++) {
		vector<long> cur_indices = grid_keypoints_train[i];
		for ( int j = 0; j < (int) cur_indices.size(); j++ ) {
			int cur_idx = (int) cur_indices[j];
			int r = cur_idx % valid_img_16U.rows;
			int c = cur_idx / valid_img_16U.rows;
			grid_keypoints_8U.at<uchar>(r, c) = 1;
		}
	}
	fprintf(stderr, "aaab\n");
	int dilation_size = 1;
	int dilation_type = MORPH_RECT;
	Mat element = getStructuringElement( dilation_type,
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );
	int total_keypoints = 0;
	for ( int i = 0; i < grid_keypoints_8U.rows; i++ ) {
		for ( int j = 0; j < grid_keypoints_8U.cols; j++ ) {
			int cur_value = grid_keypoints_8U.at<uchar>(i, j) > 0 ? 1 : 0;
			total_keypoints += cur_value;
		}
	}
	Mat grid_keypoints_dilated_8U;
  dilate( grid_keypoints_8U, grid_keypoints_dilated_8U, element );
	fprintf(stderr, "aaac\n");
	Mat valid_points_drawn = Mat::zeros( valid_img_16U.rows, valid_img_16U.cols, CV_8UC3 );
	for ( int i = 0; i < valid_points_drawn.rows; i++ ) {
		for ( int j = 0; j < valid_points_drawn.cols; j++ ) {
			if ( grid_keypoints_dilated_8U.at<uchar>(i, j) == 1 ) {
				valid_points_drawn.at<cv::Vec3b>(i, j)[0] = 0;
				valid_points_drawn.at<cv::Vec3b>(i, j)[1] = 255;
				valid_points_drawn.at<cv::Vec3b>(i, j)[2] = 0;
			} else if ( valid_img_16U.at<ushort>(i, j) > 0 ) {
				valid_points_drawn.at<cv::Vec3b>(i, j)[0] = 255;
				valid_points_drawn.at<cv::Vec3b>(i, j)[1] = 255;
				valid_points_drawn.at<cv::Vec3b>(i, j)[2] = 255;
			}
		}
	}
	fprintf(stderr, "aaad\n");
	return valid_points_drawn;
}

Mat compute_invalid_pixels( Mat valid_img_16U, Mat input_img_rgb_uchar_train ) {
	Mat input_img_gray_uchar_train;
	cvtColor( input_img_rgb_uchar_train, input_img_gray_uchar_train, CV_RGB2GRAY );
	imwrite( "input_img_rgb_uchar_train.ppm", input_img_rgb_uchar_train );
	imwrite( "input_img_gray_uchar_train.pgm", input_img_gray_uchar_train );
	Mat invalid_img_16U = Mat::zeros( valid_img_16U.rows, valid_img_16U.cols, CV_16U );
	for ( int i = 0; i < valid_img_16U.rows; i++ )
	for ( int j = 0; j < valid_img_16U.cols; j++ )
	{
		bool condition_1 = valid_img_16U.at<ushort>(i, j) < 1;
		bool condition_2 = input_img_gray_uchar_train.at<uchar>(i, j) > 25;
		if ( condition_1 && condition_2 ) {
			invalid_img_16U.at<ushort>(i, j) = 1;
		}
	}
	Mat invalid_img_8U;
	invalid_img_16U.convertTo(invalid_img_8U, CV_8U);
	Mat invalid_img_8U_temp = multiply_matrix_by_255( invalid_img_8U );
	imwrite( "invalid_before_erosion.pgm", invalid_img_8U_temp );
	
	int erosion_size = 10;
	int erosion_type = MORPH_RECT; 
	Mat element = getStructuringElement( erosion_type,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );
	Mat invalid_img_8U_eroded, invalid_img_16U_eroded;
  erode( invalid_img_8U, invalid_img_8U_eroded, element );
	invalid_img_8U_temp = multiply_matrix_by_255( invalid_img_8U_eroded );
	imwrite( "invalid_after_erosion.pgm", invalid_img_8U_temp );
	invalid_img_8U_eroded.convertTo(invalid_img_16U_eroded, CV_16U);
	return invalid_img_16U_eroded;
}

vector<orchard_classification> query_superpixel_features( flann::Index * kdtrees, vector<orchard_classification> combined_labels, const Mat cur_features_double, const Mat combined_descriptors_double ) {
	fprintf(stderr, "before conversion \n");
	
	Mat cur_features, combined_descriptors;
	cur_features_double.convertTo( cur_features, CV_32F );
	combined_descriptors_double.convertTo( combined_descriptors, CV_32F );
	fprintf(stderr, "after conversion \n");
	int k = 7;
	vector<orchard_classification> classifications(cur_features.rows, TREE_TRUNK);
	Mat distances;
	int feat_type = cur_features.type();
	FlannBasedMatcher matcher;
	vector< vector< DMatch > > matches;
	matcher.knnMatch( cur_features, combined_descriptors, matches, k );
	fprintf( stderr, "after matches computed\n" );
	for(int i = 0; i < cur_features.rows; i++) {
		int apple_count = 0, leaf_count = 0, bark_count = 0;
		for ( int j = 0; j < k; j++) {
			//int idx = matches.at<int>( i, j );
			int idx = matches[i][j].trainIdx;
			if (idx >= (int) combined_labels.size() ) runtime_error( "WTF?" );
			orchard_classification cur_type = combined_labels[(int) idx];
			if ( cur_type == APPLE ) {
				apple_count++;
			} else if ( cur_type == LEAF ) {
				leaf_count++;
			} else {
				bark_count++;
			}
		}
		if ( apple_count >= leaf_count && apple_count >= bark_count )
			classifications[i] = APPLE;
		else if ( leaf_count >= apple_count && leaf_count >= bark_count )
			classifications[i] = LEAF;
		else
			classifications[i] = TREE_TRUNK;
	}
	fprintf( stderr, "after matching classifications computed\n" );
	return classifications;
}

Mat multiply_matrix_by_255( Mat valid_img_8U ) {	
	Mat multiplier( valid_img_8U.rows, valid_img_8U.cols, CV_8U, Scalar(255) );
	Mat output_8U;
	multiply( valid_img_8U, multiplier, output_8U );
	return output_8U;
}

Mat display_superpixel_classifications( vector<orchard_classification> cur_superpixel_classifications, Mat input_img_rgb_uchar, Mat superpixel_img ) {
	Mat valid_points_drawn = Mat::zeros( input_img_rgb_uchar.rows, input_img_rgb_uchar.cols, CV_8UC3 );
	for ( int i = 0; i < valid_points_drawn.rows; i++ ) {
		for ( int j = 0; j < valid_points_drawn.cols; j++ ) {
			int cur_superpixel_idx = superpixel_img.at<ushort>( i, j );
			int r = 0, g=0, b=0;
			if (cur_superpixel_classifications[cur_superpixel_idx] == APPLE) {
				valid_points_drawn.at<cv::Vec3b>(i, j)[0] = input_img_rgb_uchar.at<cv::Vec3b>(i, j)[0];
				valid_points_drawn.at<cv::Vec3b>(i, j)[1] = input_img_rgb_uchar.at<cv::Vec3b>(i, j)[1];
				valid_points_drawn.at<cv::Vec3b>(i, j)[2] = input_img_rgb_uchar.at<cv::Vec3b>(i, j)[2];
			} else {
				valid_points_drawn.at<cv::Vec3b>(i, j)[0] = r;
				valid_points_drawn.at<cv::Vec3b>(i, j)[1] = g;
				valid_points_drawn.at<cv::Vec3b>(i, j)[2] = b;
			}
		}
	}
	return valid_points_drawn;
}

vector< Mat > split_features( Mat descriptors_object_8U, vector< vector< long > > grid_keypoints ) {
	fprintf( stderr, "There are %d keypoints in the vector\n", (int) grid_keypoints.size() );
	vector< Mat > all_superpixel_features;
	int cur_start = 0;
	for ( int i = 0; i < (int) grid_keypoints.size(); i++ ) {
		vector<long> cur_keypoints_indices = grid_keypoints[i];
		Mat dst;
		descriptors_object_8U(Rect(0, cur_start, descriptors_object_8U.cols, (int) cur_keypoints_indices.size())).copyTo(dst);
		all_superpixel_features.push_back( dst );
		cur_start = cur_start + (int) cur_keypoints_indices.size();
	}
	return all_superpixel_features;
}

Mat find_equal_16U( Mat superpixel_img, int compare_number_input ) {
	Mat valid_this_superpixel_16U;
	Mat this_superpixel_number( superpixel_img.rows, superpixel_img.cols, CV_16U, compare_number_input );
	compare( superpixel_img, this_superpixel_number, valid_this_superpixel_16U, CMP_EQ );
	valid_this_superpixel_16U.convertTo( valid_this_superpixel_16U, CV_16U );
	return valid_this_superpixel_16U;
}

Mat compute_color_features_over_image( Mat input_img_rgb_uchar, vector< KeyPoint > grid_keypoints_train ) {
	Mat all_mask_color_features_train_combined = Mat::zeros( (int) grid_keypoints_train.size(), 3, CV_8U );
	for( int i = 0; i < (int) grid_keypoints_train.size(); i++ ) {
//		Vec3b temp = input_img_rgb_uchar.at<Vec3b>( grid_keypoints_train[i].pt.y, grid_keypoints_train[i].pt.x );
		int total_r = 0, total_g = 0, total_b = 0;
		int min_dim = -9;
		int max_dim = 9;
		for( int j = min_dim; j <= max_dim; j++ ) {
			for( int k = min_dim; k <= max_dim; k++ ) {
				Vec3b temp2 = input_img_rgb_uchar.at<Vec3b>( grid_keypoints_train[i].pt.y + j, grid_keypoints_train[i].pt.x + k );
				total_r += (int) temp2[0];
				total_g += (int) temp2[1];
				total_b += (int) temp2[2];
			}
		}
		all_mask_color_features_train_combined.at<uchar>( i, 0 ) = total_r / 361;
		all_mask_color_features_train_combined.at<uchar>( i, 1 ) = total_g / 361;
		all_mask_color_features_train_combined.at<uchar>( i, 2 ) = total_b / 361;
	}
	return all_mask_color_features_train_combined;
}

Mat compute_std_by_row( Mat combined_descriptors, Mat row_mean ) {
	Mat std_by_dimension(1, combined_descriptors.cols, CV_64F);
	for( int i = 0; i < combined_descriptors.rows; i++ ) {
		for( int j = 0; j < combined_descriptors.cols; j++ ) {
			double cur_diff = ((double)combined_descriptors.at<uchar>(i,j)) - row_mean.at<double>(0, j);
			std_by_dimension.at<double>( 0, j ) += pow(cur_diff, 2);
		}
	}
	Mat rows_mat( 1, combined_descriptors.cols, CV_64F, combined_descriptors.rows );
	divide( std_by_dimension, rows_mat, std_by_dimension );
	sqrt( std_by_dimension, std_by_dimension );
	std_by_dimension.convertTo(std_by_dimension, CV_64F);
	return std_by_dimension;
}

vector<Mat> compute_mean_and_std( const Mat combined_descriptors_input ) {
	Mat combined_descriptors = combined_descriptors_input.clone();
	Mat row_mean;
	reduce( combined_descriptors, row_mean, 0, CV_REDUCE_AVG );
	row_mean.convertTo(row_mean, CV_64F);
	Mat std_by_dimension = compute_std_by_row( combined_descriptors, row_mean );
	vector<Mat> mean_and_std;
	mean_and_std.push_back( row_mean );
	mean_and_std.push_back( std_by_dimension );
	return mean_and_std;
}

vector<string> get_files_with_extension ( char * dir_path ) {
	vector< string > mask_filepaths;
	DIR* dirFile = opendir( dir_path );
	if ( dirFile ) {
		struct dirent* hFile;
		errno = 0;
		while (( hFile = readdir( dirFile )) != NULL )  {
			if ( !strcmp( hFile->d_name, "."  )) continue;
			if ( !strcmp( hFile->d_name, ".." )) continue;
			if ( ( hFile->d_name[0] == '.' )) continue;
			if ( strstr( hFile->d_name, "mask.png" )) {
				string cur_file( hFile->d_name );
				mask_filepaths.push_back( cur_file );
			}
		}
		closedir( dirFile );
	}
	return mask_filepaths;
}

Mat normalize_mat( Mat input_mat, vector<Mat> norm_factors_mean_and_std ) {
	fprintf( stderr, "input_mat: rows: %d: cols: %d\n", input_mat.rows, input_mat.cols );
	fprintf(stderr, "norm_factors_mean_and_std[0]: rows: %d: cols: %d\n", norm_factors_mean_and_std[0].rows, norm_factors_mean_and_std[0].cols );
	fprintf(stderr, "norm_factors_mean_and_std[1]: rows: %d: cols: %d\n", norm_factors_mean_and_std[1].rows, norm_factors_mean_and_std[1].cols );
	fprintf( stderr, "norm_factors_mean_and_std.size(): %d\n", (int) norm_factors_mean_and_std.size() );
	input_mat.convertTo( input_mat, CV_64F );
	Mat normalized;
	Mat subtract_mat = Mat::zeros( input_mat.rows, input_mat.cols, CV_64F );
	Mat divide_mat = Mat::ones( input_mat.rows, input_mat.cols, CV_64F );
	fprintf( stderr, "subtract_mat: rows: %d: cols: %d\n", subtract_mat.rows, subtract_mat.cols );
	fprintf( stderr, "divide_mat: rows: %d: cols: %d\n", divide_mat.rows, divide_mat.cols );
	Mat mean_mat = norm_factors_mean_and_std[0];
	mean_mat.convertTo(mean_mat, CV_64F);
	Mat std_mat = norm_factors_mean_and_std[1];
	std_mat.convertTo(std_mat, CV_64F);
	for(int i = 0; i < input_mat.rows; i++ ) {
		for(int j = 0; j < input_mat.cols; j++ ) {
			if ( i > subtract_mat.rows || i > divide_mat.rows ) throw runtime_error("a");
			if ( j > subtract_mat.cols || j > mean_mat.cols || j > divide_mat.cols || j > std_mat.cols ) throw runtime_error("b");
			subtract_mat.at<double>(i, j) = ( mean_mat.at<double>(0, j) );
			divide_mat.at<double>(i, j) = ( std_mat.at<double>(0, j) );
		}
	}
	fprintf(stderr, "before subtract\n");
	subtract(input_mat, subtract_mat, normalized);
	fprintf(stderr, "before divide\n");
	divide(normalized, divide_mat, normalized);
	return normalized;
}

Mat convert_grid_locations_to_mat( vector< vector< long > > grid_keypoints_train ) {
	int total_size_temp = 0;
	for( int j = 0; j < grid_keypoints_train.size(); j++ ) {
		total_size_temp += (j == grid_keypoints_train.size()) ? (int) grid_keypoints_train[j].size() : (int) grid_keypoints_train[j].size() + 1;
	}
	Mat grid_keypoints_train_combined( total_size_temp, 1, CV_32S );
	long cur_place = 0;
	for( int j = 0; j < grid_keypoints_train.size(); j++ ) {
		for( int k = 0; k < grid_keypoints_train[j].size(); k++ ) {
			grid_keypoints_train_combined.at< int >( cur_place, 0 ) = (int) grid_keypoints_train[j][k];
			cur_place++;
		}
		if (j < (int)grid_keypoints_train.size()-1) grid_keypoints_train_combined.at< int >(cur_place, 0) = -1;
	}
	return grid_keypoints_train_combined;
}

vector<vector<long> > convert_mat_to_grid_keypoints_vector( Mat grid_keypoints_train_combined ) {
	int count_temp = 1;
	for( int j = 0; j < grid_keypoints_train_combined.rows; j++ ) {
		long cur_el = (long) grid_keypoints_train_combined.at< int >( j, 0 );
		if( j < grid_keypoints_train_combined.rows-1 && cur_el == -1 ) {
			count_temp += 1;
		}
	}
	int idx = 0;
	vector< vector<long> > grid_keypoints_train( count_temp, vector<long>() );
	for( int j = 0; j < grid_keypoints_train_combined.rows; j++ ) {
		long cur_el = (long) grid_keypoints_train_combined.at< int >( j, 0 );
		if( j < grid_keypoints_train_combined.rows-1 && cur_el == -1 ) {
			idx++;
		} else {
			grid_keypoints_train[idx].push_back( cur_el );
		}
	}
	return grid_keypoints_train;
}

// main function that calls everything else
// 	first on one image then make a function to call everything else
int main() {
	fprintf( stderr, "Start of methods\n" );
	string feature_type = "SIFT";

	vector<string> all_mask_files;
	vector<string> all_raw_files;
	vector<string> all_image_numbers;

	int mask_characters = 9;
	// for a number of training images
	char* mask_path = (char *) malloc( sizeof(char) * 300 ), *raw_path = (char *) malloc( sizeof(char) * 300 ), *indi_mask_path = (char *) malloc( sizeof(char) * 300 ), *indi_raw_path = (char *) malloc( sizeof(char) * 300 );
	sprintf(mask_path, "/home/kyle/Dropbox/apple_images/red/mask/");
	sprintf(raw_path, "/home/kyle/Dropbox/apple_images/red/raw/");
	vector<string> mask_files = get_files_with_extension ( mask_path );
	for (int i = 0; i < (int) mask_files.size(); i++ ) {
		string cur_mask_file = mask_files[i];
		sprintf( indi_mask_path, "%s%s", mask_path, cur_mask_file.c_str() );
		string cur_mask_file_all_but_end = cur_mask_file.substr (0, cur_mask_file.length() - mask_characters );
		string img_number = cur_mask_file.substr ( cur_mask_file.length() - mask_characters - 4, cur_mask_file.length() - mask_characters );
		all_image_numbers.push_back( img_number );
		const char *cur_mask_file_all_but_end_ptr = cur_mask_file_all_but_end.c_str();
		sprintf( indi_raw_path, "%s%s.jpg", raw_path, cur_mask_file_all_but_end_ptr );
		string indi_mask_string( indi_mask_path );
		all_mask_files.push_back(indi_mask_string);
		string indi_raw_string( indi_raw_path );
		all_raw_files.push_back(indi_raw_path);
		cout << "indi_raw_path: " << indi_raw_path << endl;
	}

	float percent_apple, percent_non_apple;
	Mat temp_percents;

	vector<Mat> all_valid_texture_features;
	vector<Mat> all_invalid_texture_features;
	vector<Mat> all_valid_color_features;
	vector<Mat> all_invalid_color_features;
	vector<Mat> all_superpixel_features_vec;
	vector<Mat> all_superpixel_color_features_vec;
	vector<vector<vector<long> > > all_superpixel_grid_keypoints_processing;
	vector< Mat > superpixel_imgs;
	// for training image
	char root_directory[200] = "";
	for( int i = 0; i < (int) all_raw_files.size(); i++ ) {
		string raw_filepath = all_raw_files[i];
		string mask_filepath = all_mask_files[i];
		string img_number = all_image_numbers[i];
		char *cur_yml_filepath = (char *) malloc( sizeof(char)*300 );
		sprintf( cur_yml_filepath, "one_descriptor_%s.yml", img_number.c_str() );
		vector<vector<long> > grid_keypoints_processing;

		Mat all_mask_features_train_combined, all_mask_features_train_invalid_combined;
		Mat all_mask_color_features_train_combined, all_mask_color_features_train_invalid_combined;
		Mat all_superpixel_features_combined, all_superpixel_color_features_combined;
		Mat superpixel_img;
		bool saving_features = true;
		if ( saving_features ) {
			Mat input_img_rgb_uchar_train = imread( raw_filepath );
			transpose( input_img_rgb_uchar_train, input_img_rgb_uchar_train );
			flip(input_img_rgb_uchar_train, input_img_rgb_uchar_train, 0);
			// KYLE: CHECKING: save the image for feature tracking
			// create a folder for this image
			char img_folder[256];
			sprintf( img_folder, "/home/kyle/%s/", img_number.c_str() );
			char img_folder_create_command[256];
			sprintf( img_folder_create_command, "mkdir %s", img_folder );
			system( img_folder_create_command );
			
			imwrite( "input_img_rgb_uchar_train.ppm", input_img_rgb_uchar_train );
			vector<orchard_classification> combined_labels;
			
			Mat input_img_rgb_uchar_train_mask = imread( mask_filepath );
			// compute valid pixels
			Mat valid_img_16U = compute_valid_pixels( input_img_rgb_uchar_train_mask );
			// compute mask valid
			// compute valid pixels
			// compute invalid pixels
			vector<vector<long> > grid_keypoints_train = compute_grid_points_over_superpixels( valid_img_16U );
			Mat output_temp = draw_grid_points_on_image( valid_img_16U, grid_keypoints_train );
			Mat output_temp_8U;
			output_temp.convertTo( output_temp_8U, CV_8U );
			char output_img_path[256];
			sprintf( output_img_path, "%s/output_temp.ppm", img_folder );
			imwrite( output_img_path, output_temp_8U );
			vector<KeyPoint> grid_keypoints_train_feature_computation = convert_vector_of_indices_to_keypoints( grid_keypoints_train, input_img_rgb_uchar_train );
			all_mask_features_train_combined = compute_features_over_image( input_img_rgb_uchar_train, grid_keypoints_train_feature_computation, feature_type );
			all_mask_color_features_train_combined = compute_color_features_over_image( input_img_rgb_uchar_train, grid_keypoints_train_feature_computation );
			Mat invalid_img_16U_eroded = compute_invalid_pixels( valid_img_16U, input_img_rgb_uchar_train );
			
			Mat invalid_img_8U_eroded;
			invalid_img_16U_eroded.convertTo( invalid_img_8U_eroded, CV_8U );
			char invalid_path [256];
			sprintf( invalid_path, "%s/invalid_pixels.ppm", img_folder );
			imwrite( invalid_path, invalid_img_8U_eroded );
			vector<vector<long> > grid_keypoints_invalid = compute_grid_points_over_superpixels( invalid_img_16U_eroded );
			Mat output_temp_invalid_points_drawn = draw_grid_points_on_image( invalid_img_16U_eroded, grid_keypoints_invalid );
			Mat output_temp_invalid_points_drawn_8U;
			output_temp_invalid_points_drawn.convertTo( output_temp_invalid_points_drawn_8U, CV_8U );
			char invalid_drawn_img_path[256];
			sprintf( invalid_drawn_img_path, "%s/invalid_drawn.ppm", img_folder );
			imwrite( invalid_drawn_img_path, output_temp_invalid_points_drawn_8U );
			vector<KeyPoint> grid_keypoints_invalid_feature_computation = convert_vector_of_indices_to_keypoints( grid_keypoints_invalid, input_img_rgb_uchar_train );
			all_mask_features_train_invalid_combined = compute_features_over_image( input_img_rgb_uchar_train, grid_keypoints_invalid_feature_computation, feature_type );
			all_mask_color_features_train_invalid_combined = compute_color_features_over_image( input_img_rgb_uchar_train, grid_keypoints_invalid_feature_computation );
			// compute superpixels - check
			superpixel_img = compute_superpixels( input_img_rgb_uchar_train );
			fprintf(stderr, "superpixel image: %d: %d\n", superpixel_img.rows, superpixel_img.cols );
			if (superpixel_img.rows * superpixel_img.cols == 0) throw runtime_error( "no superpixel image? the superpixel image size is 0?" );
			grid_keypoints_processing = compute_grid_points_over_superpixels( superpixel_img );
			vector<KeyPoint> grid_keypoints_processing_feature_computation = convert_vector_of_indices_to_keypoints( grid_keypoints_processing, input_img_rgb_uchar_train );
			all_superpixel_features_combined = compute_features_over_image( input_img_rgb_uchar_train, grid_keypoints_processing_feature_computation, feature_type );
			all_superpixel_color_features_combined = compute_color_features_over_image( input_img_rgb_uchar_train, grid_keypoints_processing_feature_computation );
			
			// save the features to yml files if it seems good
			// save to big vectors
			cv::FileStorage storage(cur_yml_filepath, cv::FileStorage::WRITE);
			// features texture valid
			storage << "all_mask_features_train_combined" << all_mask_features_train_combined;
			// features texture invalid
			storage << "all_mask_features_train_invalid_combined" << all_mask_features_train_invalid_combined;
			// features color valid
			storage << "all_mask_color_features_train_combined" << all_mask_color_features_train_combined;
			// features color invalid
			storage << "all_mask_color_features_train_invalid_combined" << all_mask_color_features_train_invalid_combined;
			// texture features for superpixels
			storage << "all_superpixel_features_combined" << all_superpixel_features_combined;
			// texture features for superpixels
			storage << "all_superpixel_color_features_combined" << all_superpixel_color_features_combined;
			// grid locations for superpixels
			Mat grid_keypoints_processing_superpixels;
			grid_keypoints_processing_superpixels = convert_grid_locations_to_mat( grid_keypoints_processing );
			storage << "grid_keypoints_train_combined" << grid_keypoints_processing_superpixels;
			// superpixel_img
			storage << "superpixel_img" << superpixel_img;
			storage.release();
			fprintf(stderr, "finished initial feature computation\n");
		} else {
			Mat combined_labels_mock;
			cv::FileStorage storage( cur_yml_filepath, cv::FileStorage::READ);
			fprintf( stderr, "before loading\n" );
			// features texture valid
			storage["all_mask_features_train_combined"] >> all_mask_features_train_combined;
			// features texture invalid
			storage["all_mask_features_train_invalid_combined"] >> all_mask_features_train_invalid_combined;
			// features color valid
			storage["all_mask_color_features_train_combined"] >> all_mask_color_features_train_combined;
			// features color invalid
			storage["all_mask_color_features_train_invalid_combined"] >> all_mask_color_features_train_invalid_combined;
			// texture features for superpixels
			storage["all_superpixel_features_combined"] >> all_superpixel_features_combined;
			// color features for superpixels
			storage["all_superpixel_color_features_combined"] >> all_superpixel_color_features_combined;
			// keypoints for current superpixel
			Mat grid_keypoints_processing_superpixels;
			storage["grid_keypoints_train_combined"] >> grid_keypoints_processing_superpixels;
			grid_keypoints_processing = convert_mat_to_grid_keypoints_vector( grid_keypoints_processing_superpixels );
			// superpixel_img
			storage["superpixel_img"] >> superpixel_img;
			storage.release();
			fprintf(stderr, "loaded features correctly\n");
		}
		
		all_superpixel_grid_keypoints_processing.push_back( grid_keypoints_processing );
		// feature texture valid
		all_valid_texture_features.push_back( all_mask_features_train_combined );
		// features texture invalid
		all_invalid_texture_features.push_back( all_mask_features_train_invalid_combined );
		// features color valid
		all_valid_color_features.push_back( all_mask_color_features_train_combined );
		// features color invalid
		all_invalid_color_features.push_back( all_mask_color_features_train_invalid_combined );
		// texture features for superpixels
		all_superpixel_features_vec.push_back( all_superpixel_features_combined );
		// color features for superpixels
		all_superpixel_color_features_vec.push_back( all_superpixel_color_features_combined );
		// keypoints for current superpixel
		if (superpixel_img.rows * superpixel_img.cols == 0) throw runtime_error( "no superpixel image? the superpixel image size is 0?" );
		superpixel_imgs.push_back( superpixel_img );
	}
	// for loop over loo
	for( int i = 0; i < (int) all_valid_texture_features.size(); i++ ) {
		string input_img_rgb_uchar_path = all_raw_files[i];
		Mat input_img_rgb_uchar = imread( input_img_rgb_uchar_path );
		transpose( input_img_rgb_uchar, input_img_rgb_uchar );
		flip(input_img_rgb_uchar, input_img_rgb_uchar, 0);
		string img_number = all_image_numbers[i];

		Mat superpixel_img = superpixel_imgs[i];
		if (superpixel_img.rows * superpixel_img.cols == 0) throw runtime_error( "no superpixel image? the superpixel image size is 0?" );

		Mat combined_loo_train_all_valid_texture_features( 0, all_valid_texture_features[0].cols, all_valid_texture_features[0].type() );
		Mat combined_loo_train_all_invalid_texture_features( 0, all_invalid_texture_features[0].cols, all_invalid_texture_features[0].type() );
		Mat combined_loo_train_all_valid_color_features( 0, all_valid_color_features[0].cols, all_valid_color_features[0].type() );
		Mat combined_loo_train_all_invalid_color_features( 0, all_invalid_color_features[0].cols, all_invalid_color_features[0].type() );
		vector<Mat> loo_train_all_invalid_color_features;
		for( int j = 0; j < (int) all_valid_texture_features.size(); j++ ) {
			if( i == j ) continue;
			if (combined_loo_train_all_valid_texture_features.rows == 0 || combined_loo_train_all_valid_texture_features.cols == 0) {
				combined_loo_train_all_valid_texture_features = all_valid_texture_features[j];
			} else {
				vconcat( combined_loo_train_all_valid_texture_features, all_valid_texture_features[j], combined_loo_train_all_valid_texture_features );
			}
			if (combined_loo_train_all_invalid_texture_features.rows == 0 || combined_loo_train_all_invalid_texture_features.cols == 0) {
				combined_loo_train_all_invalid_texture_features = all_invalid_texture_features[j];
			} else {
				vconcat( combined_loo_train_all_invalid_texture_features, all_invalid_texture_features[j], combined_loo_train_all_invalid_texture_features );
			}
			if (combined_loo_train_all_valid_color_features.rows == 0 || combined_loo_train_all_valid_color_features.cols == 0) {
				combined_loo_train_all_valid_color_features = all_valid_color_features[j];
			} else {
				vconcat( combined_loo_train_all_valid_color_features, all_valid_color_features[j], combined_loo_train_all_valid_color_features );
			}
			if (combined_loo_train_all_invalid_color_features.rows == 0 || combined_loo_train_all_invalid_color_features.cols == 0) {
				combined_loo_train_all_invalid_color_features = all_invalid_color_features[j];
			} else {
				vconcat( combined_loo_train_all_invalid_color_features, all_invalid_color_features[j], combined_loo_train_all_invalid_color_features );
			}
		}
		vector<vector<long> > grid_keypoints_processing = all_superpixel_grid_keypoints_processing[i];
		Mat combined_texture_features, combined_color_features;
		vconcat( combined_loo_train_all_valid_texture_features, combined_loo_train_all_invalid_texture_features, combined_texture_features );
		vconcat( combined_loo_train_all_valid_color_features, combined_loo_train_all_invalid_color_features, combined_color_features );

		vector<orchard_classification> valid(combined_loo_train_all_valid_texture_features.rows, APPLE);
		vector<orchard_classification> invalid(combined_loo_train_all_invalid_texture_features.rows, LEAF);
		vector<orchard_classification> combined_labels;
		combined_labels.insert( combined_labels.end(), valid.begin(), valid.end() );
		combined_labels.insert( combined_labels.end(), invalid.begin(), invalid.end() );

		double total_valid = (double) valid.size();
		double total_invalid = (double) invalid.size();
		double total = total_valid + total_invalid;
		double percent_apple = total_valid / total;
		double percent_non_apple = total_invalid / total;

		// compute pca
		// do pca on combined
		Mat empty, normalized_combined_texture_and_color_descriptors_train_64F, pca_descriptors_train;
		vector<Mat> norm_factors_texture = compute_mean_and_std( combined_texture_features );
		vector<Mat> norm_factors_color = compute_mean_and_std( combined_color_features );
		fprintf( stderr, "texture norm training\n" );
		Mat normalized_texture_descriptors_train = normalize_mat( combined_texture_features, norm_factors_texture );
		Mat normalized_color_descriptors_train = normalize_mat( combined_color_features, norm_factors_color );
		fprintf( stderr, "before pca\n" );
		PCA pca( normalized_texture_descriptors_train, empty, CV_PCA_DATA_AS_ROW, 6 );
		pca.project( normalized_texture_descriptors_train, pca_descriptors_train );
		hconcat( pca_descriptors_train, normalized_color_descriptors_train, normalized_combined_texture_and_color_descriptors_train_64F );
		cv::flann::Index *kdtrees_ptr;

		Mat pca_descriptors_process, normalized_combined_texture_and_color_descriptors_process_64F;
		Mat all_superpixel_features_combined = all_superpixel_features_vec[i];
		Mat normalized_texture_descriptors_process = normalize_mat( all_superpixel_features_combined, norm_factors_texture );
		Mat all_superpixel_color_features_combined = all_superpixel_color_features_vec[i];
		Mat normalized_color_descriptors_process = normalize_mat( all_superpixel_color_features_combined, norm_factors_color );
		pca.project( normalized_texture_descriptors_process, pca_descriptors_process );
		hconcat( pca_descriptors_process, normalized_color_descriptors_process, normalized_combined_texture_and_color_descriptors_process_64F );

		fprintf(stderr, "before matching features\n");
		vector<orchard_classification> all_supperpixel_classifications_combined = query_superpixel_features( kdtrees_ptr, combined_labels, normalized_combined_texture_and_color_descriptors_process_64F, normalized_combined_texture_and_color_descriptors_train_64F );
		if ( ((int)all_supperpixel_classifications_combined.size()) != all_superpixel_features_combined.rows ) throw runtime_error("features are not equal!?");
		fprintf(stderr, "after matching features\n");
		
		fprintf( stderr, "all_superpixel_features_combined: rows: %d: cols: %d\n", all_superpixel_features_combined.rows, all_superpixel_features_combined.cols );
		vector< Mat > all_superpixel_features = split_features( all_superpixel_features_combined, grid_keypoints_processing );
		fprintf(stderr, "aaaaaabb\n");

		vector<vector<orchard_classification>> all_superpixel_classifications( (int) all_superpixel_features.size(), vector<orchard_classification>() );
		fprintf(stderr, "aaaaa\n");
		int cur_start = 0;
		for ( int i = 0; i < (int) all_superpixel_features.size(); i++ ) {
			Mat cur_superpixel_features = all_superpixel_features[i];
			int cur_end = cur_start + cur_superpixel_features.rows - 1;
			int step = 0;
			if ( cur_superpixel_features.rows != 0 && cur_superpixel_features.cols != 0 ) {
				step = cur_superpixel_features.rows;
			}
			for( int j = 0; j < step; j++ ) {
				all_superpixel_classifications[i].push_back( all_supperpixel_classifications_combined[cur_start + j] );
			}
			cur_start += step;
		}
		fprintf(stderr, "aaaab\n");

		fprintf( stderr, "percent\napple: %.2f\nnonapple: %.2f\n", percent_apple, percent_non_apple );
		float ratio_apple = percent_apple;
		float ratio_leaf = percent_non_apple;
		// for each level, for each superpixel, compute the number of valid superpixels
			// save the resulting image classifications to directories
		// later ... compute the recall and precision rates for each image
		fprintf(stderr, "aaaab\n");
		
		vector<float> levels(1, 0.5);
		for (int j = 0; j < (int) levels.size(); j++) {
			char * temp_1 = (char*) malloc( sizeof(char)*300 );
			sprintf( temp_1, "input_img_rgb_uchar_path_%s.ppm", img_number.c_str() );
			imwrite( temp_1, input_img_rgb_uchar );
			fprintf(stderr, "aaaac\n");

			int total_apple_count = 0, total_leaf_count = 0, total_bark_count = 0;
			vector<orchard_classification> cur_superpixel_classifications((int) all_superpixel_classifications.size(), TREE_TRUNK);
			for( int k = 0; k < all_superpixel_classifications.size(); k++ ) { // for superpixel
				orchard_classification cur_classification = TREE_TRUNK;
				int apple_count = 0, leaf_count = 0, bark_count = 0;
				for( int a = 0; a < all_superpixel_classifications[k].size(); a++ ) { // for grid point
					orchard_classification ct = all_superpixel_classifications[k][a];
					if( ct == APPLE ) {apple_count++;} if( ct == LEAF ) {leaf_count++; } else {bark_count++; }
				}
				if ( ( ( apple_count / ratio_apple ) >= ( leaf_count / ratio_leaf ) && apple_count >= bark_count) ) {
					cur_superpixel_classifications[k] = APPLE; total_apple_count++;
				} else if ( leaf_count >= apple_count && leaf_count >= bark_count ) {
					cur_superpixel_classifications[k] = LEAF; total_leaf_count++;
				} else {
					cur_superpixel_classifications[k] = TREE_TRUNK; total_bark_count++;
				}
			}
			fprintf(stderr, "aaaad\n");
			// display the classifications
			fprintf(stderr, "before displaying features selections\n");
			fprintf( stderr, "total_apple_count: %d\ntotal_leaf_count: %d\ntotal_bark_count: %d\n", total_apple_count, total_leaf_count, total_bark_count);
			fprintf( stderr, "cur_superpixel_classifications: size: %d\n", (int) cur_superpixel_classifications.size() );
			fprintf( stderr, "input_img_rgb_uchar: %d\t%d: type: %d\n", input_img_rgb_uchar.rows, input_img_rgb_uchar.cols, input_img_rgb_uchar.type()  );
			fprintf( stderr, "superpixel_img: %d\t%d: type: %d\n", superpixel_img.rows, superpixel_img.cols, superpixel_img.type()  );
			Mat superpixel_classifications = display_superpixel_classifications(cur_superpixel_classifications, input_img_rgb_uchar, superpixel_img);
			fprintf(stderr, "after displaying features selections\n");
			char buf[512];
			sprintf(buf, "superpixel_classifications_level_%.2f_%s.ppm", levels[j], img_number.c_str() );
			fprintf(stderr, "rows: %d\tcols: %d\n", superpixel_classifications.rows, superpixel_classifications.cols );
			imwrite( buf, superpixel_classifications );
			fprintf(stderr, "aaaax\n");
		}
		// get the testing image
		// compute features over the testing image
		// setup all of the features in OpenCV's nearest neighbor setup
	}
	fprintf(stderr, "z\n");
	return 0;
}
