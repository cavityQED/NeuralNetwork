#ifndef IMAGEREADER_H
#define IMAGEREADER_H

#include <fstream>
#include <vector>
#include <iostream>

#include "boost/multi_array.hpp"
#include "matrix.hpp"
#include "vector.hpp"

class ImageReader
{
public:
	ImageReader();
	~ImageReader() {}

	void getImage(std::vector<int> &image, int &label);
	void getImage(boost::numeric::ublas::matrix<double> &image, int &value);
	
	void getBatch(std::vector<std::vector<int>> &batch, std::vector<int> &labels, int N);
	void getBatch(std::vector<boost::numeric::ublas::matrix<double>> &batch, std::vector<int> &values, int N = 100);
	
	void getTest(std::vector<int> &image, int &label);
	void getTest(boost::numeric::ublas::matrix<double> &image, int &value);

	//Batch processing to store all images in one matrix
	void getBatch(boost::numeric::ublas::matrix<double> &images, boost::numeric::ublas::matrix<double> &values, int N = 100);
	void getBatch(boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> &images, 
					boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> &values, 
					int N = 100);

	void getBatch(boost::multi_array<double, 4> &images, boost::multi_array<double, 4> &values, int N = 100);
	void getTests(double* images, double* values, int N = 100);


	void reset();

protected:

	using b_matrix = boost::numeric::ublas::matrix<double>;
	using b_vector = boost::numeric::ublas::vector<double>;
	
	std::ifstream image_files;
	std::ifstream label_files;

	std::ifstream test_files;
	std::ifstream test_values;

	int resetCount = 0;
};

#endif