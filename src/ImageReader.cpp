#include "ImageReader.h"

ImageReader::ImageReader()
{
	image_files.open("/home/mike/Desktop/HandwrittenImages/images/train-images.idx3-ubyte");
	label_files.open("/home/mike/Desktop/HandwrittenImages/labels/train-labels.idx1-ubyte");

	test_files.open("/home/mike/Desktop/HandwrittenImages/images/t10k-images.idx3-ubyte");
	test_values.open("/home/mike/Desktop/HandwrittenImages/labels/t10k-labels.idx1-ubyte");

	int trash;

	image_files.read((char*)&trash, sizeof(trash));
	image_files.read((char*)&trash, sizeof(trash));
	image_files.read((char*)&trash, sizeof(trash));
	image_files.read((char*)&trash, sizeof(trash));

	label_files.read((char*)&trash, sizeof(trash));
	label_files.read((char*)&trash, sizeof(trash));

	test_files.read((char*)&trash, sizeof(trash));
	test_files.read((char*)&trash, sizeof(trash));
	test_files.read((char*)&trash, sizeof(trash));
	test_files.read((char*)&trash, sizeof(trash));

	test_values.read((char*)&trash, sizeof(trash));
	test_values.read((char*)&trash, sizeof(trash));

}

void ImageReader::getBatch(boost::numeric::ublas::matrix<double> &images, boost::numeric::ublas::matrix<double> &values, int N)
{
	images.resize(28*28, N);
	values.resize(10, N);

	unsigned char temp;

	for(int i = 0; i < N; i++)
	{
		if(image_files.eof())
			reset();

		for(int j = 0; j<images.size1(); j++) {
			image_files.read((char*)&temp, sizeof(temp));
			images(j,i) = (double)temp/255.0;
		}

		label_files.read((char*)&temp, sizeof(temp));
		for(int j = 0; j < values.size1(); j++) {
			if((int)temp == j)
				values(j,i) = 1;
			else
				values(j,i) = 0;
		}
	}
}

void ImageReader::getBatch(boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> &images, 
							boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major> &values, 
							int N)
{
	images.resize(28*28, N);
	values.resize(10, N);

	unsigned char temp;

	for(int i = 0; i < N; i++)
	{
		if(image_files.eof())
			reset();

		for(int j = 0; j<images.size1(); j++) {
			image_files.read((char*)&temp, sizeof(temp));
			images(j,i) = (double)temp/255.0;
		}

		label_files.read((char*)&temp, sizeof(temp));
		for(int j = 0; j < values.size1(); j++) {
			if((int)temp == j)
				values(j,i) = 1;
			else
				values(j,i) = 0;
		}
	}
}

void ImageReader::reset()
{
	resetCount++;
	std::cout << "file reset: " << resetCount << "\n";

//	image_files.seekg(0, image_files.beg);
//	label_files.seekg(0, label_files.beg);

	image_files.close();
	label_files.close();

	test_files.close();
	test_values.close();

	image_files.open("/home/mike/Desktop/HandwrittenImages/images/train-images.idx3-ubyte");
	label_files.open("/home/mike/Desktop/HandwrittenImages/labels/train-labels.idx1-ubyte");
	
	test_files.open("/home/mike/Desktop/HandwrittenImages/images/t10k-images.idx3-ubyte");
	test_values.open("/home/mike/Desktop/HandwrittenImages/labels/t10k-labels.idx1-ubyte");

	int trash;

	image_files.read((char*)&trash, sizeof(trash));
	image_files.read((char*)&trash, sizeof(trash));
	image_files.read((char*)&trash, sizeof(trash));
	image_files.read((char*)&trash, sizeof(trash));

	label_files.read((char*)&trash, sizeof(trash));
	label_files.read((char*)&trash, sizeof(trash));

	test_files.read((char*)&trash, sizeof(trash));
	test_files.read((char*)&trash, sizeof(trash));
	test_files.read((char*)&trash, sizeof(trash));
	test_files.read((char*)&trash, sizeof(trash));

	test_values.read((char*)&trash, sizeof(trash));
	test_values.read((char*)&trash, sizeof(trash));}

void ImageReader::getImage(std::vector<int> &image, int &label)
{
	if(image_files.eof())
		reset();

	image.resize(28*28);
	unsigned char temp;
	for(int i = 0; i < 28*28; i++) {
		image_files.read((char*)&temp, sizeof(temp));
		image[i] = (int)temp;
	}

	label_files.read((char*)&temp, sizeof(temp));
	label = (int)temp;
}

void ImageReader::getImage(boost::numeric::ublas::matrix<double> &image, int &value)
{
	if(image_files.eof())
		reset();

	image.resize(28,28);
	unsigned char temp;

	for(int i = 0; i < 28; i++) {
		for(int j = 0; j < 28; j++) {
			image_files.read((char*)&temp, sizeof(temp));
			image(i,j) = (double)temp;
		}
	}

	label_files.read((char*)&temp, sizeof(temp));
	value = (int)temp;
}
void ImageReader::getTest(boost::numeric::ublas::matrix<double> &image, int &value)
{	
	image.resize(28*28, 1);
	unsigned char temp;

		for(int j = 0; j < 28*28; j++) {
			test_files.read((char*)&temp, sizeof(temp));
			image(j, 0) = (double)temp/255.0;
		}

	test_values.read((char*)&temp, sizeof(temp));
	value = (int)temp;
}

void ImageReader::getTest(std::vector<int> &image, int &label)
{
	image.resize(28*28);
	unsigned char temp;
	for(int i = 0; i < 28*28; i++) {
		test_files.read((char*)&temp, sizeof(temp));
		image[i] = (int)temp;
	}

	test_values.read((char*)&temp, sizeof(temp));
	label = (int)temp;
}

void ImageReader::getBatch(std::vector<std::vector<int>> &batch, std::vector<int> &labels, int N)
{
	batch.resize(N);
	labels.resize(N);

	for(int i = 0; i < N; i++){
		getImage(batch[i], labels[i]);
	}
}

void ImageReader::getBatch(std::vector<boost::numeric::ublas::matrix<double>> &batch, std::vector<int> &values, int N)
{
	batch.resize(N);
	values.resize(N);

	for(int i = 0; i < N; i++)
	{
		getImage(batch[i], values[i]);
	}
}

void ImageReader::getBatch(boost::multi_array<double, 4> &images, boost::multi_array<double, 4> &values, int N) {
	unsigned char temp;

	double* img_ptr = &images[0][0][0][0];
	double* val_ptr = &values[0][0][0][0];

	for(int b = 0; b < N; b++) {

		if(image_files.eof())
			reset();

		for(int row = 0; row < 28; row++) {
			for(int col = 0; col < 28; col++) {
				image_files.read((char*)&temp, sizeof(temp));
				*img_ptr = ((double)temp/255);
				img_ptr++;
			}
		}

		label_files.read((char*)&temp, sizeof(temp));
		for(int j = 0; j < 10; j++) {
			if((int)temp == j)
				*val_ptr = 1;
			else
				*val_ptr = 0;
			val_ptr++;
		}
	}

}

void ImageReader::getTests(double* images, double* values, int N) {
	unsigned char temp;

	for(int b = 0; b < N; b++) {
		if(test_files.eof())
			reset();

		for(int i = 0; i < 28*28; i++) {
			test_files.read((char*)&temp, sizeof(temp));
			*images = ((double)temp/255);
			images++;
		}

		test_values.read((char*)&temp, sizeof(temp));
		for(int j = 0; j < 10; j++) {
			if(temp == j)
				*values = 1;
			else
				*values = 0;
			values++;
		}
	}
}