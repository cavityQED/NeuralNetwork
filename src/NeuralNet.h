#ifndef NEURALNET_H
#define NEURALNET_H

#include "matrix.hpp"
#include "io.hpp"

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>

#include "Utilities.h"
#include "ImageReader.h"

class NeuralNet
{
public:
	using mat = boost::numeric::ublas::matrix<double>;

	NeuralNet() {}
	NeuralNet(std::vector<int> &sizes, bool testing = false);
	void init();
	void sigmoid(mat &m);
	void getBatch();
	void forwardProp();
	void backProp();
	double avgCost();
	void learn();
	void readData(std::fstream &file);
	void saveTrainingData(const std::string &filename);

	double predict(mat &image);
	double predict(std::vector<int> &image);
	void predict(mat &image, std::vector<int> &predictedLayer);
	void predict(std::vector<int> &image, std::vector<int> &predictedLayer);

	void setLearnRate(double rate) { alpha = rate; }


protected:
	bool m_testing;

	std::ofstream dataOut;

	Utilities utils;
	ImageReader reader;

	std::vector<int> layer_sizes;
	int batch_size;
	int n_batches;
	int n_layers;
	int L;

	double min_weight = -1.0;
	double max_weight = 1.0;
	double min_bias = -1.0;
	double max_bias = 1.0;
	double alpha = 0.7;

	std::vector<mat> A;
	std::vector<mat> W;
	std::vector<mat> B;
	std::vector<mat> dSig;
	std::vector<mat> dW;
	std::vector<mat> dB;

	mat I;
	mat Y;
};

#endif