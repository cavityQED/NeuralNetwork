#include "NeuralNet.h"

NeuralNet::NeuralNet(std::vector<int> &sizes, bool testing)
{
	m_testing = testing;

	int s = sizes.size();
	layer_sizes.resize(s-2);

	n_batches = sizes[s-1];
	batch_size = sizes[s-2];

	for(int i = 0; i < s-2; i++)
		layer_sizes[i] = sizes[i];

	n_layers = layer_sizes.size();
	L = n_layers-1;

	A.resize(n_layers);
	W.resize(n_layers);
	B.resize(n_layers);
	dSig.resize(n_layers);
	dW.resize(n_layers);
	dB.resize(n_layers);

	A[0].resize(layer_sizes[0], batch_size);
	
	for(int i = 1; i < n_layers; i++) {
		A[i].resize(layer_sizes[i], batch_size);
		W[i].resize(layer_sizes[i], layer_sizes[i-1]);
		B[i].resize(layer_sizes[i], batch_size);
	}

	if(m_testing)
		std::cout << "Constructed\n";
}

void NeuralNet::init()
{
	for(int i = 1; i < n_layers; i++) {

		for(int j = 0; j < W[i].size1(); j++) {
			for(int k = 0; k < W[i].size2(); k++) {
				W[i](j,k) = utils.RNG<double>(min_weight, max_weight);
			}
		}

		for(int j = 0; j < B[i].size1(); j++) {
			B[i](j,0) = utils.RNG<double>(min_bias, max_bias);
			for(int k = 1; k < B[i].size2(); k++) {
				B[i](j,k) = B[i](j,0);
			}
		}
	}
	if(m_testing)
		std::cout << "Initialized\n";
}

void NeuralNet::sigmoid(mat &m)
{
	auto it1 = m.begin1();
	while(it1 != m.end1())
	{
		for(auto it2 = it1.begin(); it2 != it1.end(); it2++)
		{
			*it2 = 1/(1 + exp(-(*it2)));
		}
		it1++;
	}
}

void NeuralNet::getBatch()
{
	reader.getBatch(A[0], Y, batch_size);
	if(m_testing)
		std::cout << "Got Batch\n";
}

void NeuralNet::forwardProp()
{
	for(int i = 1; i < n_layers; i++) {
		A[i] = prod(W[i], A[i-1]) + B[i];
		sigmoid(A[i]);
		boost::numeric::ublas::scalar_matrix<double> I(A[i].size1(), A[i].size2(), 1.0);
		dSig[i] = element_prod(A[i], (I - A[i]));
	}
	if(m_testing)
		std::cout << "Propagated Forward\n";
}

void NeuralNet::backProp()
{
	dB[L] = 2*element_prod(dSig[L], A[L] - Y);
	dW[L] = prod(dB[L], trans(A[L-1]))/batch_size;

	for(int i = L-1; i > 0; i--) {
		dB[i] = element_prod(dSig[i], prod(trans(W[i+1]), dB[i+1]));
		dW[i] = prod(dB[i], trans(A[i-1]))/batch_size;
	}

	for(int i = 1; i < n_layers; i++)
	{
		boost::numeric::ublas::scalar_matrix<double> I(B[i].size2(), batch_size, 1.0);
		dB[i] = prod(dB[i], I)/batch_size;
		B[i] -= alpha*dB[i];
		W[i] -= alpha*dW[i];
	}
	
	if(m_testing)
		std::cout << "Propagated Backward\n";
}

double NeuralNet::avgCost()
{
	double sum = 0;
	for(int i = 0; i < A[L].size2(); i++) {
		for(int j = 0; j < A[L].size1(); j++) {
			sum += (A[L](j,i) - Y(j, i))*(A[L](j,i) - Y(j, i));
		}
	}
	sum /= A[L].size1()*batch_size;

	if(m_testing)
		std::cout << "Computed Average Cost\n";

	return sum;
}

void NeuralNet::learn()
{
	for(int i = 0; i < n_batches; i++)
	{
		utils.startTimer();
		getBatch();
		forwardProp();
		backProp();
		std::cout << "Initial Cost On Batch " << i << ": " << avgCost() << '\t';
		std::cout << "Computed in " << utils.getElapsedTime() << "ms\n";
	}
}

double NeuralNet::predict(mat &image)
{
	std::vector<mat> layers;
	layers.resize(n_layers);
	layers[0] = image;

	std::vector<mat> b;

	b.resize(n_layers);

	for(int i = 1; i < n_layers; i++)
	{
		b[i].resize(B[i].size1(), 1);
		for(int j = 0; j < B[i].size1(); j++){
			b[i](j,0) = B[i](j,0);
		}

		layers[i] = prod(W[i], layers[i-1]) + b[i];
		sigmoid(layers[i]);
	}

	double max = 0.0;
	int predicted_value;
	for(int i = 0; i < layers[L].size1(); i++)
	{
		if(layers[L](i, 0) > max) {
			max = layers[L](i,0);
			predicted_value = i;
		}
	}

	if(m_testing)
		std::cout << layers[L] << '\n';

	return predicted_value;
}

void NeuralNet::predict(mat &image, std::vector<int> &predictedLayer)
{
	std::vector<mat> layers;
	layers.resize(n_layers);
	layers[0] = image;

	std::vector<mat> b;

	b.resize(n_layers);

	for(int i = 1; i < n_layers; i++)
	{
		b[i].resize(B[i].size1(), 1);
		for(int j = 0; j < B[i].size1(); j++){
			b[i](j,0) = B[i](j,0);
		}

		layers[i] = prod(W[i], layers[i-1]) + b[i];
		sigmoid(layers[i]);
	}

	predictedLayer.resize(layers[n_layers-1].size1());
	for(int i = 0; i < predictedLayer.size(); i++)
	{
		predictedLayer[i] = 300*layers[n_layers-1](i,0);
	}
}

void NeuralNet::predict(std::vector<int> &image, std::vector<int> &predictedLayer)
{
	mat im;
	im.resize(28*28, 1);
	for(int i = 0; i < 28*28; i++)
		im(i, 0) = image[i]/255;

	predict(im, predictedLayer);
}

double NeuralNet::predict(std::vector<int> &image)
{
	mat im;
	im.resize(28*28, 1);
	for(int i = 0; i < 28*28; i++)
		im(i, 0) = image[i];

	return predict(im);
}

void NeuralNet::readData(std::fstream &file)
{
	char s[256];
	char c;

	double num;

	file.getline(s, 256);
	for(int i = 1; i < W.size(); i++) {

		for(int j = 0; j < W[i].size1(); j++) {
			for(int k = 0; k < W[i].size2(); k++) {
				file >> num;
				W[i](j,k) = num;
				file.get(c);
			}
		}
				file.get(c);
				file.get(c);
				file.get(c);
				file.get(c);
				file.get(c);
	}

	file.getline(s, 256);
	for(int i = 1; i < B.size(); i++) {

		for(int j = 0; j < B[i].size1(); j++) {
			file >> num;
			B[i](j, 0) = num;
			file.get(c);
		}
				file.get(c);
				file.get(c);
				file.get(c);
				file.get(c);
				file.get(c);
	}
}

void NeuralNet::saveTrainingData(const std::string &filename)
{
	dataOut.open(filename);

	for(int i = 1; i < n_layers; i++) {
		dataOut << "W[" << i << "]" << '\n';

		for(int j = 0; j < W[i].size1(); j++) {
			for(int k = 0; k < W[i].size2(); k++)
				dataOut << W[i](j,k) << ',';
		
			dataOut << '\n';
		}
	}

	for(int i = 1; i < n_layers; i++) {
		dataOut << "B[" << i << "]" << '\n';
		for(int j = 0; j < B[i].size1(); j++) 
			dataOut << B[i](j,0) << ',';
		dataOut << '\n';
	}
}