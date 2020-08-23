#ifndef PARALLELNET_H
#define PARALLELNET_H

//MPI Header
#include <mpi.h>

//Boost Library Stuff
#include "matrix.hpp"
#include "io.hpp"

//STL Stuff
#include <cmath>
#include <string>
#include <fstream>

//Image Reader to get train/test data
#include "ImageReader.h"

//Utilities
#include "Utilities.h"

using b_mat = boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major>;
using b_vec = boost::numeric::ublas::vector<boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major>>;

class ParallelNet {
protected:
	//MPI Stuff
	MPI_Comm	m_comm; 			//MPI communicator
	int 		m_proc_id;			//The process id (rank)
	int			m_comm_size;		//Number of processes on the communicator

	//Parameters independent of master/slave process
	int* layer_sizes;				//The size of each layer in the network
	int n_layers;					//The total number of layers in the network, including input layer (0)
	int n_batches;					//The number of sets of training data to process
	int L;							//The index of the final layer
	double avg_cost;				//Average cost

	double alpha_init	= 0.7;		//The initial learn rate, used in the backpropagation calculation W_new = W_old - alpha * dW
	double alpha_new	= alpha_init;
	double weight_min 	= -1;		//Minimum weight
	double weight_max 	= 1;		//Maximum weight
	double bias_min 	= -1;		//Minimum bias
	double bias_max 	= 1;		//Maximum bias

	ImageReader img_reader;			//Reader to fetch the train/test images;
	Utilities utils;				//Utilities class for RNG

	//Global Data Containers
	//Main process with scatter this data among the processes
	b_vec A_global;					//Stores the layer data. Each column represents a single training cases's data. The column size is equal to the batch size
	b_vec W_global;					//Stores the weights to calculate subsequent layers. Same size no matter the batch size so no need for a local one
	b_vec B_global;					//Stores the biases between layers
	b_vec dW_global;				//Stores the derivatives of the cost function wrt the weights
	b_vec dB_global;				//Stores the derivatives of the cost function wrt the biases
	b_vec dSig_global;				//Stores the derivatives of the activation function

	b_mat				Y_global;	//The expected final layer result, used to compute the cost, i.e. C = f(A[L], Y)
	int 		batch_size_global;	//The total batch size, each process will handle an equal portion of the total batch size

	//Local Data Containers
	//Forward and backward propagation will be done on these containers
	//Main process with then gather the results
	b_vec A_local;		
	b_vec B_local;		
	b_vec dSig_local;	
	b_vec dW_local;		
	b_vec dB_local;

	b_mat				Y_local;
	int 		batch_size_local;

	b_vec dW_update;

	boost::numeric::ublas::vector<double> costs;

public:
	//Constructor 	- 	Sets various member variables and correctly sizes global and local data containers
	//sizes 		- 	Array containing the sizes of each layer in the network
	//layers 		- 	Number of layers in the network
	//batches 		-	Number of batches to train on
	//batch_size 	-	Number of training examples in each batch
	//comm 			-	MPI communicator over which to distribute the global data
	ParallelNet(int* sizes, int layers, int batches, int batch_size, MPI_Comm comm);

	//Deconstructor	-	Frees allocated memory
	~ParallelNet() { delete[] layer_sizes; }

	//Initialize	-	Master process initializes W and B matrices to random values and broadcasts to all processes
	void initialize();

	//Sigmoid		-	This is the activation function, it is the final calculation to get the layer values
	//m 			- 	The function will operate on each value in the matrix individually 
	void sigmoid(b_mat &m);

	void reLU(b_mat &m, bool derivative = false);

	//Get Batch 	-	Root process will get the batch from the image reader and scatter the images accross the process
	void getBatch();

	//Forward Propagation 	-	First main calculation, which calculates every layer for every training case in the batch
	void forwardProp();

	//Backward Propagation 	-	Second main calculation, which calculates derivatives of cost function, and updates W and B parameters
	void backProp();

	void computeGradients();

	//Update 				- 	Update the W and B parameters by first finding the best learning parameter
	void update();

	//Average Cost 			- 	Compute the average cost function over all training cases in the global batch
	double avgCost();

	//Learn 				- 	Learning loop. Calls getBatch, forwardProp, backProp, and avgCost for each batch
	void learn();

	//Predict				-	Given a test case, predict the result
	void predict(boost::numeric::ublas::matrix<double> &image, boost::numeric::ublas::vector<double> &result);

	//Save Training Data 	- 	Writes out W and B to a file
	void saveData(std::string &filename);

	//Read Data
	void readData(std::string &filename);
};

#endif