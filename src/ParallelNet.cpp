#include "ParallelNet.h"

ParallelNet::ParallelNet(int* sizes, int layers, int batches, int batch_size, MPI_Comm comm) {
	//Set the MPI communicator member variable
	m_comm = comm;
	//Get the size of the communicator
	MPI_Comm_size(comm, &m_comm_size);
	//Get the process id (0 will be the master)
	MPI_Comm_rank(comm, &m_proc_id);

	//Set the total number of batches to process
	n_batches = batches;
	costs.resize(n_batches);

	//Set the batch sizes, making sure each process gets an equal number
	if(batch_size % m_comm_size == 0) {
		batch_size_global 	= batch_size;
		batch_size_local	= batch_size_global / m_comm_size;
	}
	else {
		batch_size_global 	= batch_size + (m_comm_size - batch_size % m_comm_size);
		batch_size_local 	= batch_size_global / m_comm_size;
	}

	//Set various layer information
	n_layers 	= layers;
	L 			= n_layers - 1;
	layer_sizes = new int[n_layers];
	for(int i = 0; i < layers; i++)
		layer_sizes[i] = sizes[i];

	//Resize the vectors that hold the data for each layer
	A_global.	resize(n_layers);
	W_global.	resize(n_layers);
	B_global.	resize(n_layers);
	dW_global.	resize(n_layers);
	dB_global.	resize(n_layers);
	dSig_global.resize(n_layers);
	Y_global.	resize(layer_sizes[L], batch_size_global);

	A_local.	resize(n_layers);
	B_local.	resize(n_layers);
	dSig_local.	resize(n_layers);
	dW_local.	resize(n_layers);
	dB_local.	resize(n_layers);
	Y_local.	resize(layer_sizes[L], batch_size_local);

	dW_update.resize(n_layers);

	//Resize the data containers for each layer
	A_global(0).resize(layer_sizes[0], batch_size_global);
	A_local(0).	resize(layer_sizes[0], batch_size_local);
	for(int i = 1; i < n_layers; i++) {
		A_global(i).	resize(layer_sizes[i], batch_size_global);
		W_global(i).	resize(layer_sizes[i], layer_sizes[i-1]);
		B_global(i).	resize(layer_sizes[i], 1);
		dW_global(i).	resize(W_global(i).size1(), W_global(i).size2());
		dB_global(i).	resize(B_global(i).size1(), B_global(i).size2());
		dSig_global(i).	resize(A_global(i).size1(), A_global(i).size2());

		A_local(i).		resize(layer_sizes[i], batch_size_local);
		B_local(i).		resize(layer_sizes[i], batch_size_local);
		dW_local(i).	resize(W_global(i).size1(), W_global(i).size2());
		dB_local(i).	resize(B_local(i).size1(), B_local(i).size2());
		dSig_local(i).	resize(A_local(i).size1(), A_local(i).size2());

		dW_update(i).resize(W_global(i).size1(), W_global(i).size2());
	}
}

void ParallelNet::initialize() {

	//Only the root process initializes the values
	if(m_proc_id == 0) {
		for(int i = 1; i < n_layers; i++) {
			for(int j = 0; j < W_global(i).size1(); j++) {
				for(int k = 0; k < W_global(i).size2(); k++) {
					W_global(i)(j, k) = utils.RNG<double>(weight_min, weight_max);
					dW_update(i)(j,k) = 0;
				}
			}

			for(int j = 0; j < B_global(i).size1(); j++) {
				B_global(i)(j, 0) = utils.RNG<double>(bias_min, bias_max);
			}
		}
	}

	//Broadcast W and B
	for(int i = 1; i < n_layers; i++) {
		MPI_Bcast(	&W_global(i)(0,0),
					W_global(i).size1() * W_global(i).size2(),
					MPI_DOUBLE,
					0,
					m_comm);
		MPI_Bcast(	&B_global(i)(0,0),
					B_global(i).size1(),
					MPI_DOUBLE,
					0,
					m_comm);
	}

	//Copy global B values into the local matrix
	for(int i = 1; i < n_layers; i++) {
		for(int j = 0; j < B_local(i).size1(); j++) {
			for(int k = 0; k < B_local(i).size2(); k++) {
				B_local(i)(j, k) = B_global(i)(j, 0);
			}
		}
	}
}

void ParallelNet::sigmoid(b_mat &m) {
	auto it1 = m.begin1();
	while(it1 != m.end1()) {
		for(auto it2 = it1.begin(); it2 != it1.end(); it2++) {
			*it2 = 1 / (1 + exp(-*it2));
		}
		it1++;
	}
}

void ParallelNet::reLU(b_mat &m, bool derivative) {
	auto it1 = m.begin1();
	while(it1 != m.end1()) {
		for(auto it2 = it1.begin(); it2 != it1.end(); it2++) {
			if(*it2 <= 0)
				*it2 = 0;
			if(*it2 > 0 && derivative)
				*it2 = 1.0;
		}
		it1++;
	}
}

void ParallelNet::getBatch() {
	//Root process gets the batch
	if(m_proc_id == 0)
		img_reader.getBatch(A_global(0), Y_global, batch_size_global);

	//Scatter the batch accross all process
	MPI_Scatter(	&A_global(0)(0,0),
					A_global(0).size1() * batch_size_local,
					MPI_DOUBLE,
					&A_local(0)(0,0),
					A_global(0).size1() * batch_size_local,
					MPI_DOUBLE,
					0,
					m_comm);
	
	MPI_Scatter(	&Y_global(0,0),
					Y_global.size1() * batch_size_local,
					MPI_DOUBLE,
					&Y_local(0,0),
					Y_global.size1() * batch_size_local,
					MPI_DOUBLE,
					0,
					m_comm);
}

void ParallelNet::forwardProp() {

	//Each process will proagate with their local data
	for(int i = 1; i < n_layers; i++) {
		A_local(i) = prod(W_global(i), A_local(i-1)) + B_local(i);
		sigmoid(A_local(i));
		boost::numeric::ublas::scalar_matrix<double> I(A_local(i).size1(), A_local(i).size2(), 1.0);
		dSig_local(i) = element_prod(A_local(i), I - A_local(i));
	}
}

void ParallelNet::computeGradients() {
	dB_local(L) = 2*element_prod(dSig_local(L), A_local(L) - Y_local);
	dW_local(L) = prod(dB_local(L), trans(A_local(L-1))) / batch_size_local;

	for(int i = L-1; i > 0; i--) {
		dB_local(i) = element_prod(dSig_local(i), prod(trans(W_global(i+1)), dB_local(i+1)));
		dW_local(i) = prod(dB_local(i), trans(A_local(i-1))) / batch_size_local;	//this is the average dW over all local training examples
	}

	for(int i = 0; i < n_layers; i++) {
		//Compute the average dB over all local training examples
		boost::numeric::ublas::scalar_matrix<double> I(dB_local(i).size2(), 1, 1.0);
		dB_local(i) = prod(dB_local(i), I) / batch_size_local;
	}

	for(int i = 1; i < n_layers; i++) {
		MPI_Allreduce(	&dW_local(i)(0,0),
						&dW_global(i)(0,0),
						dW_local(i).size1() * dW_local(i).size2(),
						MPI_DOUBLE,
						MPI_SUM,
						m_comm);

		MPI_Allreduce(	&dB_local(i)(0,0),
						&dB_global(i)(0,0),
						dB_local(i).size1(),
						MPI_DOUBLE,
						MPI_SUM,
						m_comm);

		dW_global(i) /= m_comm_size;
		dB_global(i) /= m_comm_size;

		dW_update(i) = dW_global(i) + 0.05*dW_update(i);

		W_global(i) -= alpha_new*dW_update(i);
		for(int j = 0; j < B_local(i).size1(); j++) {
			for(int k = 0; k < B_local(i).size2(); k++) {
				B_local(i)(j, k) -= alpha_new*dB_global(i)(j,0);
			}
		}
	}
}

void ParallelNet::backProp() {
	//Each process calculates local derivatives
	dB_local(L) = 2*element_prod(dSig_local(L), A_local(L) - Y_local);
	dW_local(L) = prod(dB_local(L), trans(A_local(L-1))) / batch_size_local;

	for(int i = L-1; i > 0; i--) {
		dB_local(i) = element_prod(dSig_local(i), prod(trans(W_global(i+1)), dB_local(i+1)));
		dW_local(i) = prod(dB_local(i), trans(A_local(i-1))) / batch_size_local;	//this is the average dW over all local training examples
	}
	for(int i = 0; i < n_layers; i++) {
		//Compute the average dB over all local training examples
		boost::numeric::ublas::scalar_matrix<double> I(dB_local(i).size2(), 1, 1.0);
		dB_local(i) = prod(dB_local(i), I) / batch_size_local;
	}

	//Gather the data from all processes
	for(int i = 1; i < n_layers; i++) {
		MPI_Allreduce(	&dW_local(i)(0,0),
						&dW_global(i)(0,0),
						dW_local(i).size1() * dW_local(i).size2(),
						MPI_DOUBLE,
						MPI_SUM,
						m_comm);

		MPI_Allreduce(	&dB_local(i)(0,0),
						&dB_global(i)(0,0),
						dB_local(i).size1(),
						MPI_DOUBLE,
						MPI_SUM,
						m_comm);

		//Update the W and B matrices
		W_global(i) -= alpha_new*dW_global(i)/m_comm_size;
		for(int j = 0; j < B_local(i).size1(); j++) {
			for(int k = 0; k < B_local(i).size2(); k++) {
				B_local(i)(j,k) -= alpha_new*dB_global(i)(j,0)/m_comm_size;
			}
		}
	}
}

double ParallelNet::avgCost() {
	double cost_local = 0;
	double cost_global = 0;

	for(int i = 0; i < batch_size_local; i++) {
		for(int j = 0; j < layer_sizes[L]; j++) {
			cost_local += (A_local(L)(j, i) - Y_local(j, i)) * (A_local(L)(j, i) - Y_local(j, i));
		}
	}

	cost_local /= (layer_sizes[L]*batch_size_local);
	MPI_Allreduce(	&cost_local,
					&cost_global,
					1,
					MPI_DOUBLE,
					MPI_SUM,
					m_comm);

	return cost_global/m_comm_size;
}

void ParallelNet::learn() {
	for(int i = 0; i < n_batches; i++) {
		utils.startTimer();
		getBatch();
		forwardProp();
		computeGradients();
//		backProp();
		avg_cost = avgCost();
		costs(i) = avg_cost;
		if(m_proc_id == 0){
			std::cout << "Initial cost on batch " << i+1 << " / " << n_batches << ": " << avg_cost << '\t';
			std::cout << "Computed in " << utils.getElapsedTime() << "ms\t";
			std::cout << "Alpha = " << alpha_new << '\n';
		}
		alpha_new = alpha_init * (1 / (1 + .0002*i));
	}
}

void ParallelNet::predict(boost::numeric::ublas::matrix<double> &image, boost::numeric::ublas::vector<double> &result) {

	for(int i = 0; i < n_layers; i++)
		A_global(i).resize(layer_sizes[i], 1);

	A_global(0) = image;

	for(int i = 1; i < n_layers; i++) {
		A_global(i) = prod(W_global(i), A_global(i-1)) + B_global(i);
		sigmoid(A_global(i));
	}

	result.resize(A_global(L).size1());
	for(int i = 0; i < layer_sizes[L]; i++)
		result(i) = A_global(L)(i, 0);
}

void ParallelNet::saveData(std::string &filename) {
	if(m_proc_id == 0) {
		std::ofstream file;
		file.open(filename);
	
		for(int i = 1; i < n_layers; i++) {
			file << "W[" << i << "]" << '\n';
	
			for(int j = 0; j < W_global(i).size1(); j++) {
				for(int k = 0; k < W_global(i).size2(); k++)
					file << W_global(i)(j,k) << ',';
			
				file << '\n';
			}
		}
	
		for(int i = 1; i < n_layers; i++) {
			file << "B[" << i << "]" << '\n';
			for(int j = 0; j < B_global(i).size1(); j++) 
				file << B_global(i)(j,0) << ',';
			file << '\n';
		}

		for(int i = 0; i < n_batches; i++) {
			file << costs(i) << '\n';
		}
	}
}

void ParallelNet::readData(std::string &filename) {
	std::fstream file;
	file.open(filename);

	char s[256];
	char c;

	double num;

	file.getline(s, 256);
	for(int i = 1; i < W_global.size(); i++) {

		for(int j = 0; j < W_global(i).size1(); j++) {
			for(int k = 0; k < W_global(i).size2(); k++) {
				file >> num;
				W_global(i)(j,k) = num;
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
	for(int i = 1; i < B_global.size(); i++) {

		for(int j = 0; j < B_global(i).size1(); j++) {
			file >> num;
			B_global(i)(j, 0) = num;
			file.get(c);
		}
				file.get(c);
				file.get(c);
				file.get(c);
				file.get(c);
				file.get(c);
	}

	for(int i = 1; i < n_layers; i++) {
		for(int j = 0; j < B_local(i).size1(); j++) {
			for(int k = 0; k < B_local(i).size2(); k++) {
				B_local(i)(j,k) = B_global(i)(j, 0);
			}
		}
	}
}
