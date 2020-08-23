#include "mpi.h"
#include "matrix.hpp"
#include "io.hpp"

#include <vector>

#include "ImageReader.h"
#include "ParallelNet.h"
#include "Utilities.h"

int main(int argc, char const *argv[])
{
	MPI_Init(NULL, NULL);

	Utilities utils;
	ImageReader reader;

	int rank, size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int layers = 4;
	int sizes[] = {28*28, 16, 16, 10};
	int batches;
	if(argc > 1)
		batches = std::stoi(argv[1]);
	else
		batches = 100;
	int batch_size = 100;
	std::vector<int> s {28*28, 24, 16, 10, batch_size, batches};

	ParallelNet net(sizes, layers, batches, batch_size, MPI_COMM_WORLD);
	utils.startTimer();
	net.initialize();
	net.learn();

	std::string filename = "ParallelData.csv";

	net.saveData(filename);

//	net.readData(filename);
	boost::numeric::ublas::matrix<double> test_img;
	boost::numeric::ublas::vector<double> result;
	int real_value;

	int tests;
	if(argc > 2)
		tests = std::stoi(argv[2]);
	else
		tests = 100;
	int guess = 0;
	int count = 0;
	if(rank == 0) {
		for(int i = 0; i < tests; i++) {
			reader.getTest(test_img, real_value);
			net.predict(test_img, result);
			for(int i = 0; i < result.size(); i++) {
				if(result(i) > result(guess))
					guess = i;
			}
			if(guess == real_value)
				count++;
		}
		std::cout << "Accuracy: " << count << " / " << tests << '\n';
	}

	MPI_Barrier(MPI_COMM_WORLD);
	std::cout << '\n';

/*
	int mat_size = 16;

	boost::numeric::ublas::matrix<int, boost::numeric::ublas::column_major> global_mat(mat_size, mat_size);
	boost::numeric::ublas::matrix<int> local_mat(mat_size/size, mat_size);

	if(rank == 0) {
		for(int i = 0; i < mat_size; i++) {
			for(int j = 0; j < mat_size; j++) {
				global_mat(i, j) = i*100 + j;
			}
		}

	//	std::cout << "Global Matrix: " << global_mat << '\n';
	}

//	MPI_Bcast(&global_mat(0, 0), mat_size*mat_size, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Scatter(	&global_mat(0,0), 
					mat_size*mat_size/size, 
					MPI_INT, 
					&local_mat(0, 0), 
					mat_size*mat_size/size, 
					MPI_INT,
					0,
					MPI_COMM_WORLD);


	boost::numeric::ublas::vector<boost::numeric::ublas::matrix<int, boost::numeric::ublas::column_major>> global_vec(1);
	boost::numeric::ublas::vector<boost::numeric::ublas::matrix<int, boost::numeric::ublas::column_major>> local_vec(1);
	boost::numeric::ublas::vector<boost::numeric::ublas::matrix<int, boost::numeric::ublas::column_major>> gather_vec(1);
	global_vec(0).resize(mat_size, mat_size);
	local_vec(0).resize(mat_size, mat_size/4);
	gather_vec(0).resize(mat_size, mat_size);

	if(rank == 0) {
		global_vec(0) = global_mat;
	}

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Scatter(	&global_vec(0)(0,0),
					mat_size*mat_size/size,
					MPI_INT,
					&local_vec(0)(0,0),
					mat_size*mat_size/size,
					MPI_INT,
					0,
					MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Gather(	&local_vec(0)(0,0),
				mat_size*mat_size/size,
				MPI_INT,
				&gather_vec(0)(0,0),
				mat_size*mat_size/size,
				MPI_INT,
				0,
				MPI_COMM_WORLD);							

	if(rank == 0)
		//std::cout << "Gathered Matrix: " << gather_vec(0) << '\n';
*/
	MPI_Finalize();
	return 0;
}