#include "InputLayer.h"

InputLayer::InputLayer(Layer* parent, MPI_Comm comm) : Layer(parent, comm) {
	m_layer_type = INPUT;
}

void InputLayer::forward_propagation() {
	if(m_child != nullptr)
		m_child->forward_propagation();
}

void InputLayer::backward_propagation() {
	return;
}

void InputLayer::set_data_size(int rows, int cols, int channels) {
	m_rows_data = rows;
	m_cols_data = cols;
	m_channels_data = channels;

	m_layer_data.resize(		boost::extents	[m_batch_size_local]
												[m_channels_data]
												[m_rows_data]
												[m_cols_data]);
	m_layer_data_global.resize(	boost::extents	[m_batch_size_global]
												[m_channels_data]
												[m_rows_data]
												[m_cols_data]);
}	

void InputLayer::set_input_data(const boost::multi_array<double, 4>& data) {
	std::cout << "setting conv input data\n";

	assert(data.num_elements() == m_layer_data_global.num_elements());

	int count = 0; 

	if(m_comm_rank == 0) {
		const double* 	in_it		= &data[0][0][0][0];
		double*			layer_it 	= &m_layer_data_global[0][0][0][0];

		for(int e = 0; e < data.num_elements(); e++) {
			*layer_it = *in_it;
			count++;
			in_it++;
			layer_it++;
		}
		std::cout << "Set " << count << " values\n";
	}

	MPI_Scatter(	&m_layer_data_global[0][0][0][0],
					m_batch_size_local * m_rows_data * m_cols_data * m_channels_data,
					MPI_DOUBLE,
					&m_layer_data[0][0][0][0],
					m_batch_size_local * m_rows_data * m_cols_data * m_channels_data,
					MPI_DOUBLE,
					0,
					m_comm);
	std::cout << "scattered input data\n";
}

void InputLayer::set_test_data(const double* data, int size) {
	int test_batches = size/elems();

	m_test_data_global.resize(	boost::extents	[test_batches]
												[m_channels_data]
												[m_rows_data]
												[m_cols_data]);

	if(m_comm_rank == 0) {
		double* m_test_ptr = &m_test_data_global[0][0][0][0];

		for(int i = 0; i < size; i++) {
			*m_test_ptr = *data;
			data++;
			m_test_ptr++;
		}
	}
}

void InputLayer::predict() {
	MPI_Scatter(	&m_test_data_global[0][0][0][0],
					m_test_data_global.num_elements() / m_comm_size,
					MPI_DOUBLE,
					&m_layer_data[0][0][0][0],
					m_test_data_global.num_elements() / m_comm_size,
					MPI_DOUBLE,
					0,
					m_comm);

	m_child->forward_propagation();
}