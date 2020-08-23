#include "OutputLayer.h"

OutputLayer::OutputLayer(Layer* parent, MPI_Comm comm) : FullyConnectedLayer(parent, comm) {
	m_layer_type = OUTPUT;
}

void OutputLayer::backward_propagation() {
	d_cost();
	/*
	for(int b = 0; b < m_batch_size_local; b++) {
		for(int out_row = 0; out_row < m_rows_data; out_row++) {
			for(int in_ch = 0; in_ch < m_parent->channels(); in_ch++) {
				for(int in_row = 0; in_row < m_parent->rows(); in_row++) {
					for(int in_col = 0; in_col < m_parent->cols(); in_col++) {
						m_weight_gradient[out_row][in_ch][in_row][in_col] += 
							(*(m_parent->data_ptr(b, in_ch, in_row, in_col))
							*m_activation_gradient[b][0][out_row][0]
							*m_cost_gradient[b][0][out_row][0])/m_batch_size_local;
					}
				}
			}
			m_bias_gradient[0][0][out_row][0] += m_activation_gradient[b][0][out_row][0]*m_cost_gradient[b][0][out_row][0]/m_batch_size_local;
		} 
	}
	*/
	const double* d_act_ptr = &m_activation_gradient[0][0][0][0];
	const double* d_cost_ptr = &m_cost_gradient[0][0][0][0];
	const double* in_ptr;

	double* d_w_ptr;
	double* d_b_ptr = &m_bias_gradient[0][0][0][0];

	for(int b = 0; b < m_batch_size_local; b++) {
		d_w_ptr = &m_weight_gradient[0][0][0][0];
	//	d_b_ptr = &m_bias_gradient[0][0][0][0];
		for(int out_row = 0; out_row < m_rows_data; out_row++) {
			in_ptr = m_parent->data_ptr(b,0,0,0);
			for(int e = 0; e < m_parent->elems(); e++) {
	  			*d_w_ptr += (*in_ptr * (*d_act_ptr) * (*d_cost_ptr))/m_batch_size_local;
				in_ptr++;
				d_w_ptr++;
			}
			*d_b_ptr += (*d_act_ptr)*(*d_cost_ptr);
			d_b_ptr++;
			d_act_ptr++;
			d_cost_ptr++;
		}
	}

	if(test && m_comm_rank == 0) {
		std::cout << "Outpur Layer Bias Gradients:\n";
		const double* d_b = &m_bias_gradient[0][0][0][0];
		for(int i = 0; i < m_bias_gradient.num_elements(); i++) {
			std::cout << *d_b << ", ";
			d_b++;
		}

		std::cout << "\nOutput Layer Weight Gradients:\n";
		const double* d_w = &m_weight_gradient[0][0][0][0];
		for(int i = 0; i < m_weight_gradient.num_elements(); i++) {
			std::cout << *d_w << ", ";
			d_w++;
		}
		std::cout << "\n";
	}

	m_parent->backward_propagation();
	update();
}

void OutputLayer::set_data_size(int rows, int cols, int channels) {
	m_channels_data = 1;
	m_rows_data = rows;
	m_cols_data = 1;

	//Resize the data
	m_layer_data.resize(			boost::extents	[m_batch_size_local]
													[m_channels_data]
													[m_rows_data]
													[m_cols_data]);

	m_layer_data_global.resize(		boost::extents	[m_batch_size_global]
													[m_channels_data]
													[m_rows_data]
													[m_cols_data]);
	//Resize the weights
	m_weights.resize(				boost::extents	[m_rows_data]
													[m_parent->channels()]
													[m_parent->rows()]
													[m_parent->cols()]);
	//Resize the biases
	m_biases.resize(				boost::extents	[1]
													[1]
													[m_rows_data]
													[1]);
	//Resize the weight gradient
	m_weight_gradient.resize(		boost::extents	[m_rows_data]
													[m_parent->channels()]
													[m_parent->rows()]
													[m_parent->cols()]);
	//Resize the global weight gradient
	m_weight_grad_global.resize(	boost::extents	[m_rows_data]
													[m_parent->channels()]
													[m_parent->rows()]
													[m_parent->cols()]);
	//Resize the bias gradient
	m_bias_gradient.resize(			boost::extents	[m_batch_size_local]
													[1]
													[m_rows_data]
													[1]);
	//Resize the global bias gradient
	m_bias_grad_global.resize(		boost::extents	[m_batch_size_local]
													[1]
													[m_rows_data]
													[1]);
	//Resize the cost gradients
	m_cost_gradient.resize(			boost::extents	[m_batch_size_local]
													[m_channels_data]
													[m_rows_data]
													[m_cols_data]);
	//Resize the activation gradient
	m_activation_gradient.resize(	boost::extents	[m_batch_size_local]
													[m_channels_data]
													[m_rows_data]
													[m_cols_data]);
	//Resize the expected output
	m_expected_output.resize(		boost::extents	[m_batch_size_local]
													[m_channels_data]
													[m_rows_data]
													[m_cols_data]);

	m_expected_output_global.resize(		boost::extents	[m_batch_size_global]
													[m_channels_data]
													[m_rows_data]
													[m_cols_data]);

}
void OutputLayer::set_output_data(const boost::multi_array<double, 4>& data) {
	if(m_comm_rank == 0) {
		boost::multi_array<double, 4>::const_iterator	in_it		= data.begin();
		boost::multi_array<double, 4>::iterator 		layer_it	= m_expected_output_global.begin();
		
		while(in_it != data.end()) {
			*layer_it = *in_it;

			in_it++;
			layer_it++;
		}
	}

	MPI_Scatter(	&m_expected_output_global[0][0][0][0],
					m_batch_size_local*m_channels_data*m_rows_data*m_cols_data,
					MPI_DOUBLE,
					&m_expected_output[0][0][0][0],
					m_batch_size_local*m_channels_data*m_rows_data*m_cols_data,
					MPI_DOUBLE,
					0,
					m_comm);
}

double OutputLayer::cost() {
	double cost = 0;

	const double* computed_ptr = &m_layer_data[0][0][0][0];
	const double* expected_ptr = &m_expected_output[0][0][0][0];

	for(int i = 0; i < m_expected_output.num_elements(); i++) {
		cost += ((*expected_ptr) - (*computed_ptr)) * ((*expected_ptr) - (*computed_ptr));
		expected_ptr++;
		computed_ptr++;
	}

	cost /=	m_batch_size_local;

	double global_cost = 0;

	MPI_Allreduce(	&cost,
					&global_cost,
					1,
					MPI_DOUBLE,
					MPI_SUM,
					m_comm);

	global_cost /= m_comm_size;

	return global_cost;
}

void OutputLayer::d_cost() {
	const double* layer_data_ptr 		= &m_layer_data[0][0][0][0];
	const double* expected_output_ptr 	= &m_expected_output[0][0][0][0];
	double* cost_gradient_ptr			= &m_cost_gradient[0][0][0][0];

	for(int i = 0; i < m_layer_data.num_elements(); i++) {
		*cost_gradient_ptr = 2*(*layer_data_ptr - *expected_output_ptr);
		layer_data_ptr++;
		expected_output_ptr++;
		cost_gradient_ptr++;
	}
}

const boost::multi_array<double, 4>& OutputLayer::get_global_output() {
	MPI_Gather(	&m_layer_data[0][0][0][0],
				m_layer_data.num_elements(),
				MPI_DOUBLE,
				&m_layer_data_global[0][0][0][0],
				m_layer_data.num_elements(),
				MPI_DOUBLE,
				0,
				m_comm);

	return m_layer_data_global;
}