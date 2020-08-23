#include "FullyConnectedLayer.h"

FullyConnectedLayer::FullyConnectedLayer(Layer* parent, MPI_Comm comm) : Layer(parent, comm) {
	m_layer_type = FULLY_CONNECTED;
}

void FullyConnectedLayer::set_data_size(int rows, int cols, int channels) {
	m_channels_data = 1;
	m_rows_data = rows;
	m_cols_data = 1;

	//Resize the data
	m_layer_data.resize(			boost::extents	[m_batch_size_local]
													[m_channels_data]
													[m_rows_data]
													[m_cols_data]);
	//Resize the cost gradient
	m_cost_gradient.resize(			boost::extents	[m_batch_size_local]
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
	//Resize the activation gradient
	m_activation_gradient.resize(	boost::extents	[m_batch_size_local]
													[m_channels_data]
													[m_rows_data]
													[m_cols_data]);
}

void FullyConnectedLayer::set_weights(const double* data, int size) {
	double* layer_weights_ptr = &m_weights[0][0][0][0];
	for(int i = 0; i < size; i++) {
		*layer_weights_ptr = *data;
		data++;
		layer_weights_ptr++;
	}
}

void FullyConnectedLayer::set_biases(const double* data, int size) {
	double* layer_biases_ptr = &m_biases[0][0][0][0];
	for(int i = 0; i < size; i++) {
		*layer_biases_ptr = *data;
		data++;
		layer_biases_ptr++;
	}
}

void FullyConnectedLayer::ReLU(boost::multi_array<double, 4> &in, bool derivative) {
	int in_elements = in.num_elements();
	double* in_ptr = &in[0][0][0][0];

	for(int i = 0; i < in_elements; i++) {
		if(*in_ptr <= 0)
			*in_ptr = 0;
		if(*in_ptr > 0 && derivative)
			*in_ptr = 1.0;
		in_ptr++;
	}
}

void FullyConnectedLayer::forward_propagation() {

	
	const double* parent_ptr;
	const double* weight_ptr;
	const double* biases_ptr;

	double* data_ptr = &m_layer_data[0][0][0][0];
	double*	d_act_ptr = &m_activation_gradient[0][0][0][0];

	//Clear the layer data before computing
	for(int i = 0; i < m_layer_data.num_elements(); i++) {
		*data_ptr = 0;
		*d_act_ptr = 0;
		data_ptr++;
		d_act_ptr++;
	}

	//Reset the pointer
	data_ptr = &m_layer_data[0][0][0][0];
	d_act_ptr = &m_activation_gradient[0][0][0][0];

	int parent_elems = m_parent->elems();

	for(int b = 0; b < m_batch_size_local; b++) {
		biases_ptr = &m_biases[0][0][0][0];
		weight_ptr = &m_weights[0][0][0][0];
		for(int row = 0; row < m_rows_data; row++) {
			parent_ptr = m_parent->data_ptr(0,0,0,0) + b*parent_elems;
			for(int elems = 0; elems < parent_elems; elems++) {
				*data_ptr += (*parent_ptr)*(*weight_ptr);
				parent_ptr++;
				weight_ptr++;
			}
			if(use_biases)
				*data_ptr += *biases_ptr;
			
			*d_act_ptr = *data_ptr;
			data_ptr++;
			d_act_ptr++;
			biases_ptr++;
		}
	}

	activation_function(m_layer_data);
	activation_function(m_activation_gradient, true);

	if(m_child != nullptr)
		m_child->forward_propagation();
}

void FullyConnectedLayer::backward_propagation() {
	switch (m_child->get_type()) {
		case FULLY_CONNECTED:
			fully_connected_back_prop();
		case OUTPUT:
			fully_connected_back_prop();
		default:
			m_parent->backward_propagation();
	}
}

void FullyConnectedLayer::fully_connected_back_prop() {
	/*
	for(int b = 0; b < m_batch_size_local; b++) {
		for(int cur_row = 0; cur_row < m_rows_data; cur_row++) {
			for(int child_row = 0; child_row < m_child->rows(); child_row++) {
				m_cost_gradient[b][0][cur_row][0] += 
					(*m_child->get_weight_ptr(child_row, 0, cur_row, 0)) * (*m_child->get_bias_grad_ptr(0,0,child_row,0));
			}
				m_bias_gradient[0][0][cur_row][0] += (m_activation_gradient[b][0][cur_row][0] * m_cost_gradient[b][0][cur_row][0])/m_batch_size_local;
		}
	}
	*/
	const double* child_bias_grad_ptr	= m_child->get_bias_grad_ptr(0,0,0,0);
	const double* child_weight_ptr 		= m_child->get_weight_ptr(0,0,0,0);
	const double* current_act_grad_ptr	= &m_activation_gradient[0][0][0][0];

	double* current_cost_grad_ptr 		= &m_cost_gradient[0][0][0][0];
	double* current_bias_grad_ptr		= &m_bias_gradient[0][0][0][0];

	for(int b = 0; b < m_batch_size_local; b++) {
	//	current_cost_grad_ptr = &m_cost_gradient[b][0][0][0];
	//	current_bias_grad_ptr = &m_bias_gradient[0][0][0][0];
		for(int cur_row = 0; cur_row < m_rows_data; cur_row++) {
			child_weight_ptr = m_child->get_weight_ptr(0,0,cur_row,0);
			child_bias_grad_ptr = m_child->get_bias_grad_ptr(b,0,0,0);
			for(int child_row = 0; child_row < m_child->rows(); child_row++) {
				*current_cost_grad_ptr += (*child_weight_ptr) * (*child_bias_grad_ptr);
				child_bias_grad_ptr++;
				child_weight_ptr += m_rows_data;
			}
			*current_bias_grad_ptr = (*current_act_grad_ptr)*(*current_cost_grad_ptr);
			current_bias_grad_ptr++;
			current_cost_grad_ptr++;
			current_act_grad_ptr++;
		}
	}

	current_bias_grad_ptr = &m_bias_gradient[0][0][0][0];
	if(m_comm_rank == 0 && test) {
		std::cout << "Layer " << m_layer_id << " Bias Gradients:\n";
		for(int i = 0; i < m_bias_gradient.num_elements(); i++) {
			std::cout << *current_bias_grad_ptr << ", ";
			current_bias_grad_ptr++;
		}
		std::cout << '\n';
	}

	/*
	for(int b = 0; b < m_batch_size_local; b++) {
		for(int cur_row = 0; cur_row < m_rows_data; cur_row++) {
			for(int in_ch = 0; in_ch < m_parent->channels(); in_ch++) {
				for(int in_row = 0; in_row < m_parent->rows(); in_row++) {
					for(int in_col = 0; in_col < m_parent->cols(); in_col++) {
						m_weight_gradient[cur_row][in_ch][in_row][in_col] += 
							(*m_parent->data_ptr(b, in_ch, in_row, in_col))*m_bias_gradient[0][0][cur_row][0]/m_batch_size_local;
					}
				}
			}
		}
	}
	*/
	double* 		current_weight_grad_ptr;
	const double* 	in_ptr;

	current_bias_grad_ptr = &m_bias_gradient[0][0][0][0];

	for(int b = 0; b < m_batch_size_local; b++) {
		current_weight_grad_ptr = &m_weight_gradient[0][0][0][0];
	//		current_bias_grad_ptr = &m_bias_gradient[0][0][0][0];
		for(int cur_row = 0; cur_row < m_rows_data; cur_row++) {
			in_ptr = m_parent->data_ptr(b,0,0,0);
			for(int e = 0; e < m_parent->elems(); e++) {
				*current_weight_grad_ptr += (*in_ptr)*(*current_bias_grad_ptr)/m_batch_size_local;
				in_ptr++;
				current_weight_grad_ptr++;
			}
			current_bias_grad_ptr++;
		}
	}

	if(m_comm_rank == 0 && test) {
		std::cout << "Layer " << m_layer_id << " Weight Gradients:\n";
		const double* d_w = &m_weight_gradient[0][0][0][0];
		for(int i = 0; i < m_weight_gradient.num_elements(); i++) {
			std::cout << *d_w << ", ";
			d_w++;
		}
		std::cout << '\n';
	}

	if(m_parent != nullptr)
		m_parent->backward_propagation();
	
	update();
}