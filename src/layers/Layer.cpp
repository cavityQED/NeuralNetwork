#include "Layer.h"

int Layer::__id = 0;

Utilities Layer::utils = Utilities();

Layer::Layer(Layer* parent, MPI_Comm comm) {
	m_layer_id = __id++;

	m_parent = parent;
	m_comm = comm;

	//Get the size and rank of the commutator
	MPI_Comm_size(m_comm, &m_comm_size);
	MPI_Comm_rank(m_comm, &m_comm_rank);

	if(m_parent != nullptr)
		m_parent->set_child(this);
}

void Layer::set_batch_size(int batch_size) {
	assert(batch_size % m_comm_size == 0 && "batch size must be a multiple of the comm size");
	
	//Set the global and local batch size
	m_batch_size_global = batch_size;
	m_batch_size_local = m_batch_size_global/m_comm_size;

	//If the layer has a child, set its batch sizes
	if(m_child != nullptr)
		m_child->set_batch_size(m_batch_size_global);

}

void Layer::set_data_size(int rows, int cols, int channels) {
	m_rows_data = rows;
	m_cols_data = cols;
	m_channels_data = channels;

	m_layer_data.resize(boost::extents	[m_batch_size_local]
										[m_channels_data]
										[m_rows_data]
										[m_cols_data]);	
	//Resize the activation gradient
	m_activation_gradient.resize(	boost::extents	[m_batch_size_local]
													[m_channels_data]
													[m_rows_data]
													[m_cols_data]);
}

void Layer::activation_function(boost::multi_array<double, 4> &in, bool derivative) {
	double* in_ptr = &in[0][0][0][0];
	int elems = in.num_elements();

	/*	ReLU activation function.
	*	Equal to zero for negative inputs.
	*	Equal to the input for non-negative inputs.
	*	Derivative is zero for negative inputs.
	*	Derivative is one for non-negative inputs.
	*/	
	for(int i = 0; i < elems; i++) {
		if(*in_ptr < 0)
			*in_ptr = 0;
		if(*in_ptr > 0 && derivative)
			*in_ptr = 1;
		in_ptr++;
	}
	/*
	double x;

	for(int i = 0; i < elems; i++) {
		x = 1/(1 + exp(-(*in_ptr)));
		if(x < 0)
			x = 0;

		if(!derivative)
			*in_ptr = x; 
		else
			*in_ptr = x*(1-x);

		in_ptr++;
	}
	*/
}

void Layer::update() {

	MPI_Allreduce(	&m_weight_gradient[0][0][0][0],
					&m_weight_grad_global[0][0][0][0],
					m_weight_gradient.num_elements(),
					MPI_DOUBLE,
					MPI_SUM,
					m_comm);
	
	MPI_Allreduce(	&m_bias_gradient[0][0][0][0],
					&m_bias_grad_global[0][0][0][0],
					m_bias_gradient.num_elements(),
					MPI_DOUBLE,
					MPI_SUM,
					m_comm);

	double* weight_grad_ptr 	= &m_weight_grad_global[0][0][0][0];
	double* bias_grad_ptr		= &m_bias_grad_global[0][0][0][0];

	double* d_w_local			= &m_weight_gradient[0][0][0][0];
	double* d_b_local			= &m_bias_gradient[0][0][0][0];

	double* weight_ptr 			= &m_weights[0][0][0][0];
	double* bias_ptr 			= &m_biases[0][0][0][0];

	double* d_cost				= &m_cost_gradient[0][0][0][0];

	for(int w = 0; w < m_weights.num_elements(); w++) {
		*weight_ptr -= __alpha*(*weight_grad_ptr)/m_comm_size;
		*d_w_local = 0;
		d_w_local++;
		weight_ptr++;
		weight_grad_ptr++;
	}

	for(int b = 0; b < m_batch_size_local; b++) {
		bias_ptr = &m_biases[0][0][0][0];
		for(int e = 0; e < elems(); e++) {
			if(use_biases)
				*bias_ptr -= __alpha*(*bias_grad_ptr)/m_comm_size/m_batch_size_local;
	
			*d_b_local = 0;
			*d_cost = 0;

			d_cost++;
			d_b_local++;
			bias_ptr++;
			bias_grad_ptr++;
		}
	}

	if(alpha_update) {
		__alpha -= __alpha*.0001;
	}
}

void Layer::init(int minW, int maxW, int minB, int maxB) {
	if(m_comm_rank == 0) {
		double* w_ptr = &m_weights[0][0][0][0];
		double* b_ptr = &m_biases[0][0][0][0];

		for(int w = 0; w < m_weights.num_elements(); w++) {
			*w_ptr = utils.RNG<double>(minW, maxW);
			w_ptr++;
		}

		for(int b = 0; b < m_biases.num_elements(); b++) {
			*b_ptr = utils.RNG<double>(minB, maxB);
			b_ptr++;
		}
	}


	MPI_Bcast(	&m_weights[0][0][0][0],
				m_weights.num_elements(),
				MPI_DOUBLE,
				0,
				m_comm);

	MPI_Bcast(	&m_biases[0][0][0][0],
				m_biases.num_elements(),
				MPI_DOUBLE,
				0,
				m_comm);

	//Initialize all gradient data to zero
	double* layer_ptr	= &m_layer_data[0][0][0][0];
	double* d_c_ptr		= &m_cost_gradient[0][0][0][0];
	double* d_a_ptr		= &m_activation_gradient[0][0][0][0];
	
	double* d_w_ptr		= &m_weight_gradient[0][0][0][0];
	double* d_w_g_ptr	= &m_weight_grad_global[0][0][0][0];

	double* d_b_ptr		= &m_bias_gradient[0][0][0][0];
	double* d_b_g_ptr	= &m_bias_grad_global[0][0][0][0];
	
	for(int i = 0; i < m_layer_data.num_elements(); i++) {
		*d_c_ptr	= 0;
		*d_a_ptr	= 0;
		*layer_ptr	= 0;

		layer_ptr++;
		d_c_ptr++;
		d_a_ptr++;
	}

	for(int i = 0; i < m_weights.num_elements(); i++) {
		*d_w_ptr	= 0;
		*d_w_g_ptr	= 0;

		d_w_ptr++;
		d_w_g_ptr++;
	}

	for(int i = 0; i < m_biases.num_elements(); i++) {
		*d_b_ptr	= 0;
		*d_b_g_ptr	= 0;

		d_b_ptr++;
		d_b_g_ptr++;
	}

	if(m_child != nullptr)
		m_child->init(minW, maxW, minW, maxW);
}

void Layer::init(int val) {
	double* w_ptr = &m_weights[0][0][0][0];
	double* b_ptr = &m_biases[0][0][0][0];

	for(int w = 0; w < m_weights.num_elements(); w++) {
		*w_ptr = val;
		w_ptr++;
	}

	for(int b = 0; b < m_biases.num_elements(); b++) {
		*b_ptr = val;
		b_ptr++;
	}

	if(m_child != nullptr)
		m_child->init(val);
}