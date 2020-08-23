#include "ConvolutionalLayer.h"

ConvolutionalLayer::ConvolutionalLayer(Layer* parent, MPI_Comm comm) : Layer(parent, comm) {
	m_layer_type = CONVOLUTIONAL;
}

void ConvolutionalLayer::set_data_size(int rows, int cols, int channels) {
	m_rows_data = m_parent->rows();
	m_cols_data = m_parent->cols();
	m_channels_data = channels;

	m_filter_rows = rows;
	m_filter_cols = cols;
	m_filter_channels = m_parent->channels();

	m_layer_data.resize(			boost::extents	[m_batch_size_local]
													[m_channels_data]
													[m_rows_data]
													[m_cols_data]);

	m_cost_gradient.resize(			boost::extents	[m_batch_size_local]
													[m_channels_data]
													[m_rows_data]
													[m_cols_data]);

	m_activation_gradient.resize(	boost::extents	[m_batch_size_local]
													[m_channels_data]
													[m_rows_data]
													[m_cols_data]);

	m_weights.resize(				boost::extents	[m_channels_data]
													[m_filter_channels]
													[m_filter_rows]
													[m_filter_cols]);

	m_weight_gradient.resize(		boost::extents	[m_channels_data]
													[m_filter_channels]
													[m_filter_rows]
													[m_filter_cols]);

	m_weight_grad_global.resize(	boost::extents	[m_channels_data]
													[m_filter_channels]
													[m_filter_rows]
													[m_filter_cols]);

	m_biases.resize(				boost::extents	[1]
													[m_channels_data]
													[m_rows_data]
													[m_cols_data]);

	m_bias_gradient.resize(			boost::extents	[m_batch_size_local]
													[m_channels_data]
													[m_rows_data]
													[m_cols_data]);

	m_bias_grad_global.resize(		boost::extents	[m_batch_size_local]
													[m_channels_data]
													[m_rows_data]
													[m_cols_data]);
}

void ConvolutionalLayer::forward_propagation() {
	const double* parent_data_ptr = m_parent->data_ptr(0,0,0,0);
	const double* cur_weight_ptr = &m_weights[0][0][0][0];

	double*	cur_data_ptr = &m_layer_data[0][0][0][0];

	int w_elems = m_filter_rows * m_filter_cols;
	int d_elems = m_rows_data * m_cols_data;

	int cur_weight_row = 0;
	int cur_weight_col = 0; 

	int data_row;
	int data_col;

	const boost::multi_array<double, 4> parent_data = m_parent->get_layer();

	for(int b = 0; b < m_batch_size_local; b++) {
		for(int d_ch = 0; d_ch < m_channels_data; d_ch++) {
			for(int w_ch = 0; w_ch < m_filter_channels; w_ch++) {
				for(int w_row = 0; w_row < m_filter_rows; w_row++) {
					for(int w_col = 0; w_col < m_filter_cols; w_col++) {
						for(int d_row = 0; d_row < m_rows_data; d_row++) {
							for(int d_col = 0; d_col < m_cols_data; d_col++) {
								data_row = d_row - (m_filter_rows>>1) + w_row;
								data_col = d_col - (m_filter_cols>>1) + w_col;

								if(data_row >= 0 && data_row < m_rows_data && data_col >= 0 && data_col < m_cols_data) {
									m_layer_data[b][d_ch][d_row][d_col] += m_weights[d_ch][w_ch][w_row][w_col] * parent_data[b][w_ch][data_row][data_col];
								}
							}
						}
					}
				}
			}
		}
	}
}