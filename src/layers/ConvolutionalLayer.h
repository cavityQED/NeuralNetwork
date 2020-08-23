#ifndef CONVOLUTIONALLAYER_H
#define CONVOLUTIONALLAYER_H

#include "Layer.h"

/*	Convolutional Layer Class
*		This layer's row and column size will equal its parent's size (zero padding by default).
*		The number of filters in this layers is determined by its channel size.  
*		One filter for each channel.
*/

class ConvolutionalLayer : public Layer {
public:
	ConvolutionalLayer(Layer* parent = nullptr, MPI_Comm comm = MPI_COMM_WORLD);

	/*	SET_DATA_SIZE
	*		Set the size of the layer data
	*
	*	in 	rows 		-	number of rows in each filter
	*	in 	cols 		- 	number of columns in each filter
	*	in 	channels	-	number of channels in the data (number of filters)
	*/
	virtual void set_data_size(int rows, int cols, int channels) override;

	virtual void forward_propagation() override;
	virtual void backward_propagation() override {}

protected:
	int m_filter_rows;		//Number of rows in each filter
	int m_filter_cols;		//Number of columns in each filter
	int m_filter_channels;	//Number of channels in each filter, equal to the number of parent channels

};

#endif