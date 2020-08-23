#ifndef FULLYCONNECTEDLAYER_H
#define FULLYCONNECTEDLAYER_H

#include "Layer.h"

/*	Fully Connected Layer Class
*		Every element in this layer's data will depend on every element in the parent layer.
*/
class FullyConnectedLayer : public Layer {
public:
	/*	CONSTRUCTOR
	*	in	parent	-	Parent layer
	*	in	comm	-	MPI commutator
	*/
	FullyConnectedLayer(Layer* parent = nullptr, MPI_Comm comm = MPI_COMM_WORLD);

	/*	SET_DATA_SIZE
	*		Overried of the base class set_data_size.  Since this is a fully connected layer,
	*		all sizes are determined soley by the size of the layer and its parent.
	*		This function will set all those sizes.
	*
	*	in rows		-	Number of rows
	*	in cols		-	Number of cols (this will be one no matter what)
	*	in channels	-	Number of channels (this will be set to one no matter what)
	*/
	virtual void set_data_size(int rows, int cols = 1, int channels = 1) override;

	/*	SET_WEIGHTS
	*		Sets the weights.
	*	in	data	-	Pointer to the data to set as the weights
	*	in 	size	-	How many values to read in
	*/
	void set_weights(const double* data, int size);

	/* 	SET_BIASES
	*		Sets the biases.
	*	in 	data	-	Pointer to the data to set as the biases
	*	in 	size	-	How many values to read in
	*/
	void set_biases(const double* data, int size);

	virtual void forward_propagation() override;
	virtual void backward_propagation() override;

	/*	FULLY_CONNECTED_BACK_PROP
	*		Computes derivatives when the child layer is fully connected
	*/
	void fully_connected_back_prop();

	void ReLU(boost::multi_array<double, 4> &in, bool derivative = false);

protected:

};

#endif