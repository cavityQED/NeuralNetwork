#ifndef OUTPUTLAYER_H
#define OUTPUTLAYER_H

#include "FullyConnectedLayer.h"

/*	Output Layer Class
*		Fully connected layer that is the final layer in the network.
*/

class OutputLayer : public FullyConnectedLayer {
public:
	/*	CONSTRUCTOR
	*	in	parent	-	Parent layer pointer
	*	in	comm	-	MPI commutator
	*/
	OutputLayer(Layer* parent = 0, MPI_Comm comm = MPI_COMM_WORLD);

	/*	BACKWARD_PROPAGATION
	*		Computes the relevant derivatives
	*/
	virtual void backward_propagation() override;

	/*	SET_DATA_SIZE
	*		Override for set_data_size.  Sets the size of the cost gradient and expected output
	*/
	virtual void set_data_size(int rows, int cols, int channels) override;

	/*	SET_OUTPUT
	*		Set the expected output
	*	in 	data	-	Pointer to the expected output data
	*	in	size	-	How many values to read in
	*/
	void set_output_data(const boost::multi_array<double, 4>& data);

	const boost::multi_array<double, 4>& get_output() {return m_expected_output;}

	const boost::multi_array<double, 4>& get_global_output();

	/*	COST
	*		Computes the cost
	*/
	double cost();

	/*	D_COST
	*		Computes the derivative of the cost function wrt the computed output
	*/
	void d_cost();

protected:
	boost::multi_array<double, 4> m_expected_output;	//Expected output. Used to compute the cost
	boost::multi_array<double, 4> m_expected_output_global;	//Global expected output
	boost::multi_array<double, 4> m_layer_data_global;
};

#endif