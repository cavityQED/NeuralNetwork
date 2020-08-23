#ifndef INPUTLAYER_H
#define INPUTLAYER_H

#include "Layer.h"
	
/*	Input Layer Class
*		This layer is responsible for gathering and distributing the batch data.
*/
class InputLayer : public Layer {
public:
	/*	CONSTRUCTOR
	*		parent	-	Parent layer (will always be nullptr for input layer)
	*		comm	-	MPI commutator
	*/
	InputLayer(Layer* parent = nullptr, MPI_Comm comm = MPI_COMM_WORLD);

	/*	FORWARD_PROPAGATION	
	*		Since the input layer gets its data from outside the network,
	*		this function does nothing but call its child's function
	*/
	virtual void forward_propagation() override;

	/*	BACKWARD_PROPAGATION
	*		Does nothing. End of the propagation
	*/
	virtual void backward_propagation() override;

	/*	SET_DATA_SIZE
	*		Override to set the global data size and local size
	*/
	virtual void set_data_size(int rows, int cols, int channels) override;

	/*	GET_GLOBAL_LAYER
	*		Returns a constant reference to the global layer data.
	*/
	const boost::multi_array<double, 4>& get_global_layer()
		{return m_layer_data_global;}

	/*	SET_INPUT_DATA
	*		Sets the global data of the input layer. 
	*	in 	data 	-	The data to be copied in
	*/
	void set_input_data(const boost::multi_array<double, 4>& data);

	/*	INIT
	*		Override for init to do nothing since input layer has no weights or biases
	*/
	virtual void init(int minW, int maxW, int minB, int maxB) override {m_child->init(minW, maxW, minB, maxB);}

	void set_test_data(const double* data, int size);

	virtual void predict() override;

protected:
	boost::multi_array<double, 4> m_layer_data_global;	//Global layer data
	boost::multi_array<double, 4> m_test_data_global;	//Global test data
};

#endif