#ifndef LAYER_H
#define LAYER_H

#include "boost/multi_array.hpp"

#include "mpi.h"

#include "Utilities.h"

#include <cmath>
#include <assert.h>

/*	LAYERTYPE
*		Enum for the type of layer. 
*		Used so layers can know the type of their parent and child.
*/
enum LayerType {
	INPUT,
	FULLY_CONNECTED,
	CONVOLUTIONAL,
	POOLING,
	OUTPUT,
};

/*	Layer Class - An abstract base class for the individual layers in a neural network
*		Derived classes will be individual layers of a specefic type such as:
*			Input, Fully Connected, Convolutional, Pooling, Output, etc
*
*		Each layer will have a parent and child except for input and output layers.
*		Each layer's forward propagation function will call its child's forward propagation function.
*		Each layer's backward propagation function will call its parent's backward propagation function.
*/
class Layer {
public:
	/*	CONSTRUCTOR
	*	in	parent	-	Pointer to the previous layer in the network
	*	in	comm 	-	MPI commutator
	*/
	Layer(Layer* parent = nullptr, MPI_Comm comm = MPI_COMM_WORLD);

	/*	DESTRUCTOR
	*		virtual destructor
	*/
	virtual ~Layer() {}

	/*	SET_DATA_SIZE
	*		Set the size of the layer data
	*
	*	in 	rows 		-	number of rows in the data
	*	in 	cols 		- 	number of columns in the data
	*	in 	channels	-	number of channels in the data
	*/
	virtual void set_data_size(int rows, int cols, int channels) = 0;

	/*	FORWARD_PROPAGATION
	*		Pure virtual function that must be defined in the derived class layer.
	*		This function will use the layer's weights and biases to compute its data from its parent's data.
	*/
	virtual void forward_propagation() = 0;

	/*	BACKWARD_PROPAGATION
	*		Pure virtual function that must be defined in the derived class layer.
	*		This function will use the child layer's derivatives to compute its own derivatives.
	*/
	virtual void backward_propagation() = 0;

	/*	ACTIVATION_FUNCTION
	*		Default activation function for layer data.
	*/
	virtual void activation_function(boost::multi_array<double, 4> &in, bool derivative = false);

	/*	SET_CHILD
	*		Used in the constructor so the layer can set itself as the child of its parent layer.
	*	
	*	in	child 	-	The layer to set as the child, will be 'this' as the child layer will call this function on its parent layer
	*/
	void set_child(Layer* child) {m_child = child;}

	/*	SET_BATCH_SIZE
	*		Sets the global and local batch size of the network.
	*		If the layer's child exists, the layer will call set_batch_size on its child.
	*		Consequently, this function should only be called on the input layer
	*	
	*	in	batch_size 	-	Global batch size
	*/
	void set_batch_size(int batch_size);

	/*	GET_TYPE
	*		Returns the type of the layer
	*/
	LayerType get_type() const {return m_layer_type;}

	//Getters for the data size
	int rows()		const {return m_rows_data;}
	int cols()		const {return m_cols_data;}
	int channels() 	const {return m_channels_data;}
	int elems()		const {return m_rows_data*m_cols_data*m_channels_data;}

	/*	DATA_PTR
	*		Returns a pointer to the data specified by the given indices
	*	
	*	in 	batch_id	-	Batch index
	*	in 	channel 	-	Channel index
	*	in 	row 		-	Row index
	*	in 	col 		- 	Column index
	*/
	virtual const double* data_ptr(int batch_id, int channel, int row, int col) const
		{return &m_layer_data[batch_id][channel][row][col];}

	/*	GET_LAYER
	*		Returns a constant reference to the layer data
	*/
	const boost::multi_array<double, 4>& get_layer() const {return m_layer_data;}

	const boost::multi_array<double, 4>& get_weights() const {return m_weights;}

	const boost::multi_array<double, 4>& get_weight_grads() const {return m_weight_grad_global;}

	const boost::multi_array<double, 4>& get_bias_grads() const {return m_bias_grad_global;}

	/*	GET_BIAS_GRAD_PTR
	*		Returns a pointer to the bias gradients at the specified index
	*
	*	in 	i	-	First index
	*	in 	j	-	Second index
	*	in	k	-	Third index
	*	in	l	-	Fourth index
	*/
	const double* get_bias_grad_ptr(int i, int j, int k, int l) const
		{return &m_bias_gradient[i][j][k][l];}

	/*	GET_WEIGHT_PTR
	*		Returns a pointer to the weights at the specified index
	*/
	const double* get_weight_ptr(int i, int j, int k, int l) const
		{return &m_weights[i][j][k][l];}

	/*	UPDATE
	*		Updates the weights and biases according to the learning
	*		parameter and the respective gradients
	*/
	virtual void update();

	/*	INIT
	*		Initializes the weights and biases to random numbers
	*	
	*	in 	minW	-	Minimum value of the weights
	*	in 	maxW	-	Maximum value of the weights
	*	in 	minB	-	Minimum value of the biases
	*	in 	maxB	-	Maximum value of the biases	
	*/
	virtual void init(int minW, int maxW, int minB, int maxB);

	/*	INIT
	*		Initializes the weights and biases to a specified number
	*
	*	in	val		-	Value to set the weights and biases
	*/
	virtual void init(int val);


	void set_alpha(double val) {__alpha = val; if(m_child != nullptr){m_child->set_alpha(val);}}

	void set_test(bool t) {test = t;}
	void set_use_biases(bool b) {use_biases = b;}

	void set_alpha_update(bool up) {alpha_update = up; if(m_child != nullptr){m_child->set_alpha_update(up);}}

	virtual void predict() {}

	double get_alpha() {return __alpha;}

protected:
	static int __id;
	
	double __alpha;

	bool test = false;
	bool use_biases = true;
	bool alpha_update = false;

	MPI_Comm 	m_comm;					//MPI Commuatator
	int			m_comm_size;			//Size of the commutator
	int			m_comm_rank;			//Rank within the commutator
	int			m_test_batch_size;		//How many test samples to process during predict
	int 		m_batch_size_global;	//Global batch size
	int 		m_batch_size_local;		//Local batch size
	int 		m_rows_data = 0;		//Number of rows in the data
	int 		m_cols_data = 0;		//Number of columns in the data
	int 		m_channels_data = 0;	//Number of channels in the data
	int 		m_layer_id;				//Unique ID for the layer among other layers
	Layer* 		m_parent;				//Parent layer
	Layer* 		m_child = nullptr;		//Child layer
	LayerType 	m_layer_type;			//Type of the layer
	
	static Utilities	utils;					//Utilities object for RNG

	boost::multi_array<double, 4> 	m_layer_data;			//Container for the layer's data after the forward propagation is complete
	boost::multi_array<double, 4> 	m_cost_gradient;		//Derivatives of the cost function wrt to the layer data
	boost::multi_array<double, 4>	m_activation_gradient;	//Derivatives of the activation function wrt its argument

	boost::multi_array<double, 4>	m_weights;				//Weights FROM the parent TO this layer
	boost::multi_array<double, 4>	m_weight_gradient;		//Derivatives of the cost wrt the weights
	boost::multi_array<double, 4> 	m_weight_grad_global;	//Global sum of weight derivatives

	boost::multi_array<double, 4>	m_biases;				//Biases FROM the parent TO this layer
	boost::multi_array<double, 4>	m_bias_gradient;		//Derivatives of the cost wrt the biases
	boost::multi_array<double, 4>	m_bias_grad_global;		//Global sum of bias derivatives
};

#endif