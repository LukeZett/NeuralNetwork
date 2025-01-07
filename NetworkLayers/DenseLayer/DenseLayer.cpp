#include "DenseLayer.h"
#include <MatrixLike.h>
#include <VectorLike.h>
#include <VectorView.h>

using namespace NN;

VecAl::VectorLike& NN::DenseLayer::ForwardProp(VecAl::VectorLike& inputs)
{
	m_inputs = VecAl::VectorView(inputs);
	m_weights.MatMulTransposed(inputs, m_outputs);
	m_outputs -= m_biases;
	return m_outputs;
}

size_t NN::DenseLayer::TrainableParameters() const
{
	return m_weights.Size() + m_biases.Size();
}
