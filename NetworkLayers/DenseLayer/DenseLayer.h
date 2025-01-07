#pragma once
#include "BaseLayer/Layer.h"
#include <MatrixLike.h>
#include "Matrix.h"
#include "Vector.h"
#include <VectorView.h>

namespace NN
{
	class DenseLayer : public Layer
	{
	protected:
		VecAl::Matrix m_weights; // [output][input]
		VecAl::Vector m_biases; // [output]
		VecAl::Vector m_outputs; // [output]
		VecAl::VectorView m_inputs;

	public:
		DenseLayer(size_t inputs, size_t outputs) : Layer({inputs}, {outputs}), m_biases(outputs),
			m_weights(inputs, outputs), m_outputs(outputs), m_inputs(0, nullptr)
		{}

		inline size_t Inputs() { return GetInputSize().TotalLength(); }

		inline size_t Outputs() { return GetOutputSize().width; }

		VecAl::VectorLike& ForwardProp(VecAl::VectorLike& inputs) override;
		
		size_t TrainableParameters() const override;

		size_t N() const override { return GetInputSize().TotalLength(); };
	};
}