#pragma once
#include "BaseLayer/Layer.h"
#include <Vector.h>
#include <ActivationFunctions/Sigmoid.h>
#include <ActivationFunctions/ReLU.h>
#include <VectorLike.h>

namespace NN
{
	template<typename Activation>
	class ActivationLayer : public Layer
	{
	protected:
		VecAl::Vector m_outputs;
	public:
		ActivationLayer(Dimensions inputs) : Layer(inputs, inputs), m_outputs(inputs.TotalLength()) {};

		VecAl::VectorLike& ForwardProp(VecAl::VectorLike& inputs) override
		{
			Activation::Activation(inputs, m_outputs);
			return m_outputs;
		}

		size_t TrainableParameters() const override { return 0; };

		size_t N() const override { return 0; };
	};
}