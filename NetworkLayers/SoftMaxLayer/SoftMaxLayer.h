#pragma once
#include "BaseLayer/Layer.h"
#include <Vector.h>
#include <utility>

namespace NN
{
	class SoftMaxLayer : public Layer
	{
	protected:
		VecAl::Vector m_outputs;
		VecAl::Vector m_inputsStable;
	public:
		SoftMaxLayer(Dimensions inputs) : Layer(inputs, inputs), m_outputs(inputs.TotalLength()), m_inputsStable(inputs.TotalLength()) {};

		virtual VecAl::VectorLike& ForwardProp(VecAl::VectorLike& inputs) override
		{
			float max = inputs[0];
			for (size_t i = 1; i < inputs.Size(); i++)
			{
				max = std::max(max, inputs[i]);
			}
			inputs.Sub(max, m_inputsStable);
			m_inputsStable.Exponentiate(2.718f, m_outputs);

			float norm = 0;

			for (size_t i = 0; i < m_outputs.Size(); i++)
			{
				norm += m_outputs[i];
			}
			m_outputs /= norm;

			return m_outputs;
		}

		virtual size_t TrainableParameters() const override { return 0; };

		size_t N() const override { return 0; };
	};
}