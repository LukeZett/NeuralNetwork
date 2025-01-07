#pragma once
#include "Optimizer.h"

namespace NN
{
	class SGD : public Optimizer
	{
		VecAl::Vector m_multipliedGradients;

	public:
		SGD(uint32_t parameters) : Optimizer(parameters), m_multipliedGradients(parameters)
		{
		}

		void Update(const VecAl::VectorLike& gradients, VecAl::VectorLike& parameters, float learningRate) override
		{
			gradients.Mul(learningRate, m_multipliedGradients);
			parameters -= m_multipliedGradients;
		}
	};
};