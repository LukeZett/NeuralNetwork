#pragma once
#include <cstdint>
#include "Vector.h"

namespace NN
{
	class Optimizer
	{
	public:
		Optimizer(uint32_t parameters)
		{}
		
		virtual void Update(const VecAl::VectorLike& gradients, VecAl::VectorLike& parameters, float learningRate) = 0;
	};
}