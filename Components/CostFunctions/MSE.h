#pragma once
#include "CostFunction.h"
#include "cmath"

namespace NN
{
	class MSE : public CostFunction
	{
	public:
		inline static float Cost(float x, float ex)
		{
			return std::pow(x - ex, 2.0f);
		}

		inline static float CostDer(float x, float ex)
		{
			return 2.f * (x - ex);
		}
	};
}