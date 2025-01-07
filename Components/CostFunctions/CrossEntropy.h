#pragma once
#include "CostFunction.h"
#include "cmath"
#include <VectorLike.h>

namespace NN
{
	class CrossEntropy
	{
	public:
		inline static float Cost(float x, float ex)
		{
			return -(std::logf(x) * ex);
		}

		inline static float CostDer(float x, float ex)
		{
			return x - ex;
		}
	};
}