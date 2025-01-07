#pragma once
#include "VectorLike.h"

namespace NN
{
	class Sigmoid
	{
		using Vec = VecAl::VectorLike;
	public:
		static inline void Activation(Vec& source, Vec& dest)
		{
			source.SubFrom(0, dest).Exponentiate(2.71f, dest).Add(1, dest).ReciprMul(1, dest);
		}

		static inline void ActivationDerivative(Vec& source, Vec& dest)
		{
			Activation(source, dest);
			for (size_t i = 0; i < source.Size(); i++)
			{
				dest[i] = dest[i] * (1.f - dest[i]);
			}
		}
	};
}