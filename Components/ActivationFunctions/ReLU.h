#pragma once
#include <VectorLike.h>
#include <exception>

namespace NN
{
	class ReLU
	{
		using Vec = VecAl::VectorLike;
	public:
		static inline void Activation(Vec& source, Vec& dest)
		{
			source.Max(0, dest);
		}

		static inline void ActivationDerivative(Vec& source, Vec& dest)
		{
			source.ArgMax(0, dest);
			//for (size_t i = 0; i < source.Size(); i++)
			//{
			//	dest[i] = source[i] > 0.f ? 1.f : 0.f;
			//}
		}
	};
}