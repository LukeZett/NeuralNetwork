#pragma once
#include <VectorLike.h>

namespace NN
{
	class LeakyReLU
	{
		using Vec = VecAl::VectorLike;
	public:
		static inline void Activation(Vec& source, Vec& dest)
		{
			for (size_t i = 0; i < source.Size(); i++)
			{

				source.Mul(0.1f, dest);
				dest[i] = source[i] > 0.f ? source[i] : dest[i];
			}
		}

		static inline void ActivationDerivative(Vec& source, Vec& dest)
		{
			
			for (size_t i = 0; i < source.Size(); i++)
			{
				dest[i] = source[i] > 0.f ? 1.f : 0.1f;
			}
		}
	};
}