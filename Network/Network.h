#pragma once
#include "BaseLayer/Layer.h"
#include <vector>
#include <memory>
#include <VectorView.h>
#include <VectorLike.h>
#include <iostream>
#include <MatrixLike.h>

namespace NN
{
	class Network
	{
		friend class NetworkBuilder;

		std::vector<std::unique_ptr<Layer>> m_layers;

		Network() = default;

		void UpdateLayers(VecAl::VectorLike& expected);

		void Epoch(VecAl::MatrixLike& data, VecAl::MatrixLike& expected, int batchSize, int maxIndex);

		float m_alpha = 0;
	public:
		VecAl::VectorView CalculateOutputs(VecAl::VectorLike& input);

		void Learn(float alpha, VecAl::MatrixLike& data, VecAl::MatrixLike& expected, int epochs, int batchSize, float validationFraction, std::ostream& file = std::cout);
	
		void TestAccuracy(VecAl::MatrixLike& data, VecAl::MatrixLike& expected, int firstIndex, int lastIndex, int iterations = 100, std::ostream& file = std::cout);
	};



	inline int maxIndex(const VecAl::VectorLike& vec)
	{
		double max = vec[0];
		int maxI = 0;
		for (int i = 1; i < vec.Size(); i++)
		{
			if (vec[i] > max)
			{
				max = vec[i];
				maxI = i;
			}
		}
		return maxI;
	}
}