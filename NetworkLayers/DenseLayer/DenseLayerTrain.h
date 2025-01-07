#pragma once
#include "DenseLayer.h"
#include "BaseLayer/Trainable.h"
#include "BaseLayer/OutputLayer.h"
#include "Optimizers/SGD.h"
#include <Matrix.h>
#include <Vector.h>
#include "CostFunctions/MSE.h"

namespace NN
{
	class DenseLayerTrain : public DenseLayer, public OutputLayer
	{
		using Cost = MSE;
		using P = DenseLayer;

		VecAl::Matrix m_gradients;
		VecAl::Vector m_gradientsBiases;
		VecAl::Vector m_outputDer;

		std::unique_ptr<Optimizer> m_optimizerWeights = nullptr;
		std::unique_ptr<Optimizer> m_optimizerBiases = nullptr;

	public:
		DenseLayerTrain(size_t inputs, size_t outputs, OptimizerType optimizer);

		void BackProp(Trainable* succLayer) override;

		void UpdateWeights(float alpha) override;

		void GetCostDer(VecAl::VectorLike& dest) override;

		void InitValues(float mean, float variance) override;

		void BackProp(const VecAl::VectorLike& expected) override;

	private:
		void UpdateGradients();
	};
}