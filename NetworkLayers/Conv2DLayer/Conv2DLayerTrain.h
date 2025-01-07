#pragma once
#include "Conv2DLayer.h"
#include <BaseLayer/Trainable.h>
#include "Optimizers/Optimizer.h"
#include <Matrix.h>
#include <cstdint>

namespace NN {
	class Conv2DLayerTrain : public Conv2DLayer, public virtual Trainable
	{

	protected:

		VecAl::MatrixArray costDer = VecAl::MatrixArray(m_output.width, m_output.height, m_output.depth); // ?E/?y
		VecAl::MatrixArray gradients = VecAl::MatrixArray(m_kernels.Width(), m_kernels.Height(), m_kernels.Depth());  // ?E/?F or ?E/?w

		VecAl::Matrix inputDer = VecAl::Matrix(m_input.width, m_input.height);
		VecAl::Matrix temp = VecAl::Matrix(m_kernels.Width(), m_kernels.Height());

		std::unique_ptr<Optimizer> m_optimizer;

	public:
		Conv2DLayerTrain(const Dimensions& input, const Size2D& kernelSizes, uint32_t kernelCount, NN::Trainable::OptimizerType optimizer);

		virtual void BackProp(Trainable* succLayer);

		virtual void UpdateWeights(float alpha) override;

		virtual void GetCostDer(VecAl::VectorLike& dest) override;

		virtual void InitValues(float mean, float variance) override;
	};
}
