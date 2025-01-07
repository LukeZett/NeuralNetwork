#pragma once
#include "BaseLayer/Layer.h"
#include "BaseLayer/Trainable.h"
#include <MatrixArray.h>
#include <Matrix.h>
#include <MatrixArrayView.h>

namespace NN
{
	class MaxPool2DLayer : public Layer, public virtual Trainable
	{
	protected:
		using Size2D = std::pair<uint32_t, uint32_t>;
		using Matrices = VecAl::MatrixArray;
		using Matrix = VecAl::Matrix;

		Size2D m_poolSize;
		VecAl::MatrixArrayView m_inputMat = VecAl::MatrixArrayView(0, 0, 0, nullptr);
		Matrices m_outputMat = Matrices(m_output.width, m_output.height, m_output.depth);
		Matrices costDer = Matrices(m_output.width, m_output.height, m_output.depth);

	public:
		MaxPool2DLayer(const Dimensions& inputDim, const Size2D& maxPoolSize);

		// Inherited via Layer
		VecAl::VectorLike& ForwardProp(VecAl::VectorLike& inputs) override;
		size_t TrainableParameters() const override;

		// Inherited via Trainable
		void BackProp(Trainable* succLayer) override;
		void UpdateWeights(float alpha) override;
		void GetCostDer(VecAl::VectorLike& dest) override;
		void InitValues(float mean, float variance) override;
		size_t N() const override { return 0; };
	};
}