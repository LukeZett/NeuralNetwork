#pragma once
#include "BaseLayer/Trainable.h"
#include "BaseLayer/Layer.h"
#include "MatrixArray.h"
#include "MatrixArrayView.h"
#include <utility>
#include <cstdint>

namespace NN
{

	class Conv2DLayer : public Layer
	{
	protected:
		using Size2D = std::pair<uint32_t, uint32_t>;

		VecAl::MatrixArray m_kernels;
		VecAl::MatrixArray m_outputVec;

		VecAl::MatrixArrayView inputMatrices = VecAl::MatrixArrayView(0, 0, 0, nullptr); //placeholder

	public:
		Conv2DLayer(const Dimensions& input, const Size2D& kernelSizes, uint32_t kernelCount);

		virtual VecAl::VectorLike& ForwardProp(VecAl::VectorLike& inputs) override;

		virtual size_t TrainableParameters() const override { return m_kernels.Size(); }

		size_t N() const override { return m_kernels.Width() * m_kernels.Height(); };
	};
}
