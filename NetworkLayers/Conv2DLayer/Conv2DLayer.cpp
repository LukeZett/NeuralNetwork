#include "Conv2DLayer.h"
#include <iostream>

NN::Conv2DLayer::Conv2DLayer(const Dimensions& input, const Size2D& kernelSizes, uint32_t kernelCount)
	: Layer(input, { input.width - kernelSizes.first + 1, input.height - kernelSizes.second + 1, kernelCount * input.depth }),
	m_kernels(kernelSizes.first, kernelSizes.second, kernelCount),
	m_outputVec(m_output.width, m_output.height, m_output.depth)
{
}

VecAl::VectorLike& NN::Conv2DLayer::ForwardProp(VecAl::VectorLike& inputs)
{
	inputMatrices = VecAl::MatrixArrayView(m_input.width, m_input.height, m_input.depth, &inputs[0]);

	m_outputVec.SetTo(0);
	inputMatrices.AllToAllConvolution(m_kernels, m_outputVec, false);

	//for (size_t j = 0; j < m_input.depth; j++)
	//{
	//	VecAl::MatrixView inputMatrix = inputMatrices[j];
	//
	//	for (size_t i = 0; i < m_kernels.Depth(); i++)
	//	{
	//		VecAl::MatrixView dest = m_outputVec[j * m_kernels.Depth() + i];
	//		inputMatrix.Convolution(m_kernels[i], dest, false);
	//	}
	//}
	return m_outputVec;
}
