#include "Conv2DLayerTrain.h"
#include <BaseLayer/Trainable.h>
#include <cstdint>
#include "Conv2DLayer.h"
#include "Optimizers/SGD.h"
#include "Optimizers/ADAM.h"
#include <MatrixView.h>
#include <VectorLike.h>
#include <BaseLayer/Layer.h>

NN::Conv2DLayerTrain::Conv2DLayerTrain(const Dimensions& input, const Size2D& kernelSizes, uint32_t kernelCount, NN::Trainable::OptimizerType optimizer)
	: Conv2DLayer(input, kernelSizes, kernelCount)
{
	switch (optimizer)
	{
	case NN::Trainable::SGD:
		m_optimizer = std::make_unique<NN::SGD>(kernelSizes.first * kernelSizes.second * kernelCount);
		break;
	case NN::Trainable::ADAM:
		m_optimizer = std::make_unique<NN::ADAM>(kernelSizes.first * kernelSizes.second * kernelCount);
		break;
	default:
		break;
	}
}

void NN::Conv2DLayerTrain::BackProp(Trainable* succLayer)
{
	succLayer->GetCostDer(costDer);

	for (size_t j = 0; j < m_input.depth; j++)
	{
		VecAl::MatrixView inputMatrix = inputMatrices[j];

		for (size_t i = 0; i < m_kernels.Depth(); i++)
		{
			VecAl::MatrixView kernelGradients = gradients[i];
			inputMatrix.Convolution(costDer[j * m_kernels.Depth() + i], kernelGradients, false);
			//kernelGradients += temp;
		}
	}
}

void NN::Conv2DLayerTrain::UpdateWeights(float alpha)
{
	m_optimizer->Update(gradients, m_kernels, alpha);
	gradients.SetTo(0);
}

void NN::Conv2DLayerTrain::GetCostDer(VecAl::VectorLike& dest)
{
	VecAl::MatrixArrayView destMatrices = VecAl::MatrixArrayView(m_input.width, m_input.height, m_input.depth, &dest[0]);
	destMatrices.SetTo(0);

	for (size_t j = 0; j < m_input.depth; j++)
	{
		VecAl::MatrixView destMatrix = destMatrices[j];
		for (size_t i = 0; i < m_kernels.Depth(); i++)
		{
			costDer[j * m_kernels.Depth() + i].Correlation(m_kernels[i], destMatrix, true);
			//destMatrix += inputDer;
		}
	}
}

void NN::Conv2DLayerTrain::InitValues(float mean, float variance)
{
	m_kernels.NormalDist(mean, variance);
	gradients.SetTo(0);
}
