#include "MaxPool2DLayer.h"
#include <MatrixArrayView.h>

NN::MaxPool2DLayer::MaxPool2DLayer(const Dimensions& inputDim, const Size2D& maxPoolSize)
	: Layer(inputDim, { inputDim.width / maxPoolSize.first, inputDim.height / maxPoolSize.second, inputDim.depth }),
	m_poolSize(maxPoolSize)
{
}

using namespace VecAl;

VecAl::VectorLike& NN::MaxPool2DLayer::ForwardProp(VecAl::VectorLike& inputs)
{
	m_inputMat = MatrixArrayView(m_input.width, m_input.height, m_input.depth, &inputs[0]);

	for (size_t i = 0; i < m_input.depth; i++)
	{
		MatrixView view = m_inputMat[i];
		MatrixView outView = m_outputMat[i];

		for (size_t y = 0; y < m_output.height; y++) {
			for (size_t x = 0; x < m_output.width; x++)
			{
				float& val = outView[y][x];
				val = std::numeric_limits<float>::min();

				for (size_t wy = 0; wy < m_poolSize.second; wy++) {
					for (size_t wx = 0; wx < m_poolSize.first; wx++)
					{
						size_t yCoord = y * m_poolSize.second + wy;
						size_t xCoord = x * m_poolSize.first + wx;
						if (val < view[yCoord][xCoord])
						{
							val = view[yCoord][xCoord];
						}
					}
				}
			}
		}
	}
	return m_outputMat;
}

size_t NN::MaxPool2DLayer::TrainableParameters() const
{
	return 0;
}

void NN::MaxPool2DLayer::BackProp(Trainable* succLayer)
{
	succLayer->GetCostDer(costDer);
}

void NN::MaxPool2DLayer::UpdateWeights(float alpha)
{
}

void NN::MaxPool2DLayer::GetCostDer(VecAl::VectorLike& dest)
{
	dest.SetTo(0);
	MatrixArrayView destView = MatrixArrayView(m_input.width, m_input.height, m_input.depth, &dest[0]);
	for (size_t i = 0; i < m_input.depth; i++)
	{
		MatrixView destMat = destView[i];
		MatrixView inputMat = m_inputMat[i];
		MatrixView outputMat = m_outputMat[i];
		MatrixView costDerMat = costDer[i];

		for (size_t y = 0; y < m_output.height; y++) {
			for (size_t x = 0; x < m_output.width; x++)
			{
				for (size_t i = 0; i < m_poolSize.second; i++) {
					for (size_t j = 0; j < m_poolSize.first; j++)
					{
						if (inputMat[y * m_poolSize.second + i][x * m_poolSize.first + j] == outputMat[y][x])
						{
							destMat[y * m_poolSize.second + i][x * m_poolSize.first + j] = costDerMat[y][x];
						}
					}
				}
			}
		}
	}
}

void NN::MaxPool2DLayer::InitValues(float mean, float variance)
{
}
