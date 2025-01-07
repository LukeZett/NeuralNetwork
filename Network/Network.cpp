#include "Network.h"
#include <utility>
#include <Matrix.h>
#include "Components/CostFunctions/MSE.h"
#include "BaseLayer/OutputLayer.h"
#include "CostFunctions/CrossEntropy.h"

void NN::Network::UpdateLayers(VecAl::VectorLike& expected)
{
	auto lastLayer = dynamic_cast<OutputLayer*>(m_layers.back().get());
	lastLayer->BackProp(expected);

	for (int layer = m_layers.size() - 2; layer >= 0; --layer)
	{
		auto trainableLayer = dynamic_cast<Trainable*>(m_layers[layer].get());
		trainableLayer->BackProp(dynamic_cast<Trainable*>(m_layers[layer + 1].get()));
	}
}

void NN::Network::Epoch(VecAl::MatrixLike& data, VecAl::MatrixLike& expected, int batchSize, int maxIndex)
{
	for (size_t i = 0; i < maxIndex / batchSize; i++)
	{
		for (size_t k = 0; k < batchSize; k++)
		{
			int index = i * batchSize + k;
			VecAl::VectorView input = VecAl::VectorView(data.Width(), data[index]);
			VecAl::VectorView output = VecAl::VectorView(expected.Width(), expected[index]);
			CalculateOutputs(input);
			UpdateLayers(output);
		}

		for (int layer = m_layers.size() - 1; layer >= 0; --layer)
		{
			auto trainableLayer = dynamic_cast<Trainable*>(m_layers[layer].get());
			trainableLayer->UpdateWeights(m_alpha / batchSize);
		}
	}
}

VecAl::VectorView NN::Network::CalculateOutputs(VecAl::VectorLike& input)
{
	VecAl::VectorLike* temp = &input;
	for (size_t i = 0; i < m_layers.size(); i++)
	{
		temp = &m_layers[i]->ForwardProp(*temp);
	}

	return VecAl::VectorView(*temp);
}

void NN::Network::Learn(float alpha, VecAl::MatrixLike& data, VecAl::MatrixLike& expected, int epochs, int batchSize, float validationFraction, std::ostream& file)
{
	m_alpha = alpha;

	for (size_t i = 0; i < epochs; i++)
	{
		Epoch(data, expected, batchSize, data.Height() * (1 - validationFraction));
		file << i + 1 << ";";
		TestAccuracy(data, expected, 0, data.Height() * (1 - validationFraction), 1000, file);
		TestAccuracy(data, expected, data.Height() * (1 - validationFraction), data.Height(), 1000, file);
		file << std::endl;
	}
}

void NN::Network::TestAccuracy(VecAl::MatrixLike& data, VecAl::MatrixLike& expected, int firstIndex, int lastIndex, int iterations, std::ostream& file)
{
	int correct = 0;
	float loss = 0;
	int setSize = lastIndex - firstIndex;

	int iters = std::min(iterations, setSize);

	for (size_t i = 0; i < iters; i++)
	{
		int index = (rand() % setSize) + firstIndex;
		VecAl::VectorView input = VecAl::VectorView(data.Width(), data[index]);
		VecAl::VectorView expectedOutput = VecAl::VectorView(expected.Width(), expected[index]);

		auto out = CalculateOutputs(input);
		if (maxIndex(out) == maxIndex(expectedOutput))
		{
			correct += 1;
		}
		for (size_t j = 0; j < out.Size(); j++)
		{
			loss += CrossEntropy::Cost(out[j], expectedOutput[j]);
		}
	}
	file << correct / (float)iters << ";";
	file << loss / iters << ";";
}
