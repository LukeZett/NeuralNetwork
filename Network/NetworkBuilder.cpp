#include "NetworkBuilder.h"
#include "DenseLayer/DenseLayerTrain.h"
#include "ActivationLayer/ActivationLayerTrain.h"
#include <ActivationFunctions/Sigmoid.h>
#include <ActivationFunctions/LeakyReLU.h>
#include "CostFunctions/CrossEntropy.h"
#include "SoftMaxLayer/SoftMaxLayerTrain.h"
#include "Maxpool2DLayer/MaxPool2DLayer.h"
#include "Conv2DLayer/Conv2DLayerTrain.h"
#include <stdlib.h>

NN::NetworkBuilder::NetworkBuilder(const Dimensions& networkInputs, size_t networkOutputs, int randomSeed)
	: m_outputs(networkOutputs), m_lastLayerOutputs(networkInputs), m_inputDimensions(networkInputs)
{
	srand(randomSeed);
}

NN::Network NN::NetworkBuilder::Build(NN::Trainable::OptimizerType outputOptimizer, NN::ActivationFunction outputActivation)
{
	m_layerTypes.push_back({ LayerType::DenseLayer, ActivationFunction::relu });
	m_layerTypes.push_back({ LayerType::ActivationLayer, outputActivation });
	Network network;
	uint32_t lastLayerOutput = m_inputDimensions.TotalLength();
	network.m_layers.reserve(m_layers.size() + 2);
	size_t trainableParameters = 0;
	for (auto& layer : m_layers)
	{
		trainableParameters += layer->TrainableParameters();
		network.m_layers.push_back(std::move(layer));
		lastLayerOutput = network.m_layers.back()->GetOutputSize().TotalLength();
	}

	network.m_layers.push_back(std::make_unique<DenseLayerTrain>(lastLayerOutput, m_outputs, outputOptimizer));
	m_lastLayerOutputs = network.m_layers.back()->GetOutputSize();
	trainableParameters += network.m_layers.back()->TrainableParameters();

	switch (outputActivation)
	{
	case NN::relu:
		network.m_layers.push_back(std::make_unique<ActivationLayerTrain<ReLU>>(m_lastLayerOutputs));
		break;
	case NN::sigmoid:
		network.m_layers.push_back(std::make_unique<ActivationLayerTrain<Sigmoid>>(m_lastLayerOutputs));
		break;
	case NN::leakyRelu:
		network.m_layers.push_back(std::make_unique<ActivationLayerTrain<LeakyReLU>>(m_lastLayerOutputs));
		break;
	case NN::softmax:
		network.m_layers.push_back(std::make_unique<SoftMaxLayerTrain>(m_lastLayerOutputs));
		break;
	default:
		break;
	}
	trainableParameters += network.m_layers.back()->TrainableParameters();

	lastLayerOutput = m_outputs;


	for (int i = 0; i < network.m_layers.size(); i++)
	{
		bool initialized = false;
		if (i + 1 < (int)network.m_layers.size() && (m_layerTypes[i].layerType == LayerType::DenseLayer))
		{
			NN::ActivationFunction act = NN::ActivationFunction::relu;

			if (m_layerTypes[i + 1].layerType == ActivationLayer)
				act = m_layerTypes[i + 1].activation;

			initialized = true;

			auto layer = network.m_layers[i].get();
			auto trainLayer = dynamic_cast<Trainable*>(layer);

			switch (act)
			{
			case ActivationFunction::relu:
				trainLayer->InitValues(0, 2.f / (layer->N()));
				break;
			case ActivationFunction::leakyRelu:
				trainLayer->InitValues(0, 2.f / (layer->N()));
				break;
			case ActivationFunction::sigmoid:
				trainLayer->InitValues(0, 2.f / (layer->N() + layer->GetOutputSize().TotalLength()));
				break;
			case ActivationFunction::softmax:
				trainLayer->InitValues(0, 2.f / (layer->N() + layer->GetOutputSize().TotalLength()));
				break;
			default:
				trainLayer->InitValues(0, 2.f / (layer->N()));
				break;
			}
		}
		if (!initialized)
		{
			std::cout << "used" << std::endl;
			auto layer = network.m_layers[i].get();
			auto trainLayer = dynamic_cast<Trainable*>(layer);
			trainLayer->InitValues(0, 1);
		}

		i++;
	}

	std::cout << "________ Built network _______ " << network.m_layers.size() << " layers" << std::endl;
	std::cout << "____ Trainable Parameters ____ " << trainableParameters << std::endl;
	m_layers.clear();
	m_layerTypes.clear();
	m_lastLayerOutputs = m_inputDimensions;
	return network;
}

NN::NetworkBuilder& NN::NetworkBuilder::Dense(size_t neurons, NN::Trainable::OptimizerType optimizer)
{
	m_layerTypes.push_back({ LayerType::DenseLayer, ActivationFunction::relu});
	m_layers.push_back(std::make_unique<DenseLayerTrain>(m_lastLayerOutputs.TotalLength(), neurons, optimizer));
	m_lastLayerOutputs = m_layers.back()->GetOutputSize();
	PrintLayerInfo(m_layers.back().get(), "Dense Layer");
	return *this;
}

NN::NetworkBuilder& NN::NetworkBuilder::Activation(ActivationFunction function)
{
	const char* name = nullptr;
	m_layerTypes.push_back({ LayerType::ActivationLayer, function });
	switch (function)
	{
	case NN::relu:
		m_layers.push_back(std::make_unique<ActivationLayerTrain<ReLU>>(m_lastLayerOutputs));
		name = "Activation Layer - ReLU";
		break;
	case NN::sigmoid:
		m_layers.push_back(std::make_unique<ActivationLayerTrain<Sigmoid>>(m_lastLayerOutputs));
		name = "Activation Layer - Sigmoid";
		break;
	case NN::leakyRelu:
		m_layers.push_back(std::make_unique<ActivationLayerTrain<Sigmoid>>(m_lastLayerOutputs));
		name = "Activation Layer - LeakyReLU";
		break;
	default:
		break;
	}

	PrintLayerInfo(m_layers.back().get(), name);
	return *this;
}

NN::NetworkBuilder& NN::NetworkBuilder::MaxPool2D(uint32_t poolWidth, uint32_t poolHeight)
{
	m_layerTypes.push_back({ LayerType::MaxPool2DLayer, ActivationFunction::relu });

	m_layers.push_back(std::make_unique<NN::MaxPool2DLayer>(m_lastLayerOutputs, std::pair(poolWidth, poolHeight)));
	m_lastLayerOutputs = m_layers.back()->GetOutputSize();
	PrintLayerInfo(m_layers.back().get(), "MaxPool 2D Layer");
	return *this;
}

NN::NetworkBuilder& NN::NetworkBuilder::Conv2D(uint32_t kernels, uint32_t kernelWidth, uint32_t kernelHeight, NN::Trainable::OptimizerType optimizer)
{
	m_layerTypes.push_back({ LayerType::Conv2DLayer, ActivationFunction::relu });
	m_layers.push_back(std::make_unique<NN::Conv2DLayerTrain>(m_lastLayerOutputs, std::pair(kernelWidth, kernelHeight), kernels, optimizer));
	m_lastLayerOutputs = m_layers.back()->GetOutputSize();
	PrintLayerInfo(m_layers.back().get(), "Conv2D Layer");
	return *this;
}

void NN::NetworkBuilder::PrintLayerInfo(const Layer* newLayer, const char* layerName)
{
	std::cout << "____ Added layer: " << layerName << " ____ (Parameters: " << newLayer->TrainableParameters() << ")" << std::endl;
	std::cout << "    Input: " << newLayer->GetInputSize().width << " x " << newLayer->GetInputSize().height << " x " << newLayer->GetInputSize().depth << std::endl;
	std::cout << "    Output: " << newLayer->GetOutputSize().width << " x " << newLayer->GetOutputSize().height << " x " << newLayer->GetOutputSize().depth << std::endl;
}
