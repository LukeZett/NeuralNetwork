#pragma once
#include "Network.h"
#include <BaseLayer/Layer.h>
#include <memory>
#include <vector>
#include <BaseLayer/Trainable.h>

namespace NN
{
	enum ActivationFunction
	{
		relu,
		sigmoid,
		leakyRelu,
		softmax
	};



	class NetworkBuilder
	{
		enum LayerType
		{
			MaxPool2DLayer,
			Conv2DLayer,
			DenseLayer,
			ActivationLayer
		};

		struct LayerInfo
		{
			LayerType layerType;
			ActivationFunction activation;
		};

		std::vector<std::unique_ptr<Layer>> m_layers;
		std::vector<LayerInfo> m_layerTypes;
		size_t m_outputs;
		Dimensions m_lastLayerOutputs;
		Dimensions m_inputDimensions;

	public:
		NetworkBuilder(const Dimensions& networkInputs, size_t networkOutputs, int randomSeed);

		Network Build(NN::Trainable::OptimizerType outputOptimizer = NN::Trainable::SGD, NN::ActivationFunction outputActivation = ActivationFunction::sigmoid);

		NetworkBuilder& Dense(size_t neurons, NN::Trainable::OptimizerType optimizer = NN::Trainable::SGD);

		NetworkBuilder& Activation(ActivationFunction function);

		NetworkBuilder& MaxPool2D(uint32_t poolWidth, uint32_t poolHeight);

		NetworkBuilder& Conv2D(uint32_t kernels, uint32_t kernelWidth, uint32_t kernelHeight, NN::Trainable::OptimizerType optimizer = NN::Trainable::SGD);

		void PrintLayerInfo(const Layer* newLayer, const char* layerName);
	};
};