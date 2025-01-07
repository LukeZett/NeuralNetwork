#include "DenseLayerTrain.h"
#include "DenseLayer.h"
#include <BaseLayer/Trainable.h>
#include <Optimizers/ADAM.h>
#include <memory>

using namespace NN;

NN::DenseLayerTrain::DenseLayerTrain(size_t inputs, size_t outputs, OptimizerType optimizer)
	: DenseLayer(inputs, outputs), m_gradients(inputs, outputs),
	  m_gradientsBiases(outputs), m_outputDer(outputs)
{
	switch (optimizer)
	{
	case NN::Trainable::SGD:
		m_optimizerWeights = std::make_unique<NN::SGD>(inputs * outputs);
		m_optimizerBiases = std::make_unique<NN::SGD>(outputs);
		break;
	case NN::Trainable::ADAM:
		m_optimizerWeights = std::make_unique<NN::ADAM>(inputs * outputs);
		m_optimizerBiases = std::make_unique<NN::ADAM>(outputs);
		break;
	default:
		break;
	}
}

void NN::DenseLayerTrain::BackProp(Trainable* succLayer)
{
	succLayer->GetCostDer(m_outputDer);
	UpdateGradients();
}

void NN::DenseLayerTrain::UpdateWeights(float alpha)
{
	m_optimizerWeights->Update(m_gradients, m_weights, alpha);
	m_optimizerBiases->Update(m_gradientsBiases, m_biases, -alpha);
	m_gradients.SetTo(0);
	m_gradientsBiases.SetTo(0);
}

void NN::DenseLayerTrain::GetCostDer(VecAl::VectorLike& dest)
{
	m_outputDer.MatMul(m_weights, dest);
}

void NN::DenseLayerTrain::InitValues(float mean, float variance)
{
	m_biases.NormalDist(mean, variance);
	m_weights.NormalDist(mean, variance);
	m_gradients.SetTo(0);
	m_gradientsBiases.SetTo(0);
}

void NN::DenseLayerTrain::BackProp(const VecAl::VectorLike& expected)
{
	for (uint32_t nodeID = 0; nodeID < Outputs(); nodeID++) {
		for (uint32_t weightID = 0; weightID < Inputs(); weightID++)
		{
			m_outputDer[nodeID] = Cost::CostDer(m_outputs[nodeID], expected[nodeID]);
			m_gradients[nodeID][weightID] += m_outputDer[nodeID] * m_inputs[weightID];
		}
	}
	m_gradientsBiases += m_outputDer;
}


void NN::DenseLayerTrain::UpdateGradients()
{
	m_outputDer.MatMul(m_inputs, m_gradients, true);
	m_gradientsBiases += m_outputDer;
}
