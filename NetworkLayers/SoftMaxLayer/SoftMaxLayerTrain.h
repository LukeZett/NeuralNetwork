#pragma once
#include "BaseLayer/OutputLayer.h"
#include "SoftMaxLayer.h"


namespace NN
{
	class SoftMaxLayerTrain : public SoftMaxLayer, public virtual OutputLayer
	{
		VecAl::Vector m_inputDer;
	public:
		SoftMaxLayerTrain(Dimensions inputs) : SoftMaxLayer(inputs), m_inputDer(inputs.TotalLength()) {};

		virtual void BackProp(Trainable* succLayer) override {}; // not implemented for softmax;

		virtual void BackProp(const VecAl::VectorLike& expected) override
		{
			m_outputs.Sub(expected, m_inputDer);
		}

		virtual void UpdateWeights(float alpha) override {};

		virtual void GetCostDer(VecAl::VectorLike& dest) override
		{
			m_inputDer.Add(0, dest);
		};

		virtual void InitValues(float mean, float variance) override {};
	};
}