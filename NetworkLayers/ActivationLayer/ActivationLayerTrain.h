#pragma once
#include "BaseLayer/Layer.h"
#include <Vector.h>
#include <VectorLike.h>
#include "ActivationLayer.h"
#include "BaseLayer/OutputLayer.h"
#include <CostFunctions/MSE.h>

namespace NN
{
	template<typename Activation>
	class ActivationLayerTrain : public ActivationLayer<Activation>, public virtual OutputLayer
	{
		using Cost = MSE;
		using P = ActivationLayer<Activation>;

		VecAl::Vector m_outputDer;
	public:
		ActivationLayerTrain(Dimensions inputs) : ActivationLayer<Activation>(inputs), m_outputDer(inputs.TotalLength()) {};

		void BackProp(const VecAl::VectorLike& expected) override
		{
			for (size_t i = 0; i < m_outputDer.Size(); i++)
			{
				m_outputDer[i] = Cost::CostDer(P::m_outputs[i], expected[i]);
			}
		}

		virtual void BackProp(Trainable* succLayer)override
		{
			succLayer->GetCostDer(m_outputDer);
		}

		virtual void UpdateWeights([[maybe_unused]] float alpha) override {}

		virtual void GetCostDer(VecAl::VectorLike& dest)
		{
			Activation::ActivationDerivative(P::m_outputs, dest);
			dest *= m_outputDer;
		}

		virtual void InitValues(float mean, float variance) override {};

		size_t TrainableParameters() const override { return 0; };
	};
}