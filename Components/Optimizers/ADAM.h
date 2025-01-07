#pragma once
#include "Optimizer.h"
#include <Vector.h>

namespace NN
{
	class ADAM : public Optimizer
	{
		float m_beta1 = 0.9f;
		float m_beta2 = 0.999f;
		float m_beta1t = m_beta1;
		float m_beta2t = m_beta2;
		const float m_eps = 0.001f;

		VecAl::Vector m_m;
		VecAl::Vector m_v;
		VecAl::Vector m_mhat;
		VecAl::Vector m_vhat;
	public:
		ADAM(uint32_t parameters) : Optimizer(parameters),
			m_m(parameters), m_v(parameters), m_mhat(parameters), m_vhat(parameters)
		{
			m_m.SetTo(0);
			m_v.SetTo(0);
			m_mhat.SetTo(0);
			m_vhat.SetTo(0);
		}

		void Update(const VecAl::VectorLike& gradients, VecAl::VectorLike& parameters, float learningRate) override
		{
			m_m *= m_beta1;
			m_v *= m_beta2;

			gradients.Mul(1 - m_beta1, m_mhat).Add(m_m, m_m);
			gradients.Mul(gradients, m_vhat).Mul(1 - m_beta2, m_vhat).Add(m_v, m_v);

			m_m.Div((1 - m_beta1t), m_mhat);
			m_v.Div((1 - m_beta2t), m_vhat);

			m_beta1t *= m_beta1t;
			m_beta2t *= m_beta2t;

			m_vhat += m_eps;
			m_vhat.Sqrt(m_vhat);

			m_mhat *= learningRate;
			m_mhat /= m_vhat;

			parameters *= 0.99999f; // L2 regularization
			parameters -= m_mhat;
		}
	};
}