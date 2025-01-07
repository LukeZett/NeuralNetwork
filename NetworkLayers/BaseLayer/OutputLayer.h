#pragma once
#include "Trainable.h"

namespace NN
{
	/**
	* @brief Abstract class for trainable output layer,
	* provides interface for backpropagation
	*/
	class OutputLayer : public Trainable
	{
	public:
		virtual void BackProp(const VecAl::VectorLike& expected) = 0;

		virtual ~OutputLayer() = default;
	};
}