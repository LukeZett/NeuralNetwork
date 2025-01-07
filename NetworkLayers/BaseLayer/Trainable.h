#pragma once
#include "VectorLike.h"

namespace NN
{
	/** @brief interface for trainable layers
	*/
	class Trainable
	{
	public:

		enum OptimizerType
		{
			SGD,
			ADAM,
		};

		/**
		* @brief compute gradient ?E/?w for every weight w in this layer
		* @param[in] succLayer - successor layer
		* @note compute error gradient for last run of train datapoint through this layer by forward pass
		*/
		virtual void BackProp(Trainable* succLayer) = 0;

		/**
		* @brief update weights using the computed gradient
		* @param[in] alpha - learning rate
		* @note undefined until gradients are computed using BackProp
		*/
		virtual void UpdateWeights(float alpha) = 0;

		/**
		* @brief compute vector of ?E/?x for every input x of this layer
		* and put the result in dest object (assuming correct dimension)
		* @note undefined until gradients are computed using BackProp
		*/
		virtual void GetCostDer(VecAl::VectorLike& dest) = 0;  // ?E/?x

		/**
		* @brief Initialize random weights, biases and optimizer objects etc.
		*/
		virtual void InitValues(float mean, float variance) = 0;

		
	};
}