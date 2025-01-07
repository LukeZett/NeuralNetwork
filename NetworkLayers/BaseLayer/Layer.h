#pragma once
#include <MatrixLike.h>

namespace NN
{
	struct Dimensions
	{
		size_t width = 1, height = 1, depth = 1;

		size_t TotalLength() const { return width * height * depth; }
	};

	/** @brief base class for all layers
    */
	class Layer
	{
	protected:
		Dimensions m_input = {};
		Dimensions m_output = {};

	public:
		Layer() {};

		Layer(const Dimensions& inputDimensions, const Dimensions& outputDimensions) : m_input(inputDimensions), m_output(outputDimensions) {};

		/** @brief computes output of this layer w.r.t. to input
		* @param[in] input - input vector
		* @return output vector reference
		*/
		virtual VecAl::VectorLike& ForwardProp(VecAl::VectorLike& inputs) = 0;

		virtual size_t TrainableParameters() const = 0;

		inline const Dimensions& GetInputSize() const { return m_input; };

		inline const Dimensions& GetOutputSize() const { return m_output; };

		virtual size_t N() const = 0;

		virtual ~Layer() = default;
	};
}