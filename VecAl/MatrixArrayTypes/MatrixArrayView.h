#pragma once
#include "MatrixArrayLike.h"

#include <VectorLike.h>

namespace VecAl
{
	class MatrixArrayView : public MatrixArrayLike
	{
	public:
		MatrixArrayView(const MatrixArrayLike& other) : MatrixArrayLike(other) {}

		MatrixArrayView(SizeType width, SizeType height, SizeType depth, ValueType* data) : MatrixArrayLike(width, height, depth, data)
		{}
	};
}