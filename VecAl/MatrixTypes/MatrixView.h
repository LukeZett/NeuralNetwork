#pragma once
#include "MatrixLike.h"
#include <VectorLike.h>
#include <type_traits>

namespace VecAl
{
	class MatrixView : public MatrixLike
	{
	public:
		MatrixView(MatrixLike& other) : MatrixLike(other) {};

		MatrixView(SizeType width, SizeType height, ValueType* data) : MatrixLike(width, height, data) {};
	};
}
