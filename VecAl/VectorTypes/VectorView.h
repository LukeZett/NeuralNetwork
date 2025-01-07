#pragma once
#include "VectorLike.h"

namespace VecAl
{
	class VectorView : public VectorLike
	{
	public:
		VectorView(VectorLike& vector) : VectorLike(vector) {};

		VectorView(SizeType size, ValueType* data) : VectorLike(size, data) {};
	};
}