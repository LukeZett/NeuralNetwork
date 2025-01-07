#pragma once
#include "MatrixArrayLike.h"
#include <memory>
#include <VectorLike.h>

namespace VecAl
{
	class MatrixArray : public MatrixArrayLike
	{
		std::unique_ptr<ValueType[]> m_data;
	public:

		MatrixArray(SizeType width, SizeType height, SizeType depth) : MatrixArrayLike(width, height, depth, m_data.get()),
			m_data(std::make_unique<ValueType[]>((width * height * depth)))
		{
			GetPtr() = m_data.get();
		}
	};
}