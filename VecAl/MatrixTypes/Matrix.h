#pragma once
#include "MatrixLike.h"
#include <memory>
#include <type_traits>

namespace VecAl
{
	class Matrix : public MatrixLike
	{
		using SmartPtr = std::unique_ptr<ValueType[]>;

		SmartPtr m_data;
	public:
		static Matrix Init(SizeType width, SizeType height, ValueType value);

		static Matrix Init(SizeType width, SizeType height);

		Matrix(SizeType width, SizeType height) : MatrixLike(width, height, m_data.get()),
			m_data(std::make_unique<ValueType[]>(width * height)) { GetPtr() = m_data.get(); }
	private:
	};
}
