#pragma once
#include "VectorLike.h"
#include <memory>
#include <type_traits>

namespace VecAl
{
	class Vector : public VectorLike
	{
		using SmartPtr = std::unique_ptr<ValueType[]>;

		SmartPtr m_data;
	public:
		static Vector Init(SizeType size, ValueType value);

		static Vector Init(SizeType size);
		
		Vector(SizeType size) : VectorLike(size, m_data.get()), m_data(std::make_unique<ValueType[]>(size)) { GetPtr() = m_data.get(); }
	};
}




