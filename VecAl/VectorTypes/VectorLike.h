#pragma once
#include <iosfwd>
#include <time.h>
#include <cstring>
#include <cmath>

#define AVX256

#ifdef AVX512
#include "MatOps/AVX512float.h"
#elif defined AVX256
#include "MatOps/AVX256float.h"
#else
#define SCALAR
#endif // AVX512

#ifdef SCALAR
#include "MatOps/Scalarfloat.h"
#endif // SCALAR


namespace VecAl
{
#define ALLIGNED(number, blockSize) ( number - (number % blockSize) )

	class MatrixLike;

	class VectorLike
	{
	public:
		using ValueType = float;
		using PtrType = ValueType*;
		using SizeType = size_t;

	private:
		SizeType m_size;
		PtrType m_ptr;
#ifndef SCALAR
		SizeType m_alignedSize;
#endif // !SCALAR

	protected:
		VectorLike(SizeType size, PtrType data) : m_size(size), 
#ifndef SCALAR
			m_alignedSize(ALLIGNED(size, MatOp::FLOATSIZE)),
#endif // !SCALAR
			m_ptr(data) {}

		inline const PtrType& GetPtr() const { return m_ptr; }

		inline PtrType& GetPtr() { return m_ptr; }

	public: // memory operations

		VectorLike& SetTo(float value);
		
		VectorLike& SetTo(const VectorLike& value);

		VectorLike& NormalDist(ValueType mean, ValueType variance);

	public: // access operators
	
		inline const SizeType& Size() const { return m_size; }

		inline ValueType& operator[](SizeType index) { return m_ptr[index]; }

		inline const ValueType& operator[](SizeType index) const { return m_ptr[index]; }

	public: // vectorized math
		VectorLike& MatMul(const MatrixLike& matrix, VectorLike& dest) const;

		MatrixLike& MatMul(const VectorLike& vector, MatrixLike& dest, bool accumulateResult = false) const;

		inline VectorLike& Add(const VectorLike& vec2, VectorLike& dest) const { return BinaryOperation<MatOp::Add>(vec2, dest); }
		
		inline VectorLike& Sub(const VectorLike& vec2, VectorLike& dest) const { return BinaryOperation<MatOp::Sub>(vec2, dest); }

		inline VectorLike& Mul(const VectorLike& vec2, VectorLike& dest) const { return BinaryOperation<MatOp::Mul>(vec2, dest); }

		inline VectorLike& Div(const VectorLike& vec2, VectorLike& dest) const { return BinaryOperation<MatOp::Div>(vec2, dest); }
		
		inline VectorLike& Max(const VectorLike& vec2, VectorLike& dest) const { return BinaryOperation<MatOp::Max>(vec2, dest); }

		inline VectorLike& ArgMax(const VectorLike& vec2, VectorLike& dest) const { return BinaryOperation<MatOp::ArgMax>(vec2, dest); }
		
		inline VectorLike& Add(const ValueType value, VectorLike& dest) const { return BinaryOperation<MatOp::Add>(value, dest); }

		inline VectorLike& Sub(const ValueType value, VectorLike& dest) const { return BinaryOperation<MatOp::Sub>(value, dest); }

		inline VectorLike& Mul(const ValueType value, VectorLike& dest) const { return BinaryOperation<MatOp::Mul>(value, dest); }

		inline VectorLike& Div(const ValueType value, VectorLike& dest) const { return BinaryOperation<MatOp::Div>(value, dest); }

		inline VectorLike& Max(const ValueType value, VectorLike& dest) const { return BinaryOperation<MatOp::Max>(value, dest); }
		
		inline VectorLike& ArgMax(const ValueType& value, VectorLike& dest) const { return BinaryOperation<MatOp::ArgMax>(value, dest); }
		
		inline VectorLike& SubFrom(const ValueType value, VectorLike& dest) const { return BinaryOperation<MatOp::SubReversed>(value, dest); }
		
		inline VectorLike& Exponentiate(const ValueType value, VectorLike& dest) const {
			for (size_t i = 0; i < m_size; i++)
			{
				dest[i] = std::pow(value, m_ptr[i]);
			}
			return dest;
		}


		inline VectorLike& ReciprMul(const ValueType value, VectorLike& dest) const { return BinaryOperation<MatOp::DivReversed>(value, dest); }

		inline VectorLike& Abs(VectorLike& dest) { return UnaryOperation<MatOp::Abs>(dest); }

		inline VectorLike& Sqrt(VectorLike& dest) { return UnaryOperation<MatOp::Sqrt>(dest); }

	public: // math operators

		inline VectorLike& operator += (const VectorLike& other) { return Add(other, *this); }

		inline VectorLike& operator *= (const VectorLike& other) { return Mul(other, *this); }

		inline VectorLike& operator -= (const VectorLike& other) { return Sub(other, *this); }

		inline VectorLike& operator /= (const VectorLike& other) { return Div(other, *this); }

		inline VectorLike& operator += (ValueType number) { return Add(number, *this); }

		inline VectorLike& operator *= (ValueType number) { return Mul(number, *this); }

		inline VectorLike& operator -= (ValueType number) { return Sub(number, *this); }

		inline VectorLike& operator /= (ValueType number) { return Div(number, *this); }

	public:

		template<MatOp::vecf OP(const MatOp::vecf&, const MatOp::vecf&)>
		inline VectorLike& BinaryOperation(const VectorLike& vec2, VectorLike& dest) const;

		template<MatOp::vecf OP(const MatOp::vecf&, const MatOp::vecf&)>
		inline VectorLike& BinaryOperation(ValueType vec2, VectorLike& dest) const;

		template<MatOp::vecf OP(const MatOp::vecf&)>
		inline VectorLike& UnaryOperation(VectorLike& dest);
	};



/*********************************
* Templated function definitions *
**********************************/

	template<MatOp::vecf OP(const MatOp::vecf&, const MatOp::vecf&)>
	inline VectorLike& VectorLike::BinaryOperation(const VectorLike& vec2, VectorLike& dest) const
	{
		const PtrType& rawPtr = GetPtr();
		const PtrType& otherPtr = vec2.GetPtr();
		const PtrType& destPtr = dest.GetPtr();

#ifdef SCALAR
		SizeType m_alignedSize = m_size;
#endif // SCALAR

		for (SizeType i = 0; i < m_alignedSize; i += MatOp::FLOATSIZE)
		{
			MatOp::vecf a1 = MatOp::Load(rawPtr + i);
			MatOp::vecf a2 = MatOp::Load(otherPtr + i);
			MatOp::vecf res = OP(a1, a2);
			MatOp::Store(destPtr + i, res);
		}

#ifndef SCALAR
		if (m_alignedSize == Size()) return dest;

		float remainders[MatOp::FLOATSIZE];
		float remaindersOther[MatOp::FLOATSIZE];
		float remainderStore[MatOp::FLOATSIZE];

		memcpy(remainders, rawPtr + m_alignedSize, (Size() - m_alignedSize) * sizeof(float));
		memcpy(remaindersOther, otherPtr + m_alignedSize, (Size() - m_alignedSize) * sizeof(float));

		MatOp::vecf remainder = MatOp::Load(remainders);
		MatOp::vecf remainderOther = MatOp::Load(remaindersOther);
		MatOp::vecf res = OP(remainder, remainderOther);
		MatOp::Store(remainderStore, res);

		memcpy(destPtr + m_alignedSize, remainderStore, (Size() - m_alignedSize) * sizeof(float));
#endif // !SCALAR

		return dest;
	}

	template<MatOp::vecf OP(const MatOp::vecf&, const MatOp::vecf&)>
	inline VectorLike& VectorLike::BinaryOperation(ValueType value, VectorLike& dest) const
	{
		const PtrType& rawPtr = GetPtr();
		const PtrType& destPtr = dest.GetPtr();
		MatOp::vecf a2 = MatOp::Load(value);

#ifdef SCALAR
		SizeType m_alignedSize = m_size;
#endif // SCALAR

		for (SizeType i = 0; i < m_alignedSize; i += MatOp::FLOATSIZE)
		{
			MatOp::vecf a1 = MatOp::Load(rawPtr + i);
			MatOp::vecf res = OP(a1, a2);
			MatOp::Store(destPtr + i, res);
		}

#ifndef SCALAR
		if (m_alignedSize == Size()) return dest;

		float remainders[MatOp::FLOATSIZE];
		float remainderStore[MatOp::FLOATSIZE];

		memcpy(remainders, rawPtr + m_alignedSize, (Size() - m_alignedSize) * sizeof(float));

		MatOp::vecf remainder = MatOp::Load(remainders);
		MatOp::vecf res = OP(remainder, a2);
		MatOp::Store(remainderStore, res);

		memcpy(destPtr + m_alignedSize, remainderStore, (Size() - m_alignedSize) * sizeof(float));
#endif // !SCALAR

		return dest;
	}

	template<MatOp::vecf OP(const MatOp::vecf&)>
	inline VectorLike& VectorLike::UnaryOperation(VectorLike& dest)
	{
		const PtrType& destPtr = dest.GetPtr();
		const PtrType& rawPtr = GetPtr();

#ifdef SCALAR
		SizeType m_alignedSize = m_size;
#endif // SCALAR

		for (SizeType i = 0; i < m_alignedSize; i += MatOp::FLOATSIZE)
		{
			MatOp::vecf a1 = MatOp::Load(rawPtr + i);
			auto res = OP(a1);
			MatOp::Store(destPtr + i, res);
		}

#ifndef SCALAR
		if (m_alignedSize == Size()) return dest;

		float remainders[MatOp::FLOATSIZE];
		float remainderStore[MatOp::FLOATSIZE];

		memcpy(remainders, rawPtr + m_alignedSize, (Size() - m_alignedSize) * sizeof(float));

		MatOp::vecf remainder = MatOp::Load(remainders);
		MatOp::vecf res = OP(remainder);
		MatOp::Store(remainderStore, res);

		memcpy(destPtr + m_alignedSize, remainderStore, (Size() - m_alignedSize) * sizeof(float));
#endif // SCALAR
		return dest;
	}

}

std::ostream& operator<<(std::ostream& strm, const VecAl::VectorLike& a);