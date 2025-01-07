#pragma once
#include "VectorLike.h"

namespace VecAl
{
	class MatrixLike : public VectorLike 
	{
		SizeType m_width;
		SizeType m_height;

	protected:
		MatrixLike(SizeType width, SizeType height, PtrType data) : VectorLike(width * height, data), m_width(width), m_height(height) {}

	public:
		
		inline const SizeType& Width() const { return m_width; }

		inline const SizeType& Cols() const { return m_width; }
		
		inline const SizeType& Height() const { return m_height; }
		
		inline const SizeType& Rows() const { return m_height; }

	public:
		MatrixLike& MatMulTransposed(const MatrixLike& transposed, MatrixLike& dest);

		VectorLike& MatMulTransposed(const VectorLike& transposed, VectorLike& dest);

		MatrixLike& Convolution(const MatrixLike& kernel, MatrixLike& dest, bool full);

		MatrixLike& Correlation(const MatrixLike& kernel, MatrixLike& dest, bool full);

		ValueType* operator[](SizeType col) { return GetPtr() + col * m_width; }

		const ValueType* operator[](SizeType col) const { return GetPtr() + col * m_width; }
	};
}