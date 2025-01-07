#pragma once

#include "MatrixView.h"

namespace VecAl
{
	class MatrixArrayLike : public VectorLike
	{
		SizeType m_width;
		SizeType m_height;
		SizeType m_depth;

	protected:
		MatrixArrayLike(SizeType width, SizeType height, SizeType depth, PtrType data) : VectorLike(width* height* depth, data), m_width(width), m_height(height), m_depth(depth) {}

	public:
		inline const SizeType& Width() const { return m_width; }

		inline const SizeType& Cols() const { return m_width; }

		inline const SizeType& Height() const { return m_height; }

		inline const SizeType& Rows() const { return m_height; }

		inline const SizeType& Depth() const { return m_depth; }

		inline const SizeType& MatrixCount() const { return m_depth; }

		inline MatrixView operator[](SizeType index) { return MatrixView(m_width, m_height, GetPtr() + index * m_width * m_height); }
	
		MatrixArrayLike& AllToAllConvolution(MatrixArrayLike& kernels, MatrixArrayLike& dest, bool full);

		MatrixArrayLike& AllToAllCorrelation(MatrixArrayLike& kernels, MatrixArrayLike& dest, bool full);
	};
}