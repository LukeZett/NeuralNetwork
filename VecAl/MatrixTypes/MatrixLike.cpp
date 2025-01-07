#include "MatrixLike.h"
#include <VectorLike.h>

using namespace VecAl;
#ifdef SCALAR
MatrixLike& VecAl::MatrixLike::MatMulTransposed(const MatrixLike& transposed, MatrixLike& dest)
{
	for (size_t i = 0; i < dest.m_height; i++)
	{
		for (size_t j = 0; j < dest.m_width; j++)
		{
			dest[i][j] = 0;
			for (size_t k = 0; k < m_width; k++)
			{
				dest[i][j] += operator[](i)[k] * transposed[j][k];
			}
		}
	}

	return dest;
}

VectorLike& VecAl::MatrixLike::MatMulTransposed(const VectorLike& transposed, VectorLike& dest)
{
	for (size_t i = 0; i < m_height; i++)
	{
		dest[i] = 0;
		for (size_t k = 0; k < m_width; k++)
		{
			dest[i] += operator[](i)[k] * transposed[k];
		}
	}

	return dest;
}
#else

MatrixLike& VecAl::MatrixLike::MatMulTransposed(const MatrixLike& transposed, MatrixLike& dest)
{
	SizeType vectorized_size = Cols() - (Cols() % MatOp::FLOATSIZE);
	float acc[MatOp::FLOATSIZE] = { 0 };

	ValueType* data = GetPtr();

	for (SizeType i = 0; i < m_height; i++)
	{
		for (SizeType j = 0; j < transposed.m_height; j++)
		{
			dest[i][j] = 0;
			MatOp::vecf s = MatOp::Load(0.f);

			for (SizeType k = 0; k < vectorized_size; k+= MatOp::FLOATSIZE)
			{
				MatOp::vecf vec_vals = MatOp::Load(&transposed[j][k]);
				MatOp::vecf mat_vals = MatOp::Load(&(operator[](i)[k]));
				MatOp::vecf multiplied = MatOp::Mul(mat_vals, vec_vals);
				s = MatOp::Add(s, multiplied);
			}

			MatOp::Store(acc, s);

			for (SizeType j = 0; j < MatOp::FLOATSIZE; j++)
			{
				dest[i][j] += acc[j];
			}

			for (SizeType k = vectorized_size; k < Cols(); k++)
			{
				dest[i][j] += transposed[j][k] * operator[](i)[k];
			}
		}
	}

	return dest;
}


VectorLike& VecAl::MatrixLike::MatMulTransposed(const VectorLike& transposed, VectorLike& dest)
{
	SizeType vectorized_size = Cols() - (Cols() % MatOp::FLOATSIZE);
	float acc[MatOp::FLOATSIZE] = { 0 };

	ValueType* data = GetPtr();

	for (SizeType row = 0; row < Rows(); row++)
	{
		MatOp::vecf s = MatOp::Load(0.f);

		for (SizeType elementIdx = 0; elementIdx < vectorized_size; elementIdx += MatOp::FLOATSIZE) {
			MatOp::vecf vec_vals = MatOp::Load(&transposed[elementIdx]);
			MatOp::vecf mat_vals = MatOp::Load(data + (row * Cols() + elementIdx));
			MatOp::vecf multiplied = MatOp::Mul(mat_vals, vec_vals);
			s = MatOp::Add(s, multiplied);
		}

		MatOp::Store(acc, s);

		dest[row] = 0;
		for (SizeType j = 0; j < MatOp::FLOATSIZE; j++)
		{
			dest[row] += acc[j];
		}

		for (SizeType j = vectorized_size; j < Cols(); j++)
		{
			dest[row] += transposed[j] * data[row * Cols() + j];
		}
	}

	return dest;
}

#endif





#ifdef SCALAR
MatrixLike& VecAl::MatrixLike::Convolution(const MatrixLike& kernel, MatrixLike& dest, bool full)
{
	size_t height = Height();
	size_t width = Width();
	if (!full)
	{
		height -= kernel.Height();
		width -= kernel.Width();
		height++;
		width++;
	}

	for (size_t y = 0; y < height; y++)
	{
		for (size_t x = 0; x < width; x++)
		{
			float result = dest[y][x];
			for (size_t i = 0; i < kernel.Height(); i++)
			{
				if (i > y) break;

				for (size_t j = 0; j < kernel.Width(); j++)
				{
					if (j > x) break;
					result += (*this)[y - i][x - j] * kernel[i][j];
				}
			}
			dest[y][x] = result;
		}
	}
	return dest;
}
#else
MatrixLike& VecAl::MatrixLike::Convolution(const MatrixLike& kernel, MatrixLike& dest, bool full)
{
	size_t height = Height();
	size_t width = Width();
	if (!full)
	{
		height -= kernel.Height();
		width -= kernel.Width();
		height++;
		width++;
	}

	SizeType vectorized_size = width - (width % MatOp::FLOATSIZE);

	for (size_t y = 0; y < height; y++)
	{
		for (size_t x = 0; x < vectorized_size; x+= MatOp::FLOATSIZE)
		{
			MatOp::vecf acc = MatOp::Load(&dest[y][x]);

			for (size_t i = 0; i < kernel.Height(); i++)
			{
				if (i > y) break;
				for (size_t j = 0; j < kernel.Width(); j++)
				{
					MatOp::vecf a1;
					if (j > x)
					{
						if (j > x + MatOp::FLOATSIZE) break;
						
						ValueType tmp[MatOp::FLOATSIZE] = { 0 };
						for (SizeType e = j; e < MatOp::FLOATSIZE; e++)
							tmp[e] = (*this)[y - i][x - j + e];
						
						a1 = MatOp::Load(tmp);
					}
					else
					{
						a1 = MatOp::Load(&(*this)[y - i][x - j]);
					}

					MatOp::vecf a2 = MatOp::Load(kernel[i][j]);

					MatOp::vecf mult = MatOp::Mul(a1, a2);
					acc = MatOp::Add(acc, mult);
				}
			}
			MatOp::Store(&dest[y][x], acc);
		}
	}

	for (size_t y = 0; y < height; y++)
	{
		for (size_t x = vectorized_size; x < width; x++)
		{
			float result = dest[y][x];
			for (size_t i = 0; i < kernel.Height(); i++)
			{
				if (i > y) break;
				for (size_t j = 0; j < kernel.Width(); j++)
				{
					if (j > x) break;
					result += (*this)[y - i][x - j] * kernel[i][j];
				}
			}
			dest[y][x] = result;
		}
	}
	return dest;
}
#endif


#ifdef SCALAR
MatrixLike& VecAl::MatrixLike::Correlation(const MatrixLike& kernel, MatrixLike& dest, bool full)
{
	size_t height = Height();
	size_t width = Width();
	if (!full)
	{
		height -= kernel.Height();
		width -= kernel.Width();
		height++;
		width++;
	}
	else
	{
		height += kernel.Height();
		width += kernel.Width();
		height--;
		width--;
	}

	for (size_t y = 0; y < height; y++)
	{
		for (size_t x = 0; x < width; x++)
		{
			float result = dest[y][x];
			for (size_t i = 0; i < kernel.Height(); i++)
			{
				if (i + y >= Height()) break;

				for (size_t j = 0; j < kernel.Width(); j++)
				{
					if (j + x >= Width()) break;
					result += (*this)[y + i][x + j] * kernel[i][j];
				}
			}
			dest[y][x] = result;
		}
	}
	return dest;
}

#else
MatrixLike& VecAl::MatrixLike::Correlation(const MatrixLike& kernel, MatrixLike& dest, bool full)
{
	size_t height = Height();
	size_t width = Width();
	if (!full)
	{
		height -= kernel.Height();
		width -= kernel.Width();
		height++;
		width++;
	}
	else
	{
		height += kernel.Height();
		width += kernel.Width();
		height--;
		width--;
	}

	SizeType vectorized_start = width % MatOp::FLOATSIZE;

	for (size_t y = 0; y < height; y++)
	{
		for (size_t x = vectorized_start; x < width; x += MatOp::FLOATSIZE)
		{
			MatOp::vecf acc = MatOp::Load(&dest[y][x]);
			for (size_t i = 0; i < kernel.Height(); i++)
			{
				if (i + y >= Height()) break;

				for (size_t j = 0; j < kernel.Width(); j++)
				{
					MatOp::vecf a1;
					if (j + x + MatOp::FLOATSIZE >= Width())
					{
						if (j + x >= Width()) break;

						ValueType tmp[MatOp::FLOATSIZE] = { 0 };
						for (SizeType e = j + x; e < Width(); e++)
							tmp[e - (j + x)] = (*this)[y - i][e];

						a1 = MatOp::Load(tmp);
					}
					else
					{
						a1 = MatOp::Load(&(*this)[y + i][x + j]);
					}
					MatOp::vecf a2 = MatOp::Load(kernel[i][j]);

					MatOp::vecf mult = MatOp::Mul(a1, a2);
					acc = MatOp::Add(acc, mult);
				}
			}
			MatOp::Store(&dest[y][x], acc);
		}
	}

	for (size_t y = 0; y < height; y++) {
		for (size_t x = 0; x < vectorized_start; x++)
		{
			float result = dest[y][x];
			for (size_t i = 0; i < kernel.Height(); i++)
			{
				if (i + y >= Height()) break;

				for (size_t j = 0; j < kernel.Width(); j++)
				{
					if (j + x >= Width()) break;
					result += (*this)[y + i][x + j] * kernel[i][j];
				}
			}
			dest[y][x] = result;
		}
	}
	return dest;
}
#endif // SCALAR
