#include "Matrix.h"

using namespace VecAl;

Matrix VecAl::Matrix::Init(SizeType width, SizeType height, ValueType value)
{
	Matrix mat = Matrix(width, height);
	mat.SetTo(value);
	return mat;
}

Matrix VecAl::Matrix::Init(SizeType width, SizeType height)
{
	return Matrix(width, height);
}
