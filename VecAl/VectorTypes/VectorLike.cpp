#include "VectorLike.h"
#include "MatrixLike.h"
#include <random>
#include <cstring>
#include <sstream>
#include <iosfwd>
#include <stdlib.h>

using namespace VecAl;

VectorLike& VecAl::VectorLike::SetTo(float value)
{
	const PtrType& data = GetPtr();
	for (SizeType i = 0; i < Size(); i++) data[i] = value;
	return *this;
}

VectorLike& VecAl::VectorLike::SetTo(const VectorLike& value)
{
	memcpy(GetPtr(), value.GetPtr(), Size());
	return *this;
}

VectorLike& VecAl::VectorLike::NormalDist(ValueType mean, ValueType variance)
{
	std::mt19937 gen{ (unsigned int)rand() };
	std::normal_distribution<ValueType> d{ mean, variance };

	auto getRandomFloat = [&d, &gen] { return d(gen); };

	const PtrType& data = GetPtr();
	for (uint32_t i = 0; i < Size(); i++) data[i] = getRandomFloat();
	return *this;
}

#ifdef SCALAR
VectorLike& VecAl::VectorLike::MatMul(const MatrixLike& matrix, VectorLike& dest) const
{
	dest.SetTo(0);
	for (size_t i = 0; i < matrix.Width(); i++)
	{
		for (size_t j = 0; j < Size(); j++)
		{
			dest[i] += operator[](j) * matrix[j][i];
		}
	}
	return dest;
}
#else
VectorLike& VecAl::VectorLike::MatMul(const MatrixLike& matrix, VectorLike& dest) const
{
	dest.SetTo(0);
	SizeType vectorizedSize = matrix.Width() - (matrix.Width() % MatOp::FLOATSIZE);

	for (SizeType i = 0; i < vectorizedSize; i+= MatOp::FLOATSIZE)
	{
		MatOp::vecf acc = MatOp::Load(dest[i]);

		for (SizeType j = 0; j < Size(); j++)
		{
			MatOp::vecf vecElement = MatOp::Load(operator[](j));
			MatOp::vecf matRow = MatOp::Load(&matrix[j][i]);
			MatOp::vecf multiplied = MatOp::Mul(matRow, vecElement);
			acc = MatOp::Add(acc, multiplied);
		}

		MatOp::Store(&dest[i], acc);
	}

	for (SizeType i = vectorizedSize; i < matrix.Width(); i++)
	{
		for (SizeType j = 0; j < Size(); j++)
		{
			dest[i] += operator[](j) * matrix[j][i];
		}
	}


	return dest;
}
#endif // SCALAR

#ifdef SCALAR
MatrixLike& VecAl::VectorLike::MatMul(const VectorLike& vector, MatrixLike& dest, bool accumulateResult) const
{
	if (!accumulateResult)
		dest.SetTo(0.f);

	for (uint32_t nodeID = 0; nodeID < Size(); nodeID++) {
		for (uint32_t weightID = 0; weightID < vector.Size(); weightID++)
		{
			dest[nodeID][weightID] += operator[](nodeID) * vector[weightID];
		}
	}
	return dest;
}
#else
MatrixLike& VecAl::VectorLike::MatMul(const VectorLike& vector, MatrixLike& dest, bool accumulateResult) const
{
	if (!accumulateResult)
		dest.SetTo(0.f);

	SizeType vectorizedSize = vector.Size() - (vector.Size() % MatOp::FLOATSIZE);

	for (uint32_t nodeID = 0; nodeID < Size(); nodeID++) {


		for (uint32_t weightID = 0; weightID < vectorizedSize; weightID+= MatOp::FLOATSIZE)
		{
			MatOp::vecf destVec = MatOp::Load(&dest[nodeID][weightID]);
			MatOp::vecf other = MatOp::Load(&vector[weightID]);
			MatOp::vecf broadcasted = MatOp::Load(operator[](nodeID));

			MatOp::vecf multiplied = MatOp::Mul(broadcasted, other);
			destVec = MatOp::Add(multiplied, destVec);
			MatOp::Store(&dest[nodeID][weightID], destVec);
		}
		for (uint32_t weightID = vectorizedSize; weightID < vector.Size(); weightID++)
		{
			dest[nodeID][weightID] += operator[](nodeID) * vector[weightID];
		}
	}
	return dest;
}
#endif // SCALAR



std::ostream& operator<<(std::ostream& strm, const VecAl::VectorLike& a)
{
	strm << "[";
	for (size_t i = 0; i < a.Size(); i++)
	{
		strm << " " << a[i];
	}
	strm << " ]";
	return strm;
}
