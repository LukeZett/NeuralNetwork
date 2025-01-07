#include "Vector.h"

using namespace VecAl;

Vector Vector::Init(SizeType size, ValueType value)
{
	Vector vec = Vector(size);
	vec.SetTo(value);
	return vec;
}

Vector Vector::Init(SizeType size)
{
	return Vector(size);
}