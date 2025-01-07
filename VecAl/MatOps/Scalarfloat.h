#pragma once
#include <utility>
#include "cmath"
#include <cstdlib>

namespace MatOp
{
	constexpr unsigned int FLOATSIZE = 1;

	typedef float vecf;

	//load and store

	inline vecf Load(const float& value)
	{
		return value;
	}

	inline vecf Load(const float* ptr)
	{
		return *ptr;
	}

	inline void Store(float* dst, const vecf& numbers)
	{
		*dst = numbers;
	}

	//math

	inline vecf Add(const vecf& num1, const vecf& num2)
	{
		return num1 + num2;
	}

	inline vecf Sub(const vecf& num1, const vecf& num2)
	{
		return num1 - num2;
	}

	inline vecf SubReversed(const vecf& num1, const vecf& num2)
	{
		return num2 - num1;
	}

	inline vecf DivReversed(const vecf& num1, const vecf& num2)
	{
		return num2 / num1;
	}

	inline vecf Mul(const vecf& num1, const vecf& num2)
	{
		return num1 * num2;
	}

	inline vecf Div(const vecf& num1, const vecf& num2)
	{
		return num1 / num2;
	}

	inline vecf Pow(const vecf& num1, const vecf& num2)
	{
		return std::powf(num1, num2);
	}

	inline vecf Exp(const vecf& num1, const vecf& num2)
	{
		return std::powf(num2, num1);
	}

	inline vecf Max(const vecf& num1, const vecf& num2)
	{
		return std::max(num1, num2);
	}

	inline vecf ArgMax(const vecf& num1, const vecf& num2)
	{
		return num1 > num2 ? 1.f : 0.f;
	}

	inline vecf Sqrt(const vecf& num1)
	{
		return std::sqrtf(num1);
	}

	inline vecf Abs(const vecf& num1)
	{
		return std::abs(num1);
	}
}