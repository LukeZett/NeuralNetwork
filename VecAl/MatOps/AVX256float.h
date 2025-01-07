#pragma once
#include <immintrin.h>

namespace MatOp
{
	constexpr unsigned int FLOATSIZE = 8;

	typedef __m256 vecf;

	//load and store

	inline vecf Load(const float& value)
	{
		return _mm256_set1_ps(value);
	}

	inline vecf Load(const float* ptr)
	{
		return _mm256_loadu_ps(ptr);
	}

	inline void Store(float* dst, const vecf& numbers)
	{
		_mm256_storeu_ps(dst, numbers);
	}

	//math

	inline vecf Add(const vecf& num1, const vecf& num2)
	{
		return _mm256_add_ps(num1, num2);
	}

	inline vecf Sub(const vecf& num1, const vecf& num2)
	{
		return _mm256_sub_ps(num1, num2);
	}

	inline vecf SubReversed(const vecf& num1, const vecf& num2)
	{
		return _mm256_sub_ps(num2, num1);
	}

	inline vecf Mul(const vecf& num1, const vecf& num2)
	{
		return _mm256_mul_ps(num1, num2);
	}

	inline vecf Div(const vecf& num1, const vecf& num2)
	{
		return _mm256_div_ps(num1, num2);
	}

	inline vecf DivReversed(const vecf& num1, const vecf& num2)
	{
		return _mm256_div_ps(num2, num1);
	}

	inline vecf Pow(const vecf& num1, const vecf& num2)
	{
		return _mm256_pow_ps(num1, num2);
	}

	inline vecf Exp(const vecf& num1, const vecf& num2)
	{
		return _mm256_pow_ps(num2, num1);
	}

	inline vecf Max(const vecf& num1, const vecf& num2)
	{
		return _mm256_max_ps(num1, num2);
	}

	inline vecf ArgMax(const vecf& num1, const vecf& num2)
	{
		vecf cmpRes = _mm256_cmp_ps(num1, num2, _CMP_GT_OS);;
		vecf one = Load(1.f);
		return _mm256_and_ps(one, cmpRes);
	}

	inline vecf Sqrt(const vecf& num1)
	{
		return _mm256_sqrt_ps(num1);
	}

	inline vecf Abs(const vecf& num1)
	{
		vecf sign_bit = _mm256_set1_ps(-0.0f);
		return _mm256_andnot_ps(sign_bit, num1);
	}

}