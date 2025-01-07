#pragma once
#include <intrin.h>

namespace MatOp
{
	constexpr unsigned int FLOATSIZE = 16;

	typedef __m512 vecf;

	//load and store

	inline vecf Load(const float& value)
	{
		return _mm512_set1_ps(value);
	}

	inline vecf Load(const float* ptr)
	{
		return _mm512_loadu_ps(ptr);
	}

	inline void Store(float* dst, const vecf& numbers)
	{
		_mm512_storeu_ps(dst, numbers);
	}

	//math

	inline vecf Add(const vecf& num1, const vecf& num2)
	{
		return _mm512_add_ps(num1, num2);
	}

	inline vecf Sub(const vecf& num1, const vecf& num2)
	{
		return _mm512_sub_ps(num1, num2);
	}

	inline vecf SubReversed(const vecf& num1, const vecf& num2)
	{
		return _mm512_sub_ps(num2, num1);
	}

	inline vecf Mul(const vecf& num1, const vecf& num2)
	{
		return _mm512_mul_ps(num1, num2);
	}

	inline vecf Div(const vecf& num1, const vecf& num2)
	{
		return _mm512_div_ps(num1, num2);
	}

	inline vecf DivReversed(const vecf& num1, const vecf& num2)
	{
		return _mm512_div_ps(num2, num1);
	}

	inline vecf Pow(const vecf& num1, const vecf& num2)
	{
		return _mm512_pow_ps(num1, num2);
	}

	inline vecf Exp(const vecf& num1, const vecf& num2)
	{
		return _mm512_pow_ps(num2, num1);
	}
	
	inline vecf Max(const vecf& num1, const vecf& num2)
	{
		return _mm512_max_ps(num1, num2);
	}

	inline vecf ArgMax(const vecf& num1, const vecf& num2)
	{
		auto mask = _mm512_cmp_ps_mask(num1, num2, _CMP_GT_OS);
		vecf ones = Load(1.f);
		vecf zeros = Load(0.f);
		return _mm512_mask_blend_ps(mask, zeros, ones);
	}

	inline vecf Sqrt(const vecf& num1)
	{
		return _mm512_sqrt_ps(num1);
	}

	inline vecf Abs(const vecf& num1)
	{
		return _mm512_abs_ps(num1);
	}


}