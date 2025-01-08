#include <gtest/gtest.h>
#include "Vector.h"

TEST(VecArrayBasic, ZeroInit)
{
	auto vec1 = VecAl::Vector::Init(24, 0);

	ASSERT_EQ(vec1.Size(), 24);

	for (size_t i = 0; i < 24; i++)
	{
		EXPECT_FLOAT_EQ(vec1[i], 0.0f);
	}
}

TEST(VecArrayBasic, Addition)
{
	auto vec1 = VecAl::Vector::Init(24, 2);
	auto vec2 = VecAl::Vector::Init(24, 5);

	ASSERT_EQ(vec1.Size(), 24);
	ASSERT_EQ(vec2.Size(), 24);

	vec1.Add(vec2, vec1);

	for (size_t i = 0; i < 24; i++)
	{
		EXPECT_FLOAT_EQ(vec1[i], 7.0f);
	}
}

TEST(VecArrayBasic, AdditionWithRandomDist)
{
	auto vec1 = VecAl::Vector::Init(27);
	auto vec2 = VecAl::Vector::Init(27);
	
	auto vec3 = VecAl::Vector::Init(27);
	
	vec1.NormalDist(0, 1);
	vec2.NormalDist(0, 1);

	ASSERT_EQ(vec1.Size(), 27);
	ASSERT_EQ(vec2.Size(), 27);

	vec1.Add(vec2, vec3);

	for (size_t i = 0; i < 27; i++)
	{
		float expected = vec1[i] + vec2[i];

		EXPECT_FLOAT_EQ(vec3[i], expected);
	}
}

TEST(VecArrayBasic, MaxWithRandomDist)
{
	auto vec1 = VecAl::Vector::Init(27000);
	auto vec2 = VecAl::Vector::Init(27000);

	auto vec3 = VecAl::Vector::Init(27000);

	vec1.NormalDist(0, 1);
	vec2.NormalDist(0, 1);

	vec1.Max(vec2, vec3);

	for (size_t i = 0; i < 27000; i++)
	{
		float expected = std::max(vec1[i], vec2[i]);

		EXPECT_FLOAT_EQ(vec3[i], expected);
	}
}

TEST(VecFloatOperations, AdditionWithRandomDist)
{
	auto vec1 = VecAl::Vector::Init(27);

	float val = 1;

	auto vec3 = VecAl::Vector::Init(27);

	vec1.NormalDist(0, 1);

	vec1.Add(val, vec3);

	for (size_t i = 0; i < 27; i++)
	{
		float expected = vec1[i] + val;

		EXPECT_FLOAT_EQ(vec3[i], expected);
	}
}

TEST(VecFloatOperations, SubtractWithRandomDist)
{
	auto vec1 = VecAl::Vector::Init(17);

	float val = 1;

	auto vec3 = VecAl::Vector::Init(17);

	vec1.NormalDist(0, 1);

	vec1.Sub(val, vec3);

	for (size_t i = 0; i < 17; i++)
	{
		float expected = vec1[i] - val;
		EXPECT_FLOAT_EQ(vec3[i], expected);
	}
}

TEST(VecFloatOperations, CompoundOperationWithRandomDist)
{
	auto vec1 = VecAl::Vector::Init(120);

	float val = 1;
	float val2 = 2;

	auto vec3 = VecAl::Vector::Init(120);

	vec1.NormalDist(0, 1);

	vec1.Add(val, vec3);
	vec3.Mul(val2, vec3);

	for (size_t i = 0; i < 120; i++)
	{
		float expected = (vec1[i] + val) * 2;
		EXPECT_FLOAT_EQ(vec3[i], expected);
	}
}

TEST(VecUnaryOperations, CompoundAbsSqrtWithRandomDist)
{
	auto vec1 = VecAl::Vector::Init(120);
	auto vec2 = VecAl::Vector::Init(120);

	vec1.NormalDist(-1, 1);

	vec1.Abs(vec2);
	vec2.Sqrt(vec2);

	for (size_t i = 0; i < 120; i++)
	{
		float expected = std::sqrt(std::abs(vec1[i]));
		EXPECT_FLOAT_EQ(vec2[i], expected);
	}
}