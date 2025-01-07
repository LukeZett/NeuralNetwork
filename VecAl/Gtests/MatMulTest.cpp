#include <gtest/gtest.h>
#include "Matrix.h"
#include "MatrixView.h"

TEST(MatMul, SmallMatMulWTransposed)
{
	VecAl::Matrix matrix1(2, 4);
	VecAl::Matrix matrix2(2, 3);
	VecAl::Matrix matrix3(3, 4);

	int c = 1;
	for (size_t h = 0; h < 4; h++)
	{
		for (size_t w = 0; w < 2; w++)
		{
			matrix1[h][w] = c;
			c++;
		}
	}

	c = 1;
	for (size_t w = 0; w < 2; w++)
	{
		for (size_t h = 0; h < 3; h++)
		{
			matrix2[h][w] = c;
			c++;
		}
	}

	matrix1.MatMulTransposed(matrix2, matrix3);

	float result[] = { 9, 12, 15, 19, 26, 33, 29, 40, 51, 39, 54, 69 };

	int id = 0;
	for (size_t h = 0; h < 4; h++)
	{
		for (size_t w = 0; w < 3; w++)
		{
			EXPECT_FLOAT_EQ(matrix3[h][w], result[id]);
			id++;
		}
	}
}

TEST(MatMul, MatMulWTransposedSimple)
{
	VecAl::Matrix matrix1(17, 18);
	VecAl::Matrix matrix2(17, 19);
	VecAl::Matrix matrix3(19, 18);

	matrix1.SetTo(1);
	matrix2.SetTo(2);

	matrix1.MatMulTransposed(matrix2, matrix3);

	for (size_t h = 0; h < 18; h++)
	{
		for (size_t w = 0; w < 19; w++)
		{
			EXPECT_FLOAT_EQ(matrix3[h][w], 17*2);
		}
	}
}