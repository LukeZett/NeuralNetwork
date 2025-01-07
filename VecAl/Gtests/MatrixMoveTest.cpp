#include <gtest/gtest.h>
#include "Matrix.h"
#include "MatrixView.h"

TEST(MatrixMove, MoveAssign)
{
	VecAl::Matrix matrix(4, 4);
	matrix.SetTo(0);
	matrix[0][0] = 1;

	VecAl::MatrixView view(0,0,nullptr);

	ASSERT_EQ(view.Width(), 0);
	ASSERT_EQ(view.Height(), 0);

	view = VecAl::MatrixView(matrix);

	ASSERT_EQ(view.Width(), 4);
	ASSERT_EQ(view.Height(), 4);

	for (size_t i = 0; i < 4; i++)
	{
		for (size_t j = 0; j < 4; j++)
		{
			if ((i + j) == 0)
			{
				EXPECT_FLOAT_EQ(view[i][j], 1);
			}
			else
			{
				EXPECT_FLOAT_EQ(view[i][j], 0);
			}

			EXPECT_FLOAT_EQ(view[i][j], matrix[i][j]);

		}
	}
}
