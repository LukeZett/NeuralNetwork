#include "MatrixArrayLike.h"
#include <MatrixView.h>
#include <future>
#include <array>

#define THREADCOUNT 4

using namespace VecAl;

static void ConvAsync(MatrixArrayLike* matrices, MatrixArrayLike* kernels, MatrixArrayLike* dest, VectorLike::SizeType taskIdx, VectorLike::SizeType taskSize, bool full)
{
	for (size_t task = taskSize * taskIdx; task <  std::min(taskSize * (taskIdx + 1), kernels->Depth() * matrices->Depth()); task++)
	{
		size_t i = task % kernels->Depth();
		size_t j = task / kernels->Depth();
		VecAl::MatrixView dst = (*dest)[task];
		(*matrices)[j].Convolution((*kernels)[i], dst, full);
	}
}

MatrixArrayLike& MatrixArrayLike::AllToAllConvolution(MatrixArrayLike& kernels, MatrixArrayLike& dest, bool full)
{
	std::array<std::future<void>, THREADCOUNT> futures = {};

	SizeType tasksPerThread = (Depth() * kernels.Depth()) / THREADCOUNT + 1;

	for (SizeType i = 0; i < THREADCOUNT; i++)
	{
		futures[i] = std::async(std::launch::async, ConvAsync, this, &kernels, &dest, i, tasksPerThread, full);
	}

	for (auto& future : futures)
	{
		future.wait();
	}
	return dest;
}

static void CorrAsync(MatrixArrayLike* matrices, MatrixArrayLike* kernels, MatrixArrayLike* dest, VectorLike::SizeType taskIdx, VectorLike::SizeType taskSize, bool full)
{
	for (size_t task = taskSize * taskIdx; task < std::min(taskSize * (taskIdx + 1), kernels->Depth() * matrices->Depth()); task++)
	{
		size_t i = task % kernels->Depth();
		size_t j = task / kernels->Depth();
		VecAl::MatrixView dst = (*dest)[task];
		(*matrices)[j].Correlation((*kernels)[i], dst, full);
	}
}

MatrixArrayLike& MatrixArrayLike::AllToAllCorrelation(MatrixArrayLike& kernels, MatrixArrayLike& dest, bool full)
{
	std::array<std::future<void>, THREADCOUNT> futures = {};

	SizeType tasksPerThread = (Depth() * kernels.Depth()) / THREADCOUNT + 1;

	for (SizeType i = 0; i < THREADCOUNT; i++)
	{
		futures[i] = std::async(std::launch::async, CorrAsync, this, &kernels, &dest, i, tasksPerThread, full);
	}

	for (auto& future : futures)
	{
		future.wait();
	}
	return dest;
}