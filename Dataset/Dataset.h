#pragma once
#include <vector>
#include <string>
#include "Vector.h"
#include "VectorView.h"
#include <filesystem>
#include <MatrixView.h>

class Dataset
{
	size_t m_width;
	size_t m_height;
	std::vector<float> m_data;
public:
	static Dataset FromCSV(const std::filesystem::path& file, char delim = ',', size_t maxSize = 0xFFFFFFFF);

	Dataset(const std::filesystem::path& file, char delim = ',', size_t maxSize = 0xFFFFFFFF);
	
	VecAl::VectorView AsVector();

	Dataset CreateLabelsVectors(uint32_t labelCount, uint32_t index);

	inline size_t GetWidth() const { return m_width; };

	inline size_t GetHeight() const { return m_height; };

	inline VecAl::VectorView GetElement(size_t index) { return VecAl::VectorView(m_width, &m_data[index * m_width]); }

	inline VecAl::MatrixView AsMatrix() { return VecAl::MatrixView(m_width, m_height, m_data.data()); }

	inline VecAl::VectorView operator[](size_t index) { return GetElement(index); }

private:
	Dataset(std::vector<float>& data, size_t width, size_t height);

	Dataset(size_t width, size_t height);

};