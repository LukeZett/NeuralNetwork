#include "Dataset.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

VecAl::VectorView Dataset::AsVector()
{
	return VecAl::VectorView(m_width * m_height, &m_data[0]);
}


Dataset Dataset::CreateLabelsVectors(uint32_t labelCount, uint32_t index)
{
	Dataset res = Dataset(labelCount, m_height);
	auto vectorView = res.AsVector();
	vectorView.SetTo(0);

	for (size_t i = 0; i < m_data.size(); i++)
	{
		res.m_data[i * labelCount + m_data[m_width * i + index]] = 1;
	}
	return res;
}


Dataset::Dataset(std::vector<float>& data, size_t width, size_t height)
{
	m_data = std::move(data);
	m_width = width;
	m_height = height;
}


Dataset::Dataset(size_t width, size_t height) : m_height(height), m_width(width)
{
	m_data.resize(height * width);
}


Dataset Dataset::FromCSV(const std::filesystem::path& file, char delim, size_t maxSize)
{
	int count = 0;
	std::vector<float> data;

	std::ifstream input{ file };
	if (!input.is_open()) {
		std::cout << "Couldn't read file: " << file << "\n";
		return Dataset(data, 0,0);
	}
	size_t width = 0, height = 0;

	for (std::string line; std::getline(input, line);)
	{
		if (height == maxSize) break;

		std::stringstream ss(line);
		width = 0;
		for (std::string token; std::getline(ss, token, delim);)
		{
			data.push_back(std::stof(token));
			width++;
		}

		height++;
	}

	input.close();
	return Dataset(data, width, height);
}

Dataset::Dataset(const std::filesystem::path& file, char delim, size_t maxSize)
{
	int count = 0;
	m_width = 0;
	m_height = 0;

	std::ifstream input{ file };
	if (!input.is_open()) {
		std::cout << "Couldn't read file: " << file << "\n";
		return;
	}

	for (std::string line; std::getline(input, line);)
	{
		if (m_height == maxSize) break;

		std::stringstream ss(line);
		m_width = 0;
		for (std::string token; std::getline(ss, token, delim);)
		{
			m_data.push_back(std::stof(token));
			m_width++;
		}

		m_height++;
	}
}
