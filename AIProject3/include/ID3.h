// ID3.h
#pragma once

#include <memory>
#include <vector>

namespace ml
{
	struct DataSet;
	struct Instance;

	namespace id3
	{
		std::size_t algorithm(const DataSet& dataset, const std::vector<Instance>& trainingSet, const std::vector<Instance>& testSet);
	}
}
