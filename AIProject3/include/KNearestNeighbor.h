// KNearestNeighbor.h
#pragma once

#include <vector>

namespace ml
{
	struct Instance;
	struct DataSet;

	namespace k_nearest_neighbor
	{
		std::size_t algorithm(const DataSet& dataset, const std::vector<Instance>& trainingSet, const std::vector<Instance>& testSet);
	}
}
