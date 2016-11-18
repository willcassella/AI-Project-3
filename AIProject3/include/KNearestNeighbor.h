// KNearestNeighbor.h - Will Cassella
#pragma once

#include <vector>

namespace ml
{
	struct Instance;
	struct DataSet;

	namespace k_nearest_neighbor
	{
		/**
		 * \brief Runs the k nearest neighbor algorithm.
		 * \param dataset The dataset to run the algorithm on.
		 * \param trainingSet The set to train the K nearest neighbor data with.
		 * \param testSet The set to test the accuracy of the algorithm against.
		 * \return The number of correctly inferred classes in the test set.
		 */
		std::size_t algorithm(const DataSet& dataset, const std::vector<Instance>& trainingSet, const std::vector<Instance>& testSet);
	}
}
