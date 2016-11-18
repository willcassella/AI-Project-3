// ID3.h - Will Cassella
#pragma once

#include <vector>

namespace ml
{
	struct DataSet;
	struct Instance;

	namespace id3_rep
	{
		/**
		 * \brief Runs the ID3 with reduceed error pruning algorithm.
		 * \param dataset The dataset to run ID3 on.
		 * \param trainingSet The training set to build the ID3 tree.
		 * \param testSet The set to calculate the accuracy of the ID3 tree on.
		 * \return The number of correctly inferred classes in the test set, this should be divided by the test set size to produce the percentage.
		 */
		std::size_t algorithm(const DataSet& dataset, const std::vector<Instance>& trainingSet, const std::vector<Instance>& testSet);
	}
}
