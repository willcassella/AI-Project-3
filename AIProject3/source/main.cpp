// main.cpp - Will Cassella

#include <algorithm>
#include <iostream>
#include <numeric>
#include "../include/DataSets.h"
#include "../include/KNearestNeighbor.h"
#include "../include/ID3.h"

/* An algorithm is just a function with the following signature: */
using IAlgorithm = std::size_t(const ml::DataSet& database, const std::vector<ml::Instance>& trainingSet, const std::vector<ml::Instance>& testSet);

/**
 * \brief Runs the given algorithm on the given dataset, by generating 10 cross folds.
 * \param dataset The dataset being tested on.
 * \param algorithm The algorithm to run.
 */
void run_algorithm(const ml::DataSet& dataset, IAlgorithm* algorithm)
{
	constexpr std::size_t NUM_FOLDS = 10;
	const std::size_t foldSize = dataset.num_instances() / NUM_FOLDS;
	std::vector<std::size_t> results;
	results.assign(NUM_FOLDS, 0);

	// Create a vector to index into the dataset
	std::vector<std::size_t> indexVec;
	indexVec.assign(NUM_FOLDS * foldSize, 0);
	std::iota(indexVec.begin(), indexVec.end(), 0);
	std::random_shuffle(indexVec.begin(), indexVec.end());

	// Run the benchmark
	for (std::size_t i = 0; i < NUM_FOLDS; ++i)
	{
		std::vector<ml::Instance> trainingSet;
		trainingSet.reserve(foldSize * (NUM_FOLDS - 1));

		std::vector<ml::Instance> testSet;
		testSet.reserve(foldSize);

		for (std::size_t index = 0; index < indexVec.size(); ++index)
		{
			if (index / foldSize == i)
			{
				testSet.push_back(dataset.get_instance(indexVec[index]));
			}
			else
			{
				trainingSet.push_back(dataset.get_instance(indexVec[index]));
			}
		}

		// Run the algorithm
		std::cout << "Run " << i << ":" << std::endl;
		results[i] = algorithm(dataset, trainingSet, testSet);
	}

	// Determine the average accuracy
	float averageAccuracy = 0;
	for (std::size_t i = 0; i < results.size(); ++i)
	{
		const auto accuracy = static_cast<float>(results[i] * 100) / foldSize;
		averageAccuracy += accuracy;
	}

	std::cout << "Average accuracy: " << averageAccuracy / results.size() << "%" << std::endl;
}

int main()
{
	// load the dataset
	auto dataset = ml::load_breast_cancer_data();

	// Run the nearest neighbor algorithm
	std::cout << "Nearest Neighbor:" << std::endl;
	run_algorithm(dataset, &ml::k_nearest_neighbor::algorithm);

	// Run the ID3 algorithm
	std::cout << std::endl << "ID3:" << std::endl;
	run_algorithm(dataset, &ml::id3_rep::algorithm);

	std::cin.get();
}
