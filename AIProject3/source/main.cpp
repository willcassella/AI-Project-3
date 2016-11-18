// main.cpp

#include <iostream>
#include <chrono>
#include "../include/DataSets.h"
#include "../include/KNearestNeighbor.h"
#include "../include/ID3.h"
#include <algorithm>
#include <numeric>

using IAlgorithm = std::size_t(const ml::DataSet& database, const std::vector<ml::Instance>& trainingSet, const std::vector<ml::Instance>& testSet);

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

	// Get the time the benchmark started
	const auto start = std::chrono::high_resolution_clock::now();

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
		results[i] = algorithm(dataset, trainingSet, testSet);
	}

	// Get the time the benchmark ended
	const auto end = std::chrono::high_resolution_clock::now();
	float averageAccuracy = 0;

	// Print out the accuracy for each run
	for (std::size_t i = 0; i < results.size(); ++i)
	{
		const auto accuracy = static_cast<float>(results[i] * 100) / foldSize;
		averageAccuracy += accuracy;
		std::cout << "Run " << i << " accuracy: " << accuracy << "%" << std::endl;
	}

	std::cout << "Average accuracy: " << averageAccuracy / results.size() << "%" << std::endl;

	// Print out the total time elapsed in milliseconds
	const auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "Total elapsed time: " << elapsedTime.count() << "ms" << std::endl << std::endl;
}

int main()
{
	// load the dataset
	auto dataset = ml::load_soybean_data();

	// Run the nearest neighbor algorithm
	std::cout << "Nearest Neighbor:" << std::endl;
	run_algorithm(dataset, &ml::k_nearest_neighbor::algorithm);

	// Run the ID3 algorithm
	std::cout << "ID3:" << std::endl;
	run_algorithm(dataset, &ml::id3::algorithm);

	std::cin.get();
}
