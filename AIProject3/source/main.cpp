// main.cpp

#include <iostream>
#include <chrono>
#include "../include/DataSets.h"
#include "../include/KNearestNeighbor.h"
#include "../include/ID3.h"

using IAlgorithm = std::size_t(const ml::DataSet& database, const std::vector<ml::Instance>& trainingSet, const std::vector<ml::Instance>& testSet);

void run_algorithm(const ml::DataSet& dataset, IAlgorithm* algorithm)
{
	constexpr std::size_t NUM_FOLDS = 10;
	const std::size_t foldSize = dataset.num_instances() / NUM_FOLDS;
	std::vector<std::size_t> results;
	results.assign(NUM_FOLDS, 0);

	// Get the time the benchmark started
	const auto start = std::chrono::high_resolution_clock::now();

	// Run the benchmark
	for (std::size_t i = 0; i < NUM_FOLDS; ++i)
	{
		std::vector<ml::Instance> trainingSet;
		trainingSet.reserve(foldSize * (NUM_FOLDS - 1));

		std::vector<ml::Instance> testSet;
		testSet.reserve(foldSize);

		for (std::size_t instance = 0; instance < foldSize * NUM_FOLDS; ++instance)
		{
			if (instance / foldSize == i)
			{
				testSet.push_back(dataset.get_instance(instance));
			}
			else
			{
				trainingSet.push_back(dataset.get_instance(instance));
			}
		}

		// Run the algorithm
		results[i] = algorithm(dataset, trainingSet, testSet);
	}

	// Get the time the benchmark ended
	const auto end = std::chrono::high_resolution_clock::now();

	// Print out the accuracy for each run
	for (auto result : results)
	{
		const auto accuracy = static_cast<float>(result * 100) / foldSize;
		std::cout << "Accuracy: " << accuracy << "%" << std::endl;
	}

	// Print out the total time elapsed in milliseconds
	const auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "Total elapsed time: " << elapsedTime.count() << "ms" << std::endl;
}

int main()
{
	// load it
	auto dataset = ml::load_soybean_data();

	// print it
	//for (std::size_t i = 0; i < dataset.num_instances(); ++i)
	//{
	//	auto instance = dataset.get_instance(i);

	//	for (ml::Attribute::Index attrib = 0; attrib < dataset.num_attributes(); ++attrib)
	//	{
	//		std::cout << dataset.get_attribute(attrib).value_name(instance.get_attrib(attrib)) << ", ";
	//	}

	//	std::cout << ": " << dataset.class_name(instance.get_class()) << std::endl;
	//}

	run_algorithm(dataset, &ml::id3::algorithm);

	std::cin.get();
}
