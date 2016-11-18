// main.cpp

#include <iostream>
#include "../include/DataSets.h"
#include "../include/KNearestNeighbor.h"

using IAlgorithm = std::size_t(const ml::DataSet& database, const std::vector<ml::Instance>& trainingSet, const std::vector<ml::Instance>& testSet);

void run_algorithm(const ml::DataSet& dataset, IAlgorithm* algorithm)
{
	const std::size_t foldSize = dataset.num_instances() / 10;

	for (std::size_t i = 0; i < 10; ++i)
	{
		std::vector<ml::Instance> trainingSet;
		trainingSet.reserve(foldSize * 9);

		std::vector<ml::Instance> testSet;
		testSet.reserve(foldSize);

		for (std::size_t instance = 0; instance < foldSize * 10; ++instance)
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
		auto numCorrect = algorithm(dataset, trainingSet, testSet);
		std::cout << (float)numCorrect / testSet.size() << std::endl;
	}
}

int main()
{
	// load it
	auto dataset = ml::load_house_votes_data_set();

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

	run_algorithm(dataset, &ml::k_nearest_neighbor::algorithm);

	std::cin.get();
}
