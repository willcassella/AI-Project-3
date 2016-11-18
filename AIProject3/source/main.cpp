// main.cpp

#include <iostream>
#include "../include/DataSets.h"

int main()
{
	// load it
	auto dataset = ml::load_house_votes_data_set();

	// print it
	for (std::size_t i = 0; i < dataset.num_instances(); ++i)
	{
		auto instance = dataset.get_instance(i);
		instance.enumerate_attributes([&dataset](std::size_t attribIndex, std::size_t valueIndex)
		{
			std::cout << dataset.attrib_value(attribIndex, valueIndex) << ",";
		});

		std::cout << std::endl;
	}

	std::cin.get();
}
