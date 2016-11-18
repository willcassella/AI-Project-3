// DataSets.cpp

#include <fstream>
#include <sstream>
#include <cstring>
#include "../include/DataSets.h"

namespace ml
{
	void load_data_set(DataSet& dataset, const char* path)
	{
		std::ifstream file{ path, std::ios::in };

		// Load all instances from the file
		std::string value;
		std::vector<std::size_t> instances;
		instances.reserve(dataset.num_attributes());

		std::string line;
		std::stringstream lineStream;
		while (std::getline(file, line))
		{
			lineStream << line;

			for (std::size_t attribIndex = 0; std::getline(lineStream, value, ','); attribIndex = (attribIndex + 1) % dataset.num_attributes())
			{
				auto valueIndex = dataset.attrib_value_index(attribIndex, value);
				instances.push_back(valueIndex);
			}

			lineStream.clear();
		}

		// Add them to the dataset
		dataset.set_instances(std::move(instances));
	}

	DataSet load_house_votes_data_set()
	{
		// Initialize the dataset
		DataSet result{
			{
				Attribute{ "class", {"democrat", "republican"} },
				Attribute{ "handicapped-infants", {"y", "n"} },
				Attribute{ "water-project-cost-sharing", {"y", "n"} },
				Attribute{ "adoption-of-the-budget-resolution", {"y", "n"} },
				Attribute{ "position-fee-freeze", {"y", "n"} },
				Attribute{ "el-salvador-aid", {"y", "n"} },
				Attribute{ "religious-groups-in-schools", {"y", "n"} },
				Attribute{ "anti-satellite-test-ban", {"y", "n"} },
				Attribute{ "aid-to-nicaraguan-contras", {"y", "n"} },
				Attribute{ "mx-missles", {"y", "n"} },
				Attribute{ "immigration", {"y", "n"} },
				Attribute{ "synfuels-corporation-cutback", {"y", "n"} },
				Attribute{ "education-spending", {"y", "n"} },
				Attribute{ "superfund-right-to-sue", {"y", "n"} },
				Attribute{ "crime", {"y", "n"} },
				Attribute{ "duty-free-exports", {"y", "n"} },
				Attribute{ "export-administration-act-south-africa", {"y", "n"} }
			}, 0};

		// Load from file
		load_data_set(result, "data/house-votes-84.data.txt");
		return result;
	}
}
