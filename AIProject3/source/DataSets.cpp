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
		std::vector<Attribute::ValueIndex> attributes;
		attributes.reserve(dataset.num_attributes());

		std::string line;
		std::stringstream lineStream;
		while (std::getline(file, line))
		{
			lineStream << line;
			std::string value;

			// The first element of the CSV is the class
			std::getline(lineStream, value, ',');
			ClassIndex classIndex = dataset.class_index(value);

			for (Attribute::Index attribIndex = 0; std::getline(lineStream, value, ','); ++attribIndex)
			{
				assert(attribIndex < dataset.num_attributes());

				auto valueIndex = dataset.get_attribute(attribIndex).value_index(value);
				attributes.push_back(valueIndex);
			}

			lineStream.clear();

			// Add the instance to the dataset
			dataset.add_instance(classIndex, attributes);
			attributes.clear();
		}
	}

	DataSet load_house_votes_data_set()
	{
		// Initialize the dataset
		DataSet result{
			{ "democrat", "republican" },
			{
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
			}};

		// Load from file
		load_data_set(result, "data/house-votes-84.data.txt");
		return result;
	}
}
