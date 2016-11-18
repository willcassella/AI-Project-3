// DataSets.cpp

#include <fstream>
#include <sstream>
#include <cstring>
#include "../include/DataSets.h"

namespace ml
{
	void load_data_set(DataSet& dataset, const char* path, bool classFirst)
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
			ClassIndex classIndex = 0;

			// If the first element of the CSV is the class
			if (classFirst)
			{
				std::getline(lineStream, value, ',');
				classIndex = dataset.class_index(value);
			}

			for (Attribute::Index attribIndex = 0; attribIndex < dataset.num_attributes(); ++attribIndex)
			{
				std::getline(lineStream, value, ',');

				auto valueIndex = dataset.get_attribute(attribIndex).value_index(value);
				attributes.push_back(valueIndex);
			}

			if (!classFirst)
			{
				std::getline(lineStream, value);
				classIndex = dataset.class_index(value);
			}

			lineStream.clear();

			// Add the instance to the dataset
			dataset.add_instance(classIndex, attributes);
			attributes.clear();
		}
	}

	DataSet load_house_votes_data()
	{
		// Initialize the dataset
		DataSet result{
			{ "republican", "democrat" },
			{
				Attribute{ "handicapped-infants", {"n", "y"} },
				Attribute{ "water-project-cost-sharing",{ "n", "y" } },
				Attribute{ "adoption-of-the-budget-resolution",{ "n", "y" } },
				Attribute{ "position-fee-freeze",{ "n", "y" } },
				Attribute{ "el-salvador-aid",{ "n", "y" } },
				Attribute{ "religious-groups-in-schools",{ "n", "y" } },
				Attribute{ "anti-satellite-test-ban",{ "n", "y" } },
				Attribute{ "aid-to-nicaraguan-contras",{ "n", "y" } },
				Attribute{ "mx-missles",{ "n", "y" } },
				Attribute{ "immigration",{ "n", "y" } },
				Attribute{ "synfuels-corporation-cutback",{ "n", "y" } },
				Attribute{ "education-spending",{ "n", "y" } },
				Attribute{ "superfund-right-to-sue",{ "n", "y" } },
				Attribute{ "crime",{ "n", "y" } },
				Attribute{ "duty-free-exports",{ "n", "y" } },
				Attribute{ "export-administration-act-south-africa",{ "n", "y" } }
			}};

		// Load from file
		load_data_set(result, "data/test.data.txt", true);
		result.finalize();
		return result;
	}

	DataSet load_breast_cancer_data()
	{
		DataSet result{
			{ "2", "4" },
			{
				Attribute{},
				Attribute{ "Clump Thickness", {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10" } },
				Attribute{ "Uniformity of Cell Size", { "1", "2", "3", "4", "5", "6", "7", "8", "9", "10" } },
				Attribute{ "Uniformity of Cell Shape", { "1", "2", "3", "4", "5", "6", "7", "8", "9", "10" } },
				Attribute{ "Marginal Adhesion", { "1", "2", "3", "4", "5", "6", "7", "8", "9", "10" } },
				Attribute{ "Single Epithelial Cell Size", { "1", "2", "3", "4", "5", "6", "7", "8", "9", "10" } },
				Attribute{ "Bare Nuclei",{ "1", "2", "3", "4", "5", "6", "7", "8", "9", "10" } },
				Attribute{ "Bland Chlomatin", { "1", "2", "3", "4", "5", "6", "7", "8", "9", "10" } },
				Attribute{ "Normal Nucleoli", { "1", "2", "3", "4", "5", "6", "7", "8", "9", "10" } },
				Attribute{ "Mitoses", { "1", "2", "3", "4", "5", "6", "7", "8", "9", "10" } }
			}
		};

		load_data_set(result, "data/breast-cancer-wisconsin.data.txt", false);
		result.finalize();
		return result;
	}
}
