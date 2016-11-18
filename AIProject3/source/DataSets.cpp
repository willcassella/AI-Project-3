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

	DataSet load_glass_data()
	{
		DataSet result{
			{ "1", "2", "3", "4", "5", "6", "7" },
			{
				Attribute{},
				Attribute::discretize("refractive index", 1.5112f, 1.5339f, 5),
				Attribute::discretize("sodium", 10.73f, 17.38f, 5),
				Attribute::discretize("magnesium", 0, 4.49f, 5),
				Attribute::discretize("aluminum", 0.29f, 3.5f, 5),
				Attribute::discretize("silicon", 69.81f, 75.41f, 5),
				Attribute::discretize("potassium", 0, 6.21f, 5),
				Attribute::discretize("calcium", 5.43f, 16.19f, 5),
				Attribute::discretize("barium", 0, 3.15f, 5),
				Attribute::discretize("iron", 0, 0.51f, 5)
			}
		};

		load_data_set(result, "data/glass.data.txt", false);
		result.finalize();
		return result;
	}

	DataSet load_house_votes_data()
	{
		// Initialize the dataset
		DataSet result{
			{ "republican", "democrat" },
			{
				Attribute{ "handicapped-infants",{ "n", "y" } },
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
			} };

		// Load from file
		load_data_set(result, "data/house-votes-84.data.txt", true);
		result.finalize();
		return result;
	}

	DataSet load_iris_data()
	{
		DataSet result{
			{ "Iris-virginica", "Iris-versicolor", "Iris-setosa" },
			{
				Attribute::discretize("sepal lenght", 4.3f, 7.9f, 10),
				Attribute::discretize("sepal width", 2.0f, 4.4f, 10),
				Attribute::discretize("petal length", 1.0f, 6.9f, 10),
				Attribute::discretize("petal width", 0.1f, 2.5f, 10)
			}
		};

		// Read from file
		load_data_set(result, "data/iris.data.txt", false);
		result.finalize();
		return result;
	}

	DataSet load_soybean_data()
	{
		DataSet result{
			{ "D1", "D2", "D3", "D4" },
			{
				Attribute{ "date", {"0", "1", "2", "3", "4", "5", "6", "7" } },
				Attribute{ "plant-stand", { "0", "1" } },
				Attribute{ "precip", {"0", "1", "2" } },
				Attribute{ "temp", { "0", "1", "2" } },
				Attribute{ "hail", { "0", "1" } },
				Attribute{ "crop-hist", { "0", "1", "2", "3" } },
				Attribute{ "area-damaged", { "0", "1", "2", "3" } },
				Attribute{ "severity", { "0", "1", "2" } },
				Attribute{ "seed-tmt", { "0", "1", "2" } },
				Attribute{ "germination", { "0", "1", "2" } },
				Attribute{ "plant-growth", { "0", "1" } },
				Attribute{ "leaves", { "0", "1" } },
				Attribute{ "leafspots-halo", { "0", "1", "2" } },
				Attribute{ "leafspots-marg", { "0", "1", "2" } },
				Attribute{ "leafspot-size", { "0", "1", "2" } },
				Attribute{ "leaf-shread", { "0", "1" } },
				Attribute{ "leaf-malf", { "0", "1" } },
				Attribute{ "leaf-mild", { "0", "1", "2" } },
				Attribute{ "stem", { "0", "1" } },
				Attribute{ "lodging", { "0", "1", } },
				Attribute{ "stem-crankers", { "0", "1", "2", "3" } },
				Attribute{ "cranker-lesion", { "0", "1", "2", "3" } },
				Attribute{ "fruiting-bodies", { "0", "1" } },
				Attribute{ "external decay", { "0", "1", "2" } },
				Attribute{ "mycelium", { "0", "1" } },
				Attribute{ "int-discolor", { "0", "1", "2" } },
				Attribute{ "sclerotia", { "0", "1" } },
				Attribute{ "fruit-pods", { "0", "1", "2", "3" } },
				Attribute{ "fuit spots", { "0", "1", "2", "3", "4" } },
				Attribute{ "seed", { "0", "1" } },
				Attribute{ "mold-growth", { "0", "1" } },
				Attribute{ "seed-discolor", { "0", "1" } },
				Attribute{ "seed-size", { "0", "1" } },
				Attribute{ "srhiveling", { "0", "1" } },
				Attribute{ "roots", { "0", "1", "2" } }
			}
		};

		// Read from file
		load_data_set(result, "data/soybean-small.data.txt", false);
		result.finalize();
		return result;
	}
}
