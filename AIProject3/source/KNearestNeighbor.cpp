// KNearestNeighbor.cpp

#include <future>
#include "../include/KNearestNeighbor.h"
#include "../include/DataSet.h"

namespace ml
{
	namespace k_nearest_neighbor
	{
		struct VDMCache
		{
			using AttributeVDM = std::vector<float>;

			///////////////////
			///   Methods   ///
		public:

			void init(
				const DataSet& dataset,
				const std::vector<Instance>& trainingSet)
			{
				const auto numAttributes = dataset.num_attributes();

				_attribute_vdms.reserve(numAttributes);

				// Fill up the VDMs for each attribute
				std::vector<std::future<AttributeVDM>> results;
				results.reserve(numAttributes);

				for (std::size_t i = 0; i < LC_NUMERIC)
				{

				}
			}

			//////////////////
			///   Fields   ///
		private:

			std::vector<AttributeVDM> _attribute_vdms;
		};

		/* Produces an array that contains the conditional probability for all values of the specified attribute across all classes. */
		VDMCache::AttributeVDM conditional_probability(
			const std::vector<Instance>& trainingSet,
			const std::size_t classAttribIndex,
			const std::size_t numClasses,
			const std::size_t attribIndex,
			const std::size_t attribSize)
		{
			// A two-dimensional vector representing the count of instances in each class that have each value of the attribute
			std::vector<float> result;
			result.assign(attribSize * numClasses, 0.f);

			// Vector containing the N(a,x) for each value
			std::vector<std::size_t> valueCount;
			valueCount.assign(attribSize, 0);

			// Count up all the values in the training set
			for (const auto& instance : trainingSet)
			{
				// Get the class this instance is a member of
				const auto classValue = instance.get_attrib_value(classAttribIndex);

				// Get the value this instance has for the specified attribute
				const auto value = instance.get_attrib_value(attribIndex);

				// Increment the number of times we've seen this value
				valueCount[value] += 1;

				// Increment the number of times we've seen this value on this class
				result[value * numClasses + classValue] += 1.f;
			}

			// Divide to get the probability
			for (std::size_t value = 0; value < attribSize; ++value)
			{
				const auto count = valueCount[value];

				// First check to make sure we actually found instances of this value
				if (count == 0)
				{
					continue;
				}

				const auto startIndex = value * numClasses;
				for (std::size_t classI = 0; classI < numClasses; ++classI)
				{
					// Divide
					result[startIndex + classI] /= valueCount[value];
				}
			}

			return result;
		}

		float value_difference_metric(
			const std::vector<Instance>& trainingSet,
			std::size_t classAttribIndex,
			std::size_t classAttribSize,
			std::size_t attribIndex,
			std::size_t valueX,
			std::size_t valueY)
		{
			// Array of the count of all instances in each class that have 'valueX' for the given attribute
			std::vector<std::size_t> xClassCount;
			xClassCount.assign(classAttribSize, 0);

			// The number of all instances that have 'valueX' for the given attribute
			std::size_t xCount = 0;

			// Array of the count of all instances in each class that have 'valueY' for the given attribute.
			std::vector<std::size_t> yClassCount = xClassCount;

			// The number of all intances that have 'valueY' for the given attribute
			std::size_t yCount = 0;

			for (auto instance : trainingSet)
			{
				if ()
			}
		}
	}
}
