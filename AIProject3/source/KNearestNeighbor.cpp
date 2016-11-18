// KNearestNeighbor.cpp

#include <map>
#include <limits>
#include <future>
#include "../include/KNearestNeighbor.h"
#include "../include/DataSet.h"

namespace ml
{
	namespace k_nearest_neighbor
	{
		using AttributeCPCache = std::vector<float>;

		/* Produces an array that contains the conditional probability for all values of the specified attribute across all classes. */
		AttributeCPCache attribute_conditional_probability(
			const std::vector<Instance>& trainingSet,
			const Attribute::Index attribIndex,
			const std::size_t attribDomainSize,
			const std::size_t numClasses)
		{
			// A two-dimensional vector representing the count of instances in each class that have each value of the attribute
			AttributeCPCache result;
			result.assign(attribDomainSize * numClasses, 0.f);

			// Vector containing the N(a,x) for each value
			std::vector<std::size_t> valueCount;
			valueCount.assign(attribDomainSize, 0);

			// Count up all the values in the training set
			for (const auto instance : trainingSet)
			{
				// Get the class this instance is a member of
				const auto classIndex = instance.get_class();

				// Get the value this instance has for the specified attribute
				const auto valueIndex = instance.get_attrib_value(attribIndex);

				// Increment the number of times we've seen this value
				valueCount[valueIndex] += 1;

				// Increment the number of times we've seen this value on this class
				result[valueIndex * numClasses + classIndex] += 1.f;
			}

			// Divide to get the probability
			for (std::size_t valueIndex = 0; valueIndex < attribDomainSize; ++valueIndex)
			{
				const auto count = valueCount[valueIndex];

				// First check to make sure we actually found instances of this value
				if (count == 0)
				{
					continue;
				}

				const auto startIndex = valueIndex * numClasses;
				for (std::size_t classIndex = 0; classIndex < numClasses; ++classIndex)
				{
					// Divide
					result[startIndex + classIndex] /= valueCount[valueIndex];
				}
			}

			return result;
		}

		using AttributeVDM = std::vector<float>;

		AttributeVDM attribute_value_difference_metric(
			const std::vector<Instance>& trainingSet,
			const AttributeCPCache& cpCache,
			const Attribute::Index attribIndex,
			const std::size_t numClasses,
			const Attribute::ValueIndex attribValue,
			const int q)
		{
			AttributeVDM result;
			result.assign(trainingSet.size(), 0.f);

			for (std::size_t i = 0; i < trainingSet.size(); ++i)
			{
				const auto trainingValueIndex = trainingSet[i].get_attrib_value(attribIndex);

				// Sum up the conditional probability differences for the attribute and the training value
				for (std::size_t classIndex = 0; classIndex < numClasses; ++classIndex)
				{
					result[i] += (cpCache[attribValue * numClasses + classIndex] - cpCache[trainingValueIndex * numClasses + classIndex]);
				}

				// Set it to the power of 'q'
				result[i] = std::pow(result[i], q);
			}

			return result;
		}

		struct VDMCache
		{
			///////////////////
			///   Methods   ///
		public:

			void init(
				const DataSet& dataset,
				const std::vector<Instance>& trainingSet)
			{
				const auto numAttributes = dataset.num_attributes();
				_attribute_conditional_probabilities.reserve(numAttributes);

				// Fill up the AttributeCPCaches for each attribute
				std::vector<std::future<AttributeCPCache>> results;
				results.reserve(numAttributes);

				// Queue up all the attributes
				for (std::size_t i = 0; i < numAttributes; ++i)
				{
					results.push_back(std::async(
						std::launch::async,
						attribute_conditional_probability,
						trainingSet,
						i,
						dataset.num_attrib_values(i),
						dataset.num_classes()));
				}

				// Retreive the results
				for (auto& result : results)
				{
					_attribute_conditional_probabilities.push_back(result.get());
				}
			}

			std::size_t classify(
				const DataSet& dataset,
				const std::vector<Instance>& trainingSet,
				const Instance instance,
				const unsigned int k) const
			{
				// Calculate the VDM for each attribute against each instance in the training set
				const auto numAttributes = dataset.num_attributes();
				std::vector<std::future<AttributeVDM>> attributeDifferences;
				attributeDifferences.reserve(numAttributes);

				// Queue up all the attributes
				for (std::size_t i = 0; i < numAttributes; ++i)
				{
					attributeDifferences.push_back(std::async(
						std::launch::async,
						attribute_value_difference_metric,
						trainingSet,
						_attribute_conditional_probabilities[i],
						i,
						dataset.num_classes(),
						instance.get_attrib_value(i),
						1));
				}

				// Build a vector to hold the results
				std::vector<AttributeVDM> results;
				results.reserve(numAttributes);

				for (auto& vdm : attributeDifferences)
				{
					results.push_back(vdm.get());
				}

				// Find the common class among k nearest neighbors
				return classify_impl(trainingSet, results, k);
			}

		private:

			using Neighbor = std::pair<float, std::size_t>;

			static std::size_t classify_impl(
				const std::vector<Instance>& trainingSet,
				const std::vector<AttributeVDM>& attributeDifferences,
				const unsigned int k)
			{
				std::vector<Neighbor> nearestNeighbors;
				nearestNeighbors.reserve(k);

				// For each element of the training set
				for (std::size_t i = 0; i < trainingSet.size(); ++i)
				{
					float distance = 0;

					// For each attribute
					for (Attribute::Index attribIndex = 0; attribIndex < attributeDifferences.size(); ++i)
					{
						// Add the attribute's difference metric and square it (distance function)
						distance += std::pow(attributeDifferences[attribIndex][i], 2);
					}

					// Take the square root to get the distance
					distance = std::sqrt(distance);

					// Add the current training set instance to the nearest neighbor vector if it's closer than any of the current ones
					insert_if_closer(nearestNeighbors, std::make_pair(distance, trainingSet[i].get_class()), k);
				}

				return most_common_class(nearestNeighbors);
			}

			static void insert_if_closer(
				std::vector<Neighbor>& nearestNeighbors,
				const Neighbor candidate,
				const unsigned int k)
			{
				// If we don't already have k neighbors
				if (nearestNeighbors.size() < k)
				{
					// Just add it
					nearestNeighbors.push_back(candidate);
					return;
				}

				// Find the nearest neighbor that this one is closer than
				auto beaten = nearestNeighbors.end();
				for (auto iter = nearestNeighbors.begin(); iter < nearestNeighbors.end(); ++iter)
				{
					// Don't bother if this one is further
					if (iter->first > candidate.first)
					{
						continue;
					}

					// If we haven't beaten one yet, or the one we've beaten is closer than this one (want to push out the furthest ones)
					if (beaten == nearestNeighbors.end() || beaten->first < iter->first)
					{
						beaten = iter;
					}
				}

				// If the candidate beat one of the current nearest neighbors
				if (beaten != nearestNeighbors.end())
				{
					*beaten = candidate;
				}
			}

			static std::size_t most_common_class(
				const std::vector<Neighbor>& nearestNeighbors)
			{
				std::map<std::size_t, std::size_t> classCounts;

				// Count up all the classes
				for (auto neighbor : nearestNeighbors)
				{
					classCounts[neighbor.second] += 1;
				}

				// Figure out which one has the most occurrences
				std::size_t classIndex = 0;
				std::size_t occurrences = 0;

				for (auto iter : classCounts)
				{
					if (iter.second > occurrences)
					{
						classIndex = iter.first;
					}
				}

				return classIndex;
			}

			//////////////////
			///   Fields   ///
		private:

			std::vector<AttributeCPCache> _attribute_conditional_probabilities;
		};
	}
}
