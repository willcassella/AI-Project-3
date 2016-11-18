// ID3.cpp

#include <cmath>
#include <tuple>
#include <future>
#include <numeric>
#include <limits>
#include "../include/ID3.h"
#include "../include/DataSet.h"

namespace ml
{
	namespace id3
	{
		struct Node
		{
			//////////////////
			///   Fields   ///
		public:

			/**
			* \brief The class index represented by this node. If this node is not a leaf, this field represents the most common class.
			*/
			ClassIndex class_index = 0;

			/**
			 * \brief The attribute that is being used to split the children of this node. If this is a leaf node this field is not used.
			 */
			Attribute::Index split_attribute = 0;

			/**
			* \brief The children of this node. If this is empty, you should consider this node a leaf node and check it's 'class_index' field.
			*/
			std::vector<std::unique_ptr<Node>> children;
		};

		auto split_subset(
			const std::vector<Instance>& subset,
			const Attribute::Index attrib,
			const std::size_t attribDomainSize)
		{
			std::vector<std::vector<Instance>> result;
			result.assign(attribDomainSize, {});

			for (auto instance : subset)
			{
				result[instance.get_attrib(attrib)].push_back(instance);
			}

			return result;
		}

		/* Calculates the entropy of the subset, taking a predicate to filter the subset further. */
		template <typename PredFnT>
		std::pair<float, ClassIndex> calculate_entropy(
			const std::vector<Instance>& subset,
			const std::size_t numClasses,
			PredFnT&& pred)
		{
			float entropy = 0;
			std::size_t instanceCount = 0;

			// Array of all
			std::vector<std::size_t> classCounter;
			classCounter.assign(numClasses, 0);

			for (const auto& instance : subset)
			{
				// If we're considering this instance when calculating the entropy
				if (pred(instance))
				{
					instanceCount += 1;

					// Increment the counter for the class this instance is a member of
					classCounter[instance.get_class()] += 1;
				}
			}

			ClassIndex mostCommonClass = 0;
			std::size_t mostCommonClassCount = 0;

			// For each class that we counted up
			for (ClassIndex i = 0; i < classCounter.size(); ++i)
			{
				const std::size_t count = classCounter[i];

				if (count == 0)
				{
					continue;
				}

				if (count > mostCommonClassCount)
				{
					mostCommonClass = i;
				}

				const float proportion = static_cast<float>(count) / instanceCount;
				entropy += proportion * std::log2(proportion);
			}

			return std::make_pair(-entropy, mostCommonClass);
		}

		/* Calculates the information gain by splitting the given subset on the given attribute. */
		float calculate_information_gain(
			const std::vector<Instance>& subset,
			const std::size_t numClasses,
			const float currentEntropy,
			const Attribute::Index splitAttribute,
			const std::size_t splitAttributeDomainSize)
		{
			float attribEntropy = 0;

			for (Attribute::ValueIndex value = 0; value < splitAttributeDomainSize; ++value)
			{
				// Keep track of the number of values in this
				float valueProportion = 0.f;
				auto predicate = [&valueProportion, value, splitAttribute](Instance instance)
				{
					if (instance.get_attrib(splitAttribute) == value)
					{
						valueProportion += 1.f;
						return true;
					}
					else
					{
						return false;
					}
				};

				// Calculate the entropy for this value's branch of the attribute
				const float valueEntropy = calculate_entropy(subset, numClasses, predicate).first;
				attribEntropy += valueProportion / subset.size() * valueEntropy;
			}

			return currentEntropy - attribEntropy;
		}

		void id3_recurse(
			const DataSet& dataset,
			std::vector<Instance> subset,
			std::vector<Attribute::Index> attributes,
			const Node* parent,
			Node& node)
		{
			// Get the current entropy of the node and most common class
			float entropy;
			auto pred = [](auto) {return true; };
			std::tie(entropy, node.class_index) = calculate_entropy(subset, dataset.num_classes(), pred);

			// If there is no entropy or no more attributes to select from
			if (entropy == 0 || attributes.empty())
			{
				// In the case that there was zero entropy because no instance remain, we need to set the class index to the parent most common class
				if (subset.empty())
				{
					node.class_index = parent->class_index;
				}

				return;
			}

			// There's still entropy and attributes to split by, so we need to recurse
			auto bestAttribute = attributes.begin();
			float bestAttributeInformationGain = std::numeric_limits<float>::lowest();

			for (auto iter = attributes.begin(); iter != attributes.end(); ++iter)
			{
				// Calculate the information gain by splitting on this attribute
				const auto informationGain = calculate_information_gain(
					subset,
					dataset.num_classes(),
					entropy,
					*iter,
					dataset.get_attribute(*iter).domain.size());

				// See if it's any better than what we've found so far
				if (informationGain > bestAttributeInformationGain)
				{
					bestAttributeInformationGain = informationGain;
					bestAttribute = iter;
				}
			}

			// Remove the attribute from the list of attributes
			auto attrib = *bestAttribute;
			attributes.erase(bestAttribute);
			node.split_attribute = attrib;

			// Get the domain size of the attribute we chose
			const auto attribSize = dataset.get_attribute(attrib).domain.size();

			// Recurse by splittin on the best attribute
			node.children.reserve(attribSize);

			for (auto childSubset : split_subset(subset, attrib, attribSize))
			{
				// Create a child node
				auto child = std::make_unique<Node>();
				auto* pChild = child.get();
				node.children.push_back(std::move(child));

				// Recurse on it
				id3_recurse(
					dataset,
					std::move(childSubset),
					attributes,
					&node,
					std::ref(*pChild));
			}
		}

		ClassIndex classify(
			const Node& node,
			Instance instance)
		{
			// If the current node is a leaf node
			if (node.children.empty())
			{
				// Return its classification
				return node.class_index;
			}

			return classify(*node.children[instance.get_attrib(node.split_attribute)], instance);
		}

		std::size_t algorithm(const DataSet& dataset, const std::vector<Instance>& trainingSet, const std::vector<Instance>& testSet)
		{
			// Build up a list of attributes
			std::vector<Attribute::Index> attributes;
			attributes.assign(dataset.num_attributes(), 0);
			std::iota(attributes.begin(), attributes.end(), 0);

			// Build the tree
			auto root = std::make_unique<Node>();
			id3_recurse(dataset, trainingSet, std::move(attributes), nullptr, *root);

			std::size_t numCorrect = 0;

			// Classify each value
			for (auto instance : testSet)
			{
				auto classIndex = classify(*root, instance);
				if (classIndex == instance.get_class())
				{
					numCorrect += 1;
				}
			}

			return numCorrect;
		}
	}
}
