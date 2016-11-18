// ID3.h
#pragma once

#include <memory>
#include <vector>

namespace ml
{
	struct DataSet;
	struct Instance;

	namespace id3
	{
		/**
		 * \brief Calculates the entropy of an attribute among the given set of values.
		 * \param dataset The dataset that holds all the attribute information.
		 * \param instanceSet The set of instances that we're calculating the information gain among.
		 * \param attribIndex The attribute we're testing the entropy of.
		 * \return The entropy value.
		 */
		float calculate_entropy(
			const DataSet& dataset,
			const std::vector<Instance>& instanceSet,
			std::size_t attribIndex);

		struct Node
		{
			//////////////////
			///   Fields   ///
		public:

			/**
			 * \brief The class index represented by this node. If this node is not a leaf, this field is meaningless.
			 */
			std::size_t class_index;

			/**
			 * \brief The children of this node. If this is empty, you should consider this node a leaf node and check it's 'class_index' field.
			 */
			std::vector<std::unique_ptr<Node>> children;
		};
	}
}
