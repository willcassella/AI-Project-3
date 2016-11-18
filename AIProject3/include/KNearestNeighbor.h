// KNearestNeighbor.h
#pragma once

#include <vector>

namespace ml
{
	struct Instance;

	namespace k_nearest_neighbor
	{
		/**
		 * \brief Calculates the value difference metric for the given attribute values on the dataset.
		 * \param trainingSet The training set to calculate the VDM on.
		 * \param attribIndex The index of the attribute we are calculating the VDM for.
		 * \param valueX The first attribute value to calculate the VDM for.
		 * \param valueY The second attribute value to calcuate the VDM for.
		 * \return The VDM of the two values.
		 */
		float value_difference_metric(
			const std::vector<Instance>& trainingSet,
			std::size_t attribIndex,
			std::size_t valueX,
			std::size_t valueY);
	}
}
