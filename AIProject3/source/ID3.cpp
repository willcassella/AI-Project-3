// ID3.cpp

#include <cmath>
#include "../include/ID3.h"
#include "../include/DataSet.h"

namespace ml
{
	namespace id3
	{
		/* Calculates the entropy */
		template <typename PredFnT>
		float calculate_entropy(
			const std::vector<Instance>& subset,
			const std::size_t classAttribIndex,
			const std::size_t classAttribSize,
			PredFnT&& pred)
		{
			float entropy = 0;

			// Array of all
			std::vector<std::size_t> classCounter{ classAttribSize, 0 };

			for (const auto& instance : subset)
			{
				// If the value for the attribute we're splitting on matches the value we're looking for
				if (instance.get_attrib_value(splitAttribIndex) == splitAttribValue)
				{
					// Increment the counter for the class this instance is a member of
					classCounter[instance.get_attrib_value(classAttribIndex)] += 1;
				}
			}

			for (auto count : classCounter)
			{
				if (count == 0)
				{
					continue;
				}

				const float proportion = static_cast<float>(count) / classAttribSize;
				entropy += proportion * std::log2(proportion);
			}

			return -entropy;
		}

		float calculate_information_gain(
			const std::vector<Instance>& subset,
			const std::size_t classAttribIndex,
			const std::size_t classAttribSize,
			const std::size_t splitAttribIndex,
			const std::size_t splitAttribSize)
		{
			return 0;
		}
	}
}
