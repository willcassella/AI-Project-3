// DataSets.h
#pragma once

#include "DataSet.h"

namespace ml
{
	/**
	 * \brief Loads the breast cancer data set.
	 * \return
	 */
	DataSet load_breast_cancer_data();

	/**
	 * \brief Loads the glass data set.
	 */
	DataSet load_glass_data();

	/**
	 * \brief Loads the houes votes data set.
	 */
	DataSet load_house_votes_data();

	/**
	 * \brief Loads the Iris data set.
	 */
	DataSet load_iris_data();

	/**
	 * \brief Loads the soybean data set.
	 */
	DataSet load_soybean_data();
}
