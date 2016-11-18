// DataSet.h
#pragma once

#include <vector>
#include <string>
#include <cassert>

namespace ml
{
	struct DataSet;
	using ClassIndex = std::size_t;

	/* Represents an attribute, including its name and domain of values. */
	struct Attribute
	{
		using Index = std::size_t;
		using ValueIndex = std::size_t;

		///////////////////
		///   Methods   ///
	public:

		ValueIndex value_index(const std::string& value) const
		{
			// Check if we're getting an unknown
			if (value == "?")
			{
				return std::rand() % domain.size();
			}

			// Search for the element of the domain
			auto iter = std::find(domain.begin(), domain.end(), value);

			// Make sure we found something
			assert(iter != domain.end());
			return iter - domain.begin();
		}

		const std::string& value_name(ValueIndex valueIndex) const
		{
			return domain.at(valueIndex);
		}

		//////////////////
		///   Fields   ///
	public:

		/* The name of the attribute. */
		std::string name;

		/* The domain of values it may take on. */
		std::vector<std::string> domain;

		std::vector<ValueIndex> instance_values;
	};

	/* Represents an instance of a value in the dataset. */
	struct Instance
	{
		////////////////////////
		///   Constructors   ///
	public:

		Instance(const DataSet& dataset, const std::size_t index)
			: _dataset(&dataset),
			_index(index)
		{
		}

		///////////////////z
		///   Methods   ///
	public:

		ClassIndex get_class() const;

		Attribute::ValueIndex get_attrib(Attribute::Index attribIndex) const;

		//////////////////
		///   Fields   ///
	private:

		const DataSet* _dataset;
		std::size_t _index;
	};

	/* Represents a complete set of data. */
	struct DataSet
	{
		friend struct Instance;

		////////////////////////
		///   Constructors   ///
	public:

		DataSet(std::vector<std::string> classes, std::vector<Attribute> attributes)
			: _classes(std::move(classes)),
			_attributes(std::move(attributes))
		{
		}

		///////////////////
		///   Methods   ///
	public:

		/* Returns the number of classes in this dataset. */
		std::size_t num_classes() const
		{
			return _classes.size();
		}

		/* Returns the number of attributes in this dataset. */
		std::size_t num_attributes() const
		{
			return _attributes.size();
		}

		/* Returns the number of instances in this dataset. */
		std::size_t num_instances() const
		{
			return _instance_classes.size();
		}

		const Attribute& get_attribute(std::size_t attribIndex) const
		{
			return _attributes[attribIndex];
		}

		ClassIndex class_index(const std::string& className) const
		{
			auto iter = std::find(_classes.begin(), _classes.end(), className);
			assert(iter != _classes.end());
			return iter - _classes.begin();
		}

		const std::string& class_name(ClassIndex index) const
		{
			return _classes.at(index);
		}

		/**
		 * \brief Returns the indexed dataset instance.
		 * \param index The index of the dataset instance.
		 */
		Instance get_instance(std::size_t index) const
		{
			return Instance{ *this, index };
		}

		void add_instance(ClassIndex classIndex, const std::vector<Attribute::ValueIndex>& attributes)
		{
			_instance_classes.push_back(classIndex);

			for (std::size_t attribIndex = 0; attribIndex < _attributes.size(); ++attribIndex)
			{
				_attributes[attribIndex].instance_values.push_back(attributes[attribIndex]);
			}
		}

		//////////////////
		///   Fields   ///
	private:

		std::vector<std::string> _classes;
		std::vector<Attribute> _attributes;
		std::vector<ClassIndex> _instance_classes;
	};

	inline ClassIndex Instance::get_class() const
	{
		return _dataset->_instance_classes[_index];
	}

	inline Attribute::ValueIndex Instance::get_attrib(Attribute::Index attribIndex) const
	{
		return _dataset->get_attribute(attribIndex).instance_values[_index];
	}
}
