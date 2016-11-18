// DataSet.h
#pragma once

#include <vector>
#include <string>
#include <cassert>

namespace ml
{
	/* Represents an attribute, including its name and domain of values. */
	struct Attribute
	{
		//////////////////
		///   Fields   ///
	public:

		/* The name of the attribute. */
		std::string name;

		/* The domain of values it may take on. */
		std::vector<std::string> domain;
	};

	/* Represents an instance of a value in the dataset. */
	struct Instance
	{
		////////////////////////
		///   Constructors   ///
	public:

		Instance(const std::size_t* offset)
			: _instance_offset(offset)
		{
		}

		///////////////////
		///   Methods   ///
	public:

		std::size_t get_attrib_value(std::size_t attribIndex) const
		{
			return _instance_offset[attribIndex];
		}

		//////////////////
		///   Fields   ///
	private:

		const std::size_t* _instance_offset;
	};

	/* Represents a complete set of data. */
	struct DataSet
	{
		////////////////////////
		///   Constructors   ///
	public:

		DataSet(std::vector<Attribute> attributes, std::size_t classAttribIndex)
			: _class_attrib_index(classAttribIndex), _attributes(std::move(attributes))
		{
		}

		///////////////////
		///   Methods   ///
	public:

		/* Returns the number of instances in this dataset. */
		std::size_t num_instances() const
		{
			return _instances.size() / _attributes.size();
		}

		/* Returns the number of attributes in this dataset. */
		std::size_t num_attributes() const
		{
			return _attributes.size();
		}

		/* Returns the attribute index that corresponds to the classification. */
		std::size_t class_attrib_index() const
		{
			return _class_attrib_index;
		}

		/* Given an attribute index, returns the name of the attribute. */
		const std::string& attrib_name(std::size_t attribIndex) const
		{
			return _attributes.at(attribIndex).name;
		}

		/* Given an attribute index and a vale index within that attribute, returns the name of the value. */
		const std::string& attrib_value(std::size_t attribIndex, std::size_t valueIndex) const
		{
			return _attributes.at(attribIndex).domain.at(valueIndex);
		}

		std::size_t num_attrib_values(std::size_t attribIndex) const
		{
			return _attributes[attribIndex].domain.size();
		}

		/**
		 * \brief Returns the value index for the value of the indexed attribute.
		 * \param attribIndex The index of the attribute to search through.
		 * \param value The value to get the index for.
		 */
		std::size_t attrib_value_index(std::size_t attribIndex, const std::string& value) const
		{
			const auto& domain = _attributes[attribIndex].domain;

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

		const Attribute& get_attribute(std::size_t attribIndex) const
		{
			return _attributes[attribIndex];
		}

		/**
		 * \brief Returns the indexed dataset instance.
		 * \param index The index of the dataset instance.
		 */
		Instance get_instance(std::size_t index) const
		{
			return Instance{ _attributes.size(), &_instances[index * _attributes.size()] };
		}

		void set_instances(std::vector<std::size_t> instances)
		{
			_instances = std::move(instances);
		}

		//////////////////
		///   Fields   ///
	private:

		std::size_t _class_attrib_index;
		std::vector<Attribute> _attributes;
		std::vector<std::size_t> _instances;
	};
}
