// DataSet.h
#pragma once

#include <vector>
#include <string>
#include <iostream>
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

		////////////////////////
		///   Constructors   ///
	public:

		static Attribute discretize(std::string name, float min, float max, int segments)
		{
			Attribute result;
			result.name = std::move(name);
			result._discretized_min = min;
			result._discretized_max = max;

			const float distance = max - min;
			const float increment = distance / segments;

			result._discretized_segment_size = increment;

			for (int i = 0; i < segments; ++i)
			{
				float element = min + increment * i;
				result.domain.push_back(std::to_string(element));
			}

			return result;
		}

		Attribute() = default;
		Attribute(std::string name, std::vector<std::string> domain)
			: name(std::move(name)), domain(std::move(domain))
		{
		}

		///////////////////
		///   Methods   ///
	public:

		ValueIndex value_index(const std::string& value) const
		{
			// If this attribute is to be ignored
			if (domain.empty())
			{
				return 0;
			}

			// Check if we're getting an unknown
			if (value == "?")
			{
				return std::rand() % domain.size();
			}

			// If this attribute has been discretized
			if (_discretized_segment_size > 0)
			{
				// Convert the string to a numeric value
				float nvalue = std::strtof(value.c_str(), nullptr);

				// If it's below the minimum, just return the minimum
				if (nvalue < _discretized_min)
				{
					return 0;
				}

				// If it's above the maximum, just return the maximum
				if (nvalue >= _discretized_max)
				{
					return domain.size() - 1;
				}

				auto index = static_cast<std::size_t>((nvalue - _discretized_min) / _discretized_segment_size);
				assert(index < domain.size());
				return index;
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

	private:

		float _discretized_segment_size = 0;
		float _discretized_min = 0.f;
		float _discretized_max = 0.f;
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

		void print() const;

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

		void finalize()
		{
			for (std::size_t i = 0; i < _attributes.size();)
			{
				if (_attributes[i].domain.empty())
				{
					_attributes.erase(_attributes.begin() + i);
				}
				else
				{
					++i;
				}
			}
		}

		//////////////////
		///   Fields   ///
	private:

		std::vector<std::string> _classes;
		std::vector<Attribute> _attributes;
		std::vector<ClassIndex> _instance_classes;
	};

	inline void Instance::print() const
	{
		for (Attribute::Index i = 0; i < _dataset->num_attributes(); ++i)
		{
			std::cout << ", " << _dataset->get_attribute(i).value_name(get_attrib(i));
		}
	}

	inline ClassIndex Instance::get_class() const
	{
		return _dataset->_instance_classes[_index];
	}

	inline Attribute::ValueIndex Instance::get_attrib(Attribute::Index attribIndex) const
	{
		return _dataset->get_attribute(attribIndex).instance_values[_index];
	}
}
