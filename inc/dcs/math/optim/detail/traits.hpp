#ifndef DCS_MATH_OPTIM_DETAIL_TRAITS_HPP
#define DCS_MATH_OPTIM_DETAIL_TRAITS_HPP


#include <boost/type_traits/is_complex.hpp>
#include <boost/type_traits/is_floating_point.hpp>
#include <boost/utility/enable_if.hpp>
#include <dcs/math/optim/optimizer_traits.hpp>
#include <dcs/math/optim/tags.hpp>
#include <dcs/math/traits/float.hpp>


namespace dcs { namespace math { namespace optim {

namespace detail {

template <typename T, typename Enabled = void>
struct value_traits;

template <typename T>
struct value_traits<T, typename ::boost::enable_if< ::boost::is_floating_point<T> >::type>: public ::dcs::math::float_traits<T>
{
};

template <typename T>
struct value_traits<T, typename ::boost::enable_if< ::boost::is_complex<T> >::type>: public ::dcs::math::float_traits<typename T::value_type>
{
};


template <typename T, typename Enabled = void>
struct direction_traits;

template <typename T>
struct direction_traits<T,
						typename ::boost::enable_if< ::boost::is_same<minimization_direction_tag,
																	  typename optimizer_traits<T>::direction_category> >::type>
{
	typedef typename optimizer_traits<T>::value_type value_type;
	typedef typename optimizer_traits<T>::value_type real_type;

	static bool compare(value_type const& lhs, value_type const& rhs, real_type tol)
	{
		return value_traits<value_type>::definitively_less(lhs, rhs, tol);
	}

	static bool compare(value_type const& lhs, value_type const& rhs)
	{
		return value_traits<value_type>::definitively_less(lhs, rhs);
	}
};

template <typename T>
struct direction_traits<T,
						typename ::boost::enable_if< ::boost::is_same<maximization_direction_tag,
																	  typename optimizer_traits<T>::direction_category> >::type>
{
	typedef typename optimizer_traits<T>::value_type value_type;
	typedef typename optimizer_traits<T>::value_type real_type;

	static bool compare(value_type const& lhs, value_type const& rhs, real_type tol)
	{
		return value_traits<value_type>::definitively_less(rhs, lhs, tol);
	}

	static bool compare(value_type const& lhs, value_type const& rhs)
	{
		return value_traits<value_type>::definitively_less(rhs, lhs);
	}
};


} // Namespace detail

}}} // Namespace dcs::math::optim

#endif // DCS_MATH_OPTIM_DETAIL_TRAITS_HPP
