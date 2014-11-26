/**
 * \file dcs/math/quadrature/rectangle.hpp
 *
 * \brief Quadrature methods based on the rectangle rule.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 *
 * <hr/>
 *
 * Copyright (C) 2013       Marco Guazzone (marco.guazzone@gmail.com)
 *                          [Distributed Computing System (DCS) Group,
 *                           Computer Science Institute,
 *                           Department of Science and Technological Innovation,
 *                           University of Piemonte Orientale,
 *                           Alessandria (Italy)]
 *
 * This file is part of dcsxx-commons (below referred to as "this program").
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef DCS_MATH_QUADRATURE_RECTANGLE_HPP
#define DCS_MATH_QUADRATURE_RECTANGLE_HPP


#include <cstddef>
#include <dcs/assert.hpp>
#include <dcs/debug.hpp>
#include <dcs/exception.hpp>
#include <stdexcept>


namespace dcs { namespace math { namespace quadrature {

/**
 * \brief Numerical integration based on the composite midpoint rectangle rule.
 *
 * Calculates the cumulative numerical integral with the composite rectangle formula
 * (mid-point variation):
 * \f[
 *  \int_l^u f(x)dx ~= H * \sum_{k = 1}^M f( x_k )
 * \f]
 * where:
 * - \f$f\f$ is the function to be integrated
 * - \f$M\f$ is the number of subdivisions (i.e. sub-intervals).
 * - \f$H\f$ is the sub-interval size (\f$H = (u-l)/M\f$, where \f$l\f$ is the lower limit
 *   and \f$u\f$ is the upper limit).
 * - \f$x_k\f$ is the \f$k\f$-th quadrature node, that is \f$x_k=l+(2k+1)\frac{H}{2}\f$.
 * .
 * If \f$M=1\f$, the formula is called simple rectangle rule.
 */
template <typename RealT, typename FuncT>
RealT midpoint_rectangle(FuncT f, RealT lower, RealT upper, ::std::size_t m = 1)
{
	// pre: upper > lower
	DCS_ASSERT(upper > lower,
			   DCS_EXCEPTION_THROW(::std::invalid_argument,
								   "Upper limit of integration interval must be greater than lower limit"));
	// pre: m > 0
	DCS_ASSERT(m > 0,
			   DCS_EXCEPTION_THROW(::std::invalid_argument,
								   "The number of subdivisions must be a positive number"));

	const RealT h((upper-lower)/m);
	const RealT hh(h/2);

	RealT s(0);
	while (m > 0)
	{
		lower += hh;
		s += f(lower);
		--m;
	}
	return h*s;
}

/**
 * \brief Numerical integration based on the composite leftpoint rectangle rule.
 *
 * Calculates the cumulative numerical integral with the composite rectangle formula
 * (left-point variation):
 * \f[
 *  \int_l^u f(x)dx ~= H * \sum_{k = 0}^{M-1} f( x_k )
 * \f]
 * where:
 * - \f$f\f$ is the function to be integrated
 * - \f$M\f$ is the number of subdivisions (i.e. sub-intervals).
 * - \f$H\f$ is the sub-interval size (\f$H = (u-l)/M\f$, where \f$l\f$ is the lower limit
 *   and \f$u\f$ is the upper limit).
 * - \f$x_k\f$ is the \f$k\f$-th quadrature node, that is \f$x_k=l+kH\f$.
 * .
 * If \f$M=1\f$, the formula is called simple rectangle rule.
 */
template <typename RealT, typename FuncT>
RealT leftpoint_rectangle(FuncT f, RealT lower, RealT upper, ::std::size_t m = 1)
{
	// pre: upper > lower
	DCS_ASSERT(upper > lower,
			   DCS_EXCEPTION_THROW(::std::invalid_argument,
								   "Upper limit of integration interval must be greater than lower limit"));
	// pre: m > 0
	DCS_ASSERT(m > 0,
			   DCS_EXCEPTION_THROW(::std::invalid_argument,
								   "The number of subdivisions must be a positive number"));

	const RealT h((upper-lower)/m);

	RealT s(0);
	while (m > 0)
	{
		s += f(lower);
		lower += h;
		--m;
	}
	return h*s;
}

/**
 * \brief Numerical integration based on the composite rightpoint rectangle rule.
 *
 * Calculates the cumulative numerical integral with the composite rectangle formula
 * (right-point variation):
 * \f[
 *  \int_l^u f(x)dx ~= H * \sum_{k = 1}^M f( x_k )
 * \f]
 * where:
 * - \f$f\f$ is the function to be integrated
 * - \f$M\f$ is the number of subdivisions (i.e. sub-intervals).
 * - \f$H\f$ is the sub-interval size (\f$H = (u-l)/M\f$, where \f$l\f$ is the lower limit
 *   and \f$u\f$ is the upper limit).
 * - \f$x_k\f$ is the \f$k\f$-th quadrature node, that is \f$x_k=l+kH\f$.
 * .
 * If \f$M=1\f$, the formula is called simple rectangle rule.
 */
template <typename RealT, typename FuncT>
RealT rightpoint_rectangle(FuncT f, RealT lower, RealT upper, ::std::size_t m = 1)
{
	// pre: upper > lower
	DCS_ASSERT(upper > lower,
			   DCS_EXCEPTION_THROW(::std::invalid_argument,
								   "Upper limit of integration interval must be greater than lower limit"));
	// pre: m > 0
	DCS_ASSERT(m > 0,
			   DCS_EXCEPTION_THROW(::std::invalid_argument,
								   "The number of subdivisions must be a positive number"));

	const RealT h((upper-lower)/m);

	RealT s(0);
	while (m > 0)
	{
		lower += h;
		s += f(lower);
		--m;
	}
	return h*s;
}

}}} // Namespace dcs::math::quadrature


#endif // DCS_MATH_QUADRATURE_RECTANGLE_HPP
