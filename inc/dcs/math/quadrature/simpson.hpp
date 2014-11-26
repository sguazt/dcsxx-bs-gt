/**
 * \file dcs/math/quadrature/simpson.hpp
 *
 * \brief Quadrature methods based on the Simpson's rule.
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

#ifndef DCS_MATH_QUADRATURE_SIMPSON_HPP
#define DCS_MATH_QUADRATURE_SIMPSON_HPP


#include <cstddef>
#include <dcs/assert.hpp>
#include <dcs/debug.hpp>
#include <dcs/exception.hpp>
#include <dcs/math/traits/float.hpp>
#include <stdexcept>


namespace dcs { namespace math { namespace quadrature {

/**
 * \brief Numerical integration based on the composite Simpson's rule.
 *
 * Calculates the numerical integral with the composite Simpson's formula:
 * \f[
 *  `int_l^u f(x)dx \simeq \frac{H}{3}\bigl\{f(l)+f(u)+4\sum_{k=1}^{n/2}{f(a+(2k-1)H}+2\sum_{k=1}^{(n-2)/2}{f(a+2kH)}\bigr\}
 * \f]
 * where:
 * * \f$f\f$ is the function to be integrated
 * * \f$H\f$ is the sub-interval size (\f$H = (u-l)/M\f$, where \f$l\f$ is the lower limit
 *   and \f$u\f$ is the upper limit)
 * * \f$M\f$ is the number of subdivisions (i.e. sub-intervals).
 * - \f$x_k\f$ is the \f$k\f$-th quadrature node, that is \f$x_k=l+(2k+1)\frac{H}{2}\f$.
 * .
 * If \f$M=1\f$, the formula is called simple trapezoidal rule.
 */
template <typename RealT, typename FuncT>
RealT simpson(FuncT f, RealT lower, RealT upper, ::std::size_t m = 1)
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

	RealT odds(0);
	RealT evens(0);
	bool is_odd(true);
	for (::std::size_t k = 1; k < m; ++k)
	{
		lower += h;
		if (is_odd)
		{
			odds += f(lower);
		}
		else
		{
			evens += f(lower);
		}
		is_odd = !is_odd;
	}

	return h/3*(f(lower)+f(upper) + 4*odds + 2*evens);
}

template <typename RealT, typename FuncT>
RealT adaptive_simpson(FuncT f, RealT lower, RealT upper, RealT tol = 1e-6, ::std::size_t maxm = 100)
{
	// pre: upper > lower
	DCS_ASSERT(upper >= lower,
			   DCS_EXCEPTION_THROW(::std::invalid_argument,
								   "Upper limit of integration interval must be greater than lower limit"));
	// pre: m > 0
	DCS_ASSERT(maxm >= 0,
			   DCS_EXCEPTION_THROW(::std::invalid_argument,
								   "The maximum number of subdivisions must be a positive number"));

	RealT res(0);

	const RealT h(upper-lower);
	const RealT center((upper+lower)/2);
	const RealT tol15(15*tol);

	const RealT simps_left(h/6*(f(lower)+4*f(center)+f(upper)));
	const RealT simps_right(h/12*(f(lower)+4*(f((lower+center)/2)+f((center+upper)/2))+2*f(center)+f(upper)));
	// Alternative
	//const RealT simps_left(h/12*(f(lower)+4*(f((lower+center)/2)+f(center))));
	//const RealT simps_right(h/12*(f(center)+4*(f((center+upper)/2)+f(upper))));

	if (maxm == 0)
	{
		return simps_right;
	}

	if (::dcs::math::float_traits<RealT>::approximately_equal(simps_left, simps_right, tol15))
	{
		res = simps_right+(simps_right-simps_left)/15;
	}
	else
	{
		res = adaptive_simpson(f, lower, center, tol/*/2*/, maxm-1) + adaptive_simpson(f, center, upper, tol/*/2*/, maxm-1);
	}

	return res;
}

}}} // Namespace dcs::math::quadrature


#endif // DCS_MATH_QUADRATURE_SIMPSON_HPP
