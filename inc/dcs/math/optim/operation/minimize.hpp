/**
 * \file dcs/math/optim/operation/minimize.hpp
 *
 * \brief Minimize a function according to the given method.
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

#ifndef DCS_MATH_OPTIM_OPERATION_MINIMIZE_HPP
#define DCS_MATH_OPTIM_OPERATION_MINIMIZE_HPP


#include <boost/type_traits/is_same.hpp>
#include <boost/utility/enable_if.hpp>
#include <dcs/math/optim/functional/sign_change.hpp>
#include <dcs/math/optim/optimization_result.hpp>
#include <dcs/math/optim/optimizer_traits.hpp>
#include <dcs/math/optim/tags.hpp>


namespace dcs { namespace math { namespace optim {


namespace detail { namespace /*<unnamed>*/ {

template <typename ValueT, typename OptimizerT, typename FuncT>
inline
typename ::boost::enable_if<
	::boost::is_same<minimization_direction_tag,
					 typename optimizer_traits<OptimizerT>::direction_category>,
	optimization_result<ValueT>
>::type minimize_impl(OptimizerT& optim, FuncT f)
{
	return optim.optimize(f);
}

template <typename ValueT, typename OptimizerT, typename FuncT>
inline
typename ::boost::enable_if<
	::boost::is_same<maximization_direction_tag,
					 typename optimizer_traits<OptimizerT>::direction_category>,
	optimization_result<ValueT>
>::type minimize_impl(OptimizerT& optim, FuncT f)
{
	optimization_result<ValueT> res = optim.optimize(make_sign_change(f));
	res.fopt = -res.fopt;

	return res;
}

}} // Namespace detail::<unnamed>


template <typename ValueT, typename OptimizerT, typename FuncT>
inline
optimization_result<ValueT> minimize(OptimizerT& optim, FuncT f)
{
	return detail::minimize_impl<ValueT>(optim, f);
}

}}} // Namespace dcs::math::optim


#endif // DCS_MATH_OPTIM_OPERATION_MINIMIZE_HPP
