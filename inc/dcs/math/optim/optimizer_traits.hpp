/**
 * \file dcs/math/optim/optimizer_traits.hpp
 *
 * \brief Type-traits class for optimizers.
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

#ifndef DCS_MATH_OPTIM_OPTIMIZER_TRAITS_HPP
#define DCS_MATH_OPTIM_OPTIMIZER_TRAITS_HPP


namespace dcs { namespace math { namespace optim {

template <typename OptimizerT>
struct optimizer_traits
{
	typedef typename OptimizerT::value_type value_type;
	typedef typename OptimizerT::real_type real_type;
	typedef typename OptimizerT::direction_category direction_category;
};

}}} // Namespace dcs::math::optim


#endif // DCS_MATH_OPTIM_OPTIMIZER_TRAITS_HPP
