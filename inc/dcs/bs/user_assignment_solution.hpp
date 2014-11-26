/**
 * \file dcs/bs/user_assignment_solution.hpp
 *
 * \brief Solution to the user assignment problem.
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
 * This file is part of dcsxx-bs (below referred to as "this program").
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

#ifndef DCS_BS_USER_ASSIGNMENT_SOLUTION_HPP
#define DCS_BS_USER_ASSIGNMENT_SOLUTION_HPP


#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <limits>


namespace dcs { namespace bs {

template <typename RealT>
struct user_assignment_solution
{
	user_assignment_solution()
	: solved(false),
	  optimal(false),
	  objective_value(::std::numeric_limits<RealT>::quiet_NaN()),
	  cost(::std::numeric_limits<RealT>::quiet_NaN()),
	  kwatt(::std::numeric_limits<RealT>::quiet_NaN())
	{
	}


	::boost::numeric::ublas::matrix<bool> bs_user_allocations;
	::boost::numeric::ublas::vector<RealT> bs_user_downlink_data_rates;
	::boost::numeric::ublas::vector<bool> bs_power_states;
	bool solved;
	bool optimal;
	RealT objective_value;
	RealT cost;
	RealT kwatt;
};

}} // Namespace dcs::bs


#endif // DCS_BS_USER_ASSIGNMENT_SOLUTION_HPP
