/**
 * \file dcs/bs/coalition_formation.hpp
 *
 * \brief Formation of coalitions of BSs.
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

#ifndef DCS_BS_COALITION_FORMATION_COMMONS_HPP
#define DCS_BS_COALITION_FORMATION_COMMONS_HPP


#include <dcs/bs/user_assignment_solution.hpp>
#include <gtpack/cooperative.hpp>
#include <limits>
#include <map>
#include <set>


namespace dcs { namespace bs {

typedef ::gtpack::player_type bsid_type;

enum coalition_formation_category
{
	nash_stable_coalition_formation,
	pareto_optimal_coalition_formation,
	social_optimum_coalition_formation
};

enum coalition_value_division_category
{
	banzhaf_coalition_value_division,
	normalized_banzhaf_coalition_value_division,
	shapley_coalition_value_division,
	chi_coalition_value_division
};

enum partition_preference_category
{
	utilitarian_partition_preference,
	pareto_partition_preference
};

template <typename RealT>
struct coalition_info
{
	coalition_info()
	: bsid_to_idx(),
	  user_assignment(),
	  value(::std::numeric_limits<RealT>::quiet_NaN()),
	  core_empty(true),
	  payoffs(),
	  payoffs_in_core(false),
	  cid(::gtpack::empty_coalition_id)
	{
	}

	::std::map<bsid_type,::std::size_t> bsid_to_idx;
	::std::map< ::std::size_t,::std::size_t> usr_to_idx;
	//::std::vector< ::std::size_t > usr_to_provs;
	user_assignment_solution<RealT> user_assignment;
	RealT value;
	bool core_empty;
	::std::map< ::gtpack::player_type, RealT> payoffs;
	bool payoffs_in_core;
	::gtpack::cid_type cid;
};

template <typename RealT>
struct partition_info
{
	partition_info()
	: value(-::std::numeric_limits<RealT>::infinity())
	{
	}

	RealT value;
	::std::set< ::gtpack::cid_type > coalitions;
	::std::map< ::gtpack::player_type, RealT > payoffs;
	::std::map< ::gtpack::player_type, RealT > coalition_change_penalties;
//	::std::map< ::gtpack::player_type, RealT > side_payments;
};

template <typename RealT>
struct coalition_formation_info
{
	::std::map< ::gtpack::cid_type, coalition_info<RealT> > coalitions;
	::std::vector< partition_info<RealT> > best_partitions;
};

}} // Namespace dcs::bs


#endif // DCS_BS_COALITION_FORMATION_COMMONS_HPP
