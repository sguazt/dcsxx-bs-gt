/**
 * \file dcs/bs/nash_stable_coalition_analyzer.hpp
 *
 * \brief Formation of Pareto-optimal coalitions of BSs.
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

#ifndef DCS_BS_COALITION_FORMATION_PARETO_OPTIMAL_HPP
#define DCS_BS_COALITION_FORMATION_PARETO_OPTIMAL_HPP


#include <cstddef>
#include <dcs/algorithm/combinatorics.hpp>
#include <dcs/assert.hpp>
#include <dcs/bs/coalition_formation/commons.hpp>
#include <dcs/debug.hpp>
#include <dcs/exception.hpp>
#include <dcs/math/traits/float.hpp>
#include <gtpack/cooperative.hpp>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>


namespace dcs { namespace bs {

template <typename RealT>
struct pareto_optimal_partition_selector
{
	::std::vector< partition_info<RealT> > operator()(::gtpack::cooperative_game<RealT> const& game, ::std::map< gtpack::cid_type, coalition_info<RealT> > const& visited_coalitions)
	{
		namespace alg = ::dcs::algorithm;

		// Generate all partitions and select the ones that are Pareto optimal

		::std::vector< partition_info<RealT> > best_partitions;

		const ::std::vector<gtpack::player_type> players(game.players());
		const ::std::size_t np(players.size());

		alg::lexicographic_partition partition(np);

		::std::vector<RealT> best_payoffs(np, ::std::numeric_limits<RealT>::quiet_NaN());

		while (partition.has_next())
		{
			typedef typename alg::partition_traits<bsid_type>::subset_container subset_container;
			typedef typename alg::partition_traits<bsid_type>::subset_const_iterator subset_iterator;

			subset_container subs;

			// Each subset is a collection of coalitions
			subs = alg::next_partition(players.begin(), players.end(), partition);

			DCS_DEBUG_TRACE("--- PARTITION: " << partition);//XXX

			partition_info<RealT> candidate_partition;

			subset_iterator sub_end_it(subs.end());
			for (subset_iterator sub_it = subs.begin();
				 sub_it != sub_end_it;
				 ++sub_it)
			{
				const gtpack::cid_type cid = gtpack::players_coalition<RealT>::make_id(sub_it->begin(), sub_it->end());

				if (visited_coalitions.count(cid) == 0)
				{
					continue;
				}

				DCS_DEBUG_TRACE("--- COALITION: " << game.coalition(cid) << " (CID=" << cid << ")");//XXX

				candidate_partition.coalitions.insert(cid);

				::std::vector<gtpack::player_type> coal_players(sub_it->begin(), sub_it->end());
				for (::std::size_t p = 0; p < coal_players.size(); ++p)
				{
					const gtpack::player_type pid = coal_players[p];

					if (visited_coalitions.at(cid).payoffs.count(pid) > 0)
					{
						candidate_partition.payoffs[pid] = visited_coalitions.at(cid).payoffs.at(pid);
					}
					else
					{
						candidate_partition.payoffs[pid] = ::std::numeric_limits<RealT>::quiet_NaN();
					}
				}
			}

			// Check Pareto optimality

			bool pareto_optimal(true);

			// For all players $p$
			for (std::size_t p = 0; p < np && pareto_optimal; ++p)
			{
				const gtpack::player_type pid(players[p]);

				if (::std::isnan(best_payoffs[p]) || candidate_partition.payoffs.at(pid) > best_payoffs[p])
				{
					best_payoffs[p] = candidate_partition.payoffs.at(pid);
				}
				else
				{
					pareto_optimal = false;
				}
			}

			if (pareto_optimal)
			{
				best_partitions.push_back(candidate_partition);
			}
		}

		return best_partitions;
	}
}; // pareto_optimal_partition_selector

}} // Namespace dcs::bs


#endif // DCS_BS_COALITION_FORMATION_PARETO_OPTIMAL_HPP
