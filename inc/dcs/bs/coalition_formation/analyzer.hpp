/**
 * \file dcs/bs/coalition_formation/analyzer.hpp
 *
 * \brief Analyze coalitions according to specific criteria
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

#ifndef DCS_BS_COALITION_FORMATION_ANALYZER_HPP
#define DCS_BS_COALITION_FORMATION_ANALYZER_HPP


#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublasx/operation/any.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <boost/numeric/ublasx/operation/size.hpp>
#include <boost/numeric/ublasx/operation/sum.hpp>
#include <dcs/algorithm/combinatorics.hpp>
#include <dcs/assert.hpp>
#include <dcs/bs/user_assignment_solution.hpp>
#include <dcs/bs/coalition_formation/commons.hpp>
#include <dcs/debug.hpp>
#include <dcs/exception.hpp>
#include <dcs/math/traits/float.hpp>
#include <dcs/logging.hpp>
#include <gtpack/cooperative.hpp>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>


namespace dcs { namespace bs {

template <typename RealT>
struct coalition_analyzer
{
	template <typename NbVectorT,
			  typename NuVectorT,
			  typename RVectorT,
			  typename WVectorT,
			  typename CVectorT,
			  typename BVectorT,
			  typename QVectorT,
			  typename OmegaMatrixT,
			  typename AlphaVectorT,
			  typename AssignmentSolverT,
			  typename PartitionSelectorT>
	coalition_formation_info<RealT> operator()(::boost::numeric::ublas::vector_expression<NbVectorT> const& Nb,
											   ::boost::numeric::ublas::vector_expression<NuVectorT> const& Nu,
											   ::boost::numeric::ublas::vector_expression<RVectorT> const& R,
											   ::boost::numeric::ublas::vector_expression<WVectorT> const& W,
											   ::boost::numeric::ublas::vector_expression<CVectorT> const& C,
											   ::boost::numeric::ublas::vector_expression<BVectorT> const& B,
											   ::boost::numeric::ublas::vector_expression<QVectorT> const& Q,
											   ::boost::numeric::ublas::matrix_expression<OmegaMatrixT> const& Omega,
											   ::boost::numeric::ublas::vector_expression<AlphaVectorT> const& alpha,
											   coalition_value_division_category coalition_value_division,
											   AssignmentSolverT assignment_solver,
											   PartitionSelectorT partition_selector,
											   RealT bs_coalition_cost,
											   RealT bs_coalition_change_cost,
											   //partition_preference_category partition_preference,
											   partition_info<RealT> const* p_last_partition) const
	{
		namespace alg = ::dcs::algorithm;
		namespace ublas = ::boost::numeric::ublas;
		namespace ublasx = ::boost::numeric::ublasx;


		const ::std::size_t np = ublasx::size(Nb); // Number of providers
		const ::std::size_t nb = ublasx::sum(Nb); // Number of base stations
		const ::std::size_t nu = ublasx::sum(Nu); // Number of users

		// pre
		DCS_ASSERT(nu == ublasx::size(R),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "R has not a conformant size"));
		DCS_ASSERT(nb == ublasx::size(W),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "W has not a conformant size"));
		DCS_ASSERT(nb == ublasx::size(C),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "C has not a conformant size"));
		DCS_ASSERT(nb == ublasx::size(B),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "B has not a conformant size"));
		DCS_ASSERT(nu == ublasx::size(Q),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "Q has not a conformant size"));
		DCS_ASSERT(nb == ublasx::num_rows(Omega) && nu == ublasx::num_columns(Omega),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "Omega has not a conformant size"));
		DCS_ASSERT(nb == ublasx::size(alpha),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "alpha has not a conformant size"));
		DCS_ASSERT(bs_coalition_cost >= 0,
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "Coalition cost cannot be a negative value"));
		DCS_ASSERT(bs_coalition_change_cost >= 0,
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "Coalition change cost cannot be a negative value"));

		// Auxiliary data structures
		::std::map< ::std::size_t, ::std::pair< ::std::size_t,::std::size_t > > prov_usr; // Map a provider to a pair of indices in Nu: provider => <begin_index_in_Nu,end_index_in_Nu>
		::std::map<bsid_type, ::std::size_t> bs_prov; // Map a base-station to the provider that owns it
		::std::vector<bsid_type> bss(nb); // The sequence of all base-stations

		// Setup auxiliary data structures
		{
			::std::size_t bs_start(0);
			::std::size_t usr_start(0);
			for (::std::size_t p = 0; p < np; ++p)
			{
				if (p > 0)
				{
					bs_start += Nb()(p-1);
					usr_start += Nu()(p-1);
				}
				::std::size_t bs_stop(bs_start+Nb()(p));

				for (::std::size_t b = bs_start; b < bs_stop; ++b)
				{
					bss[b] = bsid_type(b);
					bs_prov[b] = p;
				}

				prov_usr[p] = ::std::make_pair(usr_start, usr_start+Nu()(p)); // provider => <start,stop> index to users
			}
		}

		gtpack::cooperative_game<RealT> game(nb, ::boost::make_shared< gtpack::explicit_characteristic_function<RealT> >());

		::std::map< gtpack::cid_type, coalition_info<RealT> > visited_coalitions;
		::std::map< gtpack::player_type, ::std::vector< partition_info<RealT> > > best_partitions;
		bool found_same_struc(false);

		alg::lexicographic_subset subset(nb, false);

		while (subset.has_next())
		{
			typedef typename alg::subset_traits<bsid_type>::element_container element_container;

			DCS_DEBUG_TRACE("--- SUBSET: " << subset);//XXX

			const element_container sub = alg::next_subset(bss.begin(), bss.end(), subset);

			const gtpack::cid_type cid = gtpack::players_coalition<RealT>::make_id(sub.begin(), sub.end());

			DCS_DEBUG_TRACE("--- COALITION: " << game.coalition(cid) << " (CID=" << cid << ")");//XXX

			const ::std::vector<bsid_type> coal_bss(sub.begin(), sub.end());
			const ::std::size_t coal_nb(coal_bss.size());
			::std::size_t coal_nu(0);
			// Compute the number of users for this coalition
			{
				::std::set< ::std::size_t > coal_prov;
				for (::std::size_t b = 0; b < coal_nb; ++b)
				{
					const bsid_type bsid(coal_bss[b]);

					const ::std::size_t bsp(bs_prov.at(bsid));

					if (coal_prov.count(bsp) == 0)
					{
						const ::std::size_t bs_nu(Nu()(bsp));
						coal_prov.insert(bsp);
						coal_nu += bs_nu;
					}

					visited_coalitions[cid].bsid_to_idx[bsid] = b;
				}
			}

			// Setup suitable data structures for this coalition (and used by the assignment solver)

			ublas::vector<RealT> coal_R(coal_nu);
			ublas::vector<RealT> coal_W(coal_nb);
			ublas::vector<RealT> coal_C(coal_nb);
			ublas::vector<RealT> coal_B(coal_nb);
			ublas::vector<RealT> coal_Q(coal_nu);
			ublas::matrix<RealT> coal_Omega(coal_nb, coal_nu);
			ublas::vector<RealT> coal_alpha(coal_nb);

			// Fill the above arrays
			{
				::std::size_t nu_start(0);
				::std::set< ::std::size_t > bsp_set;
				for (::std::size_t b = 0; b < coal_nb; ++b)
				{
					const bsid_type bsid(coal_bss[b]);

					const ::std::size_t bsp(bs_prov.at(bsid));

					if (bsp_set.count(bsp) == 0)
					{
						const ::std::size_t bs_nu(Nu()(bsp));

						// check: double check for number of users
						DCS_DEBUG_ASSERT( bs_nu == (prov_usr.at(bsp).second-prov_usr.at(bsp).first) );

						bsp_set.insert(bsp);
						ublas::subrange(coal_R, nu_start, nu_start+bs_nu) = ublas::subrange(R(), prov_usr.at(bsp).first, prov_usr.at(bsp).second);
						ublas::subrange(coal_Q, nu_start, nu_start+bs_nu) = ublas::subrange(Q(), prov_usr.at(bsp).first, prov_usr.at(bsp).second);
						nu_start += bs_nu;
					}
					coal_W(b) = W()(bsid);
					coal_C(b) = C()(bsid);
					coal_B(b) = B()(bsid);
					ublas::subrange(coal_Omega, b, b+1, 0, coal_nu) = ublas::subrange(Omega(), bsid, bsid+1, 0, coal_nu);
					coal_alpha(b) = alpha()(bsid);
				}

				// check: double check for number of users in coalition
				DCS_DEBUG_ASSERT( nu_start == coal_nu );
			}

			user_assignment_solution<RealT> user_assignment;
			user_assignment = assignment_solver(coal_R,
												coal_W,
												coal_C,
												coal_B,
												coal_Q,
												coal_Omega,
												coal_alpha);

			visited_coalitions[cid].user_assignment = user_assignment;

			if (user_assignment.solved)
			{
				//game.value(cid, -user_assignment.cost);
				//game.value(cid, -user_assignment.cost-static_cast<RealT>(coal_nb-1)*bs_coalition_cost-ublasx::sum(coal_alpha));
				game.value(cid, -user_assignment.cost-static_cast<RealT>(coal_nb-1)*bs_coalition_cost);
				//game.value(cid, -user_assignment.cost-static_cast<RealT>(coal_nu*(coal_nb-1))*bs_coalition_cost);
				//game.value(cid, -user_assignment.cost-static_cast<RealT>(coal_nu*(coal_nb-1))*bs_coalition_cost-ublasx::sum(coal_alpha));
				//game.value(cid, -user_assignment.cost-static_cast<RealT>(coal_nu*(coal_nb-1))*bs_coalition_cost);
				//game.value(cid, -user_assignment.cost-static_cast<RealT>(coal_nb-1)*bs_coalition_cost*static_cast<RealT>(coal_nu));
				//[FIXME] Experimental: set the value of a game as the cost saving
				//game.value(cid, ublas::inner_prod(coal_W,coal_C)+ublasx::sum(coal_alpha)-user_assignment.cost);
				//game.value(cid, ublas::inner_prod(coal_W,coal_C)+ublasx::sum(coal_alpha)-user_assignment.cost-static_cast<RealT>(coal_nb-1)*bs_coalition_cost);
				//[/FIXME] Experimental

				visited_coalitions[cid].value = game.value(cid);

				DCS_DEBUG_TRACE( "CID: " << cid << " - User assignment objective value: " << user_assignment.objective_value << " => v(CID)=" << game.value(cid) );

				gtpack::cooperative_game<RealT> subgame = game.subgame(coal_bss.begin(), coal_bss.end());
				gtpack::core<RealT> core = gtpack::find_core(subgame);
				if (core.empty())
				{
					DCS_DEBUG_TRACE( "CID: " << cid << " - The core is empty" );

					visited_coalitions[cid].core_empty = true;
					visited_coalitions[cid].payoffs_in_core = false;

					if (subgame.num_players() == nb)
					{
						// This is the Grand coalition

						DCS_DEBUG_TRACE( "CID: " << cid << " - The Grand-Coalition has an empty core" );
					}
				}
				else
				{
					DCS_DEBUG_TRACE( "CID: " << cid << " - The core is not empty" );

					visited_coalitions[cid].core_empty = false;
				}

				if (coalition_value_division != chi_coalition_value_division)
				{
					// Compute the coalition payoffs
					::std::map<gtpack::player_type,RealT> coal_payoffs;
					switch (coalition_value_division)
					{
						case banzhaf_coalition_value_division:
							coal_payoffs = gtpack::banzhaf_value(subgame);
							break;
						case normalized_banzhaf_coalition_value_division:
							coal_payoffs = gtpack::norm_banzhaf_value(subgame);
							break;
						case shapley_coalition_value_division:
							coal_payoffs = gtpack::shapley_value(subgame);
							break;
						case chi_coalition_value_division:
							// postponed: we need to compute the Shapley value for the grand-coalition
							break;
					}
#ifdef DCS_DEBUG
					for (std::size_t b = 0; b < coal_nb; ++b)
					{
						const std::size_t bsid(coal_bss[b]);

						DCS_DEBUG_TRACE( "CID: " << cid << " - BS: " << bsid << " - Coalition payoff: " << coal_payoffs.at(bsid) );
					}
#endif // DCS_DEBUG
					visited_coalitions[cid].payoffs = coal_payoffs;

					// Check if the payoff vector is in the core *if the core != empty)
					if (!visited_coalitions.at(cid).core_empty)
					{
						if (gtpack::belongs_to_core(game.subgame(coal_bss.begin(), coal_bss.end()), coal_payoffs.begin(), coal_payoffs.end()))
						{
							DCS_DEBUG_TRACE( "CID: " << cid << " - The Coalition payoff vector belongs to the core" );

							visited_coalitions[cid].payoffs_in_core = true;
						}
						else
						{
							DCS_DEBUG_TRACE( "CID: " << cid << " - The Coaition payoff vector does not belong to the core" );

							visited_coalitions[cid].payoffs_in_core = false;
						}
					}
				}
			}
			else
			{
				DCS_DEBUG_TRACE( "CID: " << cid << " - The user assignment problem is infeasible" );

				visited_coalitions[cid].core_empty = true;
				visited_coalitions[cid].payoffs_in_core = false;

				game.value(cid, -::std::numeric_limits<RealT>::min());

				if (game.coalition(cid).num_players() == nb)
				{
					// This is the Grand coalition

					DCS_DEBUG_TRACE( "CID: " << cid << " - The Grand-Coalition has an infeasible solution and thus an empty core" );
				}
			}
		}

		// Compute the coalition payoffs for postponed coalition value divisions
		if (coalition_value_division == chi_coalition_value_division)
		{
			for (typename std::map<gtpack::cid_type,coalition_info<RealT> >::iterator it = visited_coalitions.begin();
				 it != visited_coalitions.end();
				 ++it)
			{
				const gtpack::cid_type cid = it->first;

				::std::map<gtpack::player_type,RealT> coal_payoffs;

				// Create a coalition structure $\{S,N \setminus S\}$, where $S$
				// is the current coalition identified by 'cid' and $N$ is the
				// set of all players
				::std::vector<gtpack::cid_type> old_coal_struc = game.coalition_structure();
				::std::vector<gtpack::cid_type> new_coal_struc;
				::std::vector<gtpack::player_type> players = game.players();
				gtpack::cid_type grand_cid = gtpack::players_coalition<RealT>::make_id(players.begin(), players.end());
				gtpack::cid_type other_cid = grand_cid-cid;
				new_coal_struc.push_back(cid);
				if (other_cid > 0)
				{
					new_coal_struc.push_back(other_cid);
				}
				// Set the new coalition structure
				game.coalition_structure(new_coal_struc.begin(), new_coal_struc.end());
				// Compute the payoffs
				coal_payoffs = gtpack::chi_value(game);

				// Restore the old coalition structure
				game.coalition_structure(old_coal_struc.begin(), old_coal_struc.end());

				// Store the computed payoffs
				players = game.coalition(cid).players();
				for (::std::size_t i = 0; i < players.size(); ++i)
				{
					const gtpack::player_type pid = players[i];

					DCS_DEBUG_TRACE( "CID: " << cid << " - BS: " << pid << " - Coalition payoff: " << coal_payoffs.at(pid) );

					visited_coalitions[cid].payoffs[pid] = coal_payoffs.at(pid);
				}
			}
		}

		coalition_formation_info<RealT> formed_coalitions;
		formed_coalitions.coalitions = visited_coalitions;
		formed_coalitions.best_partitions = partition_selector(game, visited_coalitions);

#ifdef DCS_DEBUG
		DCS_DEBUG_TRACE( "FORMED PARTITIONS: ");
		for (::std::size_t i = 0; i < formed_coalitions.best_partitions.size(); ++i)
		{
			const partition_info<RealT> part = formed_coalitions.best_partitions[i];

			typedef typename ::std::set<gtpack::cid_type>::const_iterator coalition_iterator;
			coalition_iterator coal_end_it(part.coalitions.end());
			DCS_DEBUG_STREAM << "  [";
			for (coalition_iterator coal_it = part.coalitions.begin();
				 coal_it != coal_end_it;
				 ++coal_it)
			{
				const gtpack::cid_type cid(*coal_it);

				DCS_DEBUG_STREAM << cid << ",";
			}
			DCS_DEBUG_STREAM << "]" << ::std::endl;
		}
#endif // DCS_DEBUG


		return formed_coalitions;
	}
}; // coalition_analyzer

}} // Namespace dcs::bs


#endif // DCS_BS_COALITION_FORMATION_ANALYZER_HPP
