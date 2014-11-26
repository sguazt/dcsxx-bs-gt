/**
 * \file dcs/bs/max_profit.hpp
 *
 * \brief Max profit computation for a set of coalitions.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 *
 * <hr/>
 *
 * Copyright 2013 Marco Guazzone (marco.guazzone@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef DCS_BS_MAX_PROFIT_HPP
#define DCS_BS_MAX_PROFIT_HPP


#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <boost/numeric/ublasx/operation/size.hpp>
#include <cmath>
#include <cstddef>
#include <dcs/assert.hpp>
#include <dcs/bs/radio_propagation_models.hpp>
#include <dcs/bs/user_assignment_solution.hpp>
#include <dcs/debug.hpp>
#include <dcs/exception.hpp>
#include <dcs/logging.hpp>
#include <dcs/math/traits/float.hpp>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#if defined(DCS_BS_GT_USE_NATIVE_CPLEX_SOLVER)
# include <ilconcert/iloalg.h>
# include <ilconcert/iloenv.h>
# include <ilconcert/iloexpression.h>
# include <ilconcert/ilomodel.h>
# include <ilcplex/ilocplex.h>
#elif defined(DCS_BS_GT_USE_NATIVE_GUROBI_SOLVER)
# include <gurobi_c++.h>
#elif defined(DCS_BS_GT_USE_OSI_SOLVER)
# include <flopc.hpp>
# include <OsiSolverInterface.hpp>
# if defined(DCS_BS_GT_USE_OSI_CBC_SOLVER)
#  include <OsiCbcSolverInterface.hpp>
# endif // DCS_BS_GT_USE_OSI_CBC_SOLVER
#endif // DCS_BS_GT_USE_NATIVE_CPLEX_SOLVER


namespace dcs { namespace bs {

/**
 * Solve the following optimization problem
 *  max z =  \sum_{i \in B}{\sum_{j \in U}{R_{P^b(i)}^{C^u(j)} x_{ij}} - (\sum_{i \in B}{K_{P^b(i)}^w (W_{C^b(i)}^c y_i + W_{C^b(i)}^l\sum_{j \in U}{x_{ij}}) + \sum_{j \in U}{(x_{ij}-\frac{r_{ij}}{D_{C^u(j)}^d})R_{P^b(i)}^{C^u(j)}}})
 *  s.t.
 *   \sum_{j \in U}{d_{ij}} \le T_{P^b(i)}^d, \forall i \in B,
 *   \sum_{i \in B}{x_{ij}} = 1, \forall j \in U,
 *   \sum_{j \in U}{x_{ij}} \le |U|y_i, \forall i \in B,
 *   r_{ij} \le x_{ij}D_{C^u(j)}^d, \forall i \in B, \forall j \in U,
 *   r_{ij} \ge x_{ij}G_{P_b(i),C^u(j)}^d, \forall i \in B, \forall j \in U,
 *   r_{ij} := \frac{n_{ij}}{N_{P^b(i)}^s}B_{P^b(i)}^b\log_2(1+\frac{P_{P^b(i)}^t\rho(d_{ij})}{\sum_{k\in B, k\ne i}P_{P^b(i)}^t\rho(d_{ij}) + P_{P^b(i)}^n}),
 *   n_{ij} \in Z^*, \forall i \in B, j \in U,
 *   x_{ij} \in \{0,1\}, \forall i \in B, j \in U,
 *   y_i \in \{0,1\}, \forall i \in B.
 */
template <typename RealT>
struct optimal_max_profit_user_assignment_solver
{
	public: explicit optimal_max_profit_user_assignment_solver(RealT relative_gap = 0,
															   RealT time_limit = -1)
	: rel_gap_(relative_gap),
	  time_lim_(time_limit)
	{
	}

	public: template <typename PVectorT,
					  typename BVectorT,
					  typename UVectorT,
					  typename BsCatsVectorT,
					  typename BsProvsVectorT,
					  //typename BsHeightsVectorT,
					  //typename BsFreqsVectorT,
					  //typename BsCapsVectorT,
					  typename BsBwsVectorT,
					  //typename BsNumSubcsVectorT,
					  //typename BsTxPowersVectorT,
					  typename BsConstPowersVectorT,
					  typename BsLoadPowersVectorT,
					  typename UsrCatsVectorT,
					  //typename UsrHeightsVectorT,
					  typename UsrQosDlRatesVectorT,
					  //typename UsrThermNoisesVectorT,
					  typename UsrBsSinrsMatrixT,
					  typename ProvEnergyCostsVectorT,
					  typename ProvUsrMinDlRatesMatrixT,
					  typename ProvUsrRevenuesMatrixT>
			user_assignment_solution<RealT> operator()(::boost::numeric::ublas::vector_expression<PVectorT> const& P,
													   ::boost::numeric::ublas::vector_expression<BVectorT> const& B,
													   ::boost::numeric::ublas::vector_expression<UVectorT> const& U,
													   ::boost::numeric::ublas::vector_expression<BsCatsVectorT> const& bs_cats,
													   ::boost::numeric::ublas::vector_expression<BsProvsVectorT> const& bs_provs,
													   //::boost::numeric::ublas::vector_expression<BsHeightsVectorT> const& bs_heights,
													   //::boost::numeric::ublas::vector_expression<BsFreqsVectorT> const& bs_freqs,
													   //::boost::numeric::ublas::vector_expression<BsCapsVectorT> const& bs_caps,
													   ::boost::numeric::ublas::vector_expression<BsBwsVectorT> const& bs_bws,
													   //::boost::numeric::ublas::vector_expression<BsNumSubcsVectorT> const& bs_num_subcs,
													   //::boost::numeric::ublas::vector_expression<BsTxPowersVectorT> const& bs_tx_powers,
													   ::boost::numeric::ublas::vector_expression<BsConstPowersVectorT> const& bs_const_powers,
													   ::boost::numeric::ublas::vector_expression<BsLoadPowersVectorT> const& bs_load_powers,
													   ::boost::numeric::ublas::vector_expression<UsrCatsVectorT> const& usr_cats,
													   //::boost::numeric::ublas::vector_expression<UsrHeightsVectorT> const& usr_heights,
													   ::boost::numeric::ublas::vector_expression<UsrQosDlRatesVectorT> const& usr_qos_dl_rates,
													   //::boost::numeric::ublas::vector_expression<UsrThermNoisesVectorT> const& usr_therm_noises,
													   ::boost::numeric::ublas::matrix_expression<UsrBsSinrsMatrixT> const& usr_bs_sinrs,
													   ::boost::numeric::ublas::vector_expression<ProvEnergyCostsVectorT> const& prov_energy_costs,
													   ::boost::numeric::ublas::matrix_expression<ProvUsrMinDlRatesMatrixT> const& prov_usr_min_dl_rates,
													   ::boost::numeric::ublas::matrix_expression<ProvUsrRevenuesMatrixT> const& prov_usr_revenues) const
	{
#if defined(DCS_BS_GT_USE_NATIVE_CPLEX_SOLVER)
		return by_native_cplex(P,
							   B,
							   U,
							   bs_cats,
							   bs_provs,
							   //bs_heights,
							   //bs_freqs,
							   //bs_caps,
							   bs_bws,
							   //bs_num_subcs,
							   //bs_tx_powers,
							   bs_const_powers,
							   bs_load_powers,
							   usr_cats,
							   //usr_heights,
							   usr_qos_dl_rates,
							   //usr_therm_noises,
							   usr_bs_sinrs,
							   prov_energy_costs,
							   prov_usr_min_dl_rates,
							   prov_usr_revenues);
//P, B, U, bs_cats, bs_provs, bs_caps, bs_const_powers, bs_load_powers, usr_cats, usr_qos_dl_rates, prov_energy_costs, prov_usr_min_dl_rates, prov_usr_revenues);
#elif defined(DCS_BS_GT_USE_NATIVE_GUROBI_SOLVER)
		throw std::runtime_error("Still to be updated (use the CPLEX version)");
		return by_native_gurobi(P,
								B,
								U,
								bs_cats,
								bs_provs,
								//bs_heights,
								//bs_freqs,
								//bs_caps,
								bs_bws,
								//bs_num_subcs,
								//bs_tx_powers,
								bs_const_powers,
								bs_load_powers,
								usr_cats,
								//usr_heights,
								usr_qos_dl_rates,
								//usr_therm_noises,
								usr_bs_sinrs,
								prov_energy_costs,
								prov_usr_min_dl_rates,
								prov_usr_revenues);
//P, B, U, bs_cats, bs_provs, bs_caps, bs_const_powers, bs_load_powers, usr_cats, usr_qos_dl_rates, prov_energy_costs, prov_usr_min_dl_rates, prov_usr_revenues);
#elif defined(DCS_BS_GT_USE_OSI_SOLVER)
		throw std::runtime_error("Still to be updated (use the CPLEX version)");
		return by_osi(P,
					  B,
					  U,
					  bs_cats,
					  bs_provs,
					  //bs_heights,
					  //bs_freqs,
					  //bs_caps,
					  bs_bws,
					  //bs_num_subcs,
					  //bs_tx_powers,
					  bs_const_powers,
					  bs_load_powers,
					  usr_cats,
					  //usr_heights,
					  usr_qos_dl_rates,
					  //usr_therm_noises,
					  usr_bs_sinrs,
					  prov_energy_costs,
					  prov_usr_min_dl_rates,
					  prov_usr_revenues);
//P, B, U, bs_cats, bs_provs, bs_caps, bs_const_powers, bs_load_powers, usr_cats, usr_qos_dl_rates, prov_energy_costs, prov_usr_min_dl_rates, prov_usr_revenues);
#else
# error No solver is available to solve Max-Profit problem.
#endif // defined(DCS_BS_GT_USE_NATIVE_CPLEX_SOLVER)
	}

#ifdef DCS_BS_GT_USE_NATIVE_CPLEX_SOLVER
	private: template <typename PVectorT,
					   typename BVectorT,
					   typename UVectorT,
					   typename BsCatsVectorT,
					   typename BsProvsVectorT,
					   //typename BsHeightsVectorT,
					   //typename BsFreqsVectorT,
					   //typename BsCapsVectorT,
					   typename BsBwsVectorT,
					   //typename BsNumSubcsVectorT,
					   //typename BsTxPowersVectorT,
					   typename BsConstPowersVectorT,
					   typename BsLoadPowersVectorT,
					   typename UsrCatsVectorT,
					   //typename UsrHeightsVectorT,
					   typename UsrQosDlRatesVectorT,
					   //typename UsrThermNoisesVectorT,
					   typename UsrBsSinrsMatrixT,
					   typename ProvEnergyCostsVectorT,
					   typename ProvUsrMinDlRatesMatrixT,
					   typename ProvUsrRevenuesMatrixT>
			user_assignment_solution<RealT> by_native_cplex(::boost::numeric::ublas::vector_expression<PVectorT> const& P,
															::boost::numeric::ublas::vector_expression<BVectorT> const& B,
															::boost::numeric::ublas::vector_expression<UVectorT> const& U,
															::boost::numeric::ublas::vector_expression<BsCatsVectorT> const& bs_cats,
															::boost::numeric::ublas::vector_expression<BsProvsVectorT> const& bs_provs,
															//::boost::numeric::ublas::vector_expression<BsHeightsVectorT> const& bs_heights,
															//::boost::numeric::ublas::vector_expression<BsFreqsVectorT> const& bs_freqs,
															//::boost::numeric::ublas::vector_expression<BsCapsVectorT> const& bs_caps,
															::boost::numeric::ublas::vector_expression<BsBwsVectorT> const& bs_bws,
															//::boost::numeric::ublas::vector_expression<BsNumSubcsVectorT> const& bs_num_subcs,
															//::boost::numeric::ublas::vector_expression<BsTxPowersVectorT> const& bs_tx_powers,
															::boost::numeric::ublas::vector_expression<BsConstPowersVectorT> const& bs_const_powers,
															::boost::numeric::ublas::vector_expression<BsLoadPowersVectorT> const& bs_load_powers,
															::boost::numeric::ublas::vector_expression<UsrCatsVectorT> const& usr_cats,
															//::boost::numeric::ublas::vector_expression<UsrHeightsVectorT> const& usr_heights,
															::boost::numeric::ublas::vector_expression<UsrQosDlRatesVectorT> const& usr_qos_dl_rates,
															//::boost::numeric::ublas::vector_expression<UsrThermNoisesVectorT> const& usr_therm_noises,
															::boost::numeric::ublas::matrix_expression<UsrBsSinrsMatrixT> const& usr_bs_sinrs,
															::boost::numeric::ublas::vector_expression<ProvEnergyCostsVectorT> const& prov_energy_costs,
															::boost::numeric::ublas::matrix_expression<ProvUsrMinDlRatesMatrixT> const& prov_usr_min_dl_rates,
															::boost::numeric::ublas::matrix_expression<ProvUsrRevenuesMatrixT> const& prov_usr_revenues) const
	{
		namespace ublas = ::boost::numeric::ublas;
		namespace ublasx = ::boost::numeric::ublasx;

		typedef IloNumVarArray var_vector_type;
		typedef IloArray<IloNumVarArray> var_matrix_type;

		user_assignment_solution<RealT> sol;

		const ::std::size_t np = ublasx::size(P); // Number of providers
		const ::std::size_t nb = ublasx::size(B); // Number of base stations
		const ::std::size_t nu = ublasx::size(U); // Number of users
		const ::std::size_t tot_np = ublasx::size(prov_energy_costs); // Total number of providers
		const ::std::size_t tot_nb = ublasx::size(bs_cats); // Total number of base stations
		const ::std::size_t tot_nu = ublasx::size(usr_cats); // Total number of users
		const ::std::size_t nbc = ublasx::size(bs_bws); // Number of base station categories
		const ::std::size_t nuc = ublasx::size(usr_qos_dl_rates); // Number of user categories

		DCS_DEBUG_TRACE("Max Profit :: np=" << np); //XXX
		DCS_DEBUG_TRACE("Max Profit :: nb=" << nb); //XXX
		DCS_DEBUG_TRACE("Max Profit :: nu=" << nu); //XXX
		DCS_DEBUG_TRACE("Max Profit :: tot_np=" << tot_np); //XXX
		DCS_DEBUG_TRACE("Max Profit :: tot_nb=" << tot_nb); //XXX
		DCS_DEBUG_TRACE("Max Profit :: tot_nu=" << tot_nu); //XXX
		DCS_DEBUG_TRACE("Max Profit :: nbc=" << nbc); //XXX
		DCS_DEBUG_TRACE("Max Profit :: nuc=" << nuc); //XXX
		DCS_DEBUG_TRACE("Max Profit :: P=" << P); //XXX
		DCS_DEBUG_TRACE("Max Profit :: B=" << B); //XXX
		DCS_DEBUG_TRACE("Max Profit :: U=" << U); //XXX
		DCS_DEBUG_TRACE("Max Profit :: bs_cats=" << bs_cats); //XXX
		DCS_DEBUG_TRACE("Max Profit :: bs_provs=" << bs_provs); //XXX
		//DCS_DEBUG_TRACE("Max Profit :: bs_heights=" << bs_heights); //XXX
		//DCS_DEBUG_TRACE("Max Profit :: bs_freqs=" << bs_freqs); //XXX
		//DCS_DEBUG_TRACE("Max Profit :: bs_caps=" << bs_caps); //XXX
		DCS_DEBUG_TRACE("Max Profit :: bs_bws=" << bs_bws); //XXX
		//DCS_DEBUG_TRACE("Max Profit :: bs_num_subcs=" << bs_num_subcs); //XXX
		//DCS_DEBUG_TRACE("Max Profit :: bs_tx_powers=" << bs_tx_powers); //XXX
		DCS_DEBUG_TRACE("Max Profit :: bs_const_powers=" << bs_const_powers); //XXX
		DCS_DEBUG_TRACE("Max Profit :: bs_load_powers=" << bs_load_powers); //XXX
		DCS_DEBUG_TRACE("Max Profit :: usr_cats=" << usr_cats); //XXX
		//DCS_DEBUG_TRACE("Max Profit :: usr_heights=" << usr_heights); //XXX
		DCS_DEBUG_TRACE("Max Profit :: usr_qos_dl_rates=" << usr_qos_dl_rates); //XXX
		//DCS_DEBUG_TRACE("Max Profit :: usr_therm_noises=" << usr_therm_noises); //XXX
		DCS_DEBUG_TRACE("Max Profit :: usr_bs_sinrs=" << usr_bs_sinrs); //XXX
		DCS_DEBUG_TRACE("Max Profit :: prov_energy_costs=" << prov_energy_costs); //XXX
		DCS_DEBUG_TRACE("Max Profit :: prov_usr_min_dl_rates=" << prov_usr_min_dl_rates); //XXX
		DCS_DEBUG_TRACE("Max Profit :: prov_usr_revenues=" << prov_usr_revenues); //XXX

		// preconditions
		DCS_ASSERT(nb <= tot_nb,
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "nb and tot_nb don't have a conformant size"));
		DCS_ASSERT(nu <= tot_nu,
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "nu and tot_nu don't have a conformant size"));
		DCS_ASSERT(tot_nb == ublasx::size(bs_cats),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "bs_cats has not a conformant size"));
		DCS_ASSERT(tot_nb == ublasx::size(bs_provs),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "bs_provs has not a conformant size"));
		//DCS_ASSERT(nbc == ublasx::size(bs_heights),
		//		   DCS_EXCEPTION_THROW(::std::invalid_argument, "bs_heights has not a conformant size"));
		//DCS_ASSERT(nbc == ublasx::size(bs_freqs),
		//		   DCS_EXCEPTION_THROW(::std::invalid_argument, "bs_freqs has not a conformant size"));
		//DCS_ASSERT(nbc == ublasx::size(bs_caps),
		//		   DCS_EXCEPTION_THROW(::std::invalid_argument, "bs_caps has not a conformant size"));
		DCS_ASSERT(nbc == ublasx::size(bs_bws),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "bs_bws has not a conformant size"));
		//DCS_ASSERT(nbc == ublasx::size(bs_num_subcs),
		//		   DCS_EXCEPTION_THROW(::std::invalid_argument, "bs_num_subcs has not a conformant size"));
		//DCS_ASSERT(nbc == ublasx::size(bs_tx_powers),
		//		   DCS_EXCEPTION_THROW(::std::invalid_argument, "bs_tx_powers has not a conformant size"));
		DCS_ASSERT(nbc == ublasx::size(bs_const_powers),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "bs_const_powers has not a conformant size"));
		DCS_ASSERT(nbc == ublasx::size(bs_load_powers),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "bs_load_powers has not a conformant size"));
		DCS_ASSERT(tot_nu == ublasx::size(usr_cats),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "usr_cats has not a conformant size"));
		//DCS_ASSERT(nu <= ublasx::size(usr_heights),
		//		   DCS_EXCEPTION_THROW(::std::invalid_argument, "usr_heights has not a conformant size"));
		DCS_ASSERT(nuc == ublasx::size(usr_qos_dl_rates),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "usr_qos_dl_rates has not a conformant size"));
		//DCS_ASSERT(nuc == ublasx::size(usr_therm_noises),
		//		   DCS_EXCEPTION_THROW(::std::invalid_argument, "usr_therm_noises has not a conformant size"));
		DCS_ASSERT(tot_nu == ublasx::num_rows(usr_bs_sinrs) && tot_nb == ublasx::num_columns(usr_bs_sinrs),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "usr_bs_sinrs has not a conformant size"));
		DCS_ASSERT(tot_np == ublasx::size(prov_energy_costs),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "prov_energy_costs has not a conformant size"));
		DCS_ASSERT(tot_np == ublasx::num_rows(prov_usr_min_dl_rates) && nuc == ublasx::num_columns(prov_usr_min_dl_rates),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "prov_usr_min_dl_rates has not a conformant size"));
		DCS_ASSERT(tot_np == ublasx::num_rows(prov_usr_revenues) && nuc == ublasx::num_columns(prov_usr_revenues),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "prov_usr_revenues has not a conformant size"));

		// Setting up vars
		try
		{
			// Initialize the Concert Technology app
			IloEnv env;

			IloModel model(env);

			model.setName("Max Profit");

			// Decision Variable

			// Variables n_{ij} \in Z^*: the number of subchannels assigned to user j when it is served by BS i
			var_matrix_type n(env, nb);
			for (::std::size_t i = 0; i < nb; ++i)
			{
				n[i] = var_vector_type(env, nu);

				for (::std::size_t j = 0 ; j < nu ; ++j)
				{
					::std::ostringstream oss;
					oss << "n[" << i << "][" << j << "]";
					n[i][j] = IloIntVar(env, 0, IloIntMax, oss.str().c_str());
					model.add(n[i][j]);
				}
			}

			// Variables x_{ij} \in \{0,1\}: 1 if user j is served by base station i, 0 otherwise.
			var_matrix_type x(env, nb);
			for (::std::size_t i = 0; i < nb; ++i)
			{
				x[i] = var_vector_type(env, nu);

				for (::std::size_t j = 0 ; j < nu ; ++j)
				{
					::std::ostringstream oss;
					oss << "x[" << i << "][" << j << "]";
					x[i][j] = IloBoolVar(env, 0, 1, oss.str().c_str());
					model.add(x[i][j]);
				}
			}

			// Variables y_{i} \in \{0,1\}: 1 if base station i is to be powered on, 0 otherwise.
			var_vector_type y(env, nb);
			for (::std::size_t i = 0; i < nb; ++i)
			{
				::std::ostringstream oss;
				oss << "y[" << i << "]";
				y[i] = IloBoolVar(env, 0, 1, oss.str().c_str());
				model.add(y[i]);
			}

			// Constraints

			::std::size_t cc(0); // Constraint counter

			// C1: \forall i \in B, \sum_{j \in U}{n_{ij}} \le \lfoor B_{C^b(i)}/F_{C^b(i)}^s \rfloor
			++cc;
			for (::std::size_t i = 0; i < nb; ++i)
			{
				const ::std::size_t b = B()(i);
				const ::std::size_t bc = bs_cats()(b);

				::std::ostringstream oss;
				oss << "C" << cc << "_{" << i << "}";

				IloExpr lhs(env);
				for (::std::size_t j = 0; j < nu; ++j)
				{
					lhs += n[i][j];
				}

				//IloConstraint cons(lhs <= y[i]*::std::floor(bs_bws()(bc)/0.18));
				IloConstraint cons(lhs <= ::std::floor(bs_bws()(bc)/0.18));
				cons.setName(oss.str().c_str());
				model.add(cons);
			}

			// C2: \forall j \in U, \sum_{i \in B}{x_{ij}} = 1
			++cc;
			for (::std::size_t j = 0; j < nu; ++j)
			{
				::std::ostringstream oss;
				oss << "C" << cc << "_{" << j << "}";

				IloExpr lhs(env);
				for (::std::size_t i = 0; i < nb; ++i)
				{
					lhs += x[i][j];
				}

				IloConstraint cons(lhs == 1);
				cons.setName(oss.str().c_str());
				model.add(cons);
			}

			// C3: \forall i \in B, \sum_{j \in U}{x_{ij}} \le |U|y_i
			++cc;
			for (::std::size_t i = 0; i < nb; ++i)
			{
				::std::ostringstream oss;
				oss << "C" << cc << "_{" << i << "}";

				IloConstraint cons(IloSum(x[i]) <= static_cast<RealT>(nu)*y[i]);
				cons.setName(oss.str().c_str());
				model.add(cons);
			}

/*
			// C4: \forall i \in B, \forall j \in U, 0.18 n_{ij} \log_2(1+SINR_{ij}) \le x_{ij}D_{C^u(j)}^d
			++cc;
			for (::std::size_t i = 0; i < nb; ++i)
			{
				const ::std::size_t b = B()(i);
				const ::std::size_t bc = bs_cats()(b);

				for (::std::size_t j = 0; j < nu; ++j)
				{
					const ::std::size_t u = U()(j);
					//const ::std::size_t uc = usr_cats()(u);

					::std::ostringstream oss;
					oss << "C" << cc << "_{" << i << "," << j << "}";

					//IloConstraint cons(0.18*n[i][j]*::log2(1+usr_bs_sinrs()(u,b)) <= x[i][j]*usr_qos_dl_rates()(uc));
					IloConstraint cons(0.18*n[i][j]*::log2(1+usr_bs_sinrs()(u,b)) <= x[i][j]*bs_bws()(bc)*6.0); //Assume 64-QAM == 6 bit/sec/Hz
					cons.setName(oss.str().c_str());
					model.add(cons);
				}
			}
*/

			// C4: \forall i \in B, \forall j \in U, n_{ij} \le x_{ij}B_{C^b(i)}/0.18
			++cc;
			for (::std::size_t i = 0; i < nb; ++i)
			{
				const ::std::size_t b = B()(i);
				const ::std::size_t bc = bs_cats()(b);

				for (::std::size_t j = 0; j < nu; ++j)
				{
					//const ::std::size_t u = U()(j);
					//const ::std::size_t uc = usr_cats()(u);

					::std::ostringstream oss;
					oss << "C" << cc << "_{" << i << "," << j << "}";

					IloConstraint cons(n[i][j] <= x[i][j]*::std::floor(bs_bws()(bc)/0.18));
					cons.setName(oss.str().c_str());
					model.add(cons);
				}
			}

			// C5: \forall i \in B, \forall j \in U, 0.18 n_{ij} \log_2(1+SINR_{ij}) \ge x_{ij}G_{P_b(i),C^u(j)}^d
			++cc;
			for (::std::size_t i = 0; i < nb; ++i)
			{
				const ::std::size_t b = B()(i);
				//const ::std::size_t bc = bs_cats()(b);
				const ::std::size_t p = bs_provs()(b);

				for (::std::size_t j = 0; j < nu; ++j)
				{
					const ::std::size_t u = U()(j);
					const ::std::size_t uc = usr_cats()(u);

					::std::ostringstream oss;
					oss << "C" << cc << "_{" << i << "," << j << "}";

					IloConstraint cons(0.18*n[i][j]*::log2(1+usr_bs_sinrs()(u,b)) >= x[i][j]*prov_usr_min_dl_rates()(p,uc));
					cons.setName(oss.str().c_str());
					model.add(cons);
				}
			}

			// C6: \forall i \in B, \sum_{j \in U} 0.18 n_{ij} \log_2(1+SINR_{ij}) \le y_{ij}C_{C^b(i)}^d
			++cc;
			for (::std::size_t i = 0; i < nb; ++i)
			{
				const ::std::size_t b = B()(i);
				const ::std::size_t bc = bs_cats()(b);

				::std::ostringstream oss;
				oss << "C" << cc << "_{" << i << "}";

				IloExpr lhs(env);
				for (::std::size_t j = 0; j < nu; ++j)
				{
					const ::std::size_t u = U()(j);
					//const ::std::size_t uc = usr_cats()(u);

					lhs += 0.18*n[i][j]*::log2(1+usr_bs_sinrs()(u,b));
				}

				IloConstraint cons(lhs <= bs_bws()(bc)*6);//FIXME: number of bit/sec/Hz is hard-coded
				cons.setName(oss.str().c_str());
				model.add(cons);
			}

/*
			// C7: \forall i \in B, \forall j \in U, 0.18 (n_{ij}-1) \log_2(1+SINR_{ij}) \le x_{ij}D_{C^u(j)}^d
			++cc;
			for (::std::size_t i = 0; i < nb; ++i)
			{
				const ::std::size_t b = B()(i);
				//const ::std::size_t bc = bs_cats()(b);
				for (::std::size_t j = 0; j < nu; ++j)
				{
					const ::std::size_t u = U()(j);
					const ::std::size_t uc = usr_cats()(u);

					::std::ostringstream oss;
					oss << "C" << cc << "_{" << i << "," << j << "}";

					IloConstraint cons(0.18*(n[i][j]-1)*::log2(1+usr_bs_sinrs()(u,b)) <= x[i][j]*usr_qos_dl_rates()(uc));
					cons.setName(oss.str().c_str());
					model.add(cons);
				}
			}
*/
 
			// Set objective
			//  max z = \sum_{i \in B}{\sum_{j \in U}^n{R_{P^b(i)}^{C^u(j)} x_{ij}}} - (\sum_{i \in B}{K_{P^b(i)}^w (W_{C^b(i)}^c y_i + W_{C^b(i)}^l\sum_{j \in U}{x_{ij}}) + \sum_{j \in U}{(x_{ij}-\frac{d_{ij}}{D_{C^u(j)}^d})R_{P^b(i)}^{C^u(j)}}})
			IloObjective z;
			{
/*
				IloExpr expr(env);
				for (::std::size_t i = 0; i < nb; ++i)
				{
					const ::std::size_t b(B()(i));
					const ::std::size_t bc(bs_cats()(b));
					const ::std::size_t p(bs_provs()(b));

					for (::std::size_t j = 0; j < nu; ++j)
					{
						const ::std::size_t u(U()(j));
						const ::std::size_t uc(usr_cats()(u));

						expr += prov_usr_revenues()(p,uc)*x[i][j];
					}
					expr -= prov_energy_costs()(p)*bs_const_powers()(bc)*y[i];
					for (::std::size_t j = 0; j < nu; ++j)
					{
						//const ::std::size_t u(U()(j));

						expr -= prov_energy_costs()(p)*bs_load_powers()(bc)*x[i][j];
					}
					for (::std::size_t j = 0; j < nu; ++j)
					{
						const ::std::size_t u(U()(j));
						const ::std::size_t uc(usr_cats()(u));

						expr -= (x[i][j]-d[i][j]/usr_qos_dl_rates()(uc))*prov_usr_revenues()(p,uc);
					}
				}
				z = IloMaximize(env, expr);
*/
				IloExpr revenue(env);
				for (::std::size_t i = 0; i < nb; ++i)
				{
					const ::std::size_t b = B()(i);
					const ::std::size_t p = bs_provs()(b);

					for (::std::size_t j = 0; j < nu; ++j)
					{
						const ::std::size_t u = U()(j);
						const ::std::size_t uc = usr_cats()(u);

						revenue += prov_usr_revenues()(p,uc)*x[i][j];
					}
				}
				IloExpr cost(env);
				for (::std::size_t i = 0; i < nb; ++i)
				{
					const ::std::size_t b = B()(i);
					const ::std::size_t bc = bs_cats()(b);
					const ::std::size_t p = bs_provs()(b);

					cost += prov_energy_costs()(p)*bs_const_powers()(bc)*y[i];
					for (::std::size_t j = 0; j < nu; ++j)
					{
						//const ::std::size_t u = U()(j);

						cost += prov_energy_costs()(p)*bs_load_powers()(bc)*x[i][j];
					}
					for (::std::size_t j = 0; j < nu; ++j)
					{
						const ::std::size_t u = U()(j);
						const ::std::size_t uc = usr_cats()(u);

						cost += (x[i][j]-IloMin(0.18*n[i][j]*::log2(1+usr_bs_sinrs()(u,b))/usr_qos_dl_rates()(uc),1.0))*prov_usr_revenues()(p,uc);
					}
				}
				z = IloMaximize(env, revenue-cost);
			}
			model.add(z);

			// Create the CPLEX solver and make 'model' the active ("extracted") model
			IloCplex solver(model);

			// write the model
#ifndef DCS_DEBUG
			solver.setOut(env.getNullStream());
			solver.setWarning(env.getNullStream());
#else // DCS_DEBUG
			solver.exportModel("cplex-maxprofit_model.lp");
#endif // DCS_DEBUG

			// Set Relative Gap to 1%: CPLEX will stop as soon as it has found a feasible integer solution proved to be within 1% of optimal.
			if (::dcs::math::float_traits<RealT>::definitely_greater(rel_gap_, 0))
			{
				solver.setParam(IloCplex::EpGap, rel_gap_);
			}
			if (::dcs::math::float_traits<RealT>::definitely_greater(time_lim_, 0))
			{
				solver.setParam(IloCplex::TiLim, time_lim_);
			}

			//solver.setParam(IloCplex::Reduce, CPX_PREREDUCE_PRIMALONLY);

			sol.solved = solver.solve();
			sol.optimal = false;

			IloAlgorithm::Status status = solver.getStatus();
			switch (status)
			{
				case IloAlgorithm::Optimal: // The algorithm found an optimal solution.
					sol.objective_value = static_cast<RealT>(solver.getObjValue());
					sol.optimal = true;
					break;
				case IloAlgorithm::Feasible: // The algorithm found a feasible solution, though it may not necessarily be optimal.

					sol.objective_value = static_cast<RealT>(solver.getObjValue());
					::dcs::log_warn(DCS_LOGGING_AT, "Optimization problem solved but non-optimal");
					break;
				case IloAlgorithm::Infeasible: // The algorithm proved the model infeasible (i.e., it is not possible to find an assignment of values to variables satisfying all the constraints in the model).
				case IloAlgorithm::Unbounded: // The algorithm proved the model unbounded.
				case IloAlgorithm::InfeasibleOrUnbounded: // The model is infeasible or unbounded.
				case IloAlgorithm::Error: // An error occurred and, on platforms that support exceptions, that an exception has been thrown.
				case IloAlgorithm::Unknown: // The algorithm has no information about the solution of the model.
				{
					::std::ostringstream oss;
					oss << "Optimization was stopped with status = " << status << " (CPLEX status = " << solver.getCplexStatus() << ", sub-status = " << solver.getCplexSubStatus() << ")";
					dcs::log_warn(DCS_LOGGING_AT, oss.str());
					return sol;
				}
			}

#ifdef DCS_DEBUG
			DCS_DEBUG_TRACE( "-------------------------------------------------------------------------------[" );
			DCS_DEBUG_TRACE( "- Objective value: " << sol.objective_value );

			DCS_DEBUG_TRACE( "- Decision variables: " );

			// Output n_{ij}
			for (::std::size_t i = 0; i < nb; ++i)
			{
				for (::std::size_t j = 0; j < nu; ++j)
				{
					DCS_DEBUG_STREAM << n[i][j].getName() << " = " << solver.getValue(n[i][j]) << " (" << static_cast<int>(IloRound(solver.getValue(n[i][j]))) << ")" << ::std::endl;
				}
			}

			// Output x_{ij}
			for (::std::size_t i = 0; i < nb; ++i)
			{
				for (::std::size_t j = 0; j < nu; ++j)
				{
					DCS_DEBUG_STREAM << x[i][j].getName() << " = " << solver.getValue(x[i][j]) << " (" << static_cast<bool>(IloRound(solver.getValue(x[i][j]))) << ")" << ::std::endl;
				}
			}

			// Output y{i}
			for (::std::size_t i = 0; i < nb; ++i)
			{
				DCS_DEBUG_STREAM << y[i].getName() << " = " << solver.getValue(y[i]) << " (" << static_cast<bool>(IloRound(solver.getValue(y[i]))) << ")" << ::std::endl;
			}

			DCS_DEBUG_TRACE( "- Derived variables: " );
			// Output d_{ij}
			for (::std::size_t i = 0; i < nb; ++i)
			{
				const ::std::size_t b = B()(i);

				for (::std::size_t j = 0; j < nu; ++j)
				{
					const ::std::size_t u = U()(j);

					DCS_DEBUG_STREAM << "d[" << i << "][" << j << "] = " << (0.18*static_cast<int>(IloRound(solver.getValue(n[i][j])))*::log2(1+usr_bs_sinrs()(u,b))) << ::std::endl;
				}
			}

			DCS_DEBUG_TRACE( "]-------------------------------------------------------------------------------" );
#endif // DCS_DEBUG

			sol.bs_user_allocations.resize(nb, nu, false);
			sol.bs_user_downlink_data_rates.resize(nu, false);
			sol.bs_power_states.resize(nb, false);
			sol.cost = 0;
			sol.kwatt = 0;
			for (::std::size_t i = 0; i < nb; ++i)
			{
				const ::std::size_t b = B()(i);
				const ::std::size_t bc = bs_cats()(b);
				const ::std::size_t p = bs_provs()(b);

				sol.bs_power_states(i) = static_cast<bool>(IloRound(solver.getValue(y[i])));
				for (::std::size_t j = 0; j < nu; ++j)
				{
					const ::std::size_t u = U()(j);
					//const ::std::size_t uc = usr_cats()(u);

					sol.bs_user_allocations(i,j) = static_cast<bool>(IloRound(solver.getValue(x[i][j])));
					if (sol.bs_user_allocations(i,j))
					{
						sol.bs_user_downlink_data_rates(j) = 0.18*static_cast<int>(IloRound(solver.getValue(n[i][j])))*::log2(1+usr_bs_sinrs()(u,b));
					}
					else
					{
						sol.bs_user_downlink_data_rates(j) =  0;
					}
				}
				if (sol.bs_power_states(i))
				{
					sol.kwatt += bs_const_powers()(bc);
					sol.cost += bs_const_powers()(bc)*prov_energy_costs()(p);
					for (::std::size_t j = 0; j < nu; ++j)
					{
						//const ::std::size_t u(U()(j));
						//const ::std::size_t uc(Ub()(u));

						if (sol.bs_user_allocations(i,j))
						{
							sol.kwatt += bs_load_powers()(bc);
							sol.cost += bs_load_powers()(bc)*prov_energy_costs()(p);
						}
					}
				}
			}
			//sol.kwatt *= 1.0e-3; // W -> kW

			z.end();
			y.end();
			x.end();
			n.end();

			// Close the Concert Technology app
			env.end();
		}
		catch (IloException const& e)
		{
			::std::ostringstream oss;
			oss << "Got exception from CPLEX: " << e.getMessage();
			DCS_EXCEPTION_THROW(::std::runtime_error, oss.str());
		}
		catch (...)
		{
			DCS_EXCEPTION_THROW(::std::runtime_error,
								"Unexpected error during the optimization");
		}

		return sol;
	}
#endif // DCS_BS_GT_USE_NATIVE_CPLEX_SOLVER

#ifdef DCS_BS_GT_USE_NATIVE_GUROBI_SOLVER
	private: template <typename PVectorT,
					   typename BVectorT,
					   typename UVectorT,
					   typename BsCatsVectorT,
					   typename BsProvsVectorT,
					   typename BsHeightsVectorT,
					   typename BsFreqsVectorT,
					   typename BsCapsVectorT,
					   typename BsBwsVectorT,
					   typename BsNumSubcsVectorT,
					   typename BsTxPowersVectorT,
					   typename BsConstPowersVectorT,
					   typename BsLoadPowersVectorT,
					   typename UsrCatsVectorT,
					   typename UsrHeightsVectorT,
					   typename UsrQosDlRatesVectorT,
					   typename UsrThermNoisesVectorT,
					   typename ProvEnergyCostsVectorT,
					   typename ProvUsrMinDlRatesMatrixT,
					   typename ProvUsrRevenuesMatrixT>
			user_assignment_solution<RealT> by_native_gurobi(::boost::numeric::ublas::vector_expression<PVectorT> const& P,
															 ::boost::numeric::ublas::vector_expression<BVectorT> const& B,
															 ::boost::numeric::ublas::vector_expression<UVectorT> const& U,
															 ::boost::numeric::ublas::vector_expression<BsCatsVectorT> const& bs_cats,
															 ::boost::numeric::ublas::vector_expression<BsProvsVectorT> const& bs_provs,
															 ::boost::numeric::ublas::vector_expression<BsHeightsVectorT> const& bs_heights,
															 ::boost::numeric::ublas::vector_expression<BsFreqsVectorT> const& bs_freqs,
															 ::boost::numeric::ublas::vector_expression<BsCapsVectorT> const& bs_caps,
															 ::boost::numeric::ublas::vector_expression<BsBwsVectorT> const& bs_bws,
															 ::boost::numeric::ublas::vector_expression<BsNumSubcsVectorT> const& bs_num_subcs,
															 ::boost::numeric::ublas::vector_expression<BsTxPowersVectorT> const& bs_tx_powers,
															 ::boost::numeric::ublas::vector_expression<BsConstPowersVectorT> const& bs_const_powers,
															 ::boost::numeric::ublas::vector_expression<BsLoadPowersVectorT> const& bs_load_powers,
															 ::boost::numeric::ublas::vector_expression<UsrCatsVectorT> const& usr_cats,
															 ::boost::numeric::ublas::vector_expression<UsrHeightsVectorT> const& usr_heights,
															 ::boost::numeric::ublas::vector_expression<UsrQosDlRatesVectorT> const& usr_qos_dl_rates,
															 ::boost::numeric::ublas::vector_expression<UsrThermNoisesVectorT> const& usr_therm_noises,
															 ::boost::numeric::ublas::vector_expression<ProvEnergyCostsVectorT> const& prov_energy_costs,
															 ::boost::numeric::ublas::matrix_expression<ProvUsrMinDlRatesMatrixT> const& prov_usr_min_dl_rates,
															 ::boost::numeric::ublas::matrix_expression<ProvUsrRevenuesMatrixT> const& prov_usr_revenues) const
	{
		namespace ublas = ::boost::numeric::ublas;
		namespace ublasx = ::boost::numeric::ublasx;

		typedef ::std::vector< ::std::vector<GRBVar> > grbvar_matrix_type;
		typedef ::std::vector<GRBVar> grbvar_vector_type;

		user_assignment_solution<RealT> sol;

		const ::std::size_t np = ublasx::size(P); // Number of providers
		const ::std::size_t nb = ublasx::size(B); // Number of base stations
		const ::std::size_t nu = ublasx::size(U); // Number of users
		const ::std::size_t nbc = ublasx::size(bs_const_powers); // Number of base station categories
		const ::std::size_t nuc = ublasx::size(usr_qos_dl_rates); // Number of user categories

		DCS_ASSERT(nb <= ublasx::size(bs_cats),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "bs_cats has not a conformant size"));
		DCS_ASSERT(nu <= ublasx::size(usr_cats),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "usr_cats has not a conformant size"));
		DCS_ASSERT(nb <= ublasx::size(bs_provs),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "bs_provs has not a conformant size"));
		DCS_ASSERT(nbc == ublasx::size(bs_load_powers),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "bs_load_powers has not a conformant size"));
		DCS_ASSERT(nbc == ublasx::size(bs_caps),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "bs_caps has not a conformant size"));
		DCS_ASSERT(nuc == ublasx::size(usr_qos_dl_rates),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "usr_qos_dl_rates has not a conformant size"));
		DCS_ASSERT(np <= ublasx::num_rows(prov_usr_min_dl_rates) && nuc == ublasx::num_columns(prov_usr_min_dl_rates),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "prov_usr_min_dl_rates has not a conformant size"));
		DCS_ASSERT(np <= ublasx::num_rows(prov_usr_revenues) && nuc == ublasx::num_columns(prov_usr_revenues),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "prov_usr_revenues has not a conformant size"));

		DCS_DEBUG_TRACE("Max Profit :: P=" << P); //XXX
		DCS_DEBUG_TRACE("Max Profit :: B=" << B); //XXX
		DCS_DEBUG_TRACE("Max Profit :: U=" << U); //XXX
		DCS_DEBUG_TRACE("Max Profit :: bs_cats=" << bs_cats); //XXX
		DCS_DEBUG_TRACE("Max Profit :: usr_cats=" << usr_cats); //XXX
		DCS_DEBUG_TRACE("Max Profit :: bs_provs=" << bs_provs); //XXX
		DCS_DEBUG_TRACE("Max Profit :: bs_const_powers=" << bs_const_powers); //XXX
		DCS_DEBUG_TRACE("Max Profit :: bs_load_powers=" << bs_load_powers); //XXX
		DCS_DEBUG_TRACE("Max Profit :: prov_energy_costs=" << prov_energy_costs); //XXX
		DCS_DEBUG_TRACE("Max Profit :: bs_caps=" << bs_caps); //XXX
		DCS_DEBUG_TRACE("Max Profit :: usr_qos_dl_rates=" << usr_qos_dl_rates); //XXX
		DCS_DEBUG_TRACE("Max Profit :: prov_usr_min_dl_rates=" << prov_usr_min_dl_rates); //XXX
		DCS_DEBUG_TRACE("Max Profit :: prov_usr_revenues=" << prov_usr_revenues); //XXX

		// Setting up vars
		try
		{
			// Initialize the Concert Technology app
			GRBEnv env;

			GRBModel model(env);

			model.set(GRB_StringAttr_ModelName, "Max Profit");
#ifdef DCS_DEBUG
			env.set(GRB_IntParam_OutputFlag, 1);
			env.set(GRB_IntParam_LogToConsole, 1);
#else
			env.set(GRB_IntParam_OutputFlag, 0);
			env.set(GRB_IntParam_LogToConsole, 0);
#endif // DCS_DEBUG

			// Decision Variable

			// Variables d_{ij} \in R^*: the downlink data rate assigned to user j when it is served by BS i
			grbvar_matrix_type d(nb, grbvar_vector_type(nu));
			for (::std::size_t i = 0; i < nb; ++i)
			{
				for (::std::size_t j = 0 ; j < nu ; ++j)
				{
					::std::ostringstream oss;
					oss << "d[" << i << "][" << j << "]";
					d[i][j] = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, oss.str());
				}
			}

			// Variables x_{ij} \in \{0,1\}: 1 if user j is served by base station i, 0 otherwise.
			grbvar_matrix_type x(nb, grbvar_vector_type(nu));
			for (::std::size_t i = 0; i < nb; ++i)
			{
				for (::std::size_t j = 0 ; j < nu ; ++j)
				{
					::std::ostringstream oss;
					oss << "x[" << i << "][" << j << "]";
					x[i][j] = model.addVar(0, 1, 0, GRB_BINARY, oss.str());
				}
			}

			// Variables y_{i} \in \{0,1\}: 1 if base station i is to be powered on, 0 otherwise.
			grbvar_vector_type y(nb);
			for (::std::size_t i = 0; i < nb; ++i)
			{
				::std::ostringstream oss;
				oss << "y[" << i << "]";
				y[i] = model.addVar(0, 1, 0, GRB_BINARY, oss.str());
			}

//			// Objective variable z
//			GRBVar z = model.addVar(-GRB_INFINITY, GRB_INFINITY, 1, GRB_CONTINUOUS, std::string("z"));
//
//			model.set(GRB_IntAttr_ModelSense, GRB_MAXIMIZE);

			// Integrates new variables
			model.update();

			// Constraints

			::std::size_t cc(0); // Constraint counter

			// C1: \forall i \in B, \sum_{j \in U}{d_{ij}} \le T_{C^b(i)}^d
			++cc;
			for (::std::size_t i = 0; i < nb; ++i)
			{
				const ::std::size_t b(B()(i));
				const ::std::size_t bc(bs_cats()(b));

				::std::ostringstream oss;
				oss << "C" << cc << "_{" << i << "}";

				GRBLinExpr lhs(0);
				for (::std::size_t j = 0; j < nu; ++j)
				{
					lhs += d[i][j];
				}

				model.addConstr(lhs, GRB_LESS_EQUAL, bs_caps()(bc), oss.str());
			}

			// C2: \forall j \in U, \sum_{i \in B}{x_{ij}} = 1
			++cc;
			for (::std::size_t j = 0; j < nu; ++j)
			{
				::std::ostringstream oss;
				oss << "C" << cc << "_{" << j << "}";

				GRBLinExpr lhs(0);
				for (::std::size_t i = 0; i < nb; ++i)
				{
					lhs += x[i][j];
				}

				model.addConstr(lhs, GRB_EQUAL, 1, oss.str());
			}

			// C3: \forall i \in B, \sum_{j \in U}{x_{ij}} \le |U|y_i
			++cc;
			for (::std::size_t i = 0; i < nb; ++i)
			{
				::std::ostringstream oss;
				oss << "C" << cc << "_{" << i << "}";

				GRBLinExpr lhs(0);
				for (::std::size_t j = 0; j < nu; ++j)
				{
					lhs += x[i][j];
				}

				model.addConstr(lhs, GRB_LESS_EQUAL, static_cast<RealT>(nu)*y[i], oss.str());
			}

			// C4: \forall i \in B, \forall j \in U, d_{ij} \le x_{ij}D_{C^u(j)}^d
			++cc;
			for (::std::size_t i = 0; i < nb; ++i)
			{
				//const ::std::size_t b(B()(i));

				for (::std::size_t j = 0; j < nu; ++j)
				{
					const ::std::size_t u(U()(j));
					const ::std::size_t uc(usr_cats()(u));

					::std::ostringstream oss;
					oss << "C" << cc << "_{" << i << "," << j << "}";

					model.addConstr(d[i][j], GRB_LESS_EQUAL, x[i][j]*usr_qos_dl_rates()(uc), oss.str());
				}
			}

			// C5: \forall i \in B, \forall j \in U, d_{ij} \ge x_{ij}G_{P_b(i),C^u(j)}^d
			++cc;
			for (::std::size_t i = 0; i < nb; ++i)
			{
				const ::std::size_t b(B()(i));

				for (::std::size_t j = 0; j < nu; ++j)
				{
					const ::std::size_t u(U()(j));
					const ::std::size_t uc(usr_cats()(u));
					const ::std::size_t p(bs_provs()(b));

					::std::ostringstream oss;
					oss << "C" << cc << "_{" << i << "," << j << "}";

					model.addConstr(d[i][j], GRB_GREATER_EQUAL, x[i][j]*prov_usr_min_dl_rates()(p,uc), oss.str());
				}
			}

			// Set objective
			//  z = \sum_{i \in B}{\sum_{j \in U}^n{R_{P^b(i)}^{C^u(j)} x_{ij}}} - (\sum_{i \in B}{K_{P^b(i)}^w (W_{C^b(i)}^c y_i + W_{C^b(i)}^l\sum_{j \in U}{x_{ij}}) + \sum_{j \in U}{(x_{ij}-\frac{d_{ij}}{D_{C^u(j)}^d})R_{P^b(i)}^{C^u(j)}}})
			++cc;
			GRBLinExpr revenue(0);
			for (::std::size_t i = 0; i < nb; ++i)
			{
				const ::std::size_t b(B()(i));
				const ::std::size_t p(bs_provs()(b));

				for (::std::size_t j = 0; j < nu; ++j)
				{
					const ::std::size_t u(U()(j));
					const ::std::size_t uc(usr_cats()(u));

					revenue += prov_usr_revenues()(p,uc)*x[i][j];
				}
			}
			GRBLinExpr cost(0);
			for (::std::size_t i = 0; i < nb; ++i)
			{
				const ::std::size_t b(B()(i));
				const ::std::size_t bc(bs_cats()(b));
				const ::std::size_t p(bs_provs()(b));

				cost += prov_energy_costs()(p)*bs_const_powers()(bc)*y[i];
				for (::std::size_t j = 0; j < nu; ++j)
				{
					//const ::std::size_t u(U()(j));

					cost += prov_energy_costs()(p)*bs_load_powers()(bc)*x[i][j];
				}
				for (::std::size_t j = 0; j < nu; ++j)
				{
					const ::std::size_t u(U()(j));
					const ::std::size_t uc(usr_cats()(u));

					cost += (x[i][j]-d[i][j]/usr_qos_dl_rates()(uc))*prov_usr_revenues()(p,uc);
				}
			}
			model.setObjective(revenue-cost, GRB_MAXIMIZE);


			model.update();

			// Set Relative Gap to 1%: CPLEX will stop as soon as it has found a feasible integer solution proved to be within 1% of optimal.
			if (::dcs::math::float_traits<RealT>::definitely_greater(rel_gap_, 0))
			{
				env.set(GRB_DoubleParam_MIPGap, rel_gap_);
			}
			if (::dcs::math::float_traits<RealT>::definitely_greater(time_lim_, 0))
			{
				env.set(GRB_DoubleParam_TimeLimit, time_lim_);
			}
			//env.set(GRB_IntParam_DualReductions, 0);

			// write the model
#ifdef DCS_DEBUG
			model.write("gurobi-maxprofit_model.lp");
			//model.write("gurobi-maxprofit_model.mps");
#endif // DCS_DEBUG

			model.optimize();

			sol.solved = false;
			sol.optimal = false;

			int status = model.get(GRB_IntAttr_Status);
			switch (status)
			{
				case GRB_OPTIMAL: // The algorithm found an optimal solution.
					sol.objective_value = static_cast<RealT>(model.get(GRB_DoubleAttr_ObjVal));
					sol.solved = true;
					sol.optimal = true;
					break;
				case GRB_SUBOPTIMAL: // The algorithm found a feasible solution, though it may not necessarily be optimal.
					sol.objective_value = static_cast<RealT>(model.get(GRB_DoubleAttr_ObjVal));
					::dcs::log_warn(DCS_LOGGING_AT, "Optimization problem solved but non-optimal");
					sol.solved = true;
					break;
				default:
				//case GRB_INFEASIBLE: // The algorithm proved the model infeasible (i.e., it is not possible to find an assignment of values to variables satisfying all the constraints in the model).
				//case GRB_UNBOUNDED: // The algorithm proved the model unbounded.
				//case GRB_INF_OR_UNBD: // The model is infeasible or unbounded.
				////case IloAlgorithm::Error: // An error occurred and, on platforms that support exceptions, that an exception has been thrown.
				////case IloAlgorithm::Unknown: // The algorithm has no information about the solution of the model.
				{
					::std::ostringstream oss;
					oss << "Optimization was stopped with status = " << status;// << " (CPLEX status = " << solver.getCplexStatus() << ", sub-status = " << solver.getCplexSubStatus() << ")";
					dcs::log_warn(DCS_LOGGING_AT, oss.str());
					return sol;
				}
			}

#ifdef DCS_DEBUG
			DCS_DEBUG_TRACE( "-------------------------------------------------------------------------------[" );
			DCS_DEBUG_TRACE( "- Objective value: " << sol.objective_value);

			DCS_DEBUG_TRACE( "- Decision variables: " );

			// Output d_{ij}
			for (::std::size_t i = 0; i < nb; ++i)
			{
				for (::std::size_t j = 0; j < nu; ++j)
				{
					DCS_DEBUG_STREAM << d[i][j].get(GRB_StringAttr_VarName) << " = " << d[i][j].get(GRB_DoubleAttr_X) << " (" << static_cast<RealT>(d[i][j].get(GRB_DoubleAttr_X)) << ")" << ::std::endl;
				}
			}

			// Output x_{ij}
			for (::std::size_t i = 0; i < nb; ++i)
			{
				for (::std::size_t j = 0; j < nu; ++j)
				{
					DCS_DEBUG_STREAM << x[i][j].get(GRB_StringAttr_VarName) << " = " << x[i][j].get(GRB_DoubleAttr_X) << " (" << static_cast<bool>(x[i][j].get(GRB_DoubleAttr_X)) << ")" << ::std::endl;
				}
			}

			// Output y{i}
			for (::std::size_t i = 0; i < nb; ++i)
			{
				DCS_DEBUG_STREAM << y[i].get(GRB_StringAttr_VarName) << " = " << y[i].get(GRB_DoubleAttr_X) << " (" << static_cast<bool>(y[i].get(GRB_DoubleAttr_X)) << ")" << ::std::endl;
			}
			DCS_DEBUG_TRACE( "]-------------------------------------------------------------------------------" );
#endif // DCS_DEBUG

			sol.bs_user_allocations.resize(nb, nu, false);
			sol.bs_user_downlink_data_rates.resize(nu, false);
			sol.bs_power_states.resize(nb, false);
			sol.cost = 0;
			sol.kwatt = 0;
			for (::std::size_t i = 0; i < nb; ++i)
			{
				const ::std::size_t b(B()(i));
				const ::std::size_t bc(bs_cats()(b));
				const ::std::size_t p(bs_provs()(b));

				sol.bs_power_states(i) = static_cast<bool>(y[i].get(GRB_DoubleAttr_X));
				for (::std::size_t j = 0; j < nu; ++j)
				{
					sol.bs_user_allocations(i,j) = static_cast<bool>(x[i][j].get(GRB_DoubleAttr_X));
					if (sol.bs_user_allocations(i,j))
					{
						sol.bs_user_downlink_data_rates(j) =  static_cast<RealT>(d[i][j].get(GRB_DoubleAttr_X));
					}
					else
					{
						sol.bs_user_downlink_data_rates(j) =  0;
					}
				}
				if (sol.bs_power_states(i))
				{
					sol.kwatt += bs_const_powers()(bc);
					sol.cost += bs_const_powers()(bc)*prov_energy_costs()(p);
					for (::std::size_t j = 0; j < nu; ++j)
					{
						const ::std::size_t u(U()(j));
						const ::std::size_t uc(usr_cats()(u));

						if (sol.bs_user_allocations(i,j))
						{
							sol.kwatt += bs_load_powers()(bc);
							sol.cost += bs_load_powers()(bc)*prov_energy_costs()(p);
							sol.cost += (1-sol.bs_user_downlink_data_rates(j)/usr_qos_dl_rates()(uc))*prov_usr_revenues()(p,uc);
						}
					}
				}
			}
			//sol.kwatt *= 1.0e-3; // W -> kW
		}
		catch (GRBException const& e)
		{
			::std::ostringstream oss;
			oss << "Got exception from GUROBI: " << e.getMessage() << " (" << e.getErrorCode() << ")";
			DCS_EXCEPTION_THROW(::std::runtime_error, oss.str());
		}
		catch (...)
		{
			DCS_EXCEPTION_THROW(::std::runtime_error,
								"Unexpected error during the optimization");
		}

		return sol;
	}
#endif // DCS_BS_GT_USE_NATIVE_GUROBI_SOLVER

#ifdef DCS_BS_GT_USE_OSI_SOLVER
	private: template <typename PVectorT,
					   typename BVectorT,
					   typename UVectorT,
					   typename BsCatsVectorT,
					   typename BsProvsVectorT,
					   typename BsHeightsVectorT,
					   typename BsFreqsVectorT,
					   typename BsCapsVectorT,
					   typename BsBwsVectorT,
					   typename BsNumSubcsVectorT,
					   typename BsTxPowersVectorT,
					   typename BsConstPowersVectorT,
					   typename BsLoadPowersVectorT,
					   typename UsrCatsVectorT,
					   typename UsrHeightsVectorT,
					   typename UsrQosDlRatesVectorT,
					   typename UsrThermNoisesVectorT,
					   typename ProvEnergyCostsVectorT,
					   typename ProvUsrMinDlRatesMatrixT,
					   typename ProvUsrRevenuesMatrixT>
			user_assignment_solution<RealT> by_osi(::boost::numeric::ublas::vector_expression<PVectorT> const& P,
												   ::boost::numeric::ublas::vector_expression<BVectorT> const& B,
												   ::boost::numeric::ublas::vector_expression<UVectorT> const& U,
												   ::boost::numeric::ublas::vector_expression<BsCatsVectorT> const& bs_cats,
												   ::boost::numeric::ublas::vector_expression<BsProvsVectorT> const& bs_provs,
												   ::boost::numeric::ublas::vector_expression<BsHeightsVectorT> const& bs_heights,
												   ::boost::numeric::ublas::vector_expression<BsFreqsVectorT> const& bs_freqs,
												   ::boost::numeric::ublas::vector_expression<BsCapsVectorT> const& bs_caps,
												   ::boost::numeric::ublas::vector_expression<BsBwsVectorT> const& bs_bws,
												   ::boost::numeric::ublas::vector_expression<BsNumSubcsVectorT> const& bs_num_subcs,
												   ::boost::numeric::ublas::vector_expression<BsTxPowersVectorT> const& bs_tx_powers,
												   ::boost::numeric::ublas::vector_expression<BsConstPowersVectorT> const& bs_const_powers,
												   ::boost::numeric::ublas::vector_expression<BsLoadPowersVectorT> const& bs_load_powers,
												   ::boost::numeric::ublas::vector_expression<UsrCatsVectorT> const& usr_cats,
												   ::boost::numeric::ublas::vector_expression<UsrHeightsVectorT> const& usr_heights,
												   ::boost::numeric::ublas::vector_expression<UsrQosDlRatesVectorT> const& usr_qos_dl_rates,
												   ::boost::numeric::ublas::vector_expression<UsrThermNoisesVectorT> const& usr_therm_noises,
												   ::boost::numeric::ublas::vector_expression<ProvEnergyCostsVectorT> const& prov_energy_costs,
												   ::boost::numeric::ublas::matrix_expression<ProvUsrMinDlRatesMatrixT> const& prov_usr_min_dl_rates,
												   ::boost::numeric::ublas::matrix_expression<ProvUsrRevenuesMatrixT> const& prov_usr_revenues) const
	{
		namespace ublas = ::boost::numeric::ublas;
		namespace ublasx = ::boost::numeric::ublasx;

		user_assignment_solution<RealT> sol;

		const ::std::size_t np = ublasx::size(P); // Number of providers
		const ::std::size_t nb = ublasx::size(B); // Number of base stations
		const ::std::size_t nu = ublasx::size(U); // Number of users
		const ::std::size_t nbc = ublasx::size(bs_const_powers); // Number of base station categories
		const ::std::size_t nuc = ublasx::size(usr_qos_dl_rates); // Number of user categories

		DCS_ASSERT(nb <= ublasx::size(bs_cats),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "bs_cats has not a conformant size"));
		DCS_ASSERT(nu <= ublasx::size(usr_cats),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "usr_cats has not a conformant size"));
		DCS_ASSERT(nb <= ublasx::size(bs_provs),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "bs_provs has not a conformant size"));
		DCS_ASSERT(nbc == ublasx::size(bs_load_powers),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "bs_load_powers has not a conformant size"));
		DCS_ASSERT(nbc == ublasx::size(bs_caps),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "bs_caps has not a conformant size"));
		DCS_ASSERT(nuc == ublasx::size(usr_qos_dl_rates),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "usr_qos_dl_rates has not a conformant size"));
		DCS_ASSERT(np <= ublasx::num_rows(prov_usr_revenues) && nuc == ublasx::num_columns(prov_usr_revenues),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "prov_usr_revenues has not a conformant size"));

		DCS_DEBUG_TRACE("Max Profit :: P=" << P); //XXX
		DCS_DEBUG_TRACE("Max Profit :: B=" << B); //XXX
		DCS_DEBUG_TRACE("Max Profit :: U=" << U); //XXX
		DCS_DEBUG_TRACE("Max Profit :: bs_cats=" << bs_cats); //XXX
		DCS_DEBUG_TRACE("Max Profit :: usr_cats=" << usr_cats); //XXX
		DCS_DEBUG_TRACE("Max Profit :: bs_provs=" << bs_provs); //XXX
		DCS_DEBUG_TRACE("Max Profit :: bs_const_powers=" << bs_const_powers); //XXX
		DCS_DEBUG_TRACE("Max Profit :: bs_load_powers=" << bs_load_powers); //XXX
		DCS_DEBUG_TRACE("Max Profit :: prov_energy_costs=" << prov_energy_costs); //XXX
		DCS_DEBUG_TRACE("Max Profit :: bs_caps=" << bs_caps); //XXX
		DCS_DEBUG_TRACE("Max Profit :: usr_qos_dl_rates=" << usr_qos_dl_rates); //XXX
		DCS_DEBUG_TRACE("Max Profit :: prov_usr_revenues=" << prov_usr_revenues); //XXX

		// Setting up vars
		try
		{
			::boost::shared_ptr<OsiSolverInterface> p_solver;
			p_solver = ::boost::make_shared<OsiCbcSolverInterface>();

			::flopc::MP_model model(p_solver.get());

#ifdef DCS_DEBUG
			model.verbose();
#else
			model.silent();
#endif //DCS_DEBUG

			//FIXME
			// Set Relative Gap to 1%: the solver will stop as soon as it has found a feasible integer solution proved to be within 1% of optimal.
			if (::dcs::math::float_traits<RealT>::definitely_greater(rel_gap_, 0))
			{
				//TODO
			}
			if (::dcs::math::float_traits<RealT>::definitely_greater(time_lim_, 0))
			{
				//TODO
			}

			::flopc::MP_set mp_B(nb);
			::flopc::MP_set mp_U(nu);

			::flopc::MP_data mp_bs_caps(mp_B);
			::flopc::MP_data mp_prov_energy_costs(mp_B);
			::flopc::MP_data mp_prov_energy_costsbs_const_powers(mp_B);
			::flopc::MP_data mp_prov_energy_costsbs_load_powers(mp_B);
			::flopc::MP_data mp_usr_qos_dl_rates(mp_U);
			::flopc::MP_data mp_prov_usr_revenues(mp_B,mp_U);
			for (::std::size_t i = 0; i < nb; ++i)
			{
				const ::std::size_t b(B()(i));
				const ::std::size_t bc(bs_cats()(b));
				const ::std::size_t p(bs_provs()(b));

				mp_bs_caps(i) = bs_caps()(bc);
				mp_prov_energy_costsbs_const_powers(i) = prov_energy_costs()(p)*bs_const_powers()(bc);
				mp_prov_energy_costsbs_load_powers(i) = prov_energy_costs()(p)*bs_load_powers()(bc);

				for (::std::size_t j = 0; j < nu; ++j)
				{
					const ::std::size_t u(U()(j));
					const ::std::size_t uc(usr_cats()(u));

					mp_prov_usr_revenues(i,j) = prov_usr_revenues()(p,uc);
				}
			}
			for (::std::size_t j = 0; j < nu; ++j)
			{
				const ::std::size_t u(U()(j));
				const ::std::size_t uc(usr_cats()(u));

				mp_usr_qos_dl_rates(j) = usr_qos_dl_rates()(uc);
			}

			// Decision Variable

			// Variables d_{ij} \in R^*: the downlink data rate assigned to user j when it is served by BS i
			::flopc::MP_variable d(mp_B,mp_U);
			d.lowerLimit(mp_B,mp_U) = 0;
			d.upperLimit(mp_B,mp_U) = model.getInfinity();;

			// Variables x_{ij} \in \{0,1\}: 1 if user j is served by base station i, 0 otherwise.
			::flopc::MP_variable x(mp_B,mp_U);
			x.binary();

			// Variables y_{i} \in \{0,1\}: 1 if base station i is to be powered on, 0 otherwise.
			::flopc::MP_variable y(mp_B);
			y.binary();

			// Constraints

			::std::ostringstream oss;
			::std::size_t cc(0); // Constraint counter

			// C1: \forall i \in B, \sum_{j \in U}{d_{ij}} \le T_{C^b(i)}^d
			++cc;
			::flopc::MP_constraint c01(mp_B);
			oss.str("");
			oss << "C" << cc;
			c01.setName(oss.str());
			c01(mp_B) = ::flopc::sum(mp_U, d(mp_B,mp_U)) <= mp_bs_caps(mp_B);
			model.add(c01);

			// C2: \forall j \in U, \sum_{i \in B}{x_{ij}} = 1
			++cc;
			::flopc::MP_constraint c02(mp_U);
			oss.str("");
			oss << "C" << cc;
			c02.setName(oss.str());
			c02(mp_U) = ::flopc::sum(mp_B, x(mp_B,mp_U)) <= 1;
			model.add(c02);

			// C3: \forall i \in B, \sum_{j \in U}{x_{ij}} \le |U|y_i
			++cc;
			::flopc::MP_constraint c03(mp_B);
			oss.str("");
			oss << "C" << cc;
			c03.setName(oss.str());
			c03(mp_B) = ::flopc::sum(mp_U, x(mp_B,mp_U)) <= static_cast<RealT>(np)*y(mp_B);
			model.add(c03);

			// C4: \forall i \in B, \forall j \in U, d_{ij} \le x_{ij}D_{C^u(j)}^d
			++cc;
			::flopc::MP_constraint c04(mp_B);
			oss.str("");
			oss << "C" << cc;
			c04.setName(oss.str());
			c04(mp_B,mp_U) = d(mp_B,mp_U) <= x(mp_B,mp_U)*mp_usr_qos_dl_rates(mp_U);
			model.add(c04);

			// Set objective
			//  max z = \sum_{i \in B}{\sum_{j \in U}^n{R_{P^b(i)}^{C^u(j)} x_{ij}}} - (\sum_{i \in B}{K_{P^b(i)}^w (W_{C^b(i)}^c y_i + W_{C^b(i)}^l\sum_{j \in U}{x_{ij}}) + \sum_{j \in U}{(x_{ij}-\frac{d_{ij}}{D_{C^u(j)}^d})R_{P^b(i)}^{C^u(j)}}})
			::flopc::MP_expression z;
			z = ::flopc::sum(mp_B, ::flopc::sum(mp_U, mp_prov_usr_revenues(mp_B,mp_U)*x(mp_B,mp_U)) - mp_prov_energy_costsbs_const_powers*y(mp_B) - ::flopc::sum(mp_U, mp_prov_energy_costsbs_load_powers*x(mp_B,mp_U)) - ::flopc::sum(mp_U, (x(mp_B,mp_U) - d(mp_B,mp_U)/mp_usr_qos_dl_rates(mp_U))*mp_prov_usr_revenues(mp_B,mp_U)));
			model.maximize(z);

			sol.optimal = false;

			if (model.getStatus() == ::flopc::MP_model::OPTIMAL)
			{
				sol.objective_value = model->getObjValue();
				sol.optimal = true;
			}
			else
			{
				::std::ostringstream oss;
				oss << "Optimization was stopped with status = " << model.getStatus();
				dcs::log_warn(DCS_LOGGING_AT, oss.str());
				return sol;
			}

#ifdef DCS_DEBUG
			DCS_DEBUG_TRACE( "-------------------------------------------------------------------------------[" );
			DCS_DEBUG_TRACE( "- Objective value: " << sol.objective_value );

			DCS_DEBUG_TRACE( "- Decision variables: " );

			// Output d_{ij}
			d.display("d");

			// Output x_{ij}
			x.display("x");

			// Output y{i}
			y.display("y");
			DCS_DEBUG_TRACE( "]-------------------------------------------------------------------------------" );
#endif // DCS_DEBUG

			sol.bs_user_allocations.resize(nb, nu, false);
			sol.bs_user_downlink_data_rates.resize(nu, false);
			sol.bs_power_states.resize(nb, false);
			sol.cost = 0;
			sol.kwatt = 0;
			for (::std::size_t i = 0; i < nb; ++i)
			{
				const ::std::size_t b(B()(i));
				const ::std::size_t bc(bs_cats()(b));
				const ::std::size_t p(bs_provs()(b));

				for (::std::size_t j = 0; j < nu; ++j)
				{
					sol.bs_user_allocations(i,j) = static_cast<bool>(x.level(i,j));
					if (sol.bs_user_allocations(i,j))
					{
						sol.bs_user_downlink_data_rates(j) =  static_cast<RealT>(d.level(i,j));
					}
					else
					{
						sol.bs_user_downlink_data_rates(j) =  0;
					}
				}
				sol.bs_power_states(i) = static_cast<bool>(y.level(i));
				if (sol.bs_power_states(i))
				{
					sol.kwatt += bs_const_powers()(bc);
					sol.cost += bs_const_powers()(bc)*prov_energy_costs()(p);
					for (::std::size_t j = 0; j < nu; ++j)
					{
						const ::std::size_t u(U()(j));
						const ::std::size_t uc(bs_cats()(u));

						if (sol.bs_user_allocations(i,j))
						{
							sol.kwatt += bs_load_powers()(bc);
							sol.cost += bs_load_powers()(bc)*prov_energy_costs()(p);
							sol.cost += (1-sol.bs_user_downlink_data_rates(j)/usr_qos_dl_rates()(uc))*prov_usr_revenues()(p,uc);
						}
					}
				}
			}
			//sol.kwatt *= 1.0e-3; // W -> kW
		}
		catch (CoinError const& e)
		{
			::std::ostringstream oss;
			oss << "Got exception from OSI: " << e.message();
			DCS_EXCEPTION_THROW(::std::runtime_error, oss.str());
		}
		catch (...)
		{
			DCS_EXCEPTION_THROW(::std::runtime_error,
								"Unexpected error during the optimization");
		}

		return sol;
	}
#endif // DCS_BS_GT_USE_OSI_SOLVER


	private: RealT rel_gap_;
	private: RealT time_lim_;
}; // optimal_max_profit_user_assignment_solver

}} // Namespace dcs::bs


#endif // DCS_BS_MAX_PROFIT_HPP
