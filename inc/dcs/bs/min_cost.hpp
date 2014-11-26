/**
 * \file dcs/bs/min_cost.hpp
 *
 * \brief Min cost computation for a set of coalitions.
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

#ifndef DCS_BS_MIN_COST_HPP
#define DCS_BS_MIN_COST_HPP


#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <boost/numeric/ublasx/operation/size.hpp>
#include <cstddef>
#include <dcs/assert.hpp>
#include <dcs/bs/user_assignment_solution.hpp>
#include <dcs/debug.hpp>
#include <dcs/exception.hpp>
#include <dcs/logging.hpp>
#include <dcs/math/traits/float.hpp>
#include <iostream>
#include <ilconcert/iloalg.h>
#include <ilconcert/iloenv.h>
#include <ilconcert/iloexpression.h>
#include <ilconcert/ilomodel.h>
#include <ilcplex/ilocplex.h>
#include <limits>
#include <sstream>
#include <stdexcept>


namespace dcs { namespace bs {

/**
 * Solve the following optimization problem
 *  min z = \sum_{i \in B}{K_{P^b(i)}^w (W_{C^b(i)}^c y_i + W_{C^b(i)}^l\sum_{j \in U}{x_{ij}}) + \sum_{j \in U}{(x_{ij}-\frac{d_{ij}}{D_{C^u(j)}^d})R_{P^b(i)}^{C^u(j)}}}
 *  s.t.
 *   \sum_{j \in U}{d_{ij}} \le T_{P^b(i)}^d, \forall i \in B,
 *   \sum_{i \in B}{x_{ij}} = 1, \forall j \in U,
 *   \sum_{j \in U}{x_{ij}} \le |U|y_i, \forall i \in B,
 *   d_{ij} \ge x_{ij}D_{C^u(j)}^d, \forall i \in B, \forall j \in U,
 *   d_{ij} \in R^*, \forall i \in B, j \in U,
 *   x_{ij} \in \{0,1\}, \forall i \in B, j \in U,
 *   y_i \in \{0,1\}, \forall i \in B.
 */
template <typename RealT>
class optimal_min_cost_user_assignment_solver
{
	public: explicit optimal_min_cost_user_assignment_solver(RealT relative_gap = 0,
															 RealT time_limit = -1)
	: rel_gap_(relative_gap),
	  time_lim_(time_limit)
	{
	}

	public: template <typename PVectorT,
					  typename BVectorT,
					  typename UVectorT,
					  typename CBVectorT,
					  typename CUVectorT,
					  typename PBVectorT,
					  typename WBCVectorT,
					  typename WBLVectorT,
					  typename KPWVectorT,
					  typename TBDVectorT,
					  typename DUDVectorT,
					  typename RPUMatrixT>
			user_assignment_solution<RealT> operator()(::boost::numeric::ublas::vector_expression<PVectorT> const& P,
													   ::boost::numeric::ublas::vector_expression<BVectorT> const& B,
													   ::boost::numeric::ublas::vector_expression<UVectorT> const& U,
													   ::boost::numeric::ublas::vector_expression<CBVectorT> const& Cb,
													   ::boost::numeric::ublas::vector_expression<CUVectorT> const& Cu,
													   ::boost::numeric::ublas::vector_expression<PBVectorT> const& Pb,
													   ::boost::numeric::ublas::vector_expression<WBCVectorT> const& Wbc,
													   ::boost::numeric::ublas::vector_expression<WBLVectorT> const& Wbl,
													   ::boost::numeric::ublas::vector_expression<KPWVectorT> const& Kpw,
													   ::boost::numeric::ublas::vector_expression<TBDVectorT> const& Tbd,
													   ::boost::numeric::ublas::vector_expression<DUDVectorT> const& Dud,
													   ::boost::numeric::ublas::matrix_expression<RPUMatrixT> const& Rpu) const
	{
		namespace ublas = ::boost::numeric::ublas;
		namespace ublasx = ::boost::numeric::ublasx;

		typedef IloNumVarArray var_vector_type;
		typedef IloArray<IloNumVarArray> var_matrix_type;

		user_assignment_solution<RealT> sol;

		const ::std::size_t np = ublasx::size(P); // Number of providers
		const ::std::size_t nb = ublasx::size(B); // Number of base stations
		const ::std::size_t nu = ublasx::size(U); // Number of users
		const ::std::size_t nbc = ublasx::size(Wbc); // Number of base station categories
		const ::std::size_t nuc = ublasx::size(Dud); // Number of user categories

		DCS_ASSERT(nb <= ublasx::size(Cb),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "Cb has not a conformant size"));
		DCS_ASSERT(nu <= ublasx::size(Cu),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "Cu has not a conformant size"));
		DCS_ASSERT(nb <= ublasx::size(Pb),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "Pb has not a conformant size"));
		DCS_ASSERT(nbc == ublasx::size(Wbl),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "Wbl has not a conformant size"));
		DCS_ASSERT(nbc == ublasx::size(Tbd),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "Tbd has not a conformant size"));
		DCS_ASSERT(nuc == ublasx::size(Dud),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "Dud has not a conformant size"));
		DCS_ASSERT(np <= ublasx::num_rows(Rpu) && nuc == ublasx::num_columns(Rpu),
				   DCS_EXCEPTION_THROW(::std::invalid_argument, "Rpu has not a conformant size"));

		DCS_DEBUG_TRACE("Min Cost :: P=" << P); //XXX
		DCS_DEBUG_TRACE("Min Cost :: B=" << B); //XXX
		DCS_DEBUG_TRACE("Min Cost :: U=" << U); //XXX
		DCS_DEBUG_TRACE("Min Cost :: Cb=" << Cb); //XXX
		DCS_DEBUG_TRACE("Min Cost :: Cu=" << Cu); //XXX
		DCS_DEBUG_TRACE("Min Cost :: Pb=" << Pb); //XXX
		DCS_DEBUG_TRACE("Min Cost :: Wbc=" << Wbc); //XXX
		DCS_DEBUG_TRACE("Min Cost :: Wbl=" << Wbl); //XXX
		DCS_DEBUG_TRACE("Min Cost :: Kpw=" << Kpw); //XXX
		DCS_DEBUG_TRACE("Min Cost :: Tbd=" << Tbd); //XXX
		DCS_DEBUG_TRACE("Min Cost :: Dud=" << Dud); //XXX
		DCS_DEBUG_TRACE("Min Cost :: Rpu=" << Rpu); //XXX

		// Setting up vars
		try
		{
			// Initialize the Concert Technology app
			IloEnv env;

			IloModel model(env);

			model.setName("Min Cost");

			// Decision Variable

			// Variables d_{ij} \in R^*: the downlink data rate assigned to user j when it is served by BS i
			var_matrix_type d(env, nb);
			for (::std::size_t i = 0; i < nb; ++i)
			{
				d[i] = var_vector_type(env, nu);

				for (::std::size_t j = 0 ; j < nu ; ++j)
				{
					::std::ostringstream oss;
					oss << "d[" << i << "][" << j << "]";
					d[i][j] = IloNumVar(env, 0, IloInfinity, IloNumVar::Float, oss.str().c_str());
					model.add(d[i][j]);
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

			// C1: \forall i \in B, \sum_{j \in U}{d_{ij}} \le T_{C^b(i)}^d
			++cc;
			for (::std::size_t i = 0; i < nb; ++i)
			{
				const ::std::size_t b(B()(i));
				const ::std::size_t bc(Cb()(b));

				::std::ostringstream oss;
				oss << "C" << cc << "_{" << i << "}";

				IloExpr lhs(env);
				for (::std::size_t j = 0; j < nu; ++j)
				{
					lhs += d[i][j];
				}

				IloConstraint cons(lhs <= Tbd()(bc));
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

			// C6: \forall i \in B, \forall j \in U, d_{ij} \le x_{ij}D_{C^u(j)}^d
			++cc;
			for (::std::size_t i = 0; i < nb; ++i)
			{
				//const ::std::size_t b(B()(i));

				for (::std::size_t j = 0; j < nu; ++j)
				{
					const ::std::size_t u(U()(j));
					const ::std::size_t uc(Cu()(u));

					::std::ostringstream oss;
					oss << "C" << cc << "_{" << i << "," << j << "}";

					IloConstraint cons(d[i][j] <= x[i][j]*Dud()(uc));
					cons.setName(oss.str().c_str());
					model.add(cons);
				}
			}

			// Set objective
			//  min z = \sum_{i \in B}{K_{P^b(i)}^w (W_{C^b(i)}^c y_i + W_{C^b(i)}^l\sum_{j \in U}{x_{ij}}) + \sum_{j \in U}{(x_{ij}-\frac{d_{ij}}{D_{C^u(j)}^d})R_{P^b(i)}^{C^u(j)}}}
			IloObjective z;
			{
				IloExpr expr(env);
				for (::std::size_t i = 0; i < nb; ++i)
				{
					const ::std::size_t b(B()(i));
					const ::std::size_t bc(Cb()(b));
					const ::std::size_t p(Pb()(b));

					expr += Kpw()(p)*Wbc()(bc)*y[i];
					for (::std::size_t j = 0; j < nu; ++j)
					{
						//const ::std::size_t u(U()(j));

						expr += Kpw()(p)*Wbl()(bc)*x[i][j];
					}
					for (::std::size_t j = 0; j < nu; ++j)
					{
						const ::std::size_t u(U()(j));
						const ::std::size_t uc(Cu()(u));

						expr += (x[i][j]-d[i][j]/Dud()(uc))*Rpu()(p,uc);
					}
				}
				z = IloMinimize(env, expr);
			}
			model.add(z);

			// Create the CPLEX solver and make 'model' the active ("extracted") model
			IloCplex solver(model);

			// write the model
#ifndef DCS_DEBUG
			solver.setOut(env.getNullStream());
			solver.setWarning(env.getNullStream());
#else // DCS_DEBUG
			solver.exportModel("cplex-mincost_model.lp");
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
					::dcs::log_warn(DCS_LOGGING_AT, oss.str());
					return sol;
				}
			}

#ifdef DCS_DEBUG
			DCS_DEBUG_TRACE( "-------------------------------------------------------------------------------[" );
			DCS_DEBUG_TRACE( "- Objective value: " << sol.objective_value );

			DCS_DEBUG_TRACE( "- Decision variables: " );

			// Output d_{ij}
			for (::std::size_t i = 0; i < nb; ++i)
			{
				for (::std::size_t j = 0; j < nu; ++j)
				{
					DCS_DEBUG_STREAM << d[i][j].getName() << " = " << solver.getValue(d[i][j]) << " (" << static_cast<RealT>(solver.getValue(d[i][j])) << ")" << ::std::endl;
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
			DCS_DEBUG_TRACE( "]-------------------------------------------------------------------------------" );
#endif // DCS_DEBUG

			sol.bs_user_allocations.resize(nb, nu, false);
			sol.bs_user_downlink_data_rates.resize(nu, false);
			sol.bs_power_states.resize(nb, false);
			sol.cost = sol.objective_value;
			sol.kwatt = 0;
			for (::std::size_t i = 0; i < nb; ++i)
			{
				const ::std::size_t b(B()(i));
				const ::std::size_t bc(Cb()(b));
				//const ::std::size_t p(Pb()(b));

				for (::std::size_t j = 0; j < nu; ++j)
				{
					sol.bs_user_allocations(i,j) = static_cast<bool>(IloRound(solver.getValue(x[i][j])));
					if (sol.bs_user_allocations(i,j))
					{
						sol.bs_user_downlink_data_rates(j) =  static_cast<RealT>(solver.getValue(d[i][j]));
					}
				}
				sol.bs_power_states(i) = static_cast<bool>(IloRound(solver.getValue(y[i])));
				if (sol.bs_power_states(i))
				{
					sol.kwatt += Wbc()(Cb()(i));
					for (::std::size_t j = 0; j < nu; ++j)
					{
						//const ::std::size_t u(U()(j));
						//const ::std::size_t uc(Ub()(u));

						if (sol.bs_user_allocations(i,j))
						{
							sol.kwatt += Wbl()(bc);
						}
					}
				}
			}
			sol.kwatt *= 1.0e-3; // W -> kW

			z.end();
			y.end();
			x.end();
			d.end();

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


	private: RealT rel_gap_;
	private: RealT time_lim_;
}; // optimal_min_cost_user_assignment_solver

}} // Namespace dcs::bs


#endif // DCS_BS_MIN_COST_HPP
