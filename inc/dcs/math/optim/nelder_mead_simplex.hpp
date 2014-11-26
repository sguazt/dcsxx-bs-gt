/**
 * \file dcs/math/optim/nelder_mead_simplex.hpp
 *
 * \brief Optimization based on the Nelder-Mead direct search method.
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

#ifndef DCS_MATH_OPTIM_NELDER_MEAD_SIMPLEX_HPP
#define DCS_MATH_OPTIM_NELDER_MEAD_SIMPLEX_HPP


#include <cmath>
#include <cstddef>
#include <dcs/assert.hpp>
#include <dcs/debug.hpp>
#include <dcs/exception.hpp>
#include <dcs/math/optim/optimization_result.hpp>
#include <dcs/math/type/matrix.hpp>
#include <limits>
#include <stdexcept>
#include <vector>


namespace dcs { namespace math { namespace optim {


struct minimization_direction_tag {};
struct maximization_direction_tag {};


template <typename RealT>
class nelder_mead_simplex_optimizer
{
	public: typedef RealT real_type;
	public: typedef minimization_direction_tag optimization_direction_tag;


	public: nelder_mead_simplex_optimizer()
	: alpha_(1),
	  beta_(0.5),
	  gamma_(2),
	  max_it_(::std::numeric_limits< ::std::size_t >::max()),
	  max_fev_(::std::numeric_limits< ::std::size_t >::max()),
	  tol_(1e-8)
	{
	}

	public: template <typename FuncT>
			optimization_result<real_type> optimize(FuncT fun, ::std::vector<real_type> const& x0)
	{
		const ::std::size_t n(x0.size());

		// pre: size(x0) > 0
		DCS_ASSERT(n > 0,
				   DCS_EXCEPTION_THROW(::std::invalid_argument,
									   "Empty initial value"));

		DCS_DEBUG_TRACE("Nelder-Mead direct search function minimizer");

		optimization_result<real_type> res;

		if (max_it_ == 0 || max_fev_ == 0)
		{
			res.fopt = fun(x0);
			res.xopt = x0;
			res.converged = true;

			return res;
		}

		::std::size_t nit(0); // Number of iterations
		::std::size_t nfev(0); // Number of function evaluations

		::std::vector<real_type> x(x0);
		real_type fx(fun(x));
		++nfev;

		// check: fun(x) is finite
		DCS_ASSERT(::std::isfinite(fx),
				   DCS_EXCEPTION_THROW(::std::runtime_error,
									   "Function cannot be evaluation at initial value"));

		DCS_DEBUG_TRACE("Function value for initial parameters = " << fx);

		real_type conv_tol(tol_ *(::std::abs(fx)+tol_));

		DCS_DEBUG_TRACE("Scaled convergence tolerance is " << conv_tol);

		bool fail(false);

		const ::std::size_t n1(n+1);

		::dcs::math::matrix<real_type> P(n+1,n1+1);
		P(n,0) = fx;
		for (::std::size_t i = 0; i < n; ++i)
		{
			P(i,0) = x[i];
		}

		real_type size(0);
		real_type step(0);
		for (::std::size_t i = 0; i < n; ++i)
		{
			if ((0.1*::std::abs(x[i])) > step)
			{
				step = 0.1*::std::abs(x[i]);
			}
		}
		if (::dcs::math::float_traits<real_type>::essentially_equal(step, 0.0))
		{
			step = 0.1;
		}
		DCS_DEBUG_TRACE("Stepsize computed as " << step);

		for (::std::size_t j = 2; j <= n1; ++j)
		{
			// action: BUILD
			for (::std::size_t i = 0; i < n; ++i)
			{
				P(i,j-1) = x[i];
			}

			real_type try_step(step);
			while (::dcs::math::float_traits<real_type>::essentially_equal(P(j-2,j-1),x[j-2]))
			{
				P(j-2,j-1) = x[j-2] + try_step;
				try_step *= 10;
			}
			size += try_step;
		}

		::std::size_t C(n+2);
		::std::size_t L(1);
		real_type oldsize(size);
		bool calcvert = true;

		do
		{
			++nit;

			if (calcvert)
			{
				for (::std::size_t j = 0; j < n1; ++j)
				{
					if ((j+1) != L)
					{
						for (::std::size_t i = 0; i < n; ++i)
						{
							x[i] = P(i,j);
						}
						fx = fun(x);
						if (!::std::isfinite(fx))
						{
							fx = ::std::numeric_limits<real_type>::max();
						}
						++nfev;
						P(n1-1,j) = fx;
					}
				}
				calcvert = false;
			}

			real_type VL(P(n1-1,L-1));
			real_type VH(VL);
			real_type H(L);

			for (::std::size_t j = 1; j <= n1; ++j)
			{
				if (j != L)
				{
					fx = P(n1-1,j-1);
					if (fx < VL)
					{
						L = j;
						VL = fx;
					}
					if (fx > VH)
					{
						H = j;
						VH = fx;
					}
				}
			}

			if (VH <= (VL+conv_tol) || VL <= tol_)
			{
				break;
			}

			DCS_DEBUG_TRACE(/*action << "   " << */nfev << " " << VH << " " << VL);

			for (::std::size_t i = 0; i < n; ++i)
			{
				real_type temp(-P(i,H-1));
				for (::std::size_t j = 0; j < n1; ++j)
				{
					temp += P(i,j);
				}
				P(i,C-1) = temp/n;
			}
			for (::std::size_t i = 0; i < n; ++i)
			{
				x[i] = (1+alpha_)*P(i,C-1)-alpha_*P(i,H-1);
			}
			fx = fun(x);
			if (!::std::isfinite(fx))
			{
				fx = ::std::numeric_limits<real_type>::max();
			}
			++nfev;
			// action: REFLECTION;
			real_type VR(fx);
			if (VR < VL)
			{
				P(n1-1,C-1) = fx;
				for (::std::size_t i = 0; i < n; ++i)
				{
					fx = gamma_*x[i]+(1-gamma_)*P(i,C-1);
					P(i,C-1) = x[i];
					x[i] = fx;
				}
				fx = fun(x);
				if (!::std::isfinite(fx))
				{
					fx = ::std::numeric_limits<real_type>::max();
				}
				++nfev;
				if (fx < VR)
				{
					for (::std::size_t i = 0; i < n; ++i)
					{
						P(i,H-1) = x[i];
					}
					P(n1-1,H-1) = fx;
					// action: EXTENSION
				}
				else
				{
					for (::std::size_t i = 0; i < n; ++i)
					{
						P(i,H-1) = P(i,C-1);
					}
					P(n1-1,H-1) = VR;
				}
			}
			else
			{
				// action: HI-REDUCTION
				if (VR < VH)
				{
					for (::std::size_t i = 0; i < n; ++i)
					{
						P(i,H-1) = x[i];
					}
					P(n1-1,H-1) = VR;
					// action: LO-REDUCTION
				}

				for (::std::size_t i = 0; i < n; ++i)
				{
					x[i] = (1-beta_)*P(i,H-1) + beta_*P(i,C-1);
				}
				fx = fun(x);
				if (!::std::isfinite(fx))
				{
					fx = ::std::numeric_limits<real_type>::max();
				}
				++nfev;

				if (fx < P(n1-1,H-1))
				{
					for (::std::size_t i = 0; i < n; ++i)
					{
						P(i,H-1) = x[i];
					}
					P(n1-1,H-1) = fx;
				}
				else
				{
					if (VR >= VH)
					{
						// action: SHRINK
						calcvert = true;
						size = 0.0;
						for (::std::size_t j = 0; j < n1; ++j)
						{
							if ((j+1) != L)
							{
								for (::std::size_t i = 0; i < n; ++i)
								{
									P(i,j) = beta_*(P(i,j) - P(i,L-1)) + P(i,L-1);
									size += ::std::abs(P(i,j) - P(i,L-1));
								}
							}
						}
						if (size < oldsize)
						{
							oldsize = size;
						}
						else
						{
							DCS_DEBUG_TRACE("Polytope size measure not decreased in shrink");
							fail = true;
							break;
						}
					}
				}
			}
		}
		while (nfev <= max_fev_ && nit <= max_it_);

		DCS_DEBUG_TRACE("Exiting from Nelder Mead minimizer");
		DCS_DEBUG_TRACE("    " << nfev << " function evaluations used");
		DCS_DEBUG_TRACE("    " << nit << " iterations performed");

		if (nfev > max_fev_ || nit > max_it_)
		{
			fail = true;
		}

		res.xopt.reserve(n);
		for (::std::size_t i = 0; i < n; ++i)
		{
			res.xopt[i] = P(i,L-1);
		}
		res.fopt = P(n1-1,L-1);
		res.converged = !fail;
		res.num_iter = nit;
		res.num_feval = nfev;

		return res;
	}


	private: real_type alpha_; ///< Reflection factor
	private: real_type beta_; ///< Contraction factor
	private: real_type gamma_; ///< Expansion factor
	private: ::std::size_t max_it_; ///< Maximum number of iterations
	private: ::std::size_t max_fev_; ///< Maximum number of function evaluations
	private: real_type tol_; ///< Convergence tolerance
}; // nelder_mead_simplex_optimizer

}}} // Namespace dcs::math::optim


#endif // DCS_MATH_OPTIM_NELDER_MEAD_SIMPLEX_HPP
