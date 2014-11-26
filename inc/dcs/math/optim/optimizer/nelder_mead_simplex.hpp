/**
 * \file dcs/math/optim/optimizer/nelder_mead_simplex.hpp
 *
 * \brief Optimization based on the Nelder-Mead simplex method.
 *
 * References
 * -# J.A. Nelder and R. Mead,
 *    "A simplex method for function minimization"
 *    The Computer Journal 7(4):308-313, Oxford Journals, 1965
 * .
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

#ifndef DCS_MATH_OPTIM_OPTIMIZER_NELDER_MEAD_SIMPLEX_HPP
#define DCS_MATH_OPTIM_OPTIMIZER_NELDER_MEAD_SIMPLEX_HPP


#include <cmath>
#include <cstddef>
#include <dcs/assert.hpp>
#include <dcs/debug.hpp>
#include <dcs/exception.hpp>
#include <dcs/math/optim/detail/traits.hpp>
#include <dcs/math/optim/optimization_result.hpp>
#include <dcs/math/optim/tags.hpp>
#include <dcs/math/type/matrix.hpp>
#include <limits>
#include <stdexcept>
#ifdef DCS_DEBUG
# include <string>
#endif // DCS_DEBUG
#include <vector>


namespace dcs { namespace math { namespace optim {


/**
 * \brief Optimizer based on the Nelder-Mead simplex method.
 *
 * The Nelder–Mead method or downhill simplex method or amoeba method is a
 * commonly used nonlinear optimization technique, which is a well-defined
 * numerical method for problems for which derivatives may not be known.
 * The Nelder–Mead technique was proposed by John Nelder and Roger Mead in
 * 1965 [1] and is a technique for minimizing an objective function in a
 * many-dimensional space.
 *
 * The method uses the concept of a simplex, which is a special polytope of N+1
 * vertices in N dimensions.
 * The Nelder–Mead technique is a heuristic search method that can
 * converge to non-stationary points on problems that can be solved by
 * alternative methods.
 *
 * References
 * -# John A. Nelder and Roger Mead,
 *    "A simplex method for function minimization"
 *    The Computer Journal 7(4):308-313, Oxford Journals, 1965
 * .
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
template <typename ValueT, typename RealT=ValueT>
class nelder_mead_simplex_optimizer
{
	private: typedef nelder_mead_simplex_optimizer<ValueT,RealT> self_type;
	public: typedef ValueT value_type;
	public: typedef RealT real_type;//XXX
	public: typedef minimization_direction_tag direction_category;


	public: nelder_mead_simplex_optimizer()
	: x0_(),
	  alpha_(1),
	  beta_(0.5),
	  gamma_(2),
	  max_it_(::std::numeric_limits< ::std::size_t >::max()),
	  max_fev_(::std::numeric_limits< ::std::size_t >::max()),
	  rel_tol_(::std::sqrt(::std::numeric_limits<value_type>::epsilon())),
	  abs_tol_(-::std::numeric_limits<value_type>::infinity())
	{
	}

	public: void x0(::std::vector<value_type> const& v)
	{
		x0_ = v;
	}

	public: ::std::vector<value_type> x0() const
	{
		return x0_;
	}

	public: void alpha(value_type v)
	{
		alpha_ = v;
	}

	public: value_type alpha() const
	{
		return alpha_;
	}

	public: void beta(value_type v)
	{
		beta_ = v;
	}

	public: value_type beta() const
	{
		return beta_;
	}

	public: void gamma(value_type v)
	{
		gamma_ = v;
	}

	public: value_type gamma() const
	{
		return gamma_;
	}

	public: void max_iterations(::std::size_t v)
	{
		max_it_ = v;
	}

	public: ::std::size_t max_iterations() const
	{
		return max_it_;
	}

	public: void max_evaluations(::std::size_t v)
	{
		max_fev_ = v;
	}

	public: ::std::size_t max_evaluations() const
	{
		return max_fev_;
	}

	public: void fx_relative_tolerance(value_type v)
	{
		rel_tol_ = v;
	}

	public: value_type fx_relative_tolerance() const
	{
		return rel_tol_;
	}

	public: template <typename FuncT>
			optimization_result<value_type> optimize(FuncT fun)
	{
		const ::std::size_t n = x0_.size();

		// pre: size(x0_) > 0
		DCS_ASSERT(n > 0,
				   DCS_EXCEPTION_THROW(::std::invalid_argument,
									   "Empty initial value"));

		DCS_DEBUG_TRACE("Nelder-Mead direct search function minimizer");

		optimization_result<value_type> res;

		if (max_it_ == 0 || max_fev_ == 0)
		{
			res.fopt = fun(x0_);
			res.xopt = x0_;
			res.converged = false;
			res.failed = false;

			return res;
		}

		::std::size_t nit = 0; // Number of iterations
		::std::size_t nfev = 0; // Number of function evaluations

		::std::vector<value_type> x(x0_);
		value_type fx = fun(x);
		++nfev;

		// check: fun(x) is finite
		DCS_ASSERT(::std::isfinite(fx),
				   DCS_EXCEPTION_THROW(::std::runtime_error,
									   "Function cannot be evaluation at initial value"));

		DCS_DEBUG_TRACE("Function value for initial parameters = " << fx);

		value_type conv_tol = rel_tol_ *(::std::abs(fx)+rel_tol_);

		DCS_DEBUG_TRACE("Scaled convergence tolerance is " << conv_tol);

		bool fail = false;

		const ::std::size_t n1 = n+1;

		::dcs::math::matrix<value_type> P(n+1,n1+1);
		P(n1-1,0) = fx;
		for (::std::size_t i = 0; i < n; ++i)
		{
			P(i,0) = x[i];
		}

		value_type size = 0;
		value_type step = 0;
		for (::std::size_t i = 0; i < n; ++i)
		{
			if ((0.1*::std::abs(x[i])) > step)
			{
				step = 0.1*::std::abs(x[i]);
			}
		}
		if (detail::value_traits<value_type>::essentially_equal(step, 0.0))
		{
			step = 0.1;
		}
		DCS_DEBUG_TRACE("Stepsize computed as " << step);

#ifdef DCS_DEBUG
		std::string action;
#endif // DCS_DEBUG

		for (::std::size_t j = 2; j <= n1; ++j)
		{
#ifdef DCS_DEBUG
			action = "BUILD          ";
#endif // DCS_DEBUG
			for (::std::size_t i = 0; i < n; ++i)
			{
				P(i,j-1) = x[i];
			}

			value_type try_step = step;
			while (detail::value_traits<value_type>::essentially_equal(P(j-2,j-1),x[j-2]))
			{
				P(j-2,j-1) = x[j-2] + try_step;
				try_step *= 10;
			}
			size += try_step;
		}

		::std::size_t C = n+2;
		::std::size_t L = 1;
		value_type oldsize = size;
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
							fx = ::std::numeric_limits<value_type>::max();
						}
						++nfev;
						P(n1-1,j) = fx;
					}
				}
				calcvert = false;
			}

			value_type VL = P(n1-1,L-1);
			value_type VH = VL;
			::std::size_t H(L);

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

			if (VH <= (VL+conv_tol) || VL <= abs_tol_)
			{
				break;
			}

			DCS_DEBUG_TRACE(action << nfev << " " << VH << " " << VL);

			for (::std::size_t i = 0; i < n; ++i)
			{
				value_type temp(-P(i,H-1));
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
				fx = ::std::numeric_limits<value_type>::max();
			}
			++nfev;
#ifdef DCS_DEBUG
			action = "REFLECTION     ";
#endif // DCS_DEBUG
			value_type VR(fx);
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
					fx = ::std::numeric_limits<value_type>::max();
				}
				++nfev;
				if (fx < VR)
				{
					for (::std::size_t i = 0; i < n; ++i)
					{
						P(i,H-1) = x[i];
					}
					P(n1-1,H-1) = fx;
#ifdef DCS_DEBUG
					action = "EXTENSION      ";
#endif // DCS_DEBUG
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
#ifdef DCS_DEBUG
				action = "HI-REDUCTION   ";
#endif // DCS_DEBUG
				if (VR < VH)
				{
					for (::std::size_t i = 0; i < n; ++i)
					{
						P(i,H-1) = x[i];
					}
					P(n1-1,H-1) = VR;
#ifdef DCS_DEBUG
				action = "LO-REDUCTION   ";
#endif // DCS_DEBUG
				}

				for (::std::size_t i = 0; i < n; ++i)
				{
					x[i] = (1-beta_)*P(i,H-1) + beta_*P(i,C-1);
				}
				fx = fun(x);
				if (!::std::isfinite(fx))
				{
					fx = ::std::numeric_limits<value_type>::max();
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
#ifdef DCS_DEBUG
						action = "SHRINK         ";
#endif // DCS_DEBUG
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
		res.failed = fail;

		return res;
	}

//	/**
//	 * \brief Optimize the given function with random-restarts
//	 */
//	public: template <typename FuncT, typename URNGT>
//			optimization_result<value_type> optimize(FuncT fun, URNGT& gen, value_type eq_tol, ::std::size_t max_rnd_it)
//	{
//		optimization_result<value_type> best_res;
//
//		for (::std::size_t i = 0; i < max_rnd_it; ++i)
//		{
//			x0_ = gen();
//
//			optimization_result<value_type> res;
//			res = this->optimize(fun);
//
//			if (i > 0)
//			{
//				if (detail::direction_traits<self_type>::compare(res.fopt, best_res.fopt))
//				{
//					best_res = res;
//				}
//				else if (detail::value_traits<value_type>::essentially_equal(res.fopt, best_res.fopt, eq_tol))
//				{
//					break;
//				}
//			}
//			else
//			{
//				best_res = res;
//			}
//		}
//
//		return best_res;
//	}


	private: ::std::vector<value_type> x0_; ///< The initial value
	private: value_type alpha_; ///< Reflection factor
	private: value_type beta_; ///< Contraction factor
	private: value_type gamma_; ///< Expansion factor
	private: ::std::size_t max_it_; ///< Maximum number of iterations
	private: ::std::size_t max_fev_; ///< Maximum number of function evaluations
	private: value_type rel_tol_; ///< Relative convergence tolerance
	private: value_type abs_tol_; ///< Absolute convergence tolerance
}; // nelder_mead_simplex_optimizer

}}} // Namespace dcs::math::optim


#endif // DCS_MATH_OPTIM_OPTIMIZER_NELDER_MEAD_SIMPLEX_HPP
