/**
 * \file dcs/math/optim/optimizer/brent_localmin.hpp
 *
 * \brief Minimizer based on the Brent's LOCALMIN method.
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

#ifndef DCS_MATH_OPTIM_OPTIMIZER_BRENT_LOCALMIN_HPP
#define DCS_MATH_OPTIM_OPTIMIZER_BRENT_LOCALMIN_HPP


#include <cmath>
#include <dcs/assert.hpp>
#include <dcs/debug.hpp>
#include <dcs/exception.hpp>
#include <dcs/math/function/sign.hpp>
#include <dcs/math/optim/detail/traits.hpp>
#include <dcs/math/optim/optimization_result.hpp>
#include <dcs/math/optim/tags.hpp>
#include <limits>
#include <stdexcept>
#include <vector>


namespace dcs { namespace math { namespace optim {

/**
 * \brief The LOCALMIN algorithm from (Brent,1973).
 *
 * The provided implementation is a slightly modified version of the LOCALMIN
 * algorithm proposed by Richard Brent in [1].
 *
 * References
 * -# Richard Brent.
 *    "Algorithms For Minimization Without Derivatives,"
 *    Prentice-Hall (1973)
 * .
 */
template <typename ValueT>
class brent_localmin_optimizer
{
	public: typedef ValueT value_type;
	public: typedef ValueT real_type;
	public: typedef minimization_direction_tag direction_category;


	public: brent_localmin_optimizer()
	: xmin_(-::std::numeric_limits<ValueT>::infinity()),
	  xmax_(+::std::numeric_limits<ValueT>::infinity()),
	  max_it_(::std::numeric_limits< ::std::size_t >::max()),
	  max_fev_(::std::numeric_limits< ::std::size_t >::max()),
	  tol_(1e-8)
	{
	}

	public: void xmin(value_type v)
	{
		xmin_ = v;
	}

	public: value_type xmin() const
	{
		return xmin_;
	}

	public: void xmax(value_type v)
	{
		xmax_ = v;
	}

	public: value_type xmax() const
	{
		return xmax_;
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

	public: template <typename FuncT>
			optimization_result<value_type> optimize(FuncT fun)
	{
		const value_type sqrt_eps(::std::numeric_limits<value_type>::epsilon());

		::std::size_t nit(0);
		::std::size_t nfev(0);
		//TODO: used an enumeration
		short info(0); // =1 if the algorithm converged to a solution, =0 if the max number of iterations of function evaluation has been reaced, =-1 otherwise.
		value_type c(0.5*(3.0-::std::sqrt(5.0))); // c is the squared inverse of the golden ratio
		value_type a(xmin_);
		value_type b(xmax_);
		value_type v(a+c*(b-a));
		value_type x(v);
		value_type w(x);
		value_type d(0);
		value_type e(0);
		value_type u(0);

		value_type fval(fun(::std::vector<value_type>(1,x)));
		value_type fw(fval);
		value_type fv(fw);
		++nfev;

		while ((nit < max_it_ || max_it_ == 0) && (nfev < max_fev_ || max_fev_ == 0))
		{
			const value_type xm(0.5*(a+b));

			//FIXME: the golden section search can actually get closer than
			//       sqrt(eps)... sometimes. Sometimes not, it depends on the
			//       function. This is the strategy from the Netlib code.
			//       Something yet smarter would be good.
			//const value_type tol(2.0*sqrt_eps*::std::abs(x)+tol_/3);
			const value_type tol(sqrt_eps*::std::abs(x)+tol_/3);
			if (::std::abs(x-xm) <= (2*tol-0.5*(b-a)))
			{
				info = 1;
				break;
			}
			bool dogs(true); // Do Golden Search
			if (::std::abs(e) > tol)
			{
				// Try inverse parabolic step (fit parabola)
				dogs = false;
				value_type r((x-w)*(fval-fv));
				value_type q((x-v)*(fval-fw));
				value_type p((x-v)*q-(x-w)*r);
				q = 2.0*(q-r);
				p *= -::dcs::math::sign(q);
				q = ::std::abs(q);
				r = e;
				e = d;

				if (::std::abs(p) < ::std::abs(0.5*q*r) && p > q*(a-x) && p < q*(b-x))
				{
					// The parabolic step is acceptable
					d = p/q;
					u = x+d;

					// fun must not be evaluated too close to ax or bx.
					if (::std::min(u-a, b-u) < 2.0*tol)
					{
						d = tol*(::dcs::math::sign(xm-x)+((detail::value_traits<value_type>::essentially_equal(xm, x)) ? 1 : 0));
					}
				}
				else
				{
					dogs = true;
				}
			}
			if (dogs)
			{
				// Default to golden section step.
				e = (x >= xm) ? (a-x) : (b-x);
				d = c*e;
			}

			// f must not be evaluated too close to x.
			value_type u(x+::std::max(::std::abs(d), tol)*(::dcs::math::sign(d)+((detail::value_traits<value_type>::essentially_equal(d, 0.0)) ? 1 : 0)));

			value_type fu(fun(::std::vector<value_type>(1,u)));
			++nfev;
			++nit;

			// Update  a, b, v, w, and x

			if (fu <= fval)
			{
				if (u < x)
				{
					b = x;
				}
				else
				{
					a = x;
				}
				v = w;
				fv = fw;
				w = x;
				fw = fval;
				x = u;
				fval = fu;
			}
			else
			{
				// The following if-statement was originally executed even if fu == fval.
				if (u < x)
				{
					a = u;
				}
				else
				{
					b = u;
				}
				if (fu <= fw || detail::value_traits<value_type>::essentially_equal(w, x))
				{
					v = w;
					fv = fw;
					w = u;
					fw = fu;
				}
				else if (fu <= fv || detail::value_traits<value_type>::essentially_equal(v, x) || detail::value_traits<value_type>::essentially_equal(v, w))
				{
					v = u;
					fv = fu;
				}
			}

	//		// If there's an output function, use it now.
	//		if (outfcn)
	//		{
	//			optv.funccount = nfev;
	//			optv.fval = fval;
	//			optv.iteration = nit;
	//			if (outfcn (x, optv, "iter"))
	//			{
	//				info = -1;
	//				break;
	//			}
	//		}
		}

	//	output.iterations = niter;
	//	output.funcCount = nfev;
	//	output.bracket = [a, b];
	//	## FIXME: bracketf possibly unavailable.
		optimization_result<value_type> res;
		res.xopt = ::std::vector<value_type>(1, x);
		res.fopt = fval;
		res.num_iter = nit;
		res.num_feval = nfev;
		res.converged = info ? true : false;
		res.failed = info < 0 ? true : false;
		//TODO: add a status code where storing the value of info
		//res.info = info;

		return res;
	}


	private: value_type xmin_;
	private: value_type xmax_;
	private: ::std::size_t max_it_; ///< Maximum number of iterations
	private: ::std::size_t max_fev_; ///< Maximum number of function evaluations
	private: value_type tol_; ///< Convergence relative tolerance
}; // brent_localmin

}}} // Namespace dcs::math::optim

#endif // DCS_MATH_OPTIM_OPTIMIZER_BRENT_LOCALMIN_HPP
