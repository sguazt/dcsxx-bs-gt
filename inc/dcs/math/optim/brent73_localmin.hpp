/**
 * \file dcs/math/optim/brent73_localmin.hpp
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

#ifndef DCS_MATH_OPTIM_BRENT73_LOCALMIN_HPP
#define DCS_MATH_OPTIM_BRENT73_LOCALMIN_HPP


#include <boost/function.hpp>
#include <cmath>
#include <dcs/debug.hpp>
#include <dcs/exception.hpp>
#include <dcs/math/function/sign.hpp>
#include <dcs/math/optim/functional/sign_change.hpp>
#include <dcs/math/traits/float.hpp>
#include <limits>
#include <stdexcept>


namespace dcs { namespace math { namespace optim {

namespace detail
{

template <typename FuncT, typename ParamT, typename ResultT>
struct guarded_feval
{
	guarded_feval(FuncT& f)
	: f_(f)
	{
	}

	ResultT operator()(ParamT const& x) const
	{
		ResultT fx = f_(x);
		if (::std::isinf(x))
		{
			DCS_EXCEPTION_THROW(::std::runtime_error, "Function evaluates to a non-real value");
		}
		if (::std::isnan(x))
		{
			DCS_EXCEPTION_THROW(::std::runtime_error, "Function evaluates to a NaN value");
		}

		return fx;
	}

	FuncT& f_;
};

} // Namespace detail

template <typename RealT>
struct minimization_result
{
	RealT xmin;
	RealT fmin;
	::std::size_t niter;
	short info;
}; // minimization_result


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
template <typename RealT, typename FuncT>
minimization_result<RealT> brent73_localmin(FuncT fun, RealT xmin, RealT xmax, ::std::size_t maxit = 0, ::std::size_t maxfeval = 0, RealT tol = 1e-8, bool check_eval = false)
{
	const RealT sqrteps(::std::numeric_limits<RealT>::epsilon());

	::std::size_t nit(0);
	::std::size_t nfev(0);
	//TODO: used an enumeration
	short info(0); // =1 if the algorithm converged to a solution, =0 if the max number of iterations of function evaluation has been reaced, =-1 otherwise.
	RealT c(0.5*(3.0-::std::sqrt(5.0)));
	RealT a(xmin);
	RealT b(xmax);
	RealT v(a+c*(b-a));
	RealT x(v);
	RealT w(x);
	RealT d(0);
	RealT e(0);
	RealT u(0);

	::boost::function<RealT (RealT)> fwrap;
	if (check_eval)
	{
		fwrap = detail::guarded_feval<FuncT,RealT,RealT>(fun);
	}
	else
	{
		fwrap = fun;
	}

	RealT fval(fwrap(x));
	RealT fw(fval);
	RealT fv(fw);
	++nfev;

	while ((nit < maxit || maxit == 0) && (nfev < maxfeval || maxfeval == 0))
	{
		const RealT xm(0.5*(a+b));

		//FIXME: the golden section search can actually get closer than
		//       sqrt(eps)... sometimes. Sometimes not, it depends on the
		//       function. This is the strategy from the Netlib code.
		//       Something yet smarter would be good.
		const RealT tol2(2.0*sqrteps*::std::abs(x)+tol/3);
		if (::std::abs(x-xm) <= (2*tol2-0.5*(b-a)))
		{
			info = 1;
			break;
		}
		bool dogs(true); // Do Golden Search
		if (::std::abs(e) > tol2)
		{
			// Try inverse parabolic step
			dogs = false;
			RealT r((x-w)*(fval-fv));
			RealT q((x-v)*(fval-fw));
			RealT p((x-v)*q-(x-w)*r);
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
				if (::std::min(u-a, b-u) < 2.0*tol2)
				{
					d = tol2*(::dcs::math::sign(xm-x)+((::dcs::math::float_traits<RealT>::essentially_equal(xm, x)) ? 1 : 0));
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
		RealT u(x+::std::max(::std::abs(d), tol2)*(::dcs::math::sign(d)+((::dcs::math::float_traits<RealT>::essentially_equal(d, 0.0)) ? 1 : 0)));

		RealT fu(fwrap(u));
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
			if (fu <= fw || ::dcs::math::float_traits<RealT>::essentially_equal(w, x))
			{
				v = w;
				fv = fw;
				w = u;
				fw = fu;
			}
			else if (fu <= fv || ::dcs::math::float_traits<RealT>::essentially_equal(v, x) || ::dcs::math::float_traits<RealT>::essentially_equal(v, w))
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
	minimization_result<RealT> res;
	res.xmin = x;
	res.fmin = fval;
	res.niter = nit;
	res.info = info;

	return res;
}

}}} // Namespace dcs::math::optim

#endif // DCS_MATH_OPTIM_BRENT73_LOCALMIN_HPP
