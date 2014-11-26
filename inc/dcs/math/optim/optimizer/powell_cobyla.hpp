/**
 * \file dcs/math/optim/powell_cobyla.hpp
 *
 * \brief The Constrained Optimization BY Linear Approximations (COBYLA)
 *  algorithm.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 *
 * <hr/>
 *
 * Copyright 2014 Marco Guazzone (marco.guazzone@gmail.com)
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
#ifndef DCS_MATH_OPTIM_OPTIMIZER_POWELL_COBYLA_HPP
#define DCS_MATH_OPTIM_OPTIMIZER_POWELL_COBYLA_HPP


#include <cstddef>
#include <dcs/macro.hpp>
#include <dcs/math/optim/optimization_result.hpp>
#include <dcs/math/optim/tags.hpp>
#include <nlopt.hpp>
#include <vector>


namespace dcs { namespace math { namespace optim {

/**
 * \brief The Constrained Optimization BY Linear Approximations (COBYLA)
 *  algorithm.
 *
 * This is a derivative of Powell's implementation of the COBYLA (Constrained
 * Optimization BY Linear Approximations) algorithm for derivative-free
 * optimization with nonlinear inequality and equality constraints, by M.J.D.
 * Powell, described in:
 *  M. J. D. Powell, "A direct search optimization method that models the
 *  objective and constraint functions by linear interpolation," in Advances in
 *  Optimization and Numerical Analysis, eds. S. Gomez and J.-P. Hennart (Kluwer
 *  Academic: Dordrecht, 1994), p. 51-67. 
 * and reviewed in:
 *  M. J. D. Powell, "Direct search algorithms for optimization calculations,"
 *  Acta Numerica 7, 287-336 (1998). 
 *
 * It constructs successive linear approximations of the objective function and
 * constraints via a simplex of n+1 points (in n dimensions), and optimizes
 * these approximations in a trust region at each step.
 *
 * This implementation uses the NLopt library:
 *  Steven G. Johnson, The NLopt nonlinear-optimization package,
 *  http://ab-initio.mit.edu/nlopt 
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
template <typename ValueT>
class powell_cobyla_optimizer
{
	public: typedef ValueT value_type;
	public: typedef value_type real_type;//XXX
	public: typedef minimization_direction_tag direction_category;


	public: explicit powell_cobyla_optimizer(::std::size_t n = 1)
	: impl_(nlopt::LN_COBYLA, n)
	{
	}

	public: void lower_bound(::std::vector<value_type> const& v)
	{
		impl_.set_lower_bounds(v);
	}

	public: ::std::vector<value_type> lower_bound() const
	{
		::std::vector<value_type> ret;
		impl_.get_lower_bounds(ret);
		return ret;
	}

	public: void upper_bound(::std::vector<value_type> const& v)
	{
		impl_.set_upper_bounds(v);
	}

	public: ::std::vector<value_type> upper_bound() const
	{
		::std::vector<value_type> ret;
		impl_.get_upper_bounds(ret);
		return ret;
	}

	public: void x0(::std::vector<value_type> const& v)
	{
		x0_ = v;
	}

	public: ::std::vector<value_type> x0() const
	{
		return x0_;
	}

	public: void fx_relative_tolerance(value_type v)
	{
		impl_.set_ftol_rel(v);
	}

	public: value_type fx_relative_tolerance() const
	{
		return impl_.get_ftol_rel();
	}

	public: void fx_absolute_tolerance(value_type v)
	{
		impl_.set_ftol_abs(v);
	}

	public: value_type fx_absolute_tolerance() const
	{
		return impl_.get_ftol_abs();
	}

	public: void x_relative_tolerance(value_type v)
	{
		impl_.set_xtol_rel(v);
	}

	public: value_type x_relative_tolerance() const
	{
		return impl_.get_xtol_rel();
	}

	public: void x_absolute_tolerance(value_type v)
	{
		impl_.set_xtol_abs(v);
	}

	public: ::std::vector<value_type> x_absolute_tolerance() const
	{
		return impl_.get_xtol_abs();
	}

	public: void max_evaluations(::std::size_t v)
	{
		impl_.set_maxeval(v);
	}

	public: ::std::size_t max_evaluations() const
	{
		return impl_.get_maxeval();
	}

	public: void max_time(double v)
	{
		impl_.set_maxtime(v);
	}

	public: double max_time() const
	{
		return impl_.get_maxtime();
	}

	public: template <typename FuncT>
			optimization_result<value_type> optimize(FuncT f)
	{
		optimization_result<value_type> res;

		res.xopt = x0_;
		impl_.set_min_objective(obj_fun_wrapper<FuncT>, &f);

		nlopt::result nlopt_res = nlopt::SUCCESS;
		res.converged = true;
		res.failed = false;

		try
		{
			nlopt_res = impl_.optimize(res.xopt, res.fopt);
		}
		catch (...)
		{
			// Occasionally, NLopt throw an exception even if the result if SUCCESS
			if (nlopt_res < 0)
			{
				res.converged = false;
				res.failed = true;
			}
//			if (nlopt_res < 0 && nlopt_res == nlopt::ROUNDOFF_LIMITED)
//
//			{
//				res.converged = false;
//			}
//			else
//			{
//	//			if nlopt_res == nlopt::FORCED_STOP ...
//				res.converged = true;
//			}
		}

		return res;
	}

	private: template <typename FuncT>
			 static value_type obj_fun_wrapper(::std::vector<value_type> const& x,
											   ::std::vector<value_type>& grad,
											   void* data)
	{
		DCS_MACRO_SUPPRESS_UNUSED_VARIABLE_WARNING( grad );

//DCS_DEBUG_TRACE("COBYLA :: x = " << dcs::debug::to_string(x.begin(),x.end()));
		return (*reinterpret_cast<FuncT*>(data))(x);
	}


	private: ::nlopt::opt impl_;
	private: ::std::vector<value_type> x0_;
}; // powell_cobyla_optimizer

}}} // Namespace dcs::math::optim


#endif // DCS_MATH_OPTIM_OPTIMIZER_POWELL_COBYLA_HPP
