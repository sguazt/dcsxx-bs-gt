/**
 * \file dcs/math/optim/optimizer/kaelo_crs2_local_mutation.hpp
 *
 * \brief The Controlled Random Search (CRS) with Local Mutation.
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
#ifndef DCS_MATH_OPTIM_OPTIMIZER_KAELO_CRS2_LOCAL_MUTATION_HPP
#define DCS_MATH_OPTIM_OPTIMIZER_KAELO_CRS2_LOCAL_MUTATION_HPP


#include <cstddef>
#include <dcs/macro.hpp>
#include <dcs/math/optim/optimization_result.hpp>
#include <dcs/math/optim/tags.hpp>
#include <nlopt.hpp>
#include <vector>


namespace dcs { namespace math { namespace optim {

/**
 * \brief The Controlled Random Search (CRS) with Local Mutation.
 *
 * This is the "controlled random search" (CRS) algorithm (in particular, the
 * CRS2 variant) with the "local mutation" modification, as defined by:
 *  P. Kaelo and M. M. Ali, "Some variants of the controlled random search
 *  algorithm for global optimization," J. Optim. Theory Appl. 130 (2), 253-264
 *  (2006). 
 *
 * The original CRS2 algorithm was described by:
 *  W. L. Price, "A controlled random search procedure for global optimization,"
 *  in Towards Global Optimization 2, p. 71-84 edited by L. C. W. Dixon and G.P.
 *  Szego (North-Holland Press, Amsterdam, 1978). 
 *  W. L. Price, "Global optimization by controlled random search," J. Optim.
 *  Theory Appl. 40 (3), p. 333-348 (1983). 
 *
 * The CRS algorithms are sometimes compared to genetic algorithms, in that they
 * start with a random "population" of points, and randomly "evolve" these
 * points by heuristic rules.
 * In this case, the "evolution" somewhat resembles a randomized Nelder-Mead
 * algorithm. The published results for CRS seem to be largely empirical;
 * limited analytical results about its convergence were derived in:
 *  Eligius M. T. Hendrix, P. M. Ortigosa, and I. García, "On success rates for
 *  controlled random search," J. Global Optim. 21, p. 239-263 (2001). 
 *
 * Only bound-constrained problems are supported by this algorithm.
 *
 * This implementation uses the NLopt library:
 *  Steven G. Johnson, The NLopt nonlinear-optimization package,
 *  http://ab-initio.mit.edu/nlopt 
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
template <typename ValueT>
class kaelo_crs2_local_mutation
{
	public: typedef ValueT value_type;
	public: typedef ValueT real_type;//XXX
	public: typedef minimization_direction_tag direction_category;


	public: explicit kaelo_crs2_local_mutation(::std::size_t n = 1)
	: impl_(nlopt::GN_CRS2_LM, n)
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
			res.converged = false;
			res.failed = true;
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
}; // kaelo_crs2_local_mutation

}}} // Namespace dcs::math::optim


#endif // DCS_MATH_OPTIM_OPTIMIZER_KAELO_CRS2_LOCAL_MUTATION_HPP
