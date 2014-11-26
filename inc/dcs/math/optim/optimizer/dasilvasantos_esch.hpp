/**
 * \file dcs/math/optim/optimizer/dasilvasantos_esch.hpp
 *
 * \brief The Evolutionary Algorithm (ESCH) algorithm by C.H. da Silva Santos
 *  et al.
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

#ifndef DCS_MATH_OPTIM_OPTIMIZER_DASILVASANTOS_ESCH_HPP
#define DCS_MATH_OPTIM_OPTIMIZER_DASILVASANTOS_ESCH_HPP


#include <cstddef>
#include <dcs/macro.hpp>
#include <dcs/math/optim/optimization_result.hpp>
#include <dcs/math/optim/tags.hpp>
#include <nlopt.hpp>
#include <vector>


namespace dcs { namespace math { namespace optim {

/**
 * \brief The Evolutionary Algorithm (ESCH) algorithm by C.H. da Silva Santos
 *  et al.
 *
 * This is the Evolutionary Algorithm for global optimization, developed by
 * Carlos Henrique da Silva Santos's and described in the following paper and
 * Ph.D. thesis:
 *  C.H. da Silva Santos, M. S. Gonçalves, and H. E. Hernandez-Figueroa,
 *  "Designing Novel Photonic Devices by Bio-Inspired Computing," IEEE Photonics
 *  Technology Letters 22 (15), pp. 1177–1179 (2010). 
 *  C.H. da Silva Santos, "Parallel and Bio-Inspired Computing Applied to
 *  Analyze Microwave and Photonic Metamaterial Strucutures," Ph.D. thesis,
 *  University of Campinas, (2010). 
 *
 * The algorithm is adapted from ideas described in:
 *  H.-G. Beyer and H.-P. Schwefel, "Evolution Strategies: A Comprehensive
 *  Introduction," Journal Natural Computing, 1 (1), pp. 3–52 (2002_. 
 *  Ingo Rechenberg, "Evolutionsstrategie – Optimierung technischer Systeme nach
 *  Prinzipien der biologischen Evolution," Ph.D. thesis (1971), Reprinted by
 *  Fromman-Holzboog (1973).
 *
 * The method supports bound constraints only (no nonlinear constraints).
 *
 * This implementation uses the NLopt library:
 *  Steven G. Johnson, The NLopt nonlinear-optimization package,
 *  http://ab-initio.mit.edu/nlopt 
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
template <typename ValueT>
class dasilvasantos_esch
{
	public: typedef ValueT value_type;
	public: typedef ValueT real_type;//XXX
	public: typedef minimization_direction_tag direction_category;


	public: explicit dasilvasantos_esch(::std::size_t n = 1)
	: impl_(nlopt::GN_ISRES, n)
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
}; // dasilvasantos_esch

}}} // Namespace dcs::math::optim


#endif // DCS_MATH_OPTIM_OPTIMIZER_DASILVASANTOS_ESCH_HPP
