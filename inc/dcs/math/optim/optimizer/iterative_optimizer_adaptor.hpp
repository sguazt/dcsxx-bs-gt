#ifndef DCS_MATH_OPTIM_OPTIMIZER_ITERATIVE_OPTIMIZER_ADAPTOR_HPP
#define DCS_MATH_OPTIM_OPTIMIZER_ITERATIVE_OPTIMIZER_ADAPTOR_HPP


#include <dcs/math/optim/optimizer_traits.hpp>


namespace dcs { namespace math { namespace optim {

template <typename OptimizerT, typename ParamGenT>
class iterative_optimizer_adaptor
{
	public: typedef OptimizerT optimizer_type;
	public: typedef ParamGenT parameter_generator_type;
	public: typedef typename optimizer_traits<optimizer_type>::value_type value_type;
	public: typedef typename optimizer_traits<optimizer_type>::real_type real_type;
	public: typedef typename optimizer_traits<optimizer_type>::param_type param_type;
	public: typedef typename optimizer_traits<optimizer_type>::direction_category direction_category;

	public: iterative_optimizer_adaptor(optimizer_type& optim, parameter_generator_type& gen, ::std::size_t max_rng_it, real_type tol)
	: optim_(optim),
	  param_gen_(gen),
	  max_rng_it_(max_rng_it),
	  tol_(tol)
	{
	}

	public: template <typename FuncT>
			optimization_result<value_type> optimize(FuncT f)
	{
		optimization_result<value_type> res_best;
		for (::std::size_t i = 0; i < max_rng_it_; ++i)
		{
			param_type param = param_gen_():

			optimization_result<value_type> res;
			res = optim_.optimize(f, param);

			if (res.converged)
			{
				int cmp = detail::direction_traits<value_type>::compare(res.fopt, res_best.fopt, tol_);
				if (cmp < 0)
				{
					res_best = res;
				}
				else if (cmp == 0)
				{
					break;
				}
			}
			else if (res.failed)
			{
				break;
			}
		}
	}


	private: optimizer_type& optim_;
	private: param_generator_type param_gen_;
	private: ::std::size_t max_rng_it_;
	private: real_type tol_;
};

}}} // Namespace dcs::math::optim


#endif // DCS_MATH_OPTIM_OPTIMIZER_ITERATIVE_OPTIMIZER_ADAPTOR_HPP
