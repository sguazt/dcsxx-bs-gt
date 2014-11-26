#ifndef DCS_MATH_OPTIM_FUNCTIONAL_MULTIVARIABLE_FUNCTION_HPP
#define DCS_MATH_OPTIM_FUNCTIONAL_MULTIVARIABLE_FUNCTION_HPP

namespace dcs { namespace math { namespace optim {

template <typename FuncT>
class multivariable_function
{
	public: multivariable_function(FuncT& f)
	: f_(f)
	{
	}

	public: template <typename ResultT, typename ParamT>
			ResultT operator()(::std::vector<ParamT> const& x)
	{
		return f_(x[0]);
	}

//	public: template <typename ResultT, typename ParamT>
//			ResultT operator()(::std::vector<ParamT> const& x) const
//	{
//		return f_(x[0]);
//	}

	public: template <typename ValueT>
			ValueT operator()(::std::vector<ValueT> const& x)
	{
		return f_(x[0]);
	}

//	public: template <typename ValueT>
//			ValueT operator()(::std::vector<ValueT> const& x) const
//	{
//		return f_(x[0]);
//	}


	FuncT& f_;
};

template <typename FuncT>
multivariable_function<FuncT> make_multivariable_function(FuncT& f)
{
	return multivariable_function<FuncT>(f);
}

}}} // Namespace dcs::math::optim

#endif // DCS_MATH_OPTIM_FUNCTIONAL_MULTIVARIABLE_FUNCTION_HPP
