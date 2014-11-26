#ifndef DCS_MATH_OPTIM_OPTIMIZATION_RESULT_HPP
#define DCS_MATH_OPTIM_OPTIMIZATION_RESULT_HPP


#include <cstddef>
#include <vector>


namespace dcs { namespace math { namespace optim {

template <typename ValueT>
struct optimization_result
{
	optimization_result()
	: num_iter(0),
	  num_feval(0),
	  converged(false),
	  failed(true)
	{
	}


	::std::vector<ValueT> xopt; ///< The x value corresponding to the found optimum
	ValueT fopt; ///< The function value corresponding to the found optimum
	::std::size_t num_iter; ///< Number of iterations
	::std::size_t num_feval; ///< Number of function evaluations
	bool converged; ///< \c true if convergence has been reached; \c false otherwise
	bool failed; ///< \c true if the method failed to complete the execution
};

}}} // Namespace dcs::math::optim

#endif // DCS_MATH_OPTIM_OPTIMIZATION_RESULT_HPP
