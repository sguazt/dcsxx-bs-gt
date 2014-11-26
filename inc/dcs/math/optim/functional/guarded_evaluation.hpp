/**
 * \file inc/dcs/math/optim/functional/guarded_evaluation.hpp
 *
 * \brief Guarded evaluation of a function.
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

#ifndef DCS_MATH_OPTIM_FUNCTIONAL_GUARDED_EVALUATION_HPP
#define DCS_MATH_OPTIM_FUNCTIONAL_GUARDED_EVALUATION_HPP

namespace dcs { namespace math { namespace optim {

template <typename FuncT>
class guarded_evalualtion
{
    guarded_feval(FuncT& f)
    : f_(f)
    {
    }

    public: template <typename ResultT, typename ParamT>
			ResultT operator()(::std::vector<ParamT> const& x) const
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


    private: FuncT& f_;
};

template <typename FuncT>
inline
guarded_evaluation<FuncT> make_guarded_evaluation(FuncT& f)
{
	return guarded_evaluation<FuncT>(f);
}

}}} // Namespace dcs::math::optim


#endif // DCS_MATH_OPTIM_FUNCTIONAL_GUARDED_EVALUATION_HPP
