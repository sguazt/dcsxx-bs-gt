/**
 * \file dcs/math/optim/functional/sign_change.hpp
 *
 * \brief Function object useful to evaluate the negation of a function.
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

#ifndef DCS_MATH_OPTIM_FUNCTIONAL_SIGN_CHANGE_HPP
#define DCS_MATH_OPTIM_FUNCTIONAL_SIGN_CHANGE_HPP


#include <vector>


namespace dcs { namespace math { namespace optim {

//template <typename FuncT, typename ParamT, typename ResultT>
//class sign_change_ex
//{
//	public: sign_change_ex(FuncT& f)
//	: f_(f)
//	{
//	}
//
//	public: ResultT operator()(ParamT const& x) const
//	{
//		return -f_(x);
//	}
//
//
//	private: FuncT& f_;
//}; // sign_change_ex


template <typename FuncT>
class sign_change
{
	public: sign_change(FuncT& f)
	: f_(f)
	{
	}

	public: template <typename ResultT, typename ParamT>
			ResultT operator()(::std::vector<ParamT> const& x)
	{
		return -f_(x);
	}

//	public: template <typename ResultT, typename ParamT>
//			ResultT operator()(::std::vector<ParamT> const& x) const
//	{
//		return -f_(x);
//	}

	public: template <typename ValueT>
			ValueT operator()(::std::vector<ValueT> const& x)
	{
		return -f_(x);
	}

//	public: template <typename ValueT>
//			ValueT operator()(::std::vector<ValueT> const& x) const
//	{
//		return -f_(x);
//	}


	private: FuncT& f_;
}; // sign_change

template <typename FuncT>
inline
sign_change<FuncT> make_sign_change(FuncT& f)
{
	return sign_change<FuncT>(f);
}

//template <typename FuncT, typename ParamT, typename ResultT>
//inline
//sign_change_ex<FuncT,ParamT,ResultT> make_sign_change(FuncT& f)
//{
//	return sign_change_ex<FuncT,ParamT,ResultT>(f);
//}

}}} // Namespace dcs::math::optim

#endif // DCS_MATH_OPTIM_FUNCTIONAL_SIGN_CHANGE_HPP
