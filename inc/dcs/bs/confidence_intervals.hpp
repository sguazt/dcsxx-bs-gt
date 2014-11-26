/**
 * \file dcs/bs/confidence_intervals.hpp
 *
 * \brief Confidence interval estimation
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

#ifndef DCS_BS_CONFIDENCE_INTERVALS_HPP
#define DCS_BS_CONFIDENCE_INTERVALS_HPP


#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <dcs/assert.hpp>
#include <dcs/debug.hpp>
#include <dcs/logging.hpp>
#include <dcs/macro.hpp>
#include <dcs/math/function/iszero.hpp>
#include <dcs/math/function/sqr.hpp>
#include <dcs/math/traits/float.hpp>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>


namespace dcs { namespace bs {

template <typename RealT>
class ci_mean_estimator
{
	public: typedef RealT real_type;
    private: typedef ::boost::accumulators::accumulator_set< real_type,
															 boost::accumulators::stats< ::boost::accumulators::tag::count,
																						 ::boost::accumulators::tag::mean,
																						 ::boost::accumulators::tag::variance > > accumulator_type;


	public: static const real_type default_ci_level;
	public: static const real_type default_relative_precision;
	public: static const ::std::size_t default_min_sample_size;
	public: static const ::std::size_t default_max_sample_size;


	public: explicit ci_mean_estimator(real_type confidence_level = default_ci_level,
									   real_type relative_precision = default_relative_precision,
									   ::std::size_t min_sample_size = default_min_sample_size,
									   ::std::size_t max_sample_size = default_max_sample_size)
	: ci_level_(confidence_level),
	  target_rel_prec_(relative_precision),
	  n_min_(min_sample_size),
	  n_max_(max_sample_size),
	  name_("Unnamed"),
	  n_target_(::std::numeric_limits< ::std::size_t >::max()),
	  n_detected_(false),
	  n_aborted_(false),
	  n_first_call_(true),
	  unstable_(false),
	  done_(false)
	{
		// pre: min sample size >= 2
		DCS_ASSERT(n_min_ >= 2,
				   DCS_EXCEPTION_THROW(::std::invalid_argument,
									   "Min sample size must be >= 2"));
		// pre: min sample size <= max sample size
		DCS_ASSERT(n_min_ <= n_max_,
				   DCS_EXCEPTION_THROW(::std::invalid_argument,
									   "Min sample size must be <= max sample size"));
	}

	public: void name(::std::string const& s)
	{
		name_ = s;
	}

	public: ::std::string name() const
	{
		return name_;
	}

	public: ::std::size_t size() const
	{
		return ::boost::accumulators::count(stat_);
	}

	public: ::std::size_t target_size() const
	{
		return n_target_;
	}

	public: real_type estimate() const
	{
		return ::boost::accumulators::mean(stat_);
	}

	public: real_type variance() const
	{
		//FIXME: Boost.Accumulators variance computes the biased sample variance
		const ::std::size_t n = this->size();
		return (n/static_cast<real_type>(n-1))*::boost::accumulators::variance(stat_);
	}

	public: real_type standard_deviation() const
	{
		return ::std::sqrt(this->variance());
	}

	public: real_type half_width() const
	{
		const ::std::size_t n = this->size();

		if (n > 1)
		{
			::boost::math::students_t_distribution<real_type> t_dist(n-1);
			const real_type t = ::boost::math::quantile(t_dist, (1+ci_level_)*0.5);

			return t*(this->standard_deviation()/::std::sqrt(n));
		}

		return ::std::numeric_limits<real_type>::infinity();
	}

	public: real_type target_relative_precision() const
	{
		return target_rel_prec_;
	}

	public: real_type relative_precision() const
	{
        if (!::dcs::math::iszero(this->estimate()) && this->size() > 1)
		{
			return this->half_width() / ::std::abs(this->estimate());
		}

		return ::std::numeric_limits<real_type>::infinity();
	}

	public: real_type lower() const
	{
		return this->estimate() - this->half_width();
	}

	public: real_type upper() const
	{
		return this->estimate() + this->half_width();
	}

	public: bool done() const
	{
		return done_;
	}

	public: bool unstable() const
	{
		return unstable_;
	}

	public: void collect(real_type obs)
	{
		if (n_aborted_)
		{
			return;
		}

		stat_(obs);

		//this->check_precision();
		this->check_precision_alt();
DCS_DEBUG_TRACE("(" << name_ << ") Statistic Info: estimate: " << this->estimate() << ", s.d.: " << this->standard_deviation() << ", size: " << this->size() << ", n_target_: " << n_target_ << ", n_min_: " << n_min_ << ", n_max_: " << n_max_ << ", rel.prec.: " << this->relative_precision() << ", n_detected_: " << ::std::boolalpha << n_detected_ << ", n_aborted_: " << n_aborted_ << ", unstable: " << unstable_ << ", done: " << done_ << ")");//XXX
	}

	public: void reset()
	{
		stat_ = accumulator_type();
		n_aborted_ = n_detected_
				   = false;
		n_first_call_ = true;
		unstable_ = false;
		done_ = false;
		n_target_ = ::std::numeric_limits< ::std::size_t >::max();
	}

	private: void check_precision()
	{
		const ::std::size_t n = this->size();

		if (n_detected_ && n >= n_target_)
		{
			if (::std::isinf(target_rel_prec_))
			{
				done_ = true;
			}
			else
			{
				done_ = ::dcs::math::float_traits<real_type>::definitely_less_equal(this->relative_precision(), target_rel_prec_);
			}
		}
		if (!n_detected_ || (n >= n_target_ && !done_))
		{
			// Sample size is still to be detected...
			// ... or needs to be redetected since, after having performed the
			//        detection, the precision has not been reached yet.

			DCS_DEBUG_TRACE("(" << name_ << ") Detecting sample size...");

			this->detect();

			if (n_detected_)
			{
				if (n >= n_target_)
				{
					if (done_)
					{
						// Ooops! The new detected number of replications is the
						// same of or greater than the one previously detected
						// AND we have already performed this number of replications
						// AND precision has not been reached yet.
						// This means that we are unable to reach the target
						// precision. So disable this statistic.

						::dcs::log_warn(DCS_LOGGING_AT, "Statistic '" + name_ + "' will be disabled: unable to reach the wanted precision.");

						unstable_ = true;
					}

					done_ = true;
				}

				DCS_DEBUG_TRACE("(" << name_ << ") Sample size detected: " << n_target_ << " (already collected: " << n << ")");
			}
		}

#ifdef DCS_DEBUG
		if (done_)
		{
		    DCS_DEBUG_TRACE("(" << name_ << ") [Sample #" << n << "] Detected precision: mean = " << this->estimate() << " - reached precision = " << this->relative_precision() << " - target precision: " << target_rel_prec_);
		}
		else if (n_detected_)
		{
			//DCS_DEBUG_TRACE("(" << name_ << ") Sample size detected: " << n_target_ << " (already collected: " << n << ")");
			if (n >= n_target_)
			{
				DCS_DEBUG_TRACE("(" << name_ << ") [Sample #" << n << "] Failed to detect precision: mean = " << this->estimate() << " - reached precision = " << this->relative_precision() << " - target precision: " << target_rel_prec_);
			}
			else
			{
				DCS_DEBUG_TRACE("(" << name_ << ") [Sample #" << n << "] Precision not yet reached: not enough replications (done: " << n << " - needed: " << n_target_ << ")");
			}
		}
#endif
	}

	private: bool detect()
	{
		::std::size_t n = this->size();

		if (n < n_min_)
		{
			n_detected_ = false;
			return false;
		}
		if (n >= n_max_)
		{
			n_aborted_ = true;
			return false;
		}
		if (::std::isinf(target_rel_prec_) && !n_detected_)
		{
			n_target_ = n;
			n_detected_ = true;
			return true;
		}

		// Use the procedure described in [1], chapter 11.
		//
		// References
		// 1. J. Banks et al.
		//    "Discrete-Event System Simulations,"
		//    4th Edition, Prentice-Hall, 2005
		//

		const real_type mean = this->estimate();
		const real_type sd = this->standard_deviation();

		if (sd < 0 || ::std::isinf(sd))
		{
			::dcs::log_warn(DCS_LOGGING_AT, "Standard deviation is negative or infinite");
			n_detected_ = false;
			return false;
		}

		const real_type half_alpha = (1-ci_level_)*0.5;

		// Compute an initial estimate of sample size
		if (n_first_call_)
		{
			n_first_call_ = false;

			::boost::math::normal_distribution<real_type> norm;
			const real_type z = ::boost::math::quantile(norm, half_alpha);
			n =  static_cast< ::std::size_t >(::dcs::math::sqr(z*sd/(target_rel_prec_*mean)));

			if (n < n_min_)
			{
				n = n_min_;
			}
		}

		real_type n_want = 0;

		// Compute the real estimate of sample size
		do
		{
			::boost::math::students_t_distribution<real_type> student_t(n-1);
			const real_type t = ::boost::math::quantile(student_t, half_alpha);
			n_want = ::dcs::math::sqr(t*sd/(target_rel_prec_*mean));

			if (n < n_want)
			{
				++n;
			}
		}
		while (n < n_want && n < n_max_);

		if (n <= n_max_)
		{
			if (n_detected_ && n >= n_target_ && !done_)
			{
				// Ooops! The new detected sample size is the
				// same of or greater than the one previously detected
				// AND we have already collected this number of samples
				// AND precision has not been reached yet.
				// This means that we are unable to reach the target
				// precision. So disable this statistic.

				::dcs::log_warn(DCS_LOGGING_AT, "Statistic '" + name_ + "' will be disabled: unable to reach the wanted precision.");

				unstable_ = true;
			}

			n_target_ = n;
			n_detected_ = true;
			//done_ = true;
		}
		else
		{
			n_target_ = n_max_;
			n_detected_ = false;
			n_aborted_ = true;
		}

DCS_DEBUG_TRACE("(" << name_ << ") Detecting Sample Size --> " << ::std::boolalpha << n_detected_ << " (n_target_: " << n_target_ << " - n_want: " << n_want << " - n_max_: " << n_max_ << " - n_aborted_: " << n_aborted_ << " - unstable: " << unstable_ << " - done: " << done_ << ")");//XXX

		return n_detected_;
	}

	private: bool check_precision_alt()
	{
		::std::size_t n = this->size();

		if (n < n_min_)
		{
			n_detected_ = false;
			return false;
		}
		if (n >= n_max_)
		{
			n_aborted_ = true;
			return false;
		}
		if (::std::isinf(target_rel_prec_))
		{
			n_target_ = n;
			n_detected_ = true;
			done_ = true;
			return true;
		}

		// Use the procedure described in [1], chapter 11.
		//
		// References
		// 1. J. Banks et al.
		//    "Discrete-Event System Simulations,"
		//    4th Edition, Prentice-Hall, 2005
		//

		const real_type mean = this->estimate();
		const real_type sd = this->standard_deviation();

		if (sd < 0 || ::std::isinf(sd))
		{
			::dcs::log_warn(DCS_LOGGING_AT, "Standard deviation is negative or infinite");
			n_detected_ = false;
			return false;
		}

		const real_type half_alpha = (1-ci_level_)*0.5;

		// Compute an initial estimate of sample size
		if (n_first_call_)
		{
			n_first_call_ = false;

			::boost::math::normal_distribution<real_type> norm;
			const real_type z = ::boost::math::quantile(norm, half_alpha);
			n =  static_cast< ::std::size_t >(::dcs::math::sqr(z*sd/(target_rel_prec_*mean)));

			if (n < n_min_)
			{
				n = n_min_;
			}
		}

		real_type n_want = 0;

		// Compute the real estimate of sample size
		do
		{
			::boost::math::students_t_distribution<real_type> student_t(n-1);
			const real_type t = ::boost::math::quantile(student_t, half_alpha);
			n_want = ::dcs::math::sqr(t*sd/(target_rel_prec_*mean));

			if (n < n_want)
			{
				++n;
			}
		}
		while (n < n_want && n < n_max_);

		if (n <= n_max_)
		{
//			if (n_detected_ && n >= n_target_ && !done_)
//			{
//				// Ooops! The new detected sample size is the
//				// same of or greater than the one previously detected
//				// AND we have already collected this number of samples
//				// AND precision has not been reached yet.
//				// This means that we are unable to reach the target
//				// precision. So disable this statistic.
//
//				::dcs::log_warn(DCS_LOGGING_AT, "Statistic '" + name_ + "' will be disabled: unable to reach the wanted precision.");
//
//				unstable_ = true;
//			}

			if (n <= this->size())
			{
//				n_target_ = this->size();
				done_ = true;
			}
//			else
//			{
//				n_target_ = n;
//			}
			n_target_ = n;
			n_detected_ = true;
			//done_ = true;
		}
		else
		{
			n_target_ = n_max_;
			n_detected_ = false;
			n_aborted_ = true;
		}

DCS_DEBUG_TRACE("(" << name_ << ") Detecting Sample Size --> " << ::std::boolalpha << n_detected_ << " (n_target_: " << n_target_ << " - n_want: " << n_want << " - n_max_: " << n_max_ << " - n_aborted_: " << n_aborted_ << " - unstable: " << unstable_ << " - done: " << done_ << ")");//XXX

		return n_detected_;
	}


	private: real_type ci_level_;
	private: real_type target_rel_prec_;
	private: ::std::size_t n_min_;
	private: ::std::size_t n_max_;
	private: ::std::string name_;
    private: accumulator_type stat_; ///< The accumulated statistics
	private: ::std::size_t n_target_; ///< The sample size needed to reach the target relative precision
	private: bool n_detected_; ///< Tells if the sample size needed to reach the target relative precision has been achieved
	private: bool n_aborted_; ///< Tells if the sample size detection process has been aborted
	private: bool n_first_call_; ///< Tells if this is the first invocation of the sample size detection process
	private: bool unstable_; ///< Tells if this statistics has shown an unstable behavior
	private: bool done_; ///< Tells if this statistics has reached the target precision
}; // ci_mean_estimator

template <typename RT>
const RT ci_mean_estimator<RT>::default_ci_level = 0.95;

template <typename RT>
const RT ci_mean_estimator<RT>::default_relative_precision = 0.04;

template <typename RT>
const ::std::size_t ci_mean_estimator<RT>::default_min_sample_size = 2;

template <typename RT>
const ::std::size_t ci_mean_estimator<RT>::default_max_sample_size = ::std::numeric_limits< ::std::size_t >::max();

}} // Namespace dcs::bs

#endif // DCS_BS_CONFIDENCE_INTERVALS_HPP
