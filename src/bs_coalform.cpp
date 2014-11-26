/**
 * \file src/bs_coalform.hpp
 *
 * \brief Form stable coalition among a set of base stations.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 *
 * <hr/>
 *
 * Copyright 2013 Marco Guazzone (marco.guazzone@gmail.com)
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


#include <algorithm>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublasx/operation/max.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <boost/numeric/ublasx/operation/seq.hpp>
#include <boost/numeric/ublasx/operation/size.hpp>
#include <boost/numeric/ublasx/operation/sum.hpp>
#include <boost/random.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/math/special_functions/round.hpp>
#include <boost/timer.hpp>
#include <cmath>
#include <cctype>
#include <cstddef>
#include <dcs/assert.hpp>
#include <dcs/bs/coalition_formation.hpp>
#include <dcs/bs/confidence_intervals.hpp>
#include <dcs/bs/max_profit.hpp>
//#include <dcs/bs/min_cost.hpp>
#include <dcs/bs/radio_propagation_models.hpp>
#include <dcs/cli.hpp>
#include <dcs/debug.hpp>
#include <dcs/exception.hpp>
#include <dcs/logging.hpp>
#include <dcs/math/function/iszero.hpp>
#include <dcs/math/curvefit/interpolation.hpp>
#include <dcs/math/optim/functional/multivariable_function.hpp>
#include <dcs/math/optim/operation/maximize.hpp>
#include <dcs/math/optim/operation/minimize.hpp>
#include <dcs/math/optim/optimizer/brent_localmin.hpp>
#include <dcs/math/optim/optimizer/jones_direct.hpp>
#include <dcs/math/optim/optimizer/nelder_mead_simplex.hpp>
#include <dcs/math/optim/optimizer/powell_cobyla.hpp>
//#include <dcs/math/quadrature/quadrature.hpp>
#include <dcs/math/traits/float.hpp>
#include <dcs/logging.hpp>
#include <dcs/macro.hpp>
#include <fstream>
#include <gtpack/cooperative.hpp>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


namespace alg = dcs::algorithm;
namespace bs = dcs::bs;
namespace cli = dcs::cli;
namespace math = dcs::math;
namespace ublas = boost::numeric::ublas;
namespace ublasx = boost::numeric::ublasx;


static const short VERBOSITY_NONE = 0;
static const short VERBOSITY_LOW = 1;
static const short VERBOSITY_LOW_MEDIUM = 2;
static const short VERBOSITY_MEDIUM = 5;
static const short VERBOSITY_HIGH = 9;


namespace /*<unnamed>*/ { namespace detail {

namespace util {

template <typename CharT, typename CharTraitsT, typename IterT, typename RealT>
std::basic_ostream<CharT,CharTraitsT>& to_stream(std::basic_ostream<CharT,CharTraitsT>& os, IterT first, IterT last)
{
	os << "{";
	while (first != last)
	{
		os << "[";
		typedef typename std::map<bs::bsid_type,RealT>::const_iterator coalition_iterator;
		coalition_iterator coal_end_it(first->end());
		for (coalition_iterator coal_it = first->begin(); coal_it != coal_end_it; ++coal_it)
		{
			os << "(" << coal_it->first << "=>" << coal_it->second << "),";
		}
		os << "],";

		++first;
	}
	os << "}";

	return os;
}

template <typename T>
std::string to_string(T const& t)
{
	std::ostringstream oss;

	oss << t;

	return oss.str();
}

template <typename T>
T euclidean_distance(T x1, T y1, T x2, T y2)
{
	const T xx = x1-x2;
	const T yy = y1-y2;

	return std::sqrt(xx*xx+yy*yy);
}

template <typename T, typename IterT>
std::vector< std::pair<T,T> > rotate2D(IterT first, IterT last, T by_rads)
{
	T mat[2][2];
	mat[0][0] = std::cos(by_rads);
	mat[0][1] = -std::sin(by_rads);
	mat[1][0] = std::sin(by_rads);
	mat[1][1] = std::cos(by_rads);

	std::vector< std::pair<T,T> > res;
	while (first != last)
	{
		const T x = first->first;
		const T y = first->second;

		std::pair<T,T> pos;
		pos = std::make_pair(mat[0][0]*x + mat[0][1]*y, mat[1][0]*x + mat[1][1]*y);
		res.push_back(pos);

		++first;
	}
	return res;
}

// n point star, points are distributed evenly around a circle.
// Assume the first point is at 0,r (top), with the circle centred on 0,0, and that we can construct it from a series of triangles rotated by 2\pi/(2n+1):
template <typename T>
std::vector< std::vector< std::pair<T,T> > > make_star(std::size_t n, T r, T xoffs=0, T yoffs=0)
{
	const T pi = 3.14159265358979323846;
	const T pi_n = pi/static_cast<T>(n);

	T triangle_base = r*std::tan(pi_n);
	std::vector< std::pair<T,T> > triangle(4);
	triangle[0] = std::make_pair(0,r);
	triangle[1] = std::make_pair(triangle_base/2.0,0);
	triangle[2] = std::make_pair(-triangle_base/2.0,0);
	triangle[3] = std::make_pair(0,r);
	std::vector< std::vector< std::pair<T,T> > > res(n);
	for (std::size_t i = 0; i < n; ++i)
	{
		// Rotate...
		res[i] = rotate2D(triangle.begin(), triangle.end(), i*(2.0*pi_n));
		// ...and shift
		for (std::size_t j = 0; j < res[i].size(); ++j)
		{
			res[i][j].first += xoffs;
			res[i][j].second += yoffs;
		}
	}

	return res;
}

template <typename T>
T relative_increment(T x, T r)
{
	return math::sign(x)*std::abs(x/r)-math::sign(r);
}

template <typename IterT>
bool check_stats(IterT first, IterT last)
{
	while (first != last)
	{
		if (!(*first)->done() && !(*first)->unstable())
		{
			return false;
		}

		++first;
	}

	return true;
}

} // Namespace util

namespace experiment {

enum user_assignment_category
{
	max_profit_user_assignment,
	min_cost_user_assignment
};

enum bs_traffic_curve_category
{
	constant_bs_traffic_curve,
	linear_bs_traffic_curve,
	cubic_spline_bs_traffic_curve
};

enum simulation_mode_category
{
	reference_profit_estimation_simulation_mode,
	coalition_formation_simulation_mode,
	both_simulation_mode
};

enum traffic_load_discretization_category
{
	max_traffic_load_discretization,
	min_traffic_load_discretization,
	median_traffic_load_discretization
};

enum bs_topology_layout_category
{
	random_bs_topology_layout,
	square_bs_topology_layout,
	diamond_bs_topology_layout,
	center_bs_topology_layout,
	polygon_bs_topology_layout ///< A regular polygon
};

//static const unsigned long default_rng_seed = 8954UL;
static const std::pair<double,double> default_topology_area_size(0,0);
static const bs_topology_layout_category default_bs_topology_layout = random_bs_topology_layout;
static const double default_bs_height = 0;
static const double default_bs_max_channel_capacity = 0;
static const double default_bs_carrier_frequency = 0;
static const double default_bs_bandwidth = 0;
//static const std::size_t default_bs_num_subcarriers = 0;
static const double default_bs_transmission_power = 0;
static const double default_bs_idle_power = 0;
static const double default_bs_load_dependent_power = 0;
static const std::size_t default_provider_num_bss = 0;
static const double default_provider_electricity_cost = 0;
static const double default_provider_alt_electricity_cost = 0;
static const std::pair<double,double> default_provider_alt_electricity_time_range(0,0);
static const double default_provider_coalition_cost = 0;
static const bs_traffic_curve_category default_provider_bs_traffic_curve(linear_bs_traffic_curve);
static const std::vector< std::pair<double,double> > default_provider_bs_traffic_data(1, std::make_pair(0,0));
static const double default_provider_bs_amortizing_cost = 0;
static const double default_provider_bs_residual_amortizing_time = 0;
static const std::size_t default_provider_bs_max_num_users = 0;
static const double default_provider_user_min_downlink_data_rate = 0;
static const double default_provider_user_revenue = 0;
static const double default_provider_user_mix = 1;
static const double default_user_downlink_data_rate = 0;
static const double default_user_thermal_noise = 0;
static const double default_path_loss_exponent = 3.5;


static const double subchannel_bandwidth = 0.18; ///< This is the LTE subchannel bandwidth (TODO: add to scenario configuration file)
static const double micro_time_step = 1.0/60.0; ///< Time step used for micro-discretization


/// Holds command-line options
template <typename RealT>
struct options
{
	options()
	: milp_relative_gap(0),
	  milp_time_limit(-1),
	  partition_preference(bs::utilitarian_partition_preference),
	  coalition_formation(bs::nash_stable_coalition_formation),
	  coalition_value_division(bs::shapley_coalition_value_division),
	  user_assignment(min_cost_user_assignment),
	  rng_seed(5489),
	  time_span(24),
	  time_steps(4),
	  find_all_best_partitions(false),
	  ci_level(0.95),
	  ci_rel_precision(0.04),
	  verbosity(0),
	  simulation_mode(both_simulation_mode),
	  traffic_load_discretization(max_traffic_load_discretization)
	{
		time_steps[0] = 1;
		time_steps[1] = 2;
		time_steps[2] = 3;
		time_steps[3] = 24;
	}

	RealT milp_relative_gap; ///< The relative gap option to set to the MILP solver
	RealT milp_time_limit; ///< The time limit option to set to the MILP solver
	bs::partition_preference_category partition_preference; ///< The preference relation to use for best partition selection
	bs::coalition_formation_category coalition_formation; ///< The strategy according which form coalitions
	bs::coalition_value_division_category coalition_value_division;
	user_assignment_category user_assignment;
	RealT rng_seed; ///< The seed used for random number generation
	RealT time_span; ///< The time span of the experiment (in hours)
	std::vector<RealT> time_steps; ///< Time steps used to evaluate the coalitions
	bool find_all_best_partitions; ///< A \c true value means that all possible best partitions are computed
	std::string output_stats_data_file; ///< The path to the output stats data file
	std::string output_trace_data_file; ///< The path to the output trace data file
	RealT ci_level; ///< Level for confidence intervals
	RealT ci_rel_precision; ///< Relative precision for the half-width of the confidence intervals
	short verbosity; ///< The verbosity level: 0 for 'minimum' and 9 for 'maximum' verbosity level
	simulation_mode_category simulation_mode; ///< The simulation mode to execute
	traffic_load_discretization_category traffic_load_discretization; ///< The category for the traffic load discretization statistic
}; // options

template <typename CharT, typename CharTraitsT, typename RealT>
std::basic_ostream<CharT,CharTraitsT>& operator<<(std::basic_ostream<CharT,CharTraitsT>& os, options<RealT> const& opts)
{
	os	<< "milp-relative-gap: " << opts.milp_relative_gap
		<< ", milp-time-limit: " << opts.milp_time_limit
		<< ", partition-preference: " << opts.partition_preference
		<< ", coalition-formation: " << opts.coalition_formation
		<< ", coalition-value-division: " << opts.coalition_value_division
		<< ", user-assignment: " << opts.user_assignment
		<< ", random-generator-seed: " << opts.rng_seed
		<< ", time-span: " << opts.time_span;
	os << ", time-steps: [";
	for (std::size_t i = 0; i < opts.time_steps.size(); ++i)
	{
		if (i > 0)
		{
			os << ",";
		}
		os << opts.time_steps[i];
	}
	os << "]";
	os	<< ", output-stats-data-file: " << opts.output_stats_data_file
		<< ", output-trace-data-file: " << opts.output_trace_data_file
		<< ", ci-level: " << opts.ci_level
		<< ", ci-relative-precision: " << opts.ci_rel_precision
		<< ", verbosity: " << opts.verbosity
		<< ", simulation-mode: " << opts.simulation_mode
		<< ", traffic-load-discretization: " << opts.traffic_load_discretization;

	return os;
}

template <typename RealT>
struct scenario
{
	std::size_t num_providers; ///< Number of providers
	std::size_t num_bs_types; ///< Number of BS types
	std::size_t num_user_classes; ///< Number of user classes
	//path_loss_model_category env_path_loss_model; ///< Path loss model
	std::pair<RealT,RealT> topology_area_size; ///< Width and height of the area (in meters)
	bs_topology_layout_category topology_bs_layout; ///< The topology layout for BSs in the area of interest
	std::vector<RealT> bs_heights; ///< BS heights (in meters)
	std::vector<RealT> bs_carrier_frequencies; ///< BS transmission frequencies (in MHz)
	//std::vector<RealT> bs_max_channel_capacities; ///< BS max supported channel capacity (in Mbit/sec)
	std::vector<RealT> bs_bandwidths; ///< BS bandwidths (in Hz)
	//std::vector<std::size_t> bs_num_subcarriers; ///< BS max number of subcarriers (for OFDMA-based cellular networks)
	std::vector<RealT> bs_transmission_powers; ///< BS transmission powers (in W)
	std::vector<RealT> bs_idle_powers; ///< BS idle power consumption (in kW)
	std::vector<RealT> bs_load_powers; ///< BS load-dependent power consumption (in kW)
	std::vector< std::vector<std::size_t> > provider_num_bss; ///< Number of BS per provider and per BS type
	std::vector<RealT> provider_electricity_costs; ///< Provider electricity cost plans (in $/kWh)
	std::vector<RealT> provider_alt_electricity_costs; ///< Provider alternative electricity costs (in $/kWh)
	std::vector< std::pair<RealT,RealT> > provider_alt_electricity_time_ranges; ///< Provider alternative electricity time ranges (in 24-hours)
	//std::vector<RealT> provider_penalties; ///< Penalties (in $/user) for violating user QoS (i.e., the min data rate)
	std::vector<RealT> provider_coalition_costs; ///< Cost due to form a coalition structure, per provider
	//std::vector<RealT> provider_bs_coalition_change_costs; ///< Cost due to a change in coalition structure, per provider
	std::vector< std::vector<bs_traffic_curve_category> > provider_bs_traffic_curves; ///< The category of the traffic curve, per provider and BS type
	std::vector< std::vector< std::vector< std::pair<RealT,RealT> > > > provider_bs_traffic_data; ///< The dataset describing the traffic curve, per provider and BS type
	std::vector< std::vector<RealT> > provider_bs_amortizing_costs; ///< BS amortizing cost rates, per provider and BS type (in $/hour)
	std::vector< std::vector<RealT> > provider_bs_residual_amortizing_times; ///< Residual BS amortizing times, per provider and BS type (in months)
	std::vector< std::vector<std::size_t> > provider_bs_max_num_users; ///< Max number of users a provider expect to be served by each BS, per provider and BS type
	std::vector< std::vector<RealT> > provider_user_min_downlink_data_rates; ///< Minimum downlink data rate (in Mbit/sec) that each provider guarantees guarantees to each user, by provider and user class
	std::vector< std::vector<RealT> > provider_user_revenues; ///< Revenue (in $/user) for serving each user, by provider and user class
	std::vector< std::vector<RealT> > provider_user_mixes; ///< User classes mix, by provider and user class
	std::vector<RealT> user_heights; ///< User heights (in meters)
	std::vector<RealT> user_downlink_data_rates; ///< User QoS representing the desired downlink data rate, per user class (in Mbit/sec)
	std::vector<RealT> user_thermal_noises; ///< User thermal noise (in W)
}; // scenario

template <typename CharT, typename CharTraitsT, typename RealT>
std::basic_ostream<CharT,CharTraitsT>& operator<<(std::basic_ostream<CharT,CharTraitsT>& os, scenario<RealT> const& s)
{
	os	<< "num_providers=" << s.num_providers
		<< ", " << "num_bs_types=" << s.num_bs_types
		<< ", " << "num_user_classes=" << s.num_user_classes
		//<< ", " << "env_path_loss_model=" << s.env_path_loss_model
		<< ", " << "topology.area_size=[" << s.topology_area_size.first << "," << s.topology_area_size.second << "]"
		<< ", " << "topology.bs_layout=" << s.topology_bs_layout;

	os << ", " << "bs.heights=[";
	for (std::size_t b = 0; b < s.num_bs_types; ++b)
	{
		if (b > 0)
		{
			os << ", ";
		}
		os << s.bs_heights[b];
	}
	os << "]";
	os << ", " << "bs.carrier_frequencies=[";
	for (std::size_t b = 0; b < s.num_bs_types; ++b)
	{
		if (b > 0)
		{
			os << ", ";
		}
		os << s.bs_carrier_frequencies[b];
	}
	os << "]";
//	os << ", " << "bs.max_channel_capacities=[";
//	for (std::size_t b = 0; b < s.num_bs_types; ++b)
//	{
//		if (b > 0)
//		{
//			os << ", ";
//		}
//		os << s.bs_max_channel_capacities[b];
//	}
//	os << "]";
	os << ", " << "bs.bandwidths=[";
	for (std::size_t b = 0; b < s.num_bs_types; ++b)
	{
		if (b > 0)
		{
			os << ", ";
		}
		os << s.bs_bandwidths[b];
	}
	os << "]";
//	os << ", " << "bs.num_subcarriers=[";
//	for (std::size_t b = 0; b < s.num_bs_types; ++b)
//	{
//		if (b > 0)
//		{
//			os << ", ";
//		}
//		os << s.bs_num_subcarriers[b];
//	}
//	os << "]";
	os << ", " << "bs.transmission_powers=[";
	for (std::size_t b = 0; b < s.num_bs_types; ++b)
	{
		if (b > 0)
		{
			os << ", ";
		}
		os << s.bs_transmission_powers[b];
	}
	os << "]";
	os << ", " << "bs.idle_powers=[";
	for (std::size_t b = 0; b < s.num_bs_types; ++b)
	{
		if (b > 0)
		{
			os << ", ";
		}
		os << s.bs_idle_powers[b];
	}
	os << "]";
	os << ", " << "bs.load_powers=[";
	for (std::size_t b = 0; b < s.num_bs_types; ++b)
	{
		if (b > 0)
		{
			os << ", ";
		}
		os << s.bs_load_powers[b];
	}
	os << "]";
	os << ", " << "provider.num_bss=[";
	for (std::size_t p = 0; p < s.num_providers; ++p)
	{
		if (p > 0)
		{
			os << ", ";
		}
		os << "[";
		for (std::size_t b = 0; b < s.num_bs_types; ++b)
		{
			if (b > 0)
			{
				os << ", ";
			}
			os << s.provider_num_bss[p][b];
		}
		os << "]";
	}
	os << "]";
	os << ", " << "provider.electricity_costs=[";
	for (std::size_t p = 0; p < s.num_providers; ++p)
	{
		if (p > 0)
		{
			os << ", ";
		}
		os << s.provider_electricity_costs[p];
	}
	os << "]";
	os << ", " << "provider.alt_electricity_costs=[";
	for (std::size_t p = 0; p < s.num_providers; ++p)
	{
		if (p > 0)
		{
			os << ", ";
		}
		os << s.provider_alt_electricity_costs[p];
	}
	os << "]";
	os << ", " << "provider.alt_electricity_time_ranges=[";
	for (std::size_t p = 0; p < s.num_providers; ++p)
	{
		if (p > 0)
		{
			os << ", ";
		}
		os << "[" << s.provider_alt_electricity_time_ranges[p].first << ", " <<  s.provider_alt_electricity_time_ranges[p].second << "]";
	}
	os << "]";
	os << ", " << "provider.coalition_costs=[";
	for (std::size_t p = 0; p < s.num_providers; ++p)
	{
		if (p > 0)
		{
			os << ", ";
		}
		os << s.provider_coalition_costs[p];
	}
	os << "]";
//	os << ", " << "provider.bs_coalition_change_costs=[";
//	for (std::size_t p = 0; p < s.num_providers; ++p)
//	{
//		if (p > 0)
//		{
//			os << ", ";
//		}
//		os << s.provider_bs_coalition_change_costs[p];
//	}
//	os << "]";
	os << ", " << "provider.bs_traffic_curves=[";
	for (std::size_t p = 0; p < s.num_providers; ++p)
	{
		if (p > 0)
		{
			os << ", ";
		}
		os << "[";
		for (std::size_t b = 0; b < s.num_bs_types; ++b)
		{
			if (b > 0)
			{
				os << ", ";
			}
			switch (s.provider_bs_traffic_curves[p][b])
			{
				case constant_bs_traffic_curve:
					os << "constant";
					break;
				case cubic_spline_bs_traffic_curve:
					os << "cubic-spline";
					break;
				case linear_bs_traffic_curve:
					os << "linear";
					break;
			}
			os << "]";
		}
		os << "]";
	}
	os << "]";
	os << ", " << "provider.bs_traffic_data=[";
	for (std::size_t p = 0; p < s.num_providers; ++p)
	{
		if (p > 0)
		{
			os << ", ";
		}
		os << "[";
		for (std::size_t b = 0; b < s.num_bs_types; ++b)
		{
			if (b > 0)
			{
				os << ", ";
			}
			os << "[";
			for (std::size_t d = 0; d < s.provider_bs_traffic_data[p][b].size(); ++d)
			{
				if (d > 0)
				{
					os << ", ";
				}
				os << "[" << s.provider_bs_traffic_data[p][b][d].first << ", " << s.provider_bs_traffic_data[p][b][d].second << "]";
			}
			os << "]";
		}
		os << "]";
	}
	os << ", " << "provider.bs_amortizing_costs=[";
	for (std::size_t p = 0; p < s.num_providers; ++p)
	{
		if (p > 0)
		{
			os << ", ";
		}
		os << "[";
		for (std::size_t b = 0; b < s.num_bs_types; ++b)
		{
			if (b > 0)
			{
				os << ", ";
			}
			os << s.provider_bs_amortizing_costs[p][b];
		}
		os << "]";
	}
	os << "]";
	os << ", " << "provider.bs_residual_amortizing_times=[";
	for (std::size_t p = 0; p < s.num_providers; ++p)
	{
		if (p > 0)
		{
			os << ", ";
		}
		os << "[";
		for (std::size_t b = 0; b < s.num_bs_types; ++b)
		{
			if (b > 0)
			{
				os << ", ";
			}
			os << s.provider_bs_residual_amortizing_times[p][b];
		}
		os << "]";
	}
	os << "]";
	os << ", " << "provider.bs_max_num_users=[";
	for (std::size_t p = 0; p < s.num_providers; ++p)
	{
		if (p > 0)
		{
			os << ", ";
		}
		os << "[";
		for (std::size_t b = 0; b < s.num_bs_types; ++b)
		{
			if (b > 0)
			{
				os << ", ";
			}
			os << s.provider_bs_max_num_users[p][b];
		}
		os << "]";
	}
	os << "]";
	os << ", " << "provider.user_min_downlink_data_rates=[";
	for (std::size_t p = 0; p < s.num_providers; ++p)
	{
		if (p > 0)
		{
			os << ", ";
		}
		os << "[";
		for (std::size_t u = 0; u < s.num_user_classes; ++u)
		{
			if (u > 0)
			{
				os << ", ";
			}
			os << s.provider_user_min_downlink_data_rates[p][u];
		}
		os << "]";
	}
	os << "]";
	os << ", " << "provider.user_revenues=[";
	for (std::size_t p = 0; p < s.num_providers; ++p)
	{
		if (p > 0)
		{
			os << ", ";
		}
		os << "[";
		for (std::size_t u = 0; u < s.num_user_classes; ++u)
		{
			if (u > 0)
			{
				os << ", ";
			}
			os << s.provider_user_revenues[p][u];
		}
		os << "]";
	}
	os << "]";
	os << ", " << "provider.user_mixes=[";
	for (std::size_t p = 0; p < s.num_providers; ++p)
	{
		if (p > 0)
		{
			os << ", ";
		}
		os << "[";
		for (std::size_t u = 0; u < s.num_user_classes; ++u)
		{
			if (u > 0)
			{
				os << ", ";
			}
			os << s.provider_user_mixes[p][u];
		}
		os << "]";
	}
	os << "]";
	os << ", " << "user.downlink_data_rates=[";
	for (std::size_t u = 0; u < s.num_user_classes; ++u)
	{
		if (u > 0)
		{
			os << ", ";
		}
		os << s.user_downlink_data_rates[u];
	}
	os << "]";
	os << ", " << "user.thermal_noises=[";
	for (std::size_t u = 0; u < s.num_user_classes; ++u)
	{
		if (u > 0)
		{
			os << ", ";
		}
		os << s.user_thermal_noises[u];
	}
	os << "]";
	os << ", " << "user.heights=[";
	for (std::size_t u = 0; u < s.num_user_classes; ++u)
	{
		if (u > 0)
		{
			os << ", ";
		}
		os << s.user_heights[u];
	}
	os << "]";

	return os;
}

template <typename RealT>
class daily_traffic_pattern
{
	public: typedef RealT real_type;


	public: daily_traffic_pattern(::boost::shared_ptr< math::curvefit::base_1d_interpolator<real_type> > const& p_interp)
	: p_interp_(p_interp)
	{
	}

	public: real_type operator()(real_type time) const
	{
		real_type t(std::fmod(time, 24.0));
		if (math::float_traits<real_type>::definitely_less(t, 0.0))
		{
			t = 24.0+t;
		}

		return (*p_interp_)(t);
	}

	private: ::boost::shared_ptr< math::curvefit::base_1d_interpolator<real_type> > p_interp_;
}; // daily_traffic_pattern

template <typename RealT>
class weekly_traffic_pattern
{
	public: typedef RealT real_type;


	public: weekly_traffic_pattern(::boost::shared_ptr< math::curvefit::base_1d_interpolator<real_type> > const& p_interp)
	: p_interp_(p_interp)
	{
	}

	public: real_type operator()(real_type time) const
	{
		real_type t(std::fmod(time, 24.0*7.0));
		if (math::float_traits<real_type>::definitely_less(t, 0.0))
		{
			t = 24.0*7.0+t;
		}

		return (*p_interp_)(t);
	}

	private: ::boost::shared_ptr< math::curvefit::base_1d_interpolator<real_type> > p_interp_;
}; // weekly_traffic_pattern

template <typename RealT>
struct bs_info
{
	bs_info()
	: num_poweron(0),
	  num_served_users(0),
	  num_served_users_sd(0),
	  num_own_users(0),
	  coalition_load(0),
	  alone_load(0)
	{
	}

	RealT num_poweron;
	RealT num_served_users;
	RealT num_served_users_sd;
	RealT num_own_users;
	RealT coalition_load;
	RealT alone_load;
};

template <typename RealT>
struct bs_stats_info
{
	boost::accumulators::accumulator_set<RealT,boost::accumulators::stats<boost::accumulators::tag::max,boost::accumulators::tag::mean,boost::accumulators::tag::variance> > num_poweron;
	boost::accumulators::accumulator_set<RealT,boost::accumulators::stats<boost::accumulators::tag::max,boost::accumulators::tag::mean,boost::accumulators::tag::variance> > num_served_users;
	boost::accumulators::accumulator_set<RealT,boost::accumulators::stats<boost::accumulators::tag::max,boost::accumulators::tag::mean,boost::accumulators::tag::variance> > num_own_users;
	boost::accumulators::accumulator_set<RealT,boost::accumulators::stats<boost::accumulators::tag::max,boost::accumulators::tag::mean,boost::accumulators::tag::variance> > coalition_load;
	boost::accumulators::accumulator_set<RealT,boost::accumulators::stats<boost::accumulators::tag::max,boost::accumulators::tag::mean,boost::accumulators::tag::variance> > alone_load;
};

template <typename RealT, typename RNGT>
std::vector< std::pair<RealT,RealT> > make_bs_topology(std::size_t nb, std::pair<RealT,RealT> area_size, bs_topology_layout_category layout, RNGT& rng, RealT aggregation_factor=0)
{
	std::vector< std::pair<RealT,RealT> > bs_positions(nb);
	switch (layout)
	{
		case random_bs_topology_layout:
			{
				boost::random::uniform_real_distribution<RealT> rvg_x_pos(0, area_size.first); // RVG for the x-coordinate
				boost::random::uniform_real_distribution<RealT> rvg_y_pos(0, area_size.second); // RVG for the y-coordinate

				for (std::size_t b = 0; b < nb; ++b)
				{
					bs_positions[b] = std::make_pair(rvg_x_pos(rng), rvg_y_pos(rng));
				}
			}
			break;
		case square_bs_topology_layout:
			if (nb > 5)
			{
				throw std::runtime_error("Square topology for # BS > 5 to be implemented");
			}
			if (nb > 0)
			{
				bs_positions[0] = std::make_pair(0,0);
			}
			if (nb > 1)
			{
				bs_positions[1] = std::make_pair(area_size.first, 0);
			}
			if (nb > 2)
			{
				bs_positions[2] = std::make_pair(area_size.first/2.0, area_size.second/2.0);
			}
			if (nb > 3)
			{
				bs_positions[3] = std::make_pair(0, area_size.second);
			}
			if (nb > 4)
			{
				bs_positions[4] = std::make_pair(area_size.first, area_size.second);
			}
			break;
		case diamond_bs_topology_layout:
			if (nb > 5)
			{
				throw std::runtime_error("Diamond topology for # BS > 5 to be implemented");
			}
			if (nb > 0)
			{
				bs_positions[0] = std::make_pair(area_size.first*aggregation_factor/2.0,area_size.second/2.0);
			}
			if (nb > 1)
			{
				bs_positions[1] = std::make_pair(area_size.first/2.0, area_size.second*aggregation_factor/2.0);
			}
			if (nb > 2)
			{
				bs_positions[2] = std::make_pair(area_size.first/2.0, area_size.second/2.0);
			}
			if (nb > 3)
			{
				bs_positions[3] = std::make_pair(area_size.first*(1.0-aggregation_factor/2.0), area_size.second/2.0);
			}
			if (nb > 4)
			{
				bs_positions[4] = std::make_pair(area_size.first/2.0, area_size.second*(1.0-aggregation_factor/2.0));
			}
			break;
		case center_bs_topology_layout:
			for (std::size_t b = 0; b < nb; ++b)
			{
				bs_positions[b] = std::make_pair(area_size.first/2.0, area_size.second/2.0);
			}
			break;
		case polygon_bs_topology_layout:
			{
				std::vector< std::vector< std::pair<RealT,RealT> > > res;
				RealT r = aggregation_factor*std::min(area_size.first, area_size.second)/2.0;
				res = util::make_star(nb, r, area_size.first/2.0, area_size.second/2.0);

				for (std::size_t b = 0; b < nb; ++b)
				{
					bs_positions[b] = res[b][0];
				}
			}
			break;
	}

	return bs_positions;
}

template <typename TrafficFuncT, typename RealT, typename RNGT>
std::vector<RealT> estimate_alone_profits(RealT time_step,
										  scenario<RealT> const& scen,
										  options<RealT> const& opts,
										  RNGT& rng,
										  std::vector<RealT>& bs_num_served_users)
{
	std::size_t tot_nb = 0; // Total number of BSs
	ublas::vector<std::size_t> prov_num_bss(scen.num_providers, 0); // Total number of BSs per provider
	ublas::vector<RealT> bs_heights(scen.num_bs_types, 0); // Height of each BS category. bs_heights[k] = x means that the class k BS is x meters tall
	ublas::vector<RealT> bs_freqs(scen.num_bs_types, 0); // Operating frequency of each BS category in MHz
//	ublas::vector<RealT> bs_caps(scen.num_bs_types, 0); // Max channel capacity of each BS category in Mbit/sec
	ublas::vector<RealT> bs_bws(scen.num_bs_types, 0); // Bandwidth of each BS category in MHz
//	ublas::vector<RealT> bs_num_subcarriers(scen.num_bs_types, 0); // Max number of subcarriers supported by each BS category.
	ublas::vector<RealT> bs_tx_powers(scen.num_bs_types, 0); // Transmission power each BS category. bs_tx_powers[k] = x means that the class k BS supports transmission power of x W
	ublas::vector<RealT> bs_const_powers(scen.num_bs_types, 0); // constant BS power consumption by category. bs_const_powers[k] = x means that the class k BS has a constant power consumption of x W
	ublas::vector<RealT> bs_load_powers(scen.num_bs_types, 0); // load-dependent BS power consumption by category. bs_load_powers[k] = x means that the class k BS has a load-depedented power consumption of x W
	ublas::vector<RealT> prov_energy_costs(scen.num_providers, 0); // Electricity cost by provider. prov_energy_costs[p] = x means that the provider p pays for electricity x $/Wh
//	ublas::matrix<RealT> prov_bs_amort_costs(scen.num_providers, scen.num_bs_types, 0); // Amortizing cost by provider and BS type. prov_bs_amort_costs[p,k] = x means that the provider p pays for BS type k a cost of x $/Wh
//	ublas::matrix<RealT> prov_bs_amort_times(scen.num_providers, scen.num_bs_types, 0); // Residual amortizing time by provider and BS type. prov_bs_amort_times[p,k] = x means that to amortize the cost of a class k BS the provider p takes x hours
	ublas::matrix<RealT> prov_usr_min_dl_rates(scen.num_providers, scen.num_user_classes, 0); // Guaranteed downlink data rate by each provider and user category. prov_usr_min_dl_rates[p,k] = x means that the provider p guarantees to a class k user a downlink data rate of x bit/sec
	ublas::matrix<RealT> prov_usr_revenues(scen.num_providers, scen.num_user_classes, 0); // Revenue rate each provider obtain by each user category. prov_usr_revenues[p,k] = x means that the provider p obtains from a class k user a revenue of x $/hour
	//std::vector< std::vector< boost::shared_ptr< daily_traffic_pattern<RealT> > > > bs_traffic_patterns(scen.num_providers); // BS traffic curve (per provider and BS category)
	std::vector< std::vector< boost::shared_ptr< TrafficFuncT > > > bs_traffic_patterns(scen.num_providers); // BS traffic curve (per provider and BS category)
	ublas::vector<RealT> usr_heights(scen.num_user_classes, 0); // Height of each user category. usr_heights[k] = x means that the class k user is x meters tall
	ublas::vector<RealT> usr_qos_dl_rates(scen.num_user_classes, 0); // Required downlink data rate by each user category. usr_qos_dl_rates[k] = x means that the class k user requires a downlink data rate of x bit/sec
	ublas::vector<RealT> usr_therm_noises(scen.num_user_classes, 0); // Thermal noise. usr_therm_noises[k] = x means that the class k user has a thermal noise of x W
	std::map< std::size_t, std::pair<std::size_t,std::size_t> > prov_to_bss;

	// Uniform random variate generators for randomly positioning BSs and users inside the area of interest
	boost::random::uniform_real_distribution<RealT> rvg_x_pos(0, scen.topology_area_size.first); // RVG for the x-coordinate
	boost::random::uniform_real_distribution<RealT> rvg_y_pos(0, scen.topology_area_size.second); // RVG for the y-coordinate

	for (std::size_t bc = 0; bc < scen.num_bs_types; ++bc)
	{
		bs_heights(bc) = scen.bs_heights[bc];
		bs_freqs(bc) = scen.bs_carrier_frequencies[bc];
//		bs_caps(bc) = scen.bs_max_channel_capacities[bc];
		bs_bws(bc) = scen.bs_bandwidths[bc];
//		bs_num_subcarriers(bc) = scen.bs_num_subcarriers[bc];
		bs_tx_powers(bc) = scen.bs_transmission_powers[bc];
		bs_const_powers(bc) = scen.bs_idle_powers[bc];
		bs_load_powers(bc) = scen.bs_load_powers[bc];
	}

	for (std::size_t uc = 0; uc < scen.num_user_classes; ++uc)
	{
		usr_qos_dl_rates(uc) = scen.user_downlink_data_rates[uc];
		usr_therm_noises(uc) = scen.user_thermal_noises[uc];
		usr_heights(uc) = scen.user_heights[uc];
	}

	for (std::size_t p = 0; p < scen.num_providers; ++p)
	{
		prov_energy_costs(p) = scen.provider_electricity_costs[p]; // Electricity cost ($/Wh)
		for (std::size_t uc = 0; uc < scen.num_user_classes; ++uc)
		{
			prov_usr_min_dl_rates(p,uc) = scen.provider_user_min_downlink_data_rates[p][uc];
			prov_usr_revenues(p,uc) = scen.provider_user_revenues[p][uc];
		}

		bs_traffic_patterns[p].resize(scen.num_bs_types);
		for (std::size_t bc = 0; bc < scen.num_bs_types; ++bc)
		{
			const std::size_t nb = scen.provider_num_bss[p][bc];

//			prov_bs_amort_costs(p,bc) = scen.provider_bs_amortizing_costs[p][bc];
//			prov_bs_amort_times(p,bc) = scen.provider_bs_residual_amortizing_times[p][bc];

			if (nb > 0)
			{
				const std::size_t old_tot_nb = tot_nb;

				prov_num_bss(p) += nb;
				tot_nb += nb;

				prov_to_bss[p] = std::make_pair(old_tot_nb, tot_nb);

				// Build the BS traffic curve (the same for each BS, but possible different for each provider)
				std::vector<RealT> bs_traffic_data_x(scen.provider_bs_traffic_data[p][bc].size(), 0);
				std::vector<RealT> bs_traffic_data_y(scen.provider_bs_traffic_data[p][bc].size(), 0);
				for (std::size_t i = 0; i < scen.provider_bs_traffic_data[p][bc].size(); ++i)
				{
					bs_traffic_data_x[i] = scen.provider_bs_traffic_data[p][bc][i].first;
					bs_traffic_data_y[i] = scen.provider_bs_traffic_data[p][bc][i].second;
				}
				boost::shared_ptr< math::curvefit::base_1d_interpolator<RealT> > p_interp;
				switch (scen.provider_bs_traffic_curves[p][bc])
				{
					case constant_bs_traffic_curve:
						p_interp = boost::make_shared< math::curvefit::constant_interpolator<RealT> >(bs_traffic_data_x.begin(),
																									  bs_traffic_data_x.end(),
																									  bs_traffic_data_y.begin(),
																									  bs_traffic_data_y.end());
						break;
					case cubic_spline_bs_traffic_curve:
						p_interp = boost::make_shared< math::curvefit::cubic_spline_interpolator<RealT> >(bs_traffic_data_x.begin(),
																										  bs_traffic_data_x.end(),
																										  bs_traffic_data_y.begin(),
																										  bs_traffic_data_y.end(),
																										  math::curvefit::periodic_spline_boundary_condition);
						break;
					case linear_bs_traffic_curve:
						p_interp = boost::make_shared< math::curvefit::linear_interpolator<RealT> >(bs_traffic_data_x.begin(),
																									bs_traffic_data_x.end(),
																									bs_traffic_data_y.begin(),
																									bs_traffic_data_y.end());
						break;
				}
				//bs_traffic_patterns[p][bc] = boost::make_shared< daily_traffic_pattern<RealT> >(p_interp);
				bs_traffic_patterns[p][bc] = boost::make_shared<TrafficFuncT>(p_interp);
			}
			else
			{
				prov_to_bss[p] = std::make_pair(tot_nb, tot_nb);
			}

		}
	}

	// Compute BS topology info
	std::vector< std::pair<RealT,RealT> > bs_positions;
	//bs_positions = make_bs_topology(tot_nb, scen.topology_area_size, random_bs_topology_layout, rng);
	//bs_positions = make_bs_topology(tot_nb, scen.topology_area_size, square_bs_topology_layout, rng);
	//bs_positions = make_bs_topology(tot_nb, scen.topology_area_size, diamond_bs_topology_layout, rng, 2.0/3.0);
	//bs_positions = make_bs_topology(tot_nb, scen.topology_area_size, center_bs_topology_layout, rng);
	//bs_positions = make_bs_topology(tot_nb, scen.topology_area_size, polygon_bs_topology_layout, rng, 0.5);
	bs_positions = make_bs_topology(tot_nb, scen.topology_area_size, scen.topology_bs_layout, rng, 0.5);

	// Maps BSs to their category
	ublas::vector<std::size_t> global_bs_to_cats(tot_nb); // BS to category s.t. if bs_to_cats[i]=k means that the i-th BS belongs to provider p
	for (std::size_t p = 0; p < scen.num_providers; ++p)
	{
		std::size_t b_offs = prov_to_bss.at(p).first;
		for (std::size_t bc = 0; bc < scen.num_bs_types; ++bc)
		{
			const std::size_t nbc = scen.provider_num_bss[p][bc];

			ublas::subrange(global_bs_to_cats, b_offs, b_offs+nbc) = ublas::scalar_vector<std::size_t>(nbc, bc);
			b_offs += nbc;
		}
	}

	std::vector< boost::shared_ptr< dcs::bs::ci_mean_estimator<RealT> > > ci_ap_stats(scen.num_providers); // Alone profits
	std::vector< boost::shared_ptr< dcs::bs::ci_mean_estimator<RealT> > > ci_anu_stats(tot_nb); // Alone number of served users

	// Initialize confidence interval variables
	for (std::size_t p = 0; p < scen.num_providers; ++p)
	{
		std::ostringstream oss;

		ci_ap_stats[p] = boost::make_shared< dcs::bs::ci_mean_estimator<RealT> >(opts.ci_level, opts.ci_rel_precision);
		oss.str("");
		oss << "AP_{" << p << "}";
		ci_ap_stats[p]->name(oss.str());
	}

	for (std::size_t b = 0; b < tot_nb; ++b)
	{
		std::ostringstream oss;

		ci_anu_stats[b] = boost::make_shared< dcs::bs::ci_mean_estimator<RealT> >(opts.ci_level, std::numeric_limits<RealT>::infinity());
		oss.str("");
		oss << "ANU_{" << b << "}";
		ci_anu_stats[b]->name(oss.str());
	}

    const std::size_t num_time_slots = ::std::ceil(opts.time_span/time_step);

	std::size_t iter = 0;
	while (!util::check_stats(ci_ap_stats.begin(), ci_ap_stats.end())
		   || !util::check_stats(ci_anu_stats.begin(), ci_anu_stats.end()))
	{
		++iter;

		DCS_DEBUG_TRACE("Iteration #" << iter << "...");//XXX

//const std::size_t seed = std::rand();//XXX
		for (std::size_t p = 0; p < scen.num_providers; ++p)
		{
			DCS_DEBUG_TRACE("Provider: " << p);//XXX

//rng.seed(seed+1);//XXX: for testing purpose
			const std::size_t nb = prov_num_bss(p);

			const ublas::scalar_vector<std::size_t> P(1, p);
			const ublas::scalar_vector<std::size_t> bs_to_provs(nb, p); // BS to provider s.t. if bs_to_provs[i]=p means that the i-th BS belongs to provider p
			ublas::vector<std::size_t> bs_to_cats(nb); // BS to category s.t. if bs_to_cats[i]=k means that the i-th BS belongs to provider p

			// Maps BSs to their category (single provider version)
			{
				std::size_t b_offs = 0;
				for (std::size_t bc = 0; bc < scen.num_bs_types; ++bc)
				{
					const std::size_t nbc = scen.provider_num_bss[p][bc];

					ublas::subrange(bs_to_cats, b_offs, b_offs+nbc) = ublas::scalar_vector<std::size_t>(nbc, bc);
					b_offs += nbc;
				}
				// check: double check on number of BSs
				DCS_DEBUG_ASSERT( b_offs <= nb );
			}

			// The set of BSs of this provider
			ublas::vector<std::size_t> B(nb);
			for (std::size_t b = 0; b < nb; ++b)
			{
				B(b) = b;
			}

			RealT tot_profit = 0; // The total profit of this provider
			//std::vector< boost::accumulators::accumulator_set< RealT, boost::accumulators::stats<boost::accumulators::tag::mean> > > bs_num_served_users_acc(nb); // The number of served users
			std::vector<RealT> bs_num_served_users(nb, 0); // The number of served users

			// Discretize the temporal interval

			for (std::size_t time_slot = 0; time_slot < num_time_slots; ++time_slot)
			{
				const RealT time_low = time_slot*time_step;
				const RealT time_up = std::min(time_low+time_step, opts.time_span);
				const RealT deltat = time_up-time_low; // The real length of this time interval

				DCS_DEBUG_TRACE("Time interval [" << time_low << "," << time_up << ") -- Slot: " << time_slot);//XXX

				// Find weights to apply to different energy costs
				RealT alt_int_len = math::float_traits<RealT>::max(math::float_traits<RealT>::min(time_up, scen.provider_alt_electricity_time_ranges[p].second)-math::float_traits<RealT>::max(time_low, scen.provider_alt_electricity_time_ranges[p].first), 0.0);

				RealT pc_alt_weights = alt_int_len/(time_up-time_low);
				RealT pc_weights = 1-pc_alt_weights;

				DCS_DEBUG_TRACE("PROVIDER " << p << " - ELECTRICITY COST: alternative coverage: " << alt_int_len << " -- normal weight: " << pc_weights << " -- alternative weight: " << pc_alt_weights);//XXX

				// Find peak user load in this time slot
				// - First uses a global optimization method
				// - Then refine the found solution with a location optimization
				//   method using as starting point the solution provided by the
				//   global method.

				std::size_t nu = 0; // Total number of (peak) users among the various BSs
				for (std::size_t bc = 0; bc < scen.num_bs_types; ++bc)
				{
					DCS_DEBUG_TRACE("Finding Peak user load of Traffic Load Curve for provider " << p << ", BS Category " << bc << " and time interval [" << time_low << "," << time_up << ")...");//XXX

					//math::optim::nelder_mead_simplex_optimizer<RealT> optim;
					//math::optim::powell_cobyla_optimizer<RealT> optim;
					math::optim::jones_direct_optimizer<RealT> glob_optim;
					glob_optim.lower_bound(std::vector<RealT>(1,time_low));
					glob_optim.upper_bound(std::vector<RealT>(1,time_up));
					glob_optim.x0(std::vector<RealT>(1,(time_low+time_up)/2.0));
					glob_optim.fx_relative_tolerance(1e-6);
					glob_optim.x_relative_tolerance(1e-2); // Needed, otherwise the DIRECT method gets stuck
					math::optim::optimization_result<RealT> optim_info;
					optim_info = math::optim::maximize<RealT>(glob_optim, math::optim::make_multivariable_function(*(bs_traffic_patterns[p][bc])));
					if (!optim_info.converged)
					{
						std::ostringstream oss;
						oss << "Unable to find peak user load for provider " << p << ", BS Category " << bc << " and time interval [" << time_low << "," << time_up << ")";
						dcs::log_error(DCS_LOGGING_AT, oss.str());
						continue;
					}

					DCS_DEBUG_TRACE("Global Peak user load of Traffic Load Curve for provider " << p << ", BS Category " << bc << " and time interval [" << time_low << "," << time_up << "): (" << optim_info.xopt[0] << "," << optim_info.fopt << ")");//XXX

					// Refine the global optimum with a local optimizer
					math::optim::powell_cobyla_optimizer<RealT> loc_optim;
					loc_optim.lower_bound(std::vector<RealT>(1,time_low));
					loc_optim.upper_bound(std::vector<RealT>(1,time_up));
					loc_optim.x0(optim_info.xopt);
					loc_optim.fx_relative_tolerance(1e-6);
					optim_info = math::optim::maximize<RealT>(loc_optim, math::optim::make_multivariable_function(*(bs_traffic_patterns[p][bc])));
					if (!optim_info.converged)
					{
						std::ostringstream oss;
						oss << "Unable to find peak user load for provider " << p << ", BS Category " << bc << " and time interval [" << time_low << "," << time_up << ")";
						dcs::log_error(DCS_LOGGING_AT, oss.str());
						continue;
					}

					DCS_DEBUG_TRACE("Refined Peak user load of Traffic Load Curve for provider " << p << ", BS Category " << bc << " and time interval [" << time_low << "," << time_up << "): (" << optim_info.xopt[0] << "," << optim_info.fopt << ")");//XXX

					//DCS_DEBUG_ASSERT( optim_info.fopt >= 0.0 && optim_info.fopt <= 1.0 );
					//DCS_DEBUG_ASSERT( optim_info.fopt >= 0.0 );

					if (optim_info.fopt < 0)
					{
						optim_info.fopt = 0;
					}

					const std::size_t bs_nu_peak = std::ceil(scen.provider_bs_max_num_users[p][bc]*optim_info.fopt);

					DCS_DEBUG_TRACE("PROV: " << p << " - BS Category: " << bc << " - BS_NU_MAX: " << scen.provider_bs_max_num_users[p][bc] << " - BS_NU_PEAK: " << bs_nu_peak);///XXX

					nu += bs_nu_peak;
				}

				// Map users to their category
				ublas::vector<std::size_t> usr_to_cats(nu, 0); // User to category s.t. if usr_to_cats[i]=k means that the i-th user belongs to category k
				{
					std::size_t u_offs = 0;

					if (nu > 0)
					{
						std::size_t tot_nuc = 0;
						for (std::size_t uc = 0; uc < scen.num_user_classes; ++uc)
						{
							const std::size_t nuc = (uc < (scen.num_user_classes-1)) ? boost::math::round(nu*scen.provider_user_mixes[p][uc]) : nu-tot_nuc; // Handle residual users

							DCS_DEBUG_TRACE("PROVIDER: " << p << " - CLASS: " << uc << " - TOT NU: " << nu << " - N: " << nuc << " (U_OFFS: " << u_offs << ")");//XXX

							if (nuc > 0)
							{
								ublas::subrange(usr_to_cats, u_offs, u_offs+nuc) = ublas::scalar_vector<std::size_t>(nuc, uc);

								u_offs += nuc;
								tot_nuc += nuc;
							}
						}
					}

					// check: double checks on # users
					DCS_DEBUG_ASSERT(u_offs <= nu); //NOTE: we use '<=' and not '==' since to compute nu we use the ceiling function and hence we can get a greater number
				}

				// Compute user topology info
				ublas::vector< std::pair<RealT,RealT> > usr_positions(nu); // Position of each user inside the area
				ublas::matrix<RealT> usr_bs_dists(nu, nb, 0); // Distance from a user to a BS
				ublas::matrix<RealT> usr_bs_path_losses(nu, nb, 0); // Path loss from a user to a BS
				for (std::size_t u = 0; u < nu; ++u)
				{
					const std::size_t uc = usr_to_cats(u);

					usr_positions(u) = std::make_pair(rvg_x_pos(rng), rvg_y_pos(rng));

					for (std::size_t b = 0; b < nb; ++b)
					{
						const std::size_t bc = bs_to_cats(b);
						const std::size_t bb = prov_to_bss.at(p).first+b;

						// check: pedantic check
						DCS_DEBUG_ASSERT( bb < prov_to_bss.at(p).second );

						// check: pedantic check
						DCS_DEBUG_ASSERT( bc == global_bs_to_cats(bb) );

						usr_bs_dists(u,b) = detail::util::euclidean_distance(usr_positions(u).first, usr_positions(u).second, bs_positions[bb].first, bs_positions[bb].second);
//usr_bs_dists(u,b) = 100;//XXX: Just to try...
						//usr_bs_path_losses(u,b) = dcs::bs::distance_based_path_loss_model(default_path_loss_exponent, usr_bs_dists(u,b)*1e-3);
						usr_bs_path_losses(u,b) = dcs::bs::cost231_hata_path_loss_model(bs_freqs(bc), bs_heights(bc), usr_heights(uc), usr_bs_dists(u,b)*1e-3, dcs::bs::urban_large_radio_propagation_area);
					}
				}

				// Compute SINR
				ublas::matrix<RealT> usr_bs_sinrs(nu, nb, 0); // SINR between each pair of BS and user
				for (std::size_t u = 0; u < nu; ++u)
				{
					const std::size_t uc = usr_to_cats(u);

					for (std::size_t b = 0; b < nb; ++b)
					{
						const std::size_t bc = bs_to_cats(b);
						const std::size_t bb = prov_to_bss.at(p).first+b;

						// check: pedantic check
						DCS_DEBUG_ASSERT( bb < prov_to_bss.at(p).second );

						// check: pedantic check
						DCS_DEBUG_ASSERT( bc == global_bs_to_cats(bb) );

//std::cerr << "BB: " << bb << std::endl;//XXX
						RealT interf = 0; // Interference from other BSs
						for (std::size_t bb2 = 0; bb2 < tot_nb; ++bb2)
						{
//std::cerr << "BB2: " << bb2 << std::endl;//XXX
							if (bb2 != bb)
							{
								const std::size_t bc2 = global_bs_to_cats(bb2);

								const RealT usr_bb2_dist = detail::util::euclidean_distance(usr_positions(u).first, usr_positions(u).second, bs_positions[bb2].first, bs_positions[bb2].second);
//const RealT usr_bb2_dist = 100;//XXX: Just to try...
								//const RealT usr_bb2_path_loss = dcs::bs::distance_based_path_loss_model(default_path_loss_exponent, usr_b2_dist*1e-3);
								const RealT usr_bb2_path_loss = dcs::bs::cost231_hata_path_loss_model(bs_freqs(bc2), bs_heights(bc2), usr_heights(uc), usr_bb2_dist*1e-3, dcs::bs::urban_large_radio_propagation_area);
								interf += bs_tx_powers(bc2)/std::pow(10.0, usr_bb2_path_loss/10.0);
							}
						}
						//usr_bs_sinrs(u,b) = bs_tx_powers(bc)/std::pow(10.0, usr_bs_path_losses(u,b)/10.0)/(usr_therm_noises(uc));
						usr_bs_sinrs(u,b) = bs_tx_powers(bc)/(std::pow(10.0, usr_bs_path_losses(u,b)/10.0)*(interf+usr_therm_noises(uc)));
					}
				}

				if (opts.verbosity >= VERBOSITY_MEDIUM)
				{
					std::cout << "-- INPUTS:" << std::endl;
					std::cout << "- Number of Providers = " << scen.num_providers << std::endl;
					std::cout << "- Number of BS Types = " << scen.num_bs_types << std::endl;
					std::cout << "- Number of User Classes = " << scen.num_user_classes << std::endl;
					std::cout << "- BS :: Height (in m) = " << bs_heights << std::endl;
					std::cout << "- BS :: Carrier Frequencies (in MHz) = " << bs_freqs << std::endl;
					std::cout << "- BS :: Bandwidth (in Hz) = " << bs_bws << std::endl;
					//std::cout << "- BS :: Max Download Data Rate (in Mbps)= " << bs_caps << std::endl;
					//std::cout << "- BS :: Max Number of Subcarriers = " << bs_num_subcarriers << std::endl;
					std::cout << "- BS :: Transmission Power (in W) = " << bs_tx_powers << std::endl;
					std::cout << "- BS :: Constant Power Consumption by Category (in W) = " << bs_const_powers << std::endl;
					std::cout << "- BS :: Load-dependent Power Consumption by Category (in W) = " << bs_load_powers << std::endl;
					std::cout << "- BS :: Positions in the Area = [" << bs_positions.size() << "](";
					for (std::size_t b = 0; b < nb; ++b)
					{
						if (b > 0)
						{
							std::cout << ",";
						}
						std::cout << "<" << bs_positions[b].first << "," << bs_positions[b].second << ">";
					}
					std::cout << ")" << std::endl;
					std::cout << "- Provider :: Energy Cost by Provider (in $) = " << prov_energy_costs << std::endl;
					//std::cout << "- Provider :: BS Amortizing Cost by Provider and BS Category (in $) = " << prov_bs_amort_costs << std::endl;
					std::cout << "- Provider :: Peak Number of Users by Provider = " << nu << std::endl;
					std::cout << "- Provider :: Min Downlink Data Rate for each User by Provider (in Mbit/sec) = " << prov_usr_min_dl_rates << std::endl;
					std::cout << "- Provider :: Revenue for each User by Provider (in $/user) = " << prov_usr_revenues << std::endl;
					std::cout << "- Provider :: Number of BS by Provider = " << prov_num_bss << std::endl;
					std::cout << "- Provider :: BS Categories by Provider = " << bs_to_cats << std::endl;
					std::cout << "- Provider :: User Categories by Provider = " << usr_to_cats << std::endl;
					std::cout << "- User :: QoS Downlink Throughput by Category (in Mbit/sec) = " << usr_qos_dl_rates << std::endl;
					std::cout << "- User :: Height (in m) = " << usr_heights << std::endl;
					std::cout << "- User :: Thermal Noise (in W) = " << usr_therm_noises << std::endl;
					std::cout << "- User :: Positions in the Area = [" << usr_positions.size() << "](";
					for (std::size_t u = 0; u < nu; ++u)
					{
						if (u > 0)
						{
							std::cout << ",";
						}
						std::cout << "<" << usr_positions(u).first << "," << usr_positions(u).second << ">";
					}
					std::cout << ")" << std::endl;
					std::cout << "- User :: Distances from BSs (in m) = " << usr_bs_dists << std::endl;
					std::cout << "- User :: Path losses from BSs (in dB) = " << usr_bs_path_losses << std::endl;
					std::cout << "- User :: SINRs from BSs = " << usr_bs_sinrs << std::endl;
				}

				ublas::vector<std::size_t> U(nu);

				for (std::size_t u = 0; u < nu; ++u)
				{
					U(u) = u;
				}

				bs::user_assignment_solution<RealT> user_assignment;

				switch (opts.user_assignment)
				{
					case max_profit_user_assignment:
						user_assignment = bs::optimal_max_profit_user_assignment_solver<RealT>()(P,
																								 B,
																								 U,
																								 bs_to_cats,
																								 bs_to_provs,
																								 bs_bws,
																								 bs_const_powers,
																								 bs_load_powers,
																								 usr_to_cats,
																								 usr_qos_dl_rates,
																								 usr_bs_sinrs,
																								 prov_energy_costs,
																								 prov_usr_min_dl_rates,
																								 prov_usr_revenues);
						break;
					case min_cost_user_assignment:
						throw std::runtime_error("Min-Cost-User-Assignment is Deprecated");
						break;
				}

				if (user_assignment.solved)
				{
					// check: pedantic check for the number of allocated users
#ifdef DCS_DEBUG
					std::size_t alloc_nu = 0;
					for (std::size_t ib = 0; ib < nb; ++ib)
					{
						for (std::size_t iu = 0; iu < nu; ++iu)
						{
							if (user_assignment.bs_user_allocations(ib,iu))
							{
								++alloc_nu;
							}
						}
					}

					DCS_DEBUG_ASSERT( alloc_nu == nu );
#endif // DCS_DEBUG

					RealT profit = 0;

					switch (opts.user_assignment)
					{
						case max_profit_user_assignment:
							profit = user_assignment.objective_value;
							break;
						case min_cost_user_assignment:
							{
								RealT R = 0;
								for (std::size_t ib = 0; ib < nb; ++ib)
								{
									for (std::size_t iu = 0; iu < nu; ++iu)
									{
										if (user_assignment.bs_user_allocations(ib,iu))
										{
											R += prov_usr_revenues(p,usr_to_cats(U(iu)));
										}
									}
								}
								profit = R - user_assignment.objective_value;
							}
							break;
					}
					// Multiply for the interval length
					profit *= deltat;
					// Add amortizing costs
					for (std::size_t ib = 0; ib < nb; ++ib)
					{
						const bs::bsid_type b = B(ib);
						const std::size_t bc = bs_to_cats(b);
						const RealT prov_bs_amort_costs = scen.provider_bs_amortizing_costs[p][bc];
						const RealT prov_bs_amort_times = std::min(std::max(0.0, scen.provider_bs_residual_amortizing_times[p][bc]-time_low), deltat);

						profit -= prov_bs_amort_costs*prov_bs_amort_times;
					}

					// Update stats

					tot_profit += profit;
					for (std::size_t ib = 0; ib < nb; ++ib)
					{
						//const bs::bsid_type b = prov_to_bss.at(p).first+ib;

						std::size_t bs_nu = 0;

						for (std::size_t iu = 0; iu < nu; ++iu)
						{
							if (user_assignment.bs_user_allocations(ib,iu))
							{
								++bs_nu;
							}
						}

						//bs_num_served_users_acc[ib](bs_nu);
						bs_num_served_users[ib] += bs_nu*deltat;
					}

					DCS_DEBUG_TRACE( "User assignment objective value: " << user_assignment.objective_value << " - Total coalition watts (kW): " << user_assignment.kwatt << " - Total coalition cost: " << user_assignment.cost << " => Profit=" << profit );
				}
			}

			// Update confidence intervals

			ci_ap_stats[p]->collect(tot_profit);
			for (std::size_t ib = 0; ib < nb; ++ib)
			{
				const bs::bsid_type b = prov_to_bss.at(p).first+ib;

				//ci_anu_stats[b]->collect(boost::accumulators::mean(bs_num_served_users_acc[ib]));
				ci_anu_stats[b]->collect(bs_num_served_users[ib]);
			}
		}

		// Print confidence intervals
		if (opts.verbosity >= VERBOSITY_LOW)
		{
			std::cout << "-- ITERATION #" << iter << std::endl;
			std::cout << " - CONFIDENCE INTERVALS OUTPUTS:" << std::endl;
			for (std::size_t p = 0; p < scen.num_providers; ++p)
			{
				std::cout << "  * NO " << p << std::endl;
				std::cout << "   - AP statistics: " << ci_ap_stats[p]->estimate() << " (s.d. " << ci_ap_stats[p]->standard_deviation() << ") [" << ci_ap_stats[p]->lower() << ", " << ci_ap_stats[p]->upper() << "] (rel. prec.: " << ci_ap_stats[p]->relative_precision() << ", size: " << ci_ap_stats[p]->size() << ")" << std::endl;

				for (std::size_t b = prov_to_bss.at(p).first; b < prov_to_bss.at(p).second; ++b)
				{
					std::cout << "   - ANU statistics: " << ci_anu_stats[b]->estimate() << " (s.d. " << ci_anu_stats[b]->standard_deviation() << ") [" << ci_anu_stats[b]->lower() << ", " << ci_anu_stats[b]->upper() << "] (rel. prec.: " << ci_anu_stats[b]->relative_precision() << ", size: " << ci_anu_stats[b]->size() << ")" << std::endl;
				}
			}
		}
	}

	// Print confidence intervals
	if (opts.verbosity >= VERBOSITY_LOW)
	{
		std::cout << "-- CONFIDENCE INTERVALS OUTPUTS:" << std::endl;
		for (std::size_t p = 0; p < scen.num_providers; ++p)
		{
			std::cout << " * NO " << p << std::endl;
			std::cout << "  - AP statistics: " << ci_ap_stats[p]->estimate() << " (s.d. " << ci_ap_stats[p]->standard_deviation() << ") [" << ci_ap_stats[p]->lower() << ", " << ci_ap_stats[p]->upper() << "] (rel. prec.: " << ci_ap_stats[p]->relative_precision() << ", size: " << ci_ap_stats[p]->size() << ")" << std::endl;

			for (std::size_t b = prov_to_bss.at(p).first; b < prov_to_bss.at(p).second; ++b)
			{
				std::cout << "  - ANU statistics: " << ci_anu_stats[b]->estimate() << " (s.d. " << ci_anu_stats[b]->standard_deviation() << ") [" << ci_anu_stats[b]->lower() << ", " << ci_anu_stats[b]->upper() << "] (rel. prec.: " << ci_anu_stats[b]->relative_precision() << ", size: " << ci_anu_stats[b]->size() << ")" << std::endl;
			}
		}
	}

	std::vector<RealT> alone_profits(scen.num_providers);
	for (std::size_t p = 0; p < scen.num_providers; ++p)
	{
		alone_profits[p] = ci_ap_stats[p]->estimate();
	}

	bs_num_served_users.resize(tot_nb, 0);
	for (std::size_t b = 0; b < scen.num_providers; ++b)
	{
		bs_num_served_users[b] = ci_anu_stats[b]->estimate();
	}

	return alone_profits;
}

template <typename TrafficFuncT,
		  typename RealT,
		  typename RNGT,
		  typename MacroBssVectorT,
		  typename MacroBsToCatsVectorT,
		  typename MacroBsToProvsVectorT,
		  //typename MacroUserVectorT,
		  typename MacroUserToCatsVectorT,
		  typename MacroUserBsSINRsMatrixT>
std::vector<RealT> estimate_coalition_profits(scenario<RealT> const& scen,
											  options<RealT> const& opts,
											  RNGT& rng,
											  RealT macro_time_low,
											  RealT macro_time_up,
											  ublas::vector<MacroBssVectorT> const& macro_bss,
											  std::vector< std::vector< boost::shared_ptr< TrafficFuncT > > > const& macro_bs_traffic_patterns,
											  ublas::vector_expression<MacroBsToCatsVectorT> const& macro_bs_to_cats,
											  ublas::vector_expression<MacroBsToProvsVectorT> const& macro_bs_to_provs,
											  //ublas::vector_expression<MacroUserVectorT> const& macro_users,
											  ublas::vector_expression<MacroUserToCatsVectorT> const& macro_user_to_cats,
											  ublas::matrix_expression<MacroUserBsSINRsMatrixT> const& macro_user_bs_sinrs,
											  std::map< std::size_t, std::pair<std::size_t,std::size_t> > const& macro_prov_to_users,
											  bs::coalition_formation_info<RealT> const& macro_formed_partitions,
											  std::vector<RealT>& bs_num_served_users)
{
	DCS_MACRO_SUPPRESS_UNUSED_VARIABLE_WARNING( opts );

	typedef typename std::set<gtpack::cid_type>::const_iterator partition_iterator;
	typedef typename std::map<gtpack::player_type,RealT>::const_iterator coalition_iterator;

	// Sample the traffic load curve with micro-intervals of 1 minute

	const RealT macro_deltat = macro_time_up-macro_time_low;
	const std::size_t num_micro_time_slots = ::std::ceil(macro_deltat/micro_time_step);
	const std::size_t nparts = macro_formed_partitions.best_partitions.size();
	const std::size_t nb = ublasx::size(macro_bss);

	std::vector< std::map<gtpack::cid_type,RealT> > part_profits(nparts);
	//std::vector< boost::accumulators::accumulator_set< RealT, boost::accumulators::stats<boost::accumulators::tag::mean> > > bs_num_served_users_acc(nb);
	bs_num_served_users.resize(nb);

	for (std::size_t micro_time_slot = 0; micro_time_slot < num_micro_time_slots; ++micro_time_slot)
	{
		const RealT micro_time_low = macro_time_low+micro_time_slot*micro_time_step;
		const RealT micro_time_up = std::min(micro_time_low+micro_time_step, macro_time_up);
		const RealT micro_deltat = micro_time_up-micro_time_low;

		DCS_DEBUG_TRACE("Micro-Time Interval [" << micro_time_low << "," << micro_time_up << ") -- Slot: " << micro_time_slot);//XXX

		ublas::vector<std::size_t> micro_prov_peak_nus(scen.num_providers);
		//ublas::vector< std::pair<std::size_t,std::size_t> > micro_prov_to_usrs(scen.num_providers);
		//ublas::vector<std::size_t> micro_usr_to_cats; // User to category s.t. if usr_to_cats[i]=k means that the i-th user belongs to category k
		std::size_t micro_tot_nu = 0;

		// Find peak user load and discretize

		for (std::size_t p = 0; p < scen.num_providers; ++p)
		{
			std::size_t prov_nu = 0;
			//std::size_t u_offs = micro_tot_nu;

			for (std::size_t bc = 0; bc < scen.num_bs_types; ++bc)
			{
				math::optim::jones_direct_optimizer<RealT> glob_optim;
				glob_optim.lower_bound(std::vector<RealT>(1,micro_time_low));
				glob_optim.upper_bound(std::vector<RealT>(1,micro_time_up));
				glob_optim.x0(std::vector<RealT>(1,(micro_time_low+micro_time_up)/2.0));
				glob_optim.fx_relative_tolerance(1e-6);
				glob_optim.x_relative_tolerance(1e-2); // Needed, otherwise the DIRECT method gets stuck
				math::optim::optimization_result<RealT> optim_info;
				optim_info = math::optim::maximize<RealT>(glob_optim, math::optim::make_multivariable_function(*(macro_bs_traffic_patterns[p][bc])));
				if (!optim_info.converged)
				{
					std::ostringstream oss;
					oss << "Unable to find peak user load for provider " << p << ", BS Category " << bc << " and time interval [" << micro_time_low << "," << micro_time_up << ")";
					dcs::log_error(DCS_LOGGING_AT, oss.str());
					continue;
				}

				DCS_DEBUG_TRACE("Global Peak user load of Traffic Load Curve for provider " << p << ", BS Category " << bc << " and time interval [" << micro_time_low << "," << micro_time_up << "): (" << optim_info.xopt[0] << "," << optim_info.fopt << ")");//XXX

				// Refine the global optimum with a local optimizer
				math::optim::powell_cobyla_optimizer<RealT> loc_optim;
				loc_optim.lower_bound(std::vector<RealT>(1,micro_time_low));
				loc_optim.upper_bound(std::vector<RealT>(1,micro_time_up));
				loc_optim.x0(optim_info.xopt);
				loc_optim.fx_relative_tolerance(1e-6);
				optim_info = math::optim::maximize<RealT>(loc_optim, math::optim::make_multivariable_function(*(macro_bs_traffic_patterns[p][bc])));
				if (!optim_info.converged)
				{
					std::ostringstream oss;
					oss << "Unable to find peak user load for provider " << p << ", BS Category " << bc << " and time interval [" << micro_time_low << "," << micro_time_up << ")";
					dcs::log_error(DCS_LOGGING_AT, oss.str());
					continue;
				}

				DCS_DEBUG_TRACE("Refined Peak user load of Traffic Load Curve for provider " << p << ", BS Category " << bc << " and time interval [" << micro_time_low << "," << micro_time_up << "): (" << optim_info.xopt[0] << "," << optim_info.fopt << ")");//XXX

				if (optim_info.fopt < 0)
				{
					optim_info.fopt = 0;
				}

				const std::size_t nu = std::ceil(scen.provider_bs_max_num_users[p][bc]*optim_info.fopt);

				DCS_DEBUG_TRACE("PROVIDER: " << p << " - BS Category: " << bc << " - BS_NU_MAX: " << scen.provider_bs_max_num_users[p][bc] << " - BS_NU_PEAK: " << nu);///XXX

				prov_nu += nu;
			}

			micro_prov_peak_nus(p) = prov_nu;
			micro_tot_nu += prov_nu;
			//micro_prov_to_usrs(p) = std::make_pair(u_offs, u_offs+prov_nu);

			DCS_DEBUG_TRACE("PROVIDER: " << p << " - NU: " << prov_nu << " (TOT NU: " << micro_tot_nu << ")");//XXX
		}

		// Sample (without replacement) from user population of macro-interval

		std::set<std::size_t> micro_users;

		for (std::size_t p = 0; p < scen.num_providers; ++p)
		{
			const std::size_t min = macro_prov_to_users.at(p).first;
			const std::size_t max = (macro_prov_to_users.at(p).second > min) ? (macro_prov_to_users.at(p).second - 1) : min;

			boost::random::uniform_int_distribution<> urvg(min, max);

			const std::size_t nu = micro_prov_peak_nus(p);

			for (std::size_t i = 0; i < nu; ++i)
			{
				std::size_t u;
				do
				{
					u = static_cast<std::size_t>(urvg(rng));
				} while (micro_users.count(u) > 0);

				micro_users.insert(u);
			}
		}

		// Iterate through all the partitions that may form at this time-slot

		std::vector< boost::accumulators::accumulator_set< RealT, boost::accumulators::stats<boost::accumulators::tag::mean> > > part_bs_num_served_users_acc(nb);

		for (std::size_t pix = 0; pix < nparts; ++pix)
		{
			const bs::partition_info<RealT>& best_partition = macro_formed_partitions.best_partitions[pix];

			//const std::size_t npc = best_partition.coalitions.size();
			const partition_iterator part_beg_it = best_partition.coalitions.begin();
			const partition_iterator part_end_it = best_partition.coalitions.end();

			for (partition_iterator part_it = part_beg_it;
				 part_it != part_end_it;
				 ++part_it)
			{
				const gtpack::cid_type cid = *part_it;

				std::set<std::size_t> coal_bss;

				const coalition_iterator coal_end_it = macro_formed_partitions.coalitions.at(cid).payoffs.end();
				for (coalition_iterator coal_it = macro_formed_partitions.coalitions.at(cid).payoffs.begin();
					 coal_it != coal_end_it;
					 ++coal_it)
				{
					const gtpack::player_type b = coal_it->first;

					coal_bss.insert(b);
				}

				// Alocate users to BSs

				std::map<std::size_t,std::set<std::size_t> > coal_bs_user_alloc;
				std::map<std::size_t,RealT> coal_bs_tot_qos_data_rates;

				const typename std::set<std::size_t>::const_iterator usr_end_it = micro_users.end();
				for (typename std::set<std::size_t>::const_iterator usr_it = micro_users.begin();
					 usr_it != usr_end_it;
					 ++usr_it)
				{
					const std::size_t u = *usr_it;

					if (macro_formed_partitions.coalitions.at(cid).usr_to_idx.count(u) == 0)
					{
						// This user wasn't on this coalition
						continue;
					}

					const std::size_t uix = macro_formed_partitions.coalitions.at(cid).usr_to_idx.at(u);

					const typename std::set<std::size_t>::const_iterator bs_end_it = coal_bss.end();
					for (typename std::set<std::size_t>::const_iterator bs_it = coal_bss.begin();
						 bs_it != bs_end_it;
						 ++bs_it)
					{
						const std::size_t b = *bs_it;
						const std::size_t bix = macro_formed_partitions.coalitions.at(cid).bsid_to_idx.at(b);

						if (macro_formed_partitions.coalitions.at(cid).user_assignment.bs_user_allocations(bix,uix))
						{
							const std::size_t uc = macro_user_to_cats()(u);
							const RealT qos_data_rate = scen.user_downlink_data_rates[uc];

							coal_bs_user_alloc[b].insert(u);
							if (coal_bs_tot_qos_data_rates.count(b) > 0)
							{
								coal_bs_tot_qos_data_rates[b] += qos_data_rate;
							}
							else
							{
								coal_bs_tot_qos_data_rates[b] = qos_data_rate;
							}

							break;
						}
					}
				}

				// Compute revenue and cost of this coalition for this micro_interval

				RealT coal_revenue = 0;
				RealT coal_cost = 0;

				//const typename std::map< std::size_t, std::set<std::size_t> >::const_iterator bs_usr_end_it = coal_bs_user_alloc.end();
				//for (typename std::map< std::size_t, std::set<std::size_t> >::const_iterator bs_usr_it = coal_bs_user_alloc.begin();
				//	 bs_usr_it != bs_usr_end_it;
				//	 ++bs_usr_it)
				const typename std::set<std::size_t>::const_iterator bs_end_it = coal_bss.end();
				for (typename std::set<std::size_t>::const_iterator bs_it = coal_bss.begin();
					 bs_it != bs_end_it;
					 ++bs_it)
				{
					const std::size_t b = *bs_it;
					const std::size_t bix = macro_formed_partitions.coalitions.at(cid).bsid_to_idx.at(b);

					// Check if this BS is powered on
					if (!macro_formed_partitions.coalitions.at(cid).user_assignment.bs_power_states(bix))
					{
						part_bs_num_served_users_acc[b](0); // Just to avoid producing a NaN...
						continue;
					}

					const std::size_t bc = macro_bs_to_cats()(b);
					const std::size_t p = macro_bs_to_provs()(b);
					const RealT bw = scen.bs_bandwidths[bc];
					const RealT energy_cost = scen.provider_electricity_costs[p];

					RealT bs_cost = scen.bs_idle_powers[bc]*energy_cost;

					if (coal_bs_user_alloc.count(b) > 0)
					{
						const RealT tot_qos_data_rate = coal_bs_tot_qos_data_rates.at(b);
						const std::size_t bs_nu = coal_bs_user_alloc.at(b).size();

						const typename std::set<std::size_t>::const_iterator usr_end_it = coal_bs_user_alloc.at(b).end();
						for (typename std::set<std::size_t>::const_iterator usr_it = coal_bs_user_alloc.at(b).begin();
							 usr_it != usr_end_it;
							 ++usr_it)
						{
							const std::size_t u = *usr_it;
							const std::size_t uc = macro_user_to_cats()(u);
							const RealT qos_data_rate = scen.user_downlink_data_rates[uc];
							const RealT revenue = scen.provider_user_revenues[p][uc];
							const RealT sinr = macro_user_bs_sinrs()(u,b);
							//const std::size_t num_subchannels = std::floor((bw/subchannel_bandwidth)*(scen.user_downlink_rates[uc]/tot_qos_data_rate));
							//const RealT data_rate = num_subchannels*subchannel_bandwidth*::log2(1+macro_user_bs_sinrs()(u,b));
							const RealT data_rate = bw*::log2(1+sinr)*(qos_data_rate/tot_qos_data_rate);

							//coal_bs_user_data_rates[b][u] *= bw/(subchannel_bandwidth*tot_qos_data_rate);
							coal_revenue += revenue;
							coal_cost += (1-std::min(data_rate/qos_data_rate, 1.0))*revenue;
						}

						bs_cost += scen.bs_load_powers[bc]*energy_cost*bs_nu;

						part_bs_num_served_users_acc[b](bs_nu);
					}
					else
					{
						part_bs_num_served_users_acc[b](0); // Just to avoid producing a NaN...
					}

					coal_cost += bs_cost;

					if (coal_bss.size() > 1)
					{
						coal_cost += scen.provider_coalition_costs[p];
					}
				}

				if (part_profits[pix].count(cid) == 0)
				{
					part_profits[pix][cid] = 0;
				}
				part_profits[pix][cid] += (coal_revenue - coal_cost)*micro_deltat;
			}
		}

		for (std::size_t b = 0; b < nb; ++b)
		{
			//bs_num_served_users_acc[b](boost::accumulators::mean(part_bs_num_served_users_acc[b]));
			bs_num_served_users[b] += boost::accumulators::mean(part_bs_num_served_users_acc[b])*micro_deltat;
		}
	}

	std::vector< boost::accumulators::accumulator_set< RealT, boost::accumulators::stats<boost::accumulators::tag::mean> > > profit_accs(nb);

	for (std::size_t pix = 0; pix < nparts; ++pix)
	{
		const bs::partition_info<RealT> best_partition = macro_formed_partitions.best_partitions[pix];

		const partition_iterator part_beg_it = best_partition.coalitions.begin();
		const partition_iterator part_end_it = best_partition.coalitions.end();

		for (partition_iterator part_it = part_beg_it;
			 part_it != part_end_it;
			 ++part_it)
		{
			const gtpack::cid_type cid = *part_it;
			const RealT cid_value = macro_formed_partitions.coalitions.at(cid).value;
			const RealT cid_real_value = part_profits[pix].at(cid);

			const coalition_iterator coal_end_it = macro_formed_partitions.coalitions.at(cid).payoffs.end();
			for (coalition_iterator coal_it = macro_formed_partitions.coalitions.at(cid).payoffs.begin();
				 coal_it != coal_end_it;
				 ++coal_it)
			{
				const std::size_t b = coal_it->first;
				const RealT payoff = coal_it->second;

				profit_accs[b](cid_real_value*payoff/cid_value);
			}
		}
	}

	std::vector<RealT> profits(nb, 0);

	//bs_num_served_users.resize(nb, 0);

	for (std::size_t b = 0; b < nb; ++b)
	{
		profits[b] = boost::accumulators::mean(profit_accs[b]);
		//bs_num_served_users[b] = boost::accumulators::mean(bs_num_served_users_acc[b]);
	}

	return profits;
}

template <typename TrafficFuncT, typename RealT, typename RNGT>
void peak_every_x_hours_scenario(RealT time_step,
								 scenario<RealT> const& scen,
								 options<RealT> const& opts,
								 RNGT& rng)
{
	std::size_t tot_nb = 0; // Total number of BSs
	ublas::vector<std::size_t> prov_num_bss(scen.num_providers, 0); // Total number of BSs per provider
	ublas::vector<RealT> bs_heights(scen.num_bs_types, 0); // Height of each BS category. bs_heights[k] = x means that the class k BS is x meters tall
	ublas::vector<RealT> bs_freqs(scen.num_bs_types, 0); // Operating frequency of each BS category in MHz
//	ublas::vector<RealT> bs_caps(scen.num_bs_types, 0); // Max channel capacity of each BS category in Mbit/sec
	ublas::vector<RealT> bs_bws(scen.num_bs_types, 0); // Bandwidth of each BS category in MHz
//	ublas::vector<RealT> bs_num_subcarriers(scen.num_bs_types, 0); // Max number of subcarriers supported by each BS category.
	ublas::vector<RealT> bs_tx_powers(scen.num_bs_types, 0); // Transmission power each BS category. bs_tx_powers[k] = x means that the class k BS supports transmission power of x W
	ublas::vector<RealT> bs_const_powers(scen.num_bs_types, 0); // constant BS power consumption by category. bs_const_powers[k] = x means that the class k BS has a constant power consumption of x W
	ublas::vector<RealT> bs_load_powers(scen.num_bs_types, 0); // load-dependent BS power consumption by category. bs_load_powers[k] = x means that the class k BS has a load-depedented power consumption of x W
	ublas::vector<RealT> prov_energy_costs(scen.num_providers, 0); // Electricity cost by provider. prov_energy_costs[p] = x means that the provider p pays for electricity x $/Wh
//	ublas::matrix<RealT> prov_bs_amort_costs(scen.num_providers, scen.num_bs_types, 0); // Amortizing cost by provider and BS type. prov_bs_amort_costs[p,k] = x means that the provider p pays for BS type k a cost of x $/Wh
//	ublas::matrix<RealT> prov_bs_amort_times(scen.num_providers, scen.num_bs_types, 0); // Residual amortizing time by provider and BS type. prov_bs_amort_times[p,k] = x means that to amortize the cost of a class k BS the provider p takes x hours
	ublas::matrix<RealT> prov_usr_min_dl_rates(scen.num_providers, scen.num_user_classes, 0); // Guaranteed downlink data rate by each provider and user category. prov_usr_min_dl_rates[p,k] = x means that the provider p guarantees to a class k user a downlink data rate of x bit/sec
	ublas::vector<bs::bsid_type> B; // The set of all BSs
	ublas::vector<std::size_t> bs_to_provs; // BS to provider s.t. if bs_to_provs[i]=p means that the i-th BS belongs to provider p
	ublas::vector<std::size_t> bs_to_cats; // BS to category s.t. if bs_to_cats[i]=k means that the i-th BS belongs to provider p
	ublas::matrix<RealT> prov_usr_revenues(scen.num_providers, scen.num_user_classes, 0); // Revenue rate each provider obtain by each user category. prov_usr_revenues[p,k] = x means that the provider p obtains from a class k user a revenue of x $/hour
	//std::vector< std::vector< boost::shared_ptr< daily_traffic_pattern<RealT> > > > bs_traffic_patterns(scen.num_providers); // BS traffic curve (per provider and BS category)
	std::vector< std::vector< boost::shared_ptr< TrafficFuncT > > > bs_traffic_patterns(scen.num_providers); // BS traffic curve (per provider and BS category)
	ublas::vector<RealT> usr_heights(scen.num_user_classes, 0); // Height of each user category. usr_heights[k] = x means that the class k user is x meters tall
	ublas::vector<RealT> usr_qos_dl_rates(scen.num_user_classes, 0); // Required downlink data rate by each user category. usr_qos_dl_rates[k] = x means that the class k user requires a downlink data rate of x bit/sec
	ublas::vector<RealT> usr_therm_noises(scen.num_user_classes, 0); // Thermal noise. usr_therm_noises[k] = x means that the class k user has a thermal noise of x W
	std::map< std::size_t,std::pair<std::size_t,std::size_t> > prov_to_bss; // Maps a provider to a pair of index pointing to the first and last entries of BS vector B owned by this provider

	// Uniform random variate generators for randomly positioning BSs and users inside the area of interest
	boost::random::uniform_real_distribution<RealT> rvg_x_pos(0, scen.topology_area_size.first); // RVG for the x-coordinate
	boost::random::uniform_real_distribution<RealT> rvg_y_pos(0, scen.topology_area_size.second); // RVG for the y-coordinate
	// Normal random variate generators for randomly positioning BSs and users inside the area of interest
	//boost::random::normal_distribution<RealT> rvg_x_pos(scen.topology_area_size.first/2.0, scen.topology_area_size.first/5.0); // RVG for the x-coordinate
	//boost::random::normal_distribution<RealT> rvg_y_pos(scen.topology_area_size.second/2.0, scen.topology_area_size.second/5.0); // RVG for the y-coordinate

	for (std::size_t bc = 0; bc < scen.num_bs_types; ++bc)
	{
		bs_heights(bc) = scen.bs_heights[bc];
		bs_freqs(bc) = scen.bs_carrier_frequencies[bc];
//		bs_caps(bc) = scen.bs_max_channel_capacities[bc];
		bs_bws(bc) = scen.bs_bandwidths[bc];
//		bs_num_subcarriers(bc) = scen.bs_num_subcarriers[bc];
		bs_tx_powers(bc) = scen.bs_transmission_powers[bc];
		bs_const_powers(bc) = scen.bs_idle_powers[bc];
		bs_load_powers(bc) = scen.bs_load_powers[bc];
	}

	for (std::size_t uc = 0; uc < scen.num_user_classes; ++uc)
	{
		usr_qos_dl_rates(uc) = scen.user_downlink_data_rates[uc];
		usr_therm_noises(uc) = scen.user_thermal_noises[uc];
		usr_heights(uc) = scen.user_heights[uc];
	}

	for (std::size_t p = 0; p < scen.num_providers; ++p)
	{
		prov_energy_costs(p) = scen.provider_electricity_costs[p];

		for (std::size_t uc = 0; uc < scen.num_user_classes; ++uc)
		{
			prov_usr_min_dl_rates(p,uc) = scen.provider_user_min_downlink_data_rates[p][uc];
			prov_usr_revenues(p,uc) = scen.provider_user_revenues[p][uc];
		}

		bs_traffic_patterns[p].resize(scen.num_bs_types);
		for (std::size_t bc = 0; bc < scen.num_bs_types; ++bc)
		{
			const std::size_t nb = scen.provider_num_bss[p][bc];

//			prov_bs_amort_costs(p,bc) = scen.provider_bs_amortizing_costs[p][bc];
//			prov_bs_amort_times(p,bc) = scen.provider_bs_residual_amortizing_times[p][bc];

			if (nb > 0)
			{
				const std::size_t old_tot_nb = tot_nb;

				prov_num_bss(p) += nb;
				tot_nb += nb;

				bs_to_cats.resize(tot_nb, true);
				ublas::subrange(bs_to_cats, old_tot_nb, tot_nb) = ublas::scalar_vector<std::size_t>(nb, bc);
				bs_to_provs.resize(tot_nb, true);
				ublas::subrange(bs_to_provs, old_tot_nb, tot_nb) = ublas::scalar_vector<std::size_t>(nb, p);
				prov_to_bss[p] = std::make_pair(old_tot_nb, tot_nb);
				B.resize(tot_nb, true);
				ublas::subrange(B, old_tot_nb, tot_nb) = ublasx::seq(old_tot_nb, nb);

				// Build the BS traffic curve (the same for each BS, but possible different for each provider)
				std::vector<RealT> bs_traffic_data_x(scen.provider_bs_traffic_data[p][bc].size(), 0);
				std::vector<RealT> bs_traffic_data_y(scen.provider_bs_traffic_data[p][bc].size(), 0);
				for (std::size_t i = 0; i < scen.provider_bs_traffic_data[p][bc].size(); ++i)
				{
					bs_traffic_data_x[i] = scen.provider_bs_traffic_data[p][bc][i].first;
					bs_traffic_data_y[i] = scen.provider_bs_traffic_data[p][bc][i].second;
				}
				boost::shared_ptr< math::curvefit::base_1d_interpolator<RealT> > p_interp;
				switch (scen.provider_bs_traffic_curves[p][bc])
				{
					case constant_bs_traffic_curve:
						p_interp = boost::make_shared< math::curvefit::constant_interpolator<RealT> >(bs_traffic_data_x.begin(),
																										   bs_traffic_data_x.end(),
																										   bs_traffic_data_y.begin(),
																										   bs_traffic_data_y.end());
						break;
					case cubic_spline_bs_traffic_curve:
						p_interp = boost::make_shared< math::curvefit::cubic_spline_interpolator<RealT> >(bs_traffic_data_x.begin(),
																											   bs_traffic_data_x.end(),
																											   bs_traffic_data_y.begin(),
																											   bs_traffic_data_y.end(),
																											   math::curvefit::periodic_spline_boundary_condition);
						break;
					case linear_bs_traffic_curve:
						p_interp = boost::make_shared< math::curvefit::linear_interpolator<RealT> >(bs_traffic_data_x.begin(),
																										 bs_traffic_data_x.end(),
																										 bs_traffic_data_y.begin(),
																										 bs_traffic_data_y.end());
						break;
				}
				//bs_traffic_patterns[p][bc] = boost::make_shared< daily_traffic_pattern<RealT> >(p_interp);
				bs_traffic_patterns[p][bc] = boost::make_shared<TrafficFuncT>(p_interp);
			}
			else
			{
				prov_to_bss[p] = std::make_pair(tot_nb, tot_nb);
			}
		}
	}

	// Compute BS topology info
	std::vector< std::pair<RealT,RealT> > bs_positions;
	//bs_positions = make_bs_topology(tot_nb, scen.topology_area_size, random_bs_topology_layout, rng);
	//bs_positions = make_bs_topology(tot_nb, scen.topology_area_size, square_bs_topology_layout, rng);
	//bs_positions = make_bs_topology(tot_nb, scen.topology_area_size, diamond_bs_topology_layout, rng/*, 1.0/3.0*/);
	//bs_positions = make_bs_topology(tot_nb, scen.topology_area_size, center_bs_topology_layout, rng);
	bs_positions = make_bs_topology(tot_nb, scen.topology_area_size, scen.topology_bs_layout, rng, 0.5);

	// Open the output data file and write the header
	const char field_quote_ch = '"';
	const char field_sep_ch = ',';

	std::ofstream stats_dat_ofs;
	std::ofstream trace_dat_ofs;

	if (!opts.output_stats_data_file.empty())
	{
		stats_dat_ofs.open(opts.output_stats_data_file.c_str());

		DCS_ASSERT(stats_dat_ofs, DCS_EXCEPTION_THROW(std::runtime_error, "Unable to open output stats data file"));

		stats_dat_ofs << field_quote_ch  << "Start Time" << field_quote_ch << field_sep_ch << field_quote_ch << "Delta t" << field_quote_ch;
		for (std::size_t b = 0; b < tot_nb; ++b)
		{
			stats_dat_ofs	<< field_sep_ch << field_quote_ch << "BS " << b << " - Coalition Profit" << field_quote_ch
							<< field_sep_ch << field_quote_ch << "BS " << b << " - Alone Profit" << field_quote_ch
							<< field_sep_ch << field_quote_ch << "BS " << b << " - Coalition Profit vs. Alone Profit" << field_quote_ch
							<< field_sep_ch << field_quote_ch << "BS " << b << " - # Powered On - Mean" << field_quote_ch
							<< field_sep_ch << field_quote_ch << "BS " << b << " - # Powered On - SD" << field_quote_ch
							<< field_sep_ch << field_quote_ch << "BS " << b << " - # Powered On vs. Delta t - Mean" << field_quote_ch
							<< field_sep_ch << field_quote_ch << "BS " << b << " - # Own Users - Mean" << field_quote_ch
							<< field_sep_ch << field_quote_ch << "BS " << b << " - # Own Users - SD" << field_quote_ch
							<< field_sep_ch << field_quote_ch << "BS " << b << " - # Served Users - Mean" << field_quote_ch
							<< field_sep_ch << field_quote_ch << "BS " << b << " - # Served Users - SD" << field_quote_ch
							<< field_sep_ch << field_quote_ch << "BS " << b << " - # Served Users vs. # Own Users - Mean" << field_quote_ch
							<< field_sep_ch << field_quote_ch << "BS " << b << " - Coalition Load - Mean" << field_quote_ch
							<< field_sep_ch << field_quote_ch << "BS " << b << " - Coalition Load - SD" << field_quote_ch
							<< field_sep_ch << field_quote_ch << "BS " << b << " - Alone Load - Mean" << field_quote_ch
							<< field_sep_ch << field_quote_ch << "BS " << b << " - Alone Load - SD" << field_quote_ch
							<< field_sep_ch << field_quote_ch << "BS " << b << " - Coalition Load vs. Alone Load - Mean" << field_quote_ch;
		}
		stats_dat_ofs << std::endl;
	}

	if (!opts.output_trace_data_file.empty())
	{
		trace_dat_ofs.open(opts.output_trace_data_file.c_str());

		DCS_ASSERT(trace_dat_ofs, DCS_EXCEPTION_THROW(std::runtime_error, "Unable to open output trace data file"));

		trace_dat_ofs	<< field_quote_ch  << "Start Time" << field_quote_ch
						<< field_sep_ch << field_quote_ch << "Delta t" << field_quote_ch
						<< field_sep_ch << field_quote_ch << "Coalition Structure" << field_quote_ch;
		for (std::size_t b = 0; b < tot_nb; ++b)
		{
			trace_dat_ofs	<< field_sep_ch << field_quote_ch << "BS " << b << " - Alone Profit" << field_quote_ch
							<< field_sep_ch << field_quote_ch << "BS " << b << " - Coalition Profit" << field_quote_ch
							<< field_sep_ch << field_quote_ch << "BS " << b << " - # Powered On" << field_quote_ch
							<< field_sep_ch << field_quote_ch << "BS " << b << " - # Own Users" << field_quote_ch
							<< field_sep_ch << field_quote_ch << "BS " << b << " - # Served Users" << field_quote_ch;
		}
		trace_dat_ofs << std::endl;
	}

	//std::vector< boost::shared_ptr< dcs::bs::ci_mean_estimator<RealT> > > ci_ap_stats(scen.num_providers); // Alone profits
	std::vector< boost::shared_ptr< dcs::bs::ci_mean_estimator<RealT> > > ci_fcp_stats(scen.num_providers); // Coalition profits for coalition formation
	std::vector< boost::shared_ptr< dcs::bs::ci_mean_estimator<RealT> > > ci_rcp_stats(scen.num_providers); // Estimation of real coalition profits
	//std::vector< boost::shared_ptr< dcs::bs::ci_mean_estimator<RealT> > > ci_rp_stats(scen.num_providers); // Relative profit increments
	//std::vector< boost::shared_ptr< dcs::bs::ci_mean_estimator<RealT> > > ci_on_stats(tot_nb); // Number of times a BS is powered on
	//std::vector< boost::shared_ptr< dcs::bs::ci_mean_estimator<RealT> > > ci_xl_stats(tot_nb); // BS load increments
	std::vector< boost::shared_ptr< dcs::bs::ci_mean_estimator<RealT> > > ci_anu_stats(tot_nb); // Number of served used when alone per BS
	std::vector< boost::shared_ptr< dcs::bs::ci_mean_estimator<RealT> > > ci_cnu_stats(tot_nb); // Number of served used when in coalition per BS
	std::vector< boost::shared_ptr< dcs::bs::ci_mean_estimator<RealT> > > ci_cno_stats(tot_nb); // Number of times a BS is powered on when in coalition per BS
	std::vector< boost::shared_ptr< dcs::bs::ci_mean_estimator<RealT> > > ci_rcnu_stats(tot_nb); // Real number of served used when in coalition per BS

	// Initialize confidence interval variables

	for (std::size_t p = 0; p < scen.num_providers; ++p)
	{
		std::ostringstream oss;

//		ci_rp_stats[p] = boost::make_shared< dcs::bs::ci_mean_estimator<RealT> >(opts.ci_level, opts.ci_rel_precision);
//		oss.str("");
//		oss << "RP_{" << p << "}";
//		ci_rp_stats[p]->name(oss.str());

//		ci_ap_stats[p] = boost::make_shared< dcs::bs::ci_mean_estimator<RealT> >(opts.ci_level, opts.ci_rel_precision);
//		oss.str("");
//		oss << "AP_{" << p << "}";
//		ci_ap_stats[p]->name(oss.str());

		ci_rcp_stats[p] = boost::make_shared< dcs::bs::ci_mean_estimator<RealT> >(opts.ci_level, opts.ci_rel_precision);
		oss.str("");
		oss << "RCP_{" << p << "}";
		ci_rcp_stats[p]->name(oss.str());

		ci_fcp_stats[p] = boost::make_shared< dcs::bs::ci_mean_estimator<RealT> >(opts.ci_level, opts.ci_rel_precision);
		oss.str("");
		oss << "FCP_{" << p << "}";
		ci_fcp_stats[p]->name(oss.str());
	}
	for (std::size_t b = 0; b < tot_nb; ++b)
	{
		std::ostringstream oss;

//		ci_on_stats[b] = boost::make_shared< dcs::bs::ci_mean_estimator<RealT> >(opts.ci_level, std::numeric_limits<RealT>::infinity());
//		oss.str("");
//		oss << "ON_{" << b << "}";
//		ci_on_stats[b]->name(oss.str());

//		ci_xl_stats[b] = boost::make_shared< dcs::bs::ci_mean_estimator<RealT> >(opts.ci_level, std::numeric_limits<RealT>::infinity());
//		oss.str("");
//		oss << "XL_{" << b << "}";
//		ci_xl_stats[b]->name(oss.str());

		ci_anu_stats[b] = boost::make_shared< dcs::bs::ci_mean_estimator<RealT> >(opts.ci_level, std::numeric_limits<RealT>::infinity());
		oss.str("");
		oss << "ANU_{" << b << "}";
		ci_anu_stats[b]->name(oss.str());

		ci_cnu_stats[b] = boost::make_shared< dcs::bs::ci_mean_estimator<RealT> >(opts.ci_level, std::numeric_limits<RealT>::infinity());
		oss.str("");
		oss << "CNU_{" << b << "}";
		ci_cnu_stats[b]->name(oss.str());

		ci_cno_stats[b] = boost::make_shared< dcs::bs::ci_mean_estimator<RealT> >(opts.ci_level, std::numeric_limits<RealT>::infinity());
		oss.str("");
		oss << "CNO_{" << b << "}";
		ci_cno_stats[b]->name(oss.str());

		ci_rcnu_stats[b] = boost::make_shared< dcs::bs::ci_mean_estimator<RealT> >(opts.ci_level, std::numeric_limits<RealT>::infinity());
		oss.str("");
		oss << "RCNU_{" << b << "}";
		ci_rcnu_stats[b]->name(oss.str());
	}

#ifdef DCS_BS_GT_ALT_USER_GENERATION_FOR_DELTAT_GT_1
	ublas::vector<std::size_t> time_slot_tot_nus(opts.time_span, 0);
	if (time_step > 1)
	{
		//ublas::matrix<std::size_t> time_slot_prov_max_num_usrs(opts.time_span, scen.num_providers, 0);
		for (std::size_t time_slot = 0; time_slot < opts.time_span; ++time_slot)
		{
			const RealT time_low = time_slot;
			const RealT time_up = std::min(time_low+1, opts.time_span);

			// Discretize the current temporal slot

			for (std::size_t p = 0; p < scen.num_providers; ++p)
			{
				for (std::size_t bc = 0; bc < scen.num_bs_types; ++bc)
				{
					// Find peak user load and discretize
					//math::optim::nelder_mead_simplex_optimizer<RealT> optim;
					//math::optim::powell_cobyla_optimizer<RealT> optim;
					math::optim::jones_direct_optimizer<RealT> glob_optim;
					glob_optim.lower_bound(std::vector<RealT>(1,time_low));
					glob_optim.upper_bound(std::vector<RealT>(1,time_up));
					glob_optim.x0(std::vector<RealT>(1,(time_low+time_up)/2.0));
					glob_optim.fx_relative_tolerance(1e-6);
					glob_optim.x_relative_tolerance(1e-2); // Needed, otherwise the DIRECT method gets stuck
					math::optim::optimization_result<RealT> optim_info;
					optim_info = math::optim::maximize<RealT>(glob_optim, math::optim::make_multivariable_function(*(bs_traffic_patterns[p][bc])));
					if (!optim_info.converged)
					{
						std::ostringstream oss;
						oss << "Unable to find peak user load for provider " << p << ", BS Category " << bc << " and time interval [" << time_low << "," << time_up << ")";
						dcs::log_error(DCS_LOGGING_AT, oss.str());
						continue;
					}
					DCS_DEBUG_TRACE("Global Peak user load of Traffic Load Curve for provider " << p << ", BS Category " << bc << " and time interval [" << time_low << "," << time_up << "): (" << optim_info.xopt[0] << "," << optim_info.fopt << ")");//XXX
					// Refine the global optimum with a local optimizer
					math::optim::powell_cobyla_optimizer<RealT> loc_optim;
					loc_optim.lower_bound(std::vector<RealT>(1,time_low));
					loc_optim.upper_bound(std::vector<RealT>(1,time_up));
					loc_optim.x0(optim_info.xopt);
					loc_optim.fx_relative_tolerance(1e-6);
					optim_info = math::optim::maximize<RealT>(loc_optim, math::optim::make_multivariable_function(*(bs_traffic_patterns[p][bc])));
					if (!optim_info.converged)
					{
						std::ostringstream oss;
						oss << "Unable to find peak user load for provider " << p << ", BS Category " << bc << " and time interval [" << time_low << "," << time_up << ")";
						dcs::log_error(DCS_LOGGING_AT, oss.str());
						continue;
					}

					DCS_DEBUG_TRACE("Refined Peak user load of Traffic Load Curve for provider " << p << ", BS Category " << bc << " and time interval [" << time_low << "," << time_up << "): (" << optim_info.xopt[0] << "," << optim_info.fopt << ")");//XXX

					//DCS_DEBUG_ASSERT( optim_info.fopt >= 0.0 && optim_info.fopt <= 1.0 );
					DCS_DEBUG_ASSERT( optim_info.fopt >= 0.0 );

					const std::size_t bs_nu_peak = std::ceil(scen.provider_bs_max_num_users[p][bc]*optim_info.fopt);
		DCS_DEBUG_TRACE("PROV: " << p << " - BS Category: " << bc << " - BS_NU_MAX: " << scen.provider_bs_max_num_users[p][bc] << " - BS_NU_PEAK: " << bs_nu_peak);///XXX

					time_slot_tot_nus(time_slot) += bs_nu_peak;
				}
			}
		}
		DCS_DEBUG_TRACE("ALL PEAKS: " << time_slot_tot_nus);//XXX
	}
#endif // DCS_BS_GT_ALT_USER_GENERATION_FOR_DELTAT_GT_1

    const std::size_t num_time_slots = ::std::ceil(opts.time_span/time_step);

	std::size_t iter = 0;

//	while (!util::check_stats(ci_ap_stats.begin(), ci_ap_stats.end())
//		   || !util::check_stats(ci_cp_stats.begin(), ci_cp_stats.end())
//		   || !util::check_stats(ci_anu_stats.begin(), ci_anu_stats.end())
//		   || !util::check_stats(ci_cnu_stats.begin(), ci_cnu_stats.end())
//		   || !util::check_stats(ci_cno_stats.begin(), ci_cno_stats.end()))
////	while (!util::check_stats(ci_rp_stats.begin(), ci_rp_stats.end())
////		   || !util::check_stats(ci_on_stats.begin(), ci_on_stats.end())
////		   || !util::check_stats(ci_xl_stats.begin(), ci_xl_stats.end()))
	while (!util::check_stats(ci_rcp_stats.begin(), ci_rcp_stats.end())
		   || !util::check_stats(ci_fcp_stats.begin(), ci_fcp_stats.end())
		   || !util::check_stats(ci_anu_stats.begin(), ci_anu_stats.end())
		   || !util::check_stats(ci_cnu_stats.begin(), ci_cnu_stats.end())
		   || !util::check_stats(ci_cno_stats.begin(), ci_cno_stats.end()))
	{
		++iter;

		DCS_DEBUG_TRACE("Iteration #" << iter << "...");//XXX

		ublas::vector<RealT> prov_tot_alone_profits(tot_nb, 0);
		ublas::vector<RealT> prov_tot_coal_profits(tot_nb, 0);
		ublas::vector<RealT> prov_tot_real_coal_profits(tot_nb, 0);
		ublas::vector<RealT> bs_tot_real_num_served_users(tot_nb, 0);
		std::map< bs::bsid_type,bs_info<RealT> > bs_tot_stats;

		//bs::coalition_formation_info<RealT> formed_coalitions;

		// Discretize the temporal interval into temporal slots

		//for (RealT time_low = 0; time_low < opts.time_span; time_low += time_step)
		for (std::size_t time_slot = 0; time_slot < num_time_slots; ++time_slot)
		{
			const RealT time_low = time_slot*time_step;
			const RealT time_up = std::min(time_low+time_step, opts.time_span);
			const RealT deltat = time_up-time_low; // The real length of this time interval

			DCS_DEBUG_TRACE("Time interval [" << time_low << "," << time_up << ") -- Slot: " << time_slot);//XXX

			ublas::vector<std::size_t> prov_max_num_usrs(scen.num_providers, 0); // Number of peak users per provider
			std::size_t tot_nu = 0; // Total number of users
			std::vector<RealT> pc_weights(scen.num_providers, 0);
			std::vector<RealT> pc_alt_weights(scen.num_providers, 0);
			for (std::size_t p = 0; p < scen.num_providers; ++p)
			{
				// Find weights to apply to different energy costs
				RealT alt_int_len = math::float_traits<RealT>::max(math::float_traits<RealT>::min(time_up, scen.provider_alt_electricity_time_ranges[p].second)-math::float_traits<RealT>::max(time_low, scen.provider_alt_electricity_time_ranges[p].first), 0.0);

				pc_alt_weights[p] = alt_int_len/(time_up-time_low);
				pc_weights[p] = 1-pc_alt_weights[p];

				DCS_DEBUG_TRACE("PROVIDER " << p << " - ELECTRICITY COST: alternative coverage: " << alt_int_len << " -- normal weight: " << pc_weights[p] << " -- alternative weight: " << pc_alt_weights[p]);//XXX

				for (std::size_t bc = 0; bc < scen.num_bs_types; ++bc)
				{
					std::size_t bs_nu_peak = 0;

					if (opts.traffic_load_discretization == max_traffic_load_discretization
						|| opts.traffic_load_discretization == min_traffic_load_discretization)
					{
						// Find peak user load and discretize
						//math::optim::nelder_mead_simplex_optimizer<RealT> optim;
						//math::optim::powell_cobyla_optimizer<RealT> optim;
						math::optim::jones_direct_optimizer<RealT> glob_optim;
						glob_optim.lower_bound(std::vector<RealT>(1,time_low));
						glob_optim.upper_bound(std::vector<RealT>(1,time_up));
						glob_optim.x0(std::vector<RealT>(1,(time_low+time_up)/2.0));
						glob_optim.fx_relative_tolerance(1e-6);
						glob_optim.x_relative_tolerance(1e-2); // Needed, otherwise the DIRECT method gets stuck
						math::optim::optimization_result<RealT> optim_info;
						if (opts.traffic_load_discretization == max_traffic_load_discretization)
						{
							optim_info = math::optim::maximize<RealT>(glob_optim, math::optim::make_multivariable_function(*(bs_traffic_patterns[p][bc])));
						}
						else
						{
							optim_info = math::optim::minimize<RealT>(glob_optim, math::optim::make_multivariable_function(*(bs_traffic_patterns[p][bc])));
						}
						if (!optim_info.converged)
						{
							std::ostringstream oss;
							oss << "Unable to find peak user load for provider " << p << ", BS Category " << bc << " and time interval [" << time_low << "," << time_up << ")";
							dcs::log_error(DCS_LOGGING_AT, oss.str());
							continue;
						}

						DCS_DEBUG_TRACE("Global Peak user load of Traffic Load Curve for provider " << p << ", BS Category " << bc << " and time interval [" << time_low << "," << time_up << "): (" << optim_info.xopt[0] << "," << optim_info.fopt << ")");//XXX

						// Refine the global optimum with a local optimizer
						math::optim::powell_cobyla_optimizer<RealT> loc_optim;
						loc_optim.lower_bound(std::vector<RealT>(1,time_low));
						loc_optim.upper_bound(std::vector<RealT>(1,time_up));
						loc_optim.x0(optim_info.xopt);
						loc_optim.fx_relative_tolerance(1e-6);
						if (opts.traffic_load_discretization == max_traffic_load_discretization)
						{
							optim_info = math::optim::maximize<RealT>(loc_optim, math::optim::make_multivariable_function(*(bs_traffic_patterns[p][bc])));
						}
						else
						{
							optim_info = math::optim::minimize<RealT>(loc_optim, math::optim::make_multivariable_function(*(bs_traffic_patterns[p][bc])));
						}
						if (!optim_info.converged)
						{
							std::ostringstream oss;
							oss << "Unable to find peak user load for provider " << p << ", BS Category " << bc << " and time interval [" << time_low << "," << time_up << ")";
							dcs::log_error(DCS_LOGGING_AT, oss.str());
							continue;
						}

						DCS_DEBUG_TRACE("Refined Peak user load of Traffic Load Curve for provider " << p << ", BS Category " << bc << " and time interval [" << time_low << "," << time_up << "): (" << optim_info.xopt[0] << "," << optim_info.fopt << ")");//XXX

						//DCS_DEBUG_ASSERT( optim_info.fopt >= 0.0 && optim_info.fopt <= 1.0 );
						//DCS_DEBUG_ASSERT( optim_info.fopt >= 0.0 );

						if (optim_info.fopt < 0)
						{
							optim_info.fopt = 0;
						}

						bs_nu_peak = std::ceil(scen.provider_bs_max_num_users[p][bc]*optim_info.fopt);
					}
					else
					{
						// Sample the traffic load curve with micro-intervals of 1 minute
						boost::accumulators::accumulator_set< RealT, boost::accumulators::stats<boost::accumulators::tag::median(boost::accumulators::with_p_square_quantile)> > acc;

						const std::size_t num_micro_time_slots = ::std::ceil(deltat/micro_time_step);

						for (std::size_t micro_time_slot = 0; micro_time_slot < num_micro_time_slots; ++micro_time_slot)
						{
							const RealT micro_time_low = time_low+micro_time_slot*micro_time_step;
							const RealT micro_time_up = std::min(micro_time_low+micro_time_step, time_up);

							// Find peak user load and discretize
							math::optim::jones_direct_optimizer<RealT> glob_optim;
							glob_optim.lower_bound(std::vector<RealT>(1,micro_time_low));
							glob_optim.upper_bound(std::vector<RealT>(1,micro_time_up));
							glob_optim.x0(std::vector<RealT>(1,(micro_time_low+micro_time_up)/2.0));
							glob_optim.fx_relative_tolerance(1e-6);
							glob_optim.x_relative_tolerance(1e-2); // Needed, otherwise the DIRECT method gets stuck
							math::optim::optimization_result<RealT> optim_info;
							optim_info = math::optim::maximize<RealT>(glob_optim, math::optim::make_multivariable_function(*(bs_traffic_patterns[p][bc])));
							if (!optim_info.converged)
							{
								std::ostringstream oss;
								oss << "Unable to find peak user load for provider " << p << ", BS Category " << bc << " and time interval [" << micro_time_low << "," << micro_time_up << ")";
								dcs::log_error(DCS_LOGGING_AT, oss.str());
								continue;
							}

							DCS_DEBUG_TRACE("Global Peak user load of Traffic Load Curve for provider " << p << ", BS Category " << bc << " and time interval [" << micro_time_low << "," << micro_time_up << "): (" << optim_info.xopt[0] << "," << optim_info.fopt << ")");//XXX

							// Refine the global optimum with a local optimizer
							math::optim::powell_cobyla_optimizer<RealT> loc_optim;
							loc_optim.lower_bound(std::vector<RealT>(1,micro_time_low));
							loc_optim.upper_bound(std::vector<RealT>(1,micro_time_up));
							loc_optim.x0(optim_info.xopt);
							loc_optim.fx_relative_tolerance(1e-6);
							optim_info = math::optim::maximize<RealT>(loc_optim, math::optim::make_multivariable_function(*(bs_traffic_patterns[p][bc])));
							if (!optim_info.converged)
							{
								std::ostringstream oss;
								oss << "Unable to find peak user load for provider " << p << ", BS Category " << bc << " and time interval [" << micro_time_low << "," << micro_time_up << ")";
								dcs::log_error(DCS_LOGGING_AT, oss.str());
								continue;
							}

							DCS_DEBUG_TRACE("Refined Peak user load of Traffic Load Curve for provider " << p << ", BS Category " << bc << " and time interval [" << micro_time_low << "," << micro_time_up << "): (" << optim_info.xopt[0] << "," << optim_info.fopt << ")");//XXX

							if (optim_info.fopt < 0)
							{
								optim_info.fopt = 0;
							}

							//const std::size_t nu = std::ceil(scen.provider_bs_max_num_users[p][bc]*optim_info.fopt);
							const RealT nu = scen.provider_bs_max_num_users[p][bc]*optim_info.fopt;

							DCS_DEBUG_TRACE("PROV: " << p << " - BS Category: " << bc << " - BS_NU_MAX: " << scen.provider_bs_max_num_users[p][bc] << " - BS_NU_PEAK: " << nu);///XXX

							acc(nu);
						}

						bs_nu_peak = std::ceil(boost::accumulators::median(acc));
					}

					DCS_DEBUG_TRACE("PROV: " << p << " - BS Category: " << bc << " - BS_NU_MAX: " << scen.provider_bs_max_num_users[p][bc] << " - BS_NU_PEAK: " << bs_nu_peak);///XXX

					prov_max_num_usrs(p) += bs_nu_peak;
					tot_nu += bs_nu_peak;
				}
			}

			std::map< std::size_t, std::pair<std::size_t, std::size_t> > prov_usr; // Map a provider to a pair of indices in prov_max_num_usrs: provider => <begin_index,end_index>
			ublas::vector<std::size_t> usr_to_cats(tot_nu, 0); // User to category s.t. if usr_to_cats[i]=k means that the i-th user belongs to category k
			{
				std::size_t u_offs = 0;
				for (std::size_t p = 0; p < scen.num_providers; ++p)
				{
					const std::size_t nu = prov_max_num_usrs(p);

					prov_usr[p] = std::make_pair(u_offs, u_offs+nu); // provider => <start,stop> index to users

					if (nu > 0)
					{
						std::size_t tot_nuc = 0;
						for (std::size_t uc = 0; uc < scen.num_user_classes; ++uc)
						{
							const std::size_t nuc = (uc < (scen.num_user_classes-1)) ? boost::math::round(nu*scen.provider_user_mixes[p][uc]) : nu-tot_nuc; // Handle residual users

							DCS_DEBUG_TRACE("PROVIDER: " << p << " - CLASS: " << uc << " - NU: " << nu << " - N: " << nuc << " (TOT NU: " << tot_nu << " - U_OFFS: " << u_offs << ")");//XXX

							if (nuc > 0)
							{
								ublas::subrange(usr_to_cats, u_offs, u_offs+nuc) = ublas::scalar_vector<std::size_t>(nuc, uc);

								u_offs += nuc;
								tot_nuc += nuc;
							}
						}
					}
				}

				// check: double checks on # users
				DCS_DEBUG_ASSERT(u_offs <= tot_nu); //NOTE: we use '<=' and not '==' since to compute tot_nu we use the ceiling function and hence we can get a greater number
			}

			// Compute user topology info
			ublas::vector< std::pair<RealT,RealT> > usr_positions(tot_nu); // Position of each user inside the area
			ublas::matrix<RealT> usr_bs_dists(tot_nu, tot_nb, 0); // Distance from a user to a BS
			ublas::matrix<RealT> usr_bs_path_losses(tot_nu, tot_nb, 0); // Path loss from a user to a BS
#ifdef DCS_BS_GT_ALT_USER_GENERATION_FOR_DELTAT_GT_1
			if (deltat > 1)
			{
				std::vector< boost::accumulators::accumulator_set< RealT, boost::accumulators::stats<boost::accumulators::tag::mean> > > usr_positions_x_acc(tot_nu);
				std::vector< boost::accumulators::accumulator_set< RealT, boost::accumulators::stats<boost::accumulators::tag::mean> > > usr_positions_y_acc(tot_nu);
				std::vector< std::vector< boost::accumulators::accumulator_set< RealT, boost::accumulators::stats<boost::accumulators::tag::mean> > > > usr_bs_dists_acc(tot_nu);
				std::vector< std::vector< boost::accumulators::accumulator_set< RealT, boost::accumulators::stats<boost::accumulators::tag::mean> > > > usr_bs_path_losses_acc(tot_nu);

				for (std::size_t u = 0; u < tot_nu; ++u)
				{
					usr_bs_dists_acc[u].resize(tot_nb);
					usr_bs_path_losses_acc[u].resize(tot_nb);
				}

				for (std::size_t t = 0; t < deltat; ++t)
				{
					std::size_t time_slot = time_up-deltat+t;
					for (std::size_t u = 0; u < time_slot_tot_nus(time_slot); ++u)
					{
						const std::size_t uc = usr_to_cats(u);

						const RealT x_pos = rvg_x_pos(rng);
						const RealT y_pos = rvg_y_pos(rng);

						usr_positions_x_acc[u](x_pos);
						usr_positions_y_acc[u](y_pos);

						for (std::size_t b = 0; b < tot_nb; ++b)
						{
							const std::size_t bc = bs_to_cats(b);

							const RealT d = detail::util::euclidean_distance(x_pos, y_pos, bs_positions[b].first, bs_positions[b].second);
							//const RealT pl = dcs::bs::distance_based_path_loss_model(default_path_loss_exponent, d*1e-3);
							const RealT pl = dcs::bs::cost231_hata_path_loss_model(bs_freqs(bc), bs_heights(bc), usr_heights(uc), d*1e-3, dcs::bs::urban_large_radio_propagation_area);

							usr_bs_dists_acc[u][b](d);
							usr_bs_path_losses_acc[u][b](std::pow(10.0, pl/10.0));
							//DCS_DEBUG_TRACE("t=" << t << " (usr: "<<u<<", bs: "<<b<<"), Position: ("<<x_pos<<", "<<y_pos<<"), Path Loss " << pl);
						}
					}
				}

				for (std::size_t u = 0; u < tot_nu; ++u)
				{
					const RealT x_pos = boost::accumulators::mean(usr_positions_x_acc[u]);
					const RealT y_pos = boost::accumulators::mean(usr_positions_y_acc[u]);

					usr_positions(u) = std::make_pair(x_pos,y_pos);

					for (std::size_t b = 0; b < tot_nb; ++b)
					{
						usr_bs_dists(u,b) = boost::accumulators::mean(usr_bs_dists_acc[u][b]);
						usr_bs_path_losses(u,b) = 10*::std::log10(boost::accumulators::mean(usr_bs_path_losses_acc[u][b]));
					}
				}
			}
			else
#endif // DCS_BS_GT_ALT_USER_GENERATION_FOR_DELTAT_GT_1
			{
				for (std::size_t u = 0; u < tot_nu; ++u)
				{
					const std::size_t uc = usr_to_cats(u);

					usr_positions(u) = std::make_pair(rvg_x_pos(rng), rvg_y_pos(rng));

					for (std::size_t b = 0; b < tot_nb; ++b)
					{
						const std::size_t bc = bs_to_cats(b);

						usr_bs_dists(u,b) = detail::util::euclidean_distance(usr_positions(u).first, usr_positions(u).second, bs_positions[b].first, bs_positions[b].second);
//usr_bs_dists(u,b) = 100;//XXX: Just to try...
						//usr_bs_path_losses(u,b) = dcs::bs::distance_based_path_loss_model(default_path_loss_exponent, usr_bs_dists(u,b)*1e-3);
						usr_bs_path_losses(u,b) = dcs::bs::cost231_hata_path_loss_model(bs_freqs(bc), bs_heights(bc), usr_heights(uc), usr_bs_dists(u,b)*1e-3, dcs::bs::urban_large_radio_propagation_area);
					}
				}
			}

			// Compute SINR
			ublas::matrix<RealT> usr_bs_sinrs(tot_nu, tot_nb, 0); // SINR between each pair of BS and user
			for (std::size_t u = 0; u < tot_nu; ++u)
			{
				const std::size_t uc = usr_to_cats(u);

				for (std::size_t b = 0; b < tot_nb; ++b)
				{
					const std::size_t bc = bs_to_cats(b);

					RealT interf = 0; // Interference from other BSs
					for (std::size_t b2 = 0; b2 < tot_nb; ++b2)
					{
						if (b2 != b)
						{
							const std::size_t bc2 = bs_to_cats(b2);

							interf += bs_tx_powers(bc2)/std::pow(10.0, usr_bs_path_losses(u,b2)/10.0);
						}
					}
					//usr_bs_sinrs(u,b) = bs_tx_powers(bc)/std::pow(10.0, usr_bs_path_losses(u,b)/10.0)/(usr_therm_noises(uc));
					usr_bs_sinrs(u,b) = bs_tx_powers(bc)/(std::pow(10.0, usr_bs_path_losses(u,b)/10.0)*(interf+usr_therm_noises(uc)));
				}
			}

			if (opts.verbosity >= VERBOSITY_MEDIUM)
			{
				std::cout << "-- INPUTS:" << std::endl;
				std::cout << "- Number of Providers = " << scen.num_providers << std::endl;
				std::cout << "- Number of BS Types = " << scen.num_bs_types << std::endl;
				std::cout << "- Number of User Classes = " << scen.num_user_classes << std::endl;
				std::cout << "- BS :: Height (in m) = " << bs_heights << std::endl;
				std::cout << "- BS :: Carrier Frequencies (in MHz) = " << bs_freqs << std::endl;
				std::cout << "- BS :: Bandwidth (in Hz) = " << bs_bws << std::endl;
				//std::cout << "- BS :: Max Download Data Rate (in Mbps)= " << bs_caps << std::endl;
				//std::cout << "- BS :: Max Number of Subcarriers = " << bs_num_subcarriers << std::endl;
				std::cout << "- BS :: Transmission Power (in W) = " << bs_tx_powers << std::endl;
				std::cout << "- BS :: Constant Power Consumption by Category (in W) = " << bs_const_powers << std::endl;
				std::cout << "- BS :: Load-dependent Power Consumption by Category (in W) = " << bs_load_powers << std::endl;
				std::cout << "- BS :: Positions in the Area = [" << bs_positions.size() << "](";
				for (std::size_t b = 0; b < tot_nb; ++b)
				{
					if (b > 0)
					{
						std::cout << ",";
					}
					std::cout << "<" << bs_positions[b].first << "," << bs_positions[b].second << ">";
				}
				std::cout << ")" << std::endl;
				std::cout << "- Provider :: Energy Cost by Provider (in $) = " << prov_energy_costs << std::endl;
				//std::cout << "- Provider :: BS Amortizing Cost by Provider and BS Category (in $) = " << prov_bs_amort_costs << std::endl;
				std::cout << "- Provider :: Peak Number of Users by Provider = " << prov_max_num_usrs << std::endl;
				std::cout << "- Provider :: Min Downlink Data Rate for each User by Provider (in Mbit/sec) = " << prov_usr_min_dl_rates << std::endl;
				std::cout << "- Provider :: Revenue for each User by Provider (in $/user) = " << prov_usr_revenues << std::endl;
				std::cout << "- Provider :: Number of BS by Provider = " << prov_num_bss << std::endl;
				std::cout << "- Provider :: BS Categories by Provider = " << bs_to_cats << std::endl;
				std::cout << "- Provider :: User Categories by Provider = " << usr_to_cats << std::endl;
				std::cout << "- User :: QoS Downlink Throughput by Category (in Mbit/sec) = " << usr_qos_dl_rates << std::endl;
				std::cout << "- User :: Height (in m) = " << usr_heights << std::endl;
				std::cout << "- User :: Thermal Noise (in W) = " << usr_therm_noises << std::endl;
				std::cout << "- User :: Positions in the Area = [" << usr_positions.size() << "](";
				for (std::size_t u = 0; u < tot_nu; ++u)
				{
					if (u > 0)
					{
						std::cout << ",";
					}
					std::cout << "<" << usr_positions(u).first << "," << usr_positions(u).second << ">";
				}
				std::cout << ")" << std::endl;
				std::cout << "- User :: Distances from BSs (in m) = " << usr_bs_dists << std::endl;
				std::cout << "- User :: Path losses from BSs (in dB) = " << usr_bs_path_losses << std::endl;
				std::cout << "- User :: SINRs from BSs = " << usr_bs_sinrs << std::endl;
			}

			ublas::vector<RealT> prov_interval_alone_profits(tot_nb, std::numeric_limits<RealT>::quiet_NaN());

			// Solve the coalition formation problem

			gtpack::cooperative_game<RealT> game(tot_nb, ::boost::make_shared< gtpack::explicit_characteristic_function<RealT> >());

			std::map< gtpack::cid_type, bs::coalition_info<RealT> > visited_coalitions;
			std::map< gtpack::player_type, std::vector< bs::partition_info<RealT> > > best_partitions;
			//bool found_same_struc(false);

			alg::lexicographic_subset subset(tot_nb, false);

			while (subset.has_next())
			{
				typedef typename alg::subset_traits<bs::bsid_type>::element_container element_container;

				DCS_DEBUG_TRACE("--- SUBSET: " << subset);//XXX

				const element_container sub = alg::next_subset(B.begin(), B.end(), subset);

				const gtpack::cid_type cid = gtpack::players_coalition<RealT>::make_id(sub.begin(), sub.end());

				DCS_DEBUG_TRACE("--- COALITION: " << game.coalition(cid) << " (CID=" << cid << ")");//XXX

				const std::size_t coal_nb = sub.size();
				ublas::vector<std::size_t> coal_P;
				ublas::vector<bs::bsid_type> coal_B(coal_nb);
				ublas::vector<std::size_t> coal_U;
				std::size_t coal_np = 0;
				std::size_t coal_nu = 0;
				// Compute the number of users for this coalition
				{
					std::set<std::size_t> coal_prov;
					for (std::size_t ib = 0; ib < coal_nb; ++ib)
					{
						const bs::bsid_type b = sub[ib];
						const std::size_t p = bs_to_provs(b);

						coal_B(ib) = b;

						if (coal_prov.count(p) == 0)
						{
							const std::size_t nu = prov_max_num_usrs(p);

							// check: double check for number of users
							DCS_DEBUG_ASSERT( nu == (prov_usr.at(p).second-prov_usr.at(p).first) );

							coal_prov.insert(p);

							coal_P.resize(coal_np+1, true);
							coal_P(coal_np) = p;
							++coal_np;

							if (nu > 0)
							{
								coal_U.resize(coal_nu+nu, true);
								//ublas::subrange(coal_U, coal_nu, coal_nu+nu) = ublasx::seq(prov_usr.at(p).first, nu);
								//visited_coalitions[cid].usr_to_provs.resize(coal_nu+nu, p);
								for (std::size_t u = 0; u < nu; ++u)
								{
									const std::size_t uix = coal_nu+u;

									visited_coalitions[cid].usr_to_idx[prov_usr.at(p).first+u] = uix;
									coal_U(uix) = prov_usr.at(p).first+u;
								}
								coal_nu += nu;
							}
						}

						visited_coalitions[cid].bsid_to_idx[b] = ib;
					}
				}

				bs::user_assignment_solution<RealT> user_assignment;

				switch (opts.user_assignment)
				{
					case max_profit_user_assignment:
						user_assignment = bs::optimal_max_profit_user_assignment_solver<RealT>()(coal_P,
																								 coal_B,
																								 coal_U,
																								 bs_to_cats,
																								 bs_to_provs,
																								 //bs_heights,
																								 //bs_freqs,
																								 //bs_caps,
																								 bs_bws,
																								 //bs_num_subcarriers,
																								 //bs_tx_powers,
																								 bs_const_powers,
																								 bs_load_powers,
																								 usr_to_cats,
																								 //usr_heights,
																								 usr_qos_dl_rates,
																								 //usr_therm_noises,
																								 usr_bs_sinrs,
																								 prov_energy_costs,
																								 prov_usr_min_dl_rates,
																								 prov_usr_revenues);
						break;
					case min_cost_user_assignment:
						throw std::runtime_error("TODO: Add prov_usr_min_dl_rates in min-profit");
	//					user_assignment = bs::optimal_min_cost_user_assignment_solver<RealT>()(coal_P,
	//																						   coal_B,
	//																						   coal_U,
	//																						   bs_to_cats,
	//																						   bs_to_provs,
	//																						   bs_caps,
	//																						   bs_const_powers,
	//																						   bs_load_powers,
	//																						   usr_to_cats,
	//																						   usr_qos_dl_rates,
	//																						   prov_energy_costs,
	//																						   //TODO:prov_usr_min_dl_rates
	//																						   prov_usr_revenues);
						break;
				}

				visited_coalitions[cid].user_assignment = user_assignment;

				if (user_assignment.solved)
				{
					// check: pedantic check for the number of allocated users
#ifdef DCS_DEBUG
					std::size_t alloc_nu = 0;
					for (std::size_t ib = 0; ib < coal_nb; ++ib)
					{
						for (std::size_t iu = 0; iu < coal_nu; ++iu)
						{
							if (user_assignment.bs_user_allocations(ib,iu))
							{
								++alloc_nu;
							}
						}
					}

					DCS_DEBUG_ASSERT( alloc_nu == coal_nu );
#endif // DCS_DEBUG

					RealT v = 0;

					switch (opts.user_assignment)
					{
						case max_profit_user_assignment:
							v = user_assignment.objective_value;
							break;
						case min_cost_user_assignment:
							{
								RealT coal_R = 0;
								for (std::size_t ib = 0; ib < coal_nb; ++ib)
								{
									for (std::size_t iu = 0; iu < coal_nu; ++iu)
									{
										if (user_assignment.bs_user_allocations(ib,iu))
										{
											coal_R += prov_usr_revenues(bs_to_provs(coal_B(ib)),usr_to_cats(coal_U(iu)));
										}
									}
								}
								v = coal_R - user_assignment.objective_value;
							}
							break;
					}
					// Add coalition formation costs
					if (coal_nb > 1)
					{
						for (std::size_t ib = 0; ib < coal_nb; ++ib)
						{
							const bs::bsid_type b = coal_B(ib);

							v -= scen.provider_coalition_costs[bs_to_provs(b)];
						}
					}
					// Multiply for the interval length
					v *= deltat;
					// Add amortizing costs
					for (std::size_t ib = 0; ib < coal_nb; ++ib)
					{
						const bs::bsid_type b = coal_B(ib);
						const std::size_t p = bs_to_provs(b);
						const std::size_t bc = bs_to_cats(b);
						const RealT prov_bs_amort_costs = scen.provider_bs_amortizing_costs[p][bc];
						const RealT prov_bs_amort_times = std::min(std::max(0.0, scen.provider_bs_residual_amortizing_times[p][bc]-time_low), deltat);

						v -= prov_bs_amort_costs*prov_bs_amort_times;
					}
					game.value(cid, v);
					visited_coalitions[cid].value = v;

					if (coal_nb == 1)
					{
						const bs::bsid_type b = coal_B(0);

						prov_interval_alone_profits(b) = v;
						prov_tot_alone_profits(b) += prov_interval_alone_profits(b);
					}

					DCS_DEBUG_TRACE( "CID: " << cid << " - User assignment objective value: " << user_assignment.objective_value << " - Total coalition watts (kW): " << user_assignment.kwatt << " - Total coalition cost: " << user_assignment.cost << " => v(CID)=" << game.value(cid) );

					gtpack::cooperative_game<RealT> subgame = game.subgame(coal_B.begin(), coal_B.end());
					gtpack::core<RealT> core = gtpack::find_core(subgame);
					if (core.empty())
					{
						DCS_DEBUG_TRACE( "CID: " << cid << " - The core is empty" );

						visited_coalitions[cid].core_empty = true;
						visited_coalitions[cid].payoffs_in_core = false;

						if (subgame.num_players() == tot_nb)
						{
							// This is the Grand coalition

							DCS_DEBUG_TRACE( "CID: " << cid << " - The Grand-Coalition has an empty core" );
						}
					}
					else
					{
						DCS_DEBUG_TRACE( "CID: " << cid << " - The core is not empty" );

						visited_coalitions[cid].core_empty = false;
					}

					if (opts.coalition_value_division != bs::chi_coalition_value_division)
					{
						// Compute the coalition payoffs
						std::map<gtpack::player_type,RealT> coal_payoffs;
						switch (opts.coalition_value_division)
						{
							case bs::banzhaf_coalition_value_division:
								coal_payoffs = gtpack::banzhaf_value(subgame);
								break;
							case bs::normalized_banzhaf_coalition_value_division:
								coal_payoffs = gtpack::norm_banzhaf_value(subgame);
								break;
							case bs::shapley_coalition_value_division:
								coal_payoffs = gtpack::shapley_value(subgame);
								break;
							case bs::chi_coalition_value_division:
								// postponed: we need to compute the Shapley value for the grand-coalition
								break;
						}
#ifdef DCS_DEBUG
						if (coal_payoffs.size() > 0)
						{
							for (std::size_t ib = 0; ib < coal_nb; ++ib)
							{
								const std::size_t b = coal_B[ib];

								DCS_DEBUG_TRACE( "CID: " << cid << " - BS: " << b << " - Coalition payoff: " << coal_payoffs.at(b) );
							}
						}
#endif // DCS_DEBUG
						visited_coalitions[cid].payoffs = coal_payoffs;

						// Check if the payoff vector is in the core *if the core != empty)
						if (!visited_coalitions.at(cid).core_empty)
						{
							if (gtpack::belongs_to_core(game.subgame(coal_B.begin(), coal_B.end()), coal_payoffs.begin(), coal_payoffs.end()))
							{
								DCS_DEBUG_TRACE( "CID: " << cid << " - The Coalition payoff vector belongs to the core" );

								visited_coalitions[cid].payoffs_in_core = true;
							}
							else
							{
								DCS_DEBUG_TRACE( "CID: " << cid << " - The Coaition payoff vector does not belong to the core" );

								visited_coalitions[cid].payoffs_in_core = false;
							}
						}
					}
				}
				else
				{
					DCS_DEBUG_TRACE( "CID: " << cid << " - The user assignment problem is infeasible" );

					visited_coalitions[cid].core_empty = true;
					visited_coalitions[cid].payoffs_in_core = false;

					game.value(cid, -std::numeric_limits<RealT>::min());

					if (game.coalition(cid).num_players() == tot_nb)
					{
						// This is the Grand coalition

						DCS_DEBUG_TRACE( "CID: " << cid << " - The Grand-Coalition has an infeasible solution and thus an empty core" );
					}
				}
			}

			// Compute the coalition payoffs for postponed coalition value divisions
			if (opts.coalition_value_division == bs::chi_coalition_value_division)
			{
				for (typename std::map<gtpack::cid_type,bs::coalition_info<RealT> >::iterator it = visited_coalitions.begin();
					 it != visited_coalitions.end();
					 ++it)
				{
					const gtpack::cid_type cid = it->first;

					std::map<gtpack::player_type,RealT> coal_payoffs;

					// Create a coalition structure $\{S,N \setminus S\}$, where $S$
					// is the current coalition identified by 'cid' and $N$ is the
					// set of all players
					std::vector<gtpack::cid_type> old_coal_struc = game.coalition_structure();
					std::vector<gtpack::cid_type> new_coal_struc;
					std::vector<gtpack::player_type> players = game.players();
					gtpack::cid_type grand_cid = gtpack::players_coalition<RealT>::make_id(players.begin(), players.end());
					gtpack::cid_type other_cid = grand_cid-cid;
					new_coal_struc.push_back(cid);
					if (other_cid > 0)
					{
						new_coal_struc.push_back(other_cid);
					}
					// Set the new coalition structure
					game.coalition_structure(new_coal_struc.begin(), new_coal_struc.end());
					// Compute the payoffs
					coal_payoffs = gtpack::chi_value(game);

					// Restore the old coalition structure
					game.coalition_structure(old_coal_struc.begin(), old_coal_struc.end());

					// Store the computed payoffs
					players = game.coalition(cid).players();
					for (std::size_t i = 0; i < players.size(); ++i)
					{
						const gtpack::player_type pid = players[i];

						DCS_DEBUG_TRACE( "CID: " << cid << " - BS: " << pid << " - Coalition payoff: " << coal_payoffs.at(pid) );

						visited_coalitions[cid].payoffs[pid] = coal_payoffs.at(pid);
					}
				}
			}

			bs::coalition_formation_info<RealT> formed_coalitions;

			formed_coalitions.coalitions = visited_coalitions;
			switch (opts.coalition_formation)
			{
				case bs::nash_stable_coalition_formation:
					formed_coalitions.best_partitions = bs::nash_stable_partition_selector<RealT>()(game, visited_coalitions);
					break;
				case bs::pareto_optimal_coalition_formation:
					formed_coalitions.best_partitions = bs::pareto_optimal_partition_selector<RealT>()(game, visited_coalitions);
					break;
				case bs::social_optimum_coalition_formation:
					formed_coalitions.best_partitions = bs::social_optimum_partition_selector<RealT>()(game, visited_coalitions);
					break;
			}

#ifdef DCS_DEBUG
			DCS_DEBUG_TRACE( "FORMED PARTITIONS: ");
			for (std::size_t i = 0; i < formed_coalitions.best_partitions.size(); ++i)
			{
				const bs::partition_info<RealT> part = formed_coalitions.best_partitions[i];

				typedef typename std::set<gtpack::cid_type>::const_iterator coalition_iterator;
				coalition_iterator coal_end_it(part.coalitions.end());
				DCS_DEBUG_STREAM << "  [";
				for (coalition_iterator coal_it = part.coalitions.begin();
					 coal_it != coal_end_it;
					 ++coal_it)
				{
					const gtpack::cid_type cid = *coal_it;

					DCS_DEBUG_STREAM << cid << ",";
				}
				DCS_DEBUG_STREAM << "]" << std::endl;
			}
#endif // DCS_DEBUG

			// Compute output information

			typedef typename std::set<gtpack::cid_type>::const_iterator partition_iterator;
			typedef typename std::map<bs::bsid_type,RealT>::const_iterator coalition_iterator;

			std::vector< boost::accumulators::accumulator_set<RealT, boost::accumulators::stats<boost::accumulators::tag::max,boost::accumulators::tag::mean,boost::accumulators::tag::variance> > > prov_summary_interval_coal_profits(tot_nb);
			std::map< bs::bsid_type,bs_stats_info<RealT> > bs_summary_interval_stats;

			if (trace_dat_ofs.is_open())
			{
				trace_dat_ofs << time_low << field_sep_ch << deltat;
			}

			const std::size_t np = formed_coalitions.best_partitions.size();
			if (opts.find_all_best_partitions)
			{
				for (std::size_t p = 0; p < np; ++p)
				{
					const bs::partition_info<RealT> best_partition = formed_coalitions.best_partitions[p];

					const std::size_t nc = best_partition.coalitions.size();
					const partition_iterator part_beg_it = best_partition.coalitions.begin();
					const partition_iterator part_end_it = best_partition.coalitions.end();

					ublas::vector<RealT> prov_interval_coal_profits(tot_nb, std::numeric_limits<RealT>::quiet_NaN());
					std::map< bs::bsid_type,bs_info<RealT> > bs_interval_stats;

					for (partition_iterator part_it = part_beg_it;
						 part_it != part_end_it;
						 ++part_it)
					{
						const gtpack::cid_type cid = *part_it;

						// Compute the interval and the total payoffs, and the total payoff errors
						const coalition_iterator coal_end_it= formed_coalitions.coalitions.at(cid).payoffs.end();
						for (coalition_iterator coal_it = formed_coalitions.coalitions.at(cid).payoffs.begin();
							 coal_it != coal_end_it;
							 ++coal_it)
						{
							const bs::bsid_type b = coal_it->first;
							const RealT payoff = coal_it->second;

							// check: valid BS id
							DCS_DEBUG_ASSERT( b < tot_nb );

							prov_interval_coal_profits(b) = payoff;

							prov_summary_interval_coal_profits[b](prov_interval_coal_profits(b));
						}

						// Collect info about BS assignment
						std::vector<gtpack::player_type> players = game.coalition(cid).players();
						for (std::size_t i = 0; i < players.size(); ++i)
						{
							const bs::bsid_type b = players[i];

							// check: valid BS id
							DCS_DEBUG_ASSERT( b < tot_nb );

							if (bs_interval_stats.count(b) == 0)
							{
								bs_interval_stats[b].num_poweron = 0;
							}

							const std::size_t bidx = formed_coalitions.coalitions.at(cid).bsid_to_idx.at(b);

							bool powered_on = formed_coalitions.coalitions.at(cid).user_assignment.bs_power_states(bidx);
							if (powered_on)
							{
								bs_interval_stats[b].num_poweron += 1;
							}

							const std::size_t onu = prov_max_num_usrs(bs_to_provs(b)); // number of own users
							const std::size_t mnu = scen.provider_bs_max_num_users[bs_to_provs(b)][bs_to_cats(b)]; // max number of users
							std::size_t snu = 0; // number of served users
							for (std::size_t j = 0; j < ublasx::num_columns(formed_coalitions.coalitions.at(cid).user_assignment.bs_user_allocations); ++j)
							{
								if (formed_coalitions.coalitions.at(cid).user_assignment.bs_user_allocations(bidx,j))
								{
									++snu;
								}
							}
							bs_interval_stats[b].num_served_users = snu;
							bs_interval_stats[b].num_own_users = onu;
							bs_interval_stats[b].coalition_load = static_cast<RealT>(snu)/mnu;
							bs_interval_stats[b].alone_load = static_cast<RealT>(onu)/mnu;

							bs_summary_interval_stats[b].num_served_users(bs_interval_stats[b].num_served_users);
							bs_summary_interval_stats[b].num_poweron(bs_interval_stats[b].num_poweron);
							bs_summary_interval_stats[b].num_served_users(bs_interval_stats[b].num_served_users);
							bs_summary_interval_stats[b].num_own_users(bs_interval_stats[b].num_own_users);
							bs_summary_interval_stats[b].coalition_load(bs_interval_stats[b].coalition_load);
							bs_summary_interval_stats[b].alone_load(bs_interval_stats[b].alone_load);
						}
					}

					// Output results

					if (opts.verbosity >= VERBOSITY_MEDIUM)
					{
						std::cout << "-- INTERVAL OUTPUTS FOR BEST PARTITION #" << p << ": (" << nc << "]{";
						for (partition_iterator part_it = part_beg_it;
							 part_it != part_end_it;
							 ++part_it)
						{
							const gtpack::cid_type cid = *part_it;

							if (part_it != part_beg_it)
							{
								std::cout << ",";
							}

							const coalition_iterator coal_end_it = formed_coalitions.coalitions.at(cid).payoffs.end();
							const coalition_iterator coal_beg_it = formed_coalitions.coalitions.at(cid).payoffs.begin();
							std::cout << "{";
							for (coalition_iterator coal_it = coal_beg_it;
								 coal_it != coal_end_it;
								 ++coal_it)
							{
								const bs::bsid_type bsid = coal_it->first;

								if (coal_it != coal_beg_it)
								{
									std::cout << ",";
								}

								std::cout << bsid;
							}
							std::cout << "}";
						}
						std::cout << "}" << std::endl;
						std::cout << "- Interval Coalition Profits (in $): [" << nc << "]{";
						for (partition_iterator part_it = part_beg_it;
							 part_it != part_end_it;
							 ++part_it)
						{
							const gtpack::cid_type cid(*part_it);

							if (part_it != part_beg_it)
							{
								std::cout << ",";
							}

							const coalition_iterator coal_end_it = formed_coalitions.coalitions.at(cid).payoffs.end();
							const coalition_iterator coal_beg_it = formed_coalitions.coalitions.at(cid).payoffs.begin();
							std::cout << "{";
							for (coalition_iterator coal_it = coal_beg_it;
								 coal_it != coal_end_it;
								 ++coal_it)
							{
								const bs::bsid_type bsid = coal_it->first;

								if (coal_it != coal_beg_it)
								{
									std::cout << ",";
								}

								std::cout << "(" << bsid << "=>" << prov_interval_coal_profits[bsid] << ")";
							}
							std::cout << "}";
						}
						std::cout << "}" << std::endl;
						std::cout << "- Interval Alone Profits (in $): [" << nc << "]{";
						for (partition_iterator part_it = part_beg_it;
							 part_it != part_end_it;
							 ++part_it)
						{
							const gtpack::cid_type cid = *part_it;

							if (part_it != part_beg_it)
							{
								std::cout << ",";
							}

							const coalition_iterator coal_end_it = formed_coalitions.coalitions.at(cid).payoffs.end();
							const coalition_iterator coal_beg_it = formed_coalitions.coalitions.at(cid).payoffs.begin();
							std::cout << "{";
							for (coalition_iterator coal_it = coal_beg_it;
								 coal_it != coal_end_it;
								 ++coal_it)
							{
								const bs::bsid_type bsid = coal_it->first;

								if (coal_it != coal_beg_it)
								{
									std::cout << ",";
								}

								std::cout << "(" << bsid << "=>" << prov_interval_alone_profits[bsid] << ")";
							}
							std::cout << "}";
						}
						std::cout << "}" << std::endl;
						std::cout << "- Interval BS Stats:" << std::endl;
						for (std::size_t b = 0; b < tot_nb; ++b)
						{
							std::cout << " * BS " << b << std::endl;
							std::cout << "  - # Powered On: " << bs_interval_stats.at(b).num_poweron << std::endl;
							std::cout << "  - # Powered On vs. Interval Length: " << (bs_interval_stats.at(b).num_poweron/deltat) << std::endl;
							std::cout << "  - # Own Users: " << bs_interval_stats.at(b).num_own_users << std::endl;
							std::cout << "  - # Served Users: " << bs_interval_stats.at(b).num_served_users << std::endl;
							std::cout << "  - # Served Users vs. # Own Users: " << util::relative_increment(bs_interval_stats.at(b).num_served_users, bs_interval_stats.at(b).num_own_users) << std::endl;
							std::cout << "  - Expected Coalition Load: " << bs_interval_stats.at(b).coalition_load << std::endl;
							std::cout << "  - Expected Alone Load: " << bs_interval_stats.at(b).alone_load << std::endl;
							//std::cout << "  - Expected Coalition vs. Alone Load: " << (bs_interval_stats.at(b).coalition_load/bs_interval_stats.at(b).alone_load-1) << std::endl;
							std::cout << "  - Expected Coalition vs. Alone Load: " << util::relative_increment(bs_interval_stats.at(b).coalition_load, bs_interval_stats.at(b).alone_load) << std::endl;
						}
						std::cout << "-----------------------------------------------------------------------" << std::endl;
					}

					if (trace_dat_ofs.is_open())
					{
						trace_dat_ofs << field_sep_ch << field_quote_ch << "{";
						for (partition_iterator part_it = part_beg_it;
							 part_it != part_end_it;
							 ++part_it)
						{
							const gtpack::cid_type cid(*part_it);

							if (part_it != part_beg_it)
							{
								trace_dat_ofs << ",";
							}

							coalition_iterator coal_end_it(formed_coalitions.coalitions.at(cid).payoffs.end());
							coalition_iterator coal_beg_it(formed_coalitions.coalitions.at(cid).payoffs.begin());
							trace_dat_ofs << "{";
							for (coalition_iterator coal_it = coal_beg_it;
								 coal_it != coal_end_it;
								 ++coal_it)
							{
								const bs::bsid_type bsid(coal_it->first);

								if (coal_it != coal_beg_it)
								{
									trace_dat_ofs << ",";
								}

								trace_dat_ofs << bsid;
							}
							trace_dat_ofs << "}";
						}
						trace_dat_ofs << "}" << field_quote_ch;

						for (std::size_t b = 0; b < tot_nb; ++b)
						{
							trace_dat_ofs	<< field_sep_ch << prov_interval_alone_profits[b]
											<< field_sep_ch << prov_interval_coal_profits[b]
											<< field_sep_ch << bs_interval_stats.at(b).num_poweron
											<< field_sep_ch << bs_interval_stats.at(b).num_own_users
											<< field_sep_ch << bs_interval_stats.at(b).num_served_users;
						}
					}
				}
			}
			else
			{
				bs::partition_info<RealT> max_best_partition;

				for (std::size_t p = 0; p < np; ++p)
				{
					const bs::partition_info<RealT> best_partition = formed_coalitions.best_partitions[p];

					if (best_partition.value > max_best_partition.value)
					{
						max_best_partition = best_partition;
					}
				}

				const std::size_t nc = max_best_partition.coalitions.size();
				const partition_iterator part_beg_it = max_best_partition.coalitions.begin();
				const partition_iterator part_end_it = max_best_partition.coalitions.end();

				ublas::vector<RealT> prov_interval_coal_profits(tot_nb, std::numeric_limits<RealT>::quiet_NaN());
				std::map< bs::bsid_type,bs_info<RealT> > bs_interval_stats;

				for (partition_iterator part_it = part_beg_it;
					 part_it != part_end_it;
					 ++part_it)
				{
					const gtpack::cid_type cid = *part_it;

					// Compute the interval and the total payoffs, and the total payoff errors
					const coalition_iterator coal_end_it= formed_coalitions.coalitions.at(cid).payoffs.end();
					for (coalition_iterator coal_it = formed_coalitions.coalitions.at(cid).payoffs.begin();
						 coal_it != coal_end_it;
						 ++coal_it)
					{
						const bs::bsid_type b = coal_it->first;
						const RealT payoff = coal_it->second;

						// check: valid BS id
						DCS_DEBUG_ASSERT( b < tot_nb );

						prov_interval_coal_profits(b) = payoff;

						prov_summary_interval_coal_profits[b](prov_interval_coal_profits(b));
					}

					// Collect info about BS assignment
					std::vector<gtpack::player_type> players = game.coalition(cid).players();
					for (std::size_t i = 0; i < players.size(); ++i)
					{
						const bs::bsid_type b = players[i];

						// check: valid BS id
						DCS_DEBUG_ASSERT( b < tot_nb );

						if (bs_interval_stats.count(b) == 0)
						{
							bs_interval_stats[b].num_poweron = 0;
						}

						const std::size_t bidx = formed_coalitions.coalitions.at(cid).bsid_to_idx.at(b);

						bool powered_on = formed_coalitions.coalitions.at(cid).user_assignment.bs_power_states(bidx);
						if (powered_on)
						{
							bs_interval_stats[b].num_poweron += 1;
						}

						const std::size_t onu = prov_max_num_usrs(bs_to_provs(b)); // number of own users
						const std::size_t mnu = scen.provider_bs_max_num_users[bs_to_provs(b)][bs_to_cats(b)]; // max number of users
						std::size_t snu = 0; // number of served users
						for (std::size_t j = 0; j < ublasx::num_columns(formed_coalitions.coalitions.at(cid).user_assignment.bs_user_allocations); ++j)
						{
							if (formed_coalitions.coalitions.at(cid).user_assignment.bs_user_allocations(bidx,j))
							{
								++snu;
							}
						}
						bs_interval_stats[b].num_served_users = snu;
						bs_interval_stats[b].num_own_users = onu;
						bs_interval_stats[b].coalition_load = static_cast<RealT>(snu)/mnu;
						bs_interval_stats[b].alone_load = static_cast<RealT>(onu)/mnu;

						bs_summary_interval_stats[b].num_served_users(bs_interval_stats[b].num_served_users);
						bs_summary_interval_stats[b].num_poweron(bs_interval_stats[b].num_poweron);
						bs_summary_interval_stats[b].num_served_users(bs_interval_stats[b].num_served_users);
						bs_summary_interval_stats[b].num_own_users(bs_interval_stats[b].num_own_users);
						bs_summary_interval_stats[b].coalition_load(bs_interval_stats[b].coalition_load);
						bs_summary_interval_stats[b].alone_load(bs_interval_stats[b].alone_load);
					}
				}

				// Output results

				if (opts.verbosity >= VERBOSITY_MEDIUM)
				{
					std::cout << "-- INTERVAL OUTPUTS FOR BEST PARTITION: (" << nc << "]{";
					for (partition_iterator part_it = part_beg_it;
						 part_it != part_end_it;
						 ++part_it)
					{
						const gtpack::cid_type cid = *part_it;

						if (part_it != part_beg_it)
						{
							std::cout << ",";
						}

						const coalition_iterator coal_end_it = formed_coalitions.coalitions.at(cid).payoffs.end();
						const coalition_iterator coal_beg_it = formed_coalitions.coalitions.at(cid).payoffs.begin();
						std::cout << "{";
						for (coalition_iterator coal_it = coal_beg_it;
							 coal_it != coal_end_it;
							 ++coal_it)
						{
							const bs::bsid_type bsid = coal_it->first;

							if (coal_it != coal_beg_it)
							{
								std::cout << ",";
							}

							std::cout << bsid;
						}
						std::cout << "}";
					}
					std::cout << "}" << std::endl;
					std::cout << "- Interval Coalition Profits (in $): [" << nc << "]{";
					for (partition_iterator part_it = part_beg_it;
						 part_it != part_end_it;
						 ++part_it)
					{
						const gtpack::cid_type cid(*part_it);

						if (part_it != part_beg_it)
						{
							std::cout << ",";
						}

						const coalition_iterator coal_end_it = formed_coalitions.coalitions.at(cid).payoffs.end();
						const coalition_iterator coal_beg_it = formed_coalitions.coalitions.at(cid).payoffs.begin();
						std::cout << "{";
						for (coalition_iterator coal_it = coal_beg_it;
							 coal_it != coal_end_it;
							 ++coal_it)
						{
							const bs::bsid_type bsid = coal_it->first;

							if (coal_it != coal_beg_it)
							{
								std::cout << ",";
							}

							std::cout << "(" << bsid << "=>" << prov_interval_coal_profits[bsid] << ")";
						}
						std::cout << "}";
					}
					std::cout << "}" << std::endl;
					std::cout << "- Interval Alone Profits (in $): [" << nc << "]{";
					for (partition_iterator part_it = part_beg_it;
						 part_it != part_end_it;
						 ++part_it)
					{
						const gtpack::cid_type cid = *part_it;

						if (part_it != part_beg_it)
						{
							std::cout << ",";
						}

						const coalition_iterator coal_end_it = formed_coalitions.coalitions.at(cid).payoffs.end();
						const coalition_iterator coal_beg_it = formed_coalitions.coalitions.at(cid).payoffs.begin();
						std::cout << "{";
						for (coalition_iterator coal_it = coal_beg_it;
							 coal_it != coal_end_it;
							 ++coal_it)
						{
							const bs::bsid_type bsid = coal_it->first;

							if (coal_it != coal_beg_it)
							{
								std::cout << ",";
							}

							std::cout << "(" << bsid << "=>" << prov_interval_alone_profits[bsid] << ")";
						}
						std::cout << "}";
					}
					std::cout << "}" << std::endl;
					std::cout << "- Interval BS Stats:" << std::endl;
					for (std::size_t b = 0; b < tot_nb; ++b)
					{
						std::cout << " * BS " << b << std::endl;
						std::cout << "  - # Powered On: " << bs_interval_stats.at(b).num_poweron << std::endl;
						std::cout << "  - # Powered On vs. Interval Length: " << (bs_interval_stats.at(b).num_poweron/deltat) << std::endl;
						std::cout << "  - # Own Users: " << bs_interval_stats.at(b).num_own_users << std::endl;
						std::cout << "  - # Served Users: " << bs_interval_stats.at(b).num_served_users << std::endl;
						std::cout << "  - # Served Users vs. # Own Users: " << util::relative_increment(bs_interval_stats.at(b).num_served_users, bs_interval_stats.at(b).num_own_users) << std::endl;
						std::cout << "  - Expected Coalition Load: " << bs_interval_stats.at(b).coalition_load << std::endl;
						std::cout << "  - Expected Alone Load: " << bs_interval_stats.at(b).alone_load << std::endl;
						//std::cout << "  - Expected Coalition vs. Alone Load: " << (bs_interval_stats.at(b).coalition_load/bs_interval_stats.at(b).alone_load-1) << std::endl;
						std::cout << "  - Expected Coalition vs. Alone Load: " << util::relative_increment(bs_interval_stats.at(b).coalition_load, bs_interval_stats.at(b).alone_load) << std::endl;
					}
					std::cout << "-----------------------------------------------------------------------" << std::endl;
				}

				if (trace_dat_ofs.is_open())
				{
					trace_dat_ofs << field_sep_ch << field_quote_ch << "{";
					for (partition_iterator part_it = part_beg_it;
						 part_it != part_end_it;
						 ++part_it)
					{
						const gtpack::cid_type cid(*part_it);

						if (part_it != part_beg_it)
						{
							trace_dat_ofs << ",";
						}

						coalition_iterator coal_end_it(formed_coalitions.coalitions.at(cid).payoffs.end());
						coalition_iterator coal_beg_it(formed_coalitions.coalitions.at(cid).payoffs.begin());
						trace_dat_ofs << "{";
						for (coalition_iterator coal_it = coal_beg_it;
							 coal_it != coal_end_it;
							 ++coal_it)
						{
							const bs::bsid_type bsid(coal_it->first);

							if (coal_it != coal_beg_it)
							{
								trace_dat_ofs << ",";
							}

							trace_dat_ofs << bsid;
						}
						trace_dat_ofs << "}";
					}
					trace_dat_ofs << "}" << field_quote_ch;

					for (std::size_t b = 0; b < tot_nb; ++b)
					{
						trace_dat_ofs	<< field_sep_ch << prov_interval_alone_profits[b]
										<< field_sep_ch << prov_interval_coal_profits[b]
										<< field_sep_ch << bs_interval_stats.at(b).num_poweron
										<< field_sep_ch << bs_interval_stats.at(b).num_own_users
										<< field_sep_ch << bs_interval_stats.at(b).num_served_users;
					}
				}
			}

			if (trace_dat_ofs.is_open())
			{
				trace_dat_ofs << std::endl;
			}


			// Try to estimate the real interval profit
			std::vector<RealT> prov_interval_real_coal_profits;
			std::vector<RealT> bs_interval_real_num_served_users;
			prov_interval_real_coal_profits = estimate_coalition_profits(scen,
																		 opts,
																		 rng,
																		 time_low,
																		 time_up,
																		 B,
																		 bs_traffic_patterns,
																		 bs_to_cats,
																		 bs_to_provs,
																		 //U,
																		 usr_to_cats,
																		 usr_bs_sinrs,
																		 prov_usr,
																		 formed_coalitions,
																		 bs_interval_real_num_served_users);

			// Accumulate interval profits
			for (std::size_t b = 0; b < tot_nb; ++b)
			{
				//prov_summary_interval_coal_profits(b) /= np;
				//bs_summary_interval_stats[b].num_served_users /= np;
				//bs_summary_interval_stats[b].num_own_users /= np;
				//bs_summary_interval_stats[b].num_poweron /= np;
				//bs_summary_interval_stats[b].coalition_load /= np;
				//bs_summary_interval_stats[b].alone_load /= np;

				prov_tot_coal_profits(b) += boost::accumulators::mean(prov_summary_interval_coal_profits[b]);
				bs_tot_stats[b].num_served_users += boost::accumulators::mean(bs_summary_interval_stats[b].num_served_users)*deltat;
				bs_tot_stats[b].num_own_users += boost::accumulators::mean(bs_summary_interval_stats[b].num_own_users)*deltat;
				bs_tot_stats[b].num_poweron += boost::accumulators::mean(bs_summary_interval_stats[b].num_poweron)*deltat;
				bs_tot_stats[b].coalition_load += boost::accumulators::mean(bs_summary_interval_stats[b].coalition_load);
				bs_tot_stats[b].alone_load += boost::accumulators::mean(bs_summary_interval_stats[b].alone_load);
				prov_tot_real_coal_profits(b) += prov_interval_real_coal_profits[b];
				bs_tot_real_num_served_users(b) += bs_interval_real_num_served_users[b];
			}

			if (opts.verbosity >= VERBOSITY_MEDIUM)
			{
				std::cout << "-- AVERAGED INTERVAL OUTPUTS:" << std::endl;
				std::cout << "- Average Coalition Profits: [" << tot_nb << "]{";
				for (std::size_t b = 0; b < tot_nb; ++b)
				{
					if (b > 0)
					{
						std::cout << ",";
					}

					std::cout << boost::accumulators::mean(prov_summary_interval_coal_profits[b]) << " (+/- " << std::sqrt(boost::accumulators::variance(prov_summary_interval_coal_profits[b])) << ")";
				}
				std::cout << "}" << std::endl;
				std::cout << "- Alone Profits: [" << tot_nb << "]{";
				for (std::size_t b = 0; b < tot_nb; ++b)
				{
					if (b > 0)
					{
						std::cout << ",";
					}

					std::cout << prov_interval_alone_profits(b);
				}
				std::cout << "}" << std::endl;
				std::cout << "- Average BS Stats:";
				for (std::size_t b = 0; b < tot_nb; ++b)
				{
					std::cout << " * BS " << b << std::endl;
					std::cout << "  - Expected # Powered On: " << boost::accumulators::mean(bs_summary_interval_stats.at(b).num_poweron) << " (+/- " << std::sqrt(boost::accumulators::variance(bs_summary_interval_stats.at(b).num_poweron)) << ")" << std::endl;
					std::cout << "  - Expected # Powered On vs. Interval Length: " << (boost::accumulators::mean(bs_summary_interval_stats.at(b).num_poweron)/deltat) << std::endl;
					std::cout << "  - Expected # Own Users: " << boost::accumulators::mean(bs_summary_interval_stats.at(b).num_own_users) << " (+/- " << std::sqrt(boost::accumulators::variance(bs_summary_interval_stats.at(b).num_own_users)) << ")" << std::endl;
					std::cout << "  - Expected # Served Users: " << boost::accumulators::mean(bs_summary_interval_stats.at(b).num_served_users) << " (+/- " << std::sqrt(boost::accumulators::variance(bs_summary_interval_stats.at(b).num_served_users)) << ")" << std::endl;
					std::cout << "  - Expected # Served Users vs. # Own Users: " << util::relative_increment(boost::accumulators::mean(bs_summary_interval_stats.at(b).num_served_users), boost::accumulators::mean(bs_summary_interval_stats.at(b).num_own_users)) << std::endl;
					std::cout << "  - Expected Coalition Load: " << boost::accumulators::mean(bs_summary_interval_stats.at(b).coalition_load) << " (+/- " << std::sqrt(boost::accumulators::variance(bs_summary_interval_stats.at(b).coalition_load)) << ")" << std::endl;
					std::cout << "  - Expected Alone Load: " << boost::accumulators::mean(bs_summary_interval_stats.at(b).alone_load) << " (+/- " << std::sqrt(boost::accumulators::variance(bs_summary_interval_stats.at(b).alone_load)) << ")" << std::endl;
					//std::cout << "  - Expected Coalition vs. Alone Load: " << (boost::accumulators::mean(bs_summary_interval_stats.at(b).coalition_load)/boost::accumulators::mean(bs_summary_interval_stats.at(b).alone_load)-1) << std::endl;
					std::cout << "  - Expected Coalition vs. Alone Load: " << util::relative_increment(boost::accumulators::mean(bs_summary_interval_stats.at(b).coalition_load), boost::accumulators::mean(bs_summary_interval_stats.at(b).alone_load)) << std::endl;
					std::cout << "  - Expected Real # Served Users: " << bs_interval_real_num_served_users[b] << std::endl;
				}

				if (opts.verbosity >= VERBOSITY_HIGH)
				{
					std::cout << "-- INCREMENTAL AVERAGED INTERVAL OUTPUTS:" << std::endl;
					std::cout << "- Incremental Average Coalition Profits: [" << tot_nb << "]{";
					for (std::size_t p = 0; p < scen.num_providers; ++p)
					{
						if (p > 0)
						{
							std::cout << ",";
						}

						std::cout << prov_tot_coal_profits(p);
					}
					std::cout << "}" << std::endl;
					std::cout << "- Incremental Alone Profits: [" << tot_nb << "]{";
					for (std::size_t p = 0; p < scen.num_providers; ++p)
					{
						if (p > 0)
						{
							std::cout << ",";
						}

						std::cout << prov_tot_alone_profits(p);
					}
					std::cout << "}" << std::endl;
					std::cout << "- Incremental Average BS Stats:" << std::endl;
					for (std::size_t p = 0; p < scen.num_providers; ++p)
					{
						std::cout << " * NO " << p << std::endl;
						for (std::size_t b = prov_to_bss.at(p).first; b < prov_to_bss.at(p).second; ++b)
						{
							std::cout << "  * BS " << b << std::endl;
							std::cout << "   - Expected # Powered On: " << bs_tot_stats.at(b).num_poweron << std::endl;
							std::cout << "   - Expected # Powered On vs. Simulation Time: " << (bs_tot_stats.at(b).num_poweron*deltat/opts.time_span) << std::endl;
							std::cout << "   - Expected # Own Users: " << bs_tot_stats.at(b).num_own_users << std::endl;
							std::cout << "   - Expected # Served Users: " << bs_tot_stats.at(b).num_served_users << std::endl;
							std::cout << "   - Expected # Served Users vs. # Own Users: " << util::relative_increment(bs_tot_stats.at(b).num_served_users, bs_tot_stats.at(b).num_own_users) << std::endl;
							std::cout << "   - Expected Coalition Load: " << bs_tot_stats.at(b).coalition_load << std::endl;
							std::cout << "   - Expected Alone Load: " << bs_tot_stats.at(b).alone_load << std::endl;
							//std::cout << "   - Expected Coalition vs. Alone Load: " << (bs_tot_stats.at(b).coalition_load/bs_tot_stats.at(b).alone_load-1) << std::endl;
							std::cout << "   - Expected Coalition vs. Alone Load: " << util::relative_increment(bs_tot_stats.at(b).coalition_load, bs_tot_stats.at(b).alone_load) << std::endl;
							std::cout << "   - Expected Real # Served Users: " << bs_tot_real_num_served_users[b] << std::endl;
						}
					}
				}
			}

			// Output to file
			if (stats_dat_ofs.is_open())
			{
				stats_dat_ofs << time_low << field_sep_ch << deltat;
				for (std::size_t b = 0; b < tot_nb; ++b)
				{
//TODO: add "real" profit and # served users
					stats_dat_ofs	<< field_sep_ch << boost::accumulators::mean(prov_summary_interval_coal_profits[b])
									<< field_sep_ch << prov_interval_alone_profits(b)
									//<< field_sep_ch << (boost::accumulators::mean(prov_summary_interval_coal_profits[b])/prov_interval_alone_profits(b)-1)
									<< field_sep_ch << util::relative_increment(boost::accumulators::mean(prov_summary_interval_coal_profits[b]), prov_interval_alone_profits(b))
									<< field_sep_ch << boost::accumulators::mean(bs_summary_interval_stats.at(b).num_poweron)
									<< field_sep_ch << std::sqrt(boost::accumulators::variance(bs_summary_interval_stats.at(b).num_poweron))
									<< field_sep_ch << (boost::accumulators::mean(bs_summary_interval_stats.at(b).num_poweron)/deltat)
									<< field_sep_ch << boost::accumulators::mean(bs_summary_interval_stats.at(b).num_own_users)
									<< field_sep_ch << std::sqrt(boost::accumulators::variance(bs_summary_interval_stats.at(b).num_own_users))
									<< field_sep_ch << boost::accumulators::mean(bs_summary_interval_stats.at(b).num_served_users)
									<< field_sep_ch << std::sqrt(boost::accumulators::variance(bs_summary_interval_stats.at(b).num_served_users))
									<< field_sep_ch << boost::accumulators::mean(bs_summary_interval_stats.at(b).coalition_load)
									<< field_sep_ch << util::relative_increment(boost::accumulators::mean(bs_summary_interval_stats.at(b).num_served_users), boost::accumulators::mean(bs_summary_interval_stats.at(b).num_own_users))
									<< field_sep_ch << std::sqrt(boost::accumulators::variance(bs_summary_interval_stats.at(b).coalition_load))
									<< field_sep_ch << boost::accumulators::mean(bs_summary_interval_stats.at(b).alone_load)
									<< field_sep_ch << std::sqrt(boost::accumulators::variance(bs_summary_interval_stats.at(b).alone_load))
									//<< field_sep_ch << (boost::accumulators::mean(bs_summary_interval_stats.at(b).coalition_load)/boost::accumulators::mean(bs_summary_interval_stats.at(b).alone_load)-1);
									<< field_sep_ch << util::relative_increment(boost::accumulators::mean(bs_summary_interval_stats.at(b).coalition_load), boost::accumulators::mean(bs_summary_interval_stats.at(b).alone_load));
				}
				stats_dat_ofs << std::endl;
			}
		}

		// Update stats for confidence interval estimation
		for (std::size_t p = 0; p < scen.num_providers; ++p)
		{
////		ci_rp_stats[p]->collect(util::relative_increment(prov_tot_coal_profits(p), prov_tot_alone_profits(p)));
//			ci_ap_stats[p]->collect(prov_tot_alone_profits(p));
			ci_rcp_stats[p]->collect(prov_tot_real_coal_profits(p));
			ci_fcp_stats[p]->collect(prov_tot_coal_profits(p));
			for (std::size_t b = prov_to_bss.at(p).first; b < prov_to_bss.at(p).second; ++b)
			{
////			ci_on_stats[b]->collect(bs_tot_stats.at(b).num_poweron*time_step/opts.time_span);
////			ci_xl_stats[b]->collect(util::relative_increment(bs_tot_stats.at(b).num_served_users, bs_tot_stats.at(b).num_own_users));
				ci_anu_stats[b]->collect(bs_tot_stats.at(b).num_own_users);
				ci_cnu_stats[b]->collect(bs_tot_stats.at(b).num_served_users);
				ci_cno_stats[b]->collect(bs_tot_stats.at(b).num_poweron);
				ci_rcnu_stats[b]->collect(bs_tot_real_num_served_users(b));
			}
		}

		if (opts.verbosity >= VERBOSITY_LOW)
		{
			std::cout << "-- ITERATION #" << iter << std::endl;

			if (opts.verbosity >= VERBOSITY_LOW_MEDIUM)
			{
				std::cout << " - SUMMARY OUTPUTS:" << std::endl;
				std::cout << "  - Total Coalition Profits: [" << scen.num_providers << "]{";
				for (std::size_t p = 0; p < scen.num_providers; ++p)
				{
					if (p > 0)
					{
						std::cout << ",";
					}

					std::cout << prov_tot_coal_profits(p);
				}
				std::cout << "}" << std::endl;
				std::cout << "  - Total Alone Profits: [" << scen.num_providers << "]{";
				for (std::size_t p = 0; p < scen.num_providers; ++p)
				{
					if (p > 0)
					{
						std::cout << ",";
					}

					std::cout << prov_tot_alone_profits(p);
				}
				std::cout << "}" << std::endl;
				std::cout << "  - Total NO Stats:" << std::endl;
				for (std::size_t p = 0; p < scen.num_providers; ++p)
				{
					std::cout << "    - Expected Coalition vs. Alone Profit: " << util::relative_increment(prov_tot_coal_profits(p), prov_tot_alone_profits(p)) << std::endl;

					for (std::size_t b = prov_to_bss.at(p).first; b < prov_to_bss.at(p).second; ++b)
					{
						std::cout << "   * BS " << b << std::endl;
						std::cout << "    - Expected # Powered On: " << bs_tot_stats.at(b).num_poweron << std::endl;
						std::cout << "    - Expected # Powered On vs. Simulation Time: " << (bs_tot_stats.at(b).num_poweron*time_step/opts.time_span) << std::endl;
						std::cout << "    - Expected # Own Users: " << bs_tot_stats.at(b).num_own_users << std::endl;
						std::cout << "    - Expected # Served Users: " << bs_tot_stats.at(b).num_served_users << std::endl;
						std::cout << "    - Expected # Served Users vs. # Own Users: " << util::relative_increment(bs_tot_stats.at(b).num_served_users, bs_tot_stats.at(b).num_own_users) << std::endl;
						std::cout << "    - Expected Coalition Load: " << bs_tot_stats.at(b).coalition_load << std::endl;
						std::cout << "    - Expected Alone Load: " << bs_tot_stats.at(b).alone_load << std::endl;
						std::cout << "    - Expected Coalition vs. Alone Load: " << util::relative_increment(bs_tot_stats.at(b).coalition_load, bs_tot_stats.at(b).alone_load) << std::endl;
						std::cout << "    - Expected Real # Served Users: " << bs_tot_real_num_served_users[b] << std::endl;
					}
				}
			}

			std::cout << " - CONFIDENCE INTERVALS OUTPUTS:" << std::endl;
			for (std::size_t p = 0; p < scen.num_providers; ++p)
			{
				std::cout << "  * NO " << p << std::endl;
////			std::cout << "   - RP statistics: " << ci_rp_stats[p]->estimate() << " (s.d. " << ci_rp_stats[p]->standard_deviation() << ") [" << ci_rp_stats[p]->lower() << ", " << ci_rp_stats[p]->upper() << "] (rel. prec.: " << ci_rp_stats[p]->relative_precision() << ")" << std::endl;
////			std::cout << "   - ON statistics: " << ci_on_stats[p]->estimate() << " (s.d. " << ci_on_stats[p]->standard_deviation() << ") [" << ci_on_stats[p]->lower() << ", " << ci_on_stats[p]->upper() << "] (rel. prec.: " << ci_on_stats[p]->relative_precision() << ")" << std::endl;
////			std::cout << "   - XL statistics: " << ci_xl_stats[p]->estimate() << " (s.d. " << ci_xl_stats[p]->standard_deviation() << ") [" << ci_xl_stats[p]->lower() << ", " << ci_xl_stats[p]->upper() << "] (rel. prec.: " << ci_xl_stats[p]->relative_precision() << ")" << std::endl;
//				std::cout << "   - AP statistics: " << ci_ap_stats[p]->estimate() << " (s.d. " << ci_ap_stats[p]->standard_deviation() << ") [" << ci_ap_stats[p]->lower() << ", " << ci_ap_stats[b]->upper() << "] (rel. prec.: " << ci_ap_stats[p]->relative_precision() << ")" << std::endl;
				std::cout << "   - FCP statistics: " << ci_fcp_stats[p]->estimate() << " (s.d. " << ci_fcp_stats[p]->standard_deviation() << ") [" << ci_fcp_stats[p]->lower() << ", " << ci_fcp_stats[p]->upper() << "] (rel. prec.: " << ci_fcp_stats[p]->relative_precision() << ", size: " << ci_fcp_stats[p]->size() << ")" << std::endl;
				std::cout << "   - RCP statistics: " << ci_rcp_stats[p]->estimate() << " (s.d. " << ci_rcp_stats[p]->standard_deviation() << ") [" << ci_rcp_stats[p]->lower() << ", " << ci_rcp_stats[p]->upper() << "] (rel. prec.: " << ci_rcp_stats[p]->relative_precision() << ", size: " << ci_rcp_stats[p]->size() << ")" << std::endl;

				for (std::size_t b = prov_to_bss.at(p).first; b < prov_to_bss.at(p).second; ++b)
				{
					std::cout << "  * BS " << b << std::endl;
					std::cout << "   - ANU statistics: " << ci_anu_stats[b]->estimate() << " (s.d. " << ci_anu_stats[b]->standard_deviation() << ") [" << ci_anu_stats[p]->lower() << ", " << ci_anu_stats[b]->upper() << "] (rel. prec.: " << ci_anu_stats[b]->relative_precision() << ", size: " << ci_anu_stats[b]->size() << ")" << std::endl;
					std::cout << "   - CNU statistics: " << ci_cnu_stats[b]->estimate() << " (s.d. " << ci_cnu_stats[b]->standard_deviation() << ") [" << ci_cnu_stats[p]->lower() << ", " << ci_cnu_stats[b]->upper() << "] (rel. prec.: " << ci_cnu_stats[b]->relative_precision() << ", size: " << ci_cnu_stats[b]->size() << ")" << std::endl;
					std::cout << "   - CNO statistics: " << ci_cno_stats[b]->estimate() << " (s.d. " << ci_cno_stats[b]->standard_deviation() << ") [" << ci_cno_stats[p]->lower() << ", " << ci_cno_stats[b]->upper() << "] (rel. prec.: " << ci_cno_stats[b]->relative_precision() << ", size: " << ci_cno_stats[b]->size() << ")" << std::endl;
					std::cout << "   - RCNU statistics: " << ci_rcnu_stats[b]->estimate() << " (s.d. " << ci_rcnu_stats[b]->standard_deviation() << ") [" << ci_rcnu_stats[p]->lower() << ", " << ci_rcnu_stats[b]->upper() << "] (rel. prec.: " << ci_rcnu_stats[b]->relative_precision() << ", size: " << ci_rcnu_stats[b]->size() << ")" << std::endl;
				}
			}
		}
	}

	if (stats_dat_ofs.is_open())
	{
		stats_dat_ofs.close();
	}
	if (trace_dat_ofs.is_open())
	{
		trace_dat_ofs.close();
	}

	// Print confidence intervals
	std::cout << "-- CONFIDENCE INTERVALS OUTPUTS:" << std::endl;
	for (std::size_t p = 0; p < scen.num_providers; ++p)
	{
		std::cout << " * NO " << p << std::endl;
////		std::cout << "  - RP statistics: " << ci_rp_stats[p]->estimate() << " (s.d. " << ci_rp_stats[p]->standard_deviation() << ") [" << ci_rp_stats[p]->lower() << ", " << ci_rp_stats[p]->upper() << "] (rel. prec.: " << ci_rp_stats[b]->relative_precision() << ", size: " << ci_rp_stats[p]->size() << ")" << std::endl;
////		std::cout << "  - ON statistics: " << ci_on_stats[p]->estimate() << " (s.d. " << ci_on_stats[p]->standard_deviation() << ") [" << ci_on_stats[p]->lower() << ", " << ci_on_stats[p]->upper() << "] (rel. prec.: " << ci_on_stats[b]->relative_precision() << ", size: " << ci_on_stats[p]->size() << ")" << std::endl;
////		std::cout << "  - XL statistics: " << ci_xl_stats[p]->estimate() << " (s.d. " << ci_xl_stats[p]->standard_deviation() << ") [" << ci_xl_stats[p]->lower() << ", " << ci_xl_stats[p]->upper() << "] (rel. prec.: " << ci_xl_stats[b]->relative_precision() << ", size: " << ci_xl_stats[p]->size() << ")" << std::endl;
//		std::cout << "  - AP statistics: " << ci_ap_stats[p]->estimate() << " (s.d. " << ci_ap_stats[p]->standard_deviation() << ") [" << ci_ap_stats[p]->lower() << ", " << ci_ap_stats[p]->upper() << "] (rel. prec.: " << ci_ap_stats[b]->relative_precision() << ", size: " << ci_ap_stats[p]->size() << ")" << std::endl;
		std::cout << "  - FCP statistics: " << ci_fcp_stats[p]->estimate() << " (s.d. " << ci_fcp_stats[p]->standard_deviation() << ") [" << ci_fcp_stats[p]->lower() << ", " << ci_fcp_stats[p]->upper() << "] (rel. prec.: " << ci_fcp_stats[p]->relative_precision() << ", size: " << ci_fcp_stats[p]->size() << ")" << std::endl;
		std::cout << "  - RCP statistics: " << ci_rcp_stats[p]->estimate() << " (s.d. " << ci_rcp_stats[p]->standard_deviation() << ") [" << ci_rcp_stats[p]->lower() << ", " << ci_rcp_stats[p]->upper() << "] (rel. prec.: " << ci_rcp_stats[p]->relative_precision() << ", size: " << ci_rcp_stats[p]->size() << ")" << std::endl;
		for (std::size_t b = prov_to_bss.at(p).first; b < prov_to_bss.at(p).second; ++b)
		{
			std::cout << "  * BS " << b << std::endl;
			std::cout << "   - ANU statistics: " << ci_anu_stats[b]->estimate() << " (s.d. " << ci_anu_stats[b]->standard_deviation() << ") [" << ci_anu_stats[p]->lower() << ", " << ci_anu_stats[b]->upper() << "] (rel. prec.: " << ci_anu_stats[b]->relative_precision() << ", size: " << ci_anu_stats[b]->size() << ")" << std::endl;
			std::cout << "   - CNU statistics: " << ci_cnu_stats[b]->estimate() << " (s.d. " << ci_cnu_stats[b]->standard_deviation() << ") [" << ci_cnu_stats[p]->lower() << ", " << ci_cnu_stats[b]->upper() << "] (rel. prec.: " << ci_cnu_stats[b]->relative_precision() << ", size: " << ci_cnu_stats[b]->size() << ")" << std::endl;
			std::cout << "   - CNO statistics: " << ci_cno_stats[b]->estimate() << " (s.d. " << ci_cno_stats[b]->standard_deviation() << ") [" << ci_cno_stats[p]->lower() << ", " << ci_cno_stats[b]->upper() << "] (rel. prec.: " << ci_cno_stats[b]->relative_precision() << ", size: " << ci_cno_stats[b]->size() << ")" << std::endl;
			std::cout << "   - RCNU statistics: " << ci_rcnu_stats[b]->estimate() << " (s.d. " << ci_rcnu_stats[b]->standard_deviation() << ") [" << ci_rcnu_stats[p]->lower() << ", " << ci_rcnu_stats[b]->upper() << "] (rel. prec.: " << ci_rcnu_stats[b]->relative_precision() << ", size: " << ci_rcnu_stats[b]->size() << ")" << std::endl;
		}
	}
	std::cout << "-- DERIVED STATISTICS OUTPUTS:" << std::endl;
	for (std::size_t p = 0; p < scen.num_providers; ++p)
	{
		std::cout << " * NO " << p << std::endl;
//		std::cout << "  - RP statistics: " << util::relative_increment(ci_rp_stats[p]->estimate(), scen.provider_reference_profits[p]) << std::endl;
		std::cout << "  - ON statistics: " << (ci_cno_stats[p]->estimate()/opts.time_span) << std::endl;
		std::cout << "  - XL statistics: " << util::relative_increment(ci_cnu_stats[p]->estimate(), ci_anu_stats[p]->estimate()) << std::endl;
	}
}

template <typename RealT>
scenario<RealT> make_scenario(std::string const& fname)
{
	DCS_ASSERT(!fname.empty(),
			   DCS_EXCEPTION_THROW(std::invalid_argument, "Invalid scenario file name"));

	scenario<RealT> s;
	std::vector< std::vector<std::size_t> > provider_bs_traffic_data_sizes;

	std::ifstream ifs(fname.c_str());

	DCS_ASSERT(ifs,
			   DCS_EXCEPTION_THROW(std::runtime_error, "Cannot open scenario file"));

	std::ostringstream oss;
	std::size_t lineno(0);
	for (std::string line; std::getline(ifs, line); )
	{
		++lineno;

		std::size_t pos(0);
		for (; pos < line.length() && std::isspace(line[pos]); ++pos)
		{
			; // empty
		}
		if (pos > 0)
		{
			line = line.substr(pos);
		}
		if (line.empty() || line.at(0) == '#')
		{
			// Skip either empty or comment lines
			continue;
		}

		boost::to_lower(line);
		if (boost::starts_with(line, "num_providers"))
		{
			std::istringstream iss(line);

			// Move to '='
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '=');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('=' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			iss >> s.num_providers;
		}
		else if (boost::starts_with(line, "num_bs_types"))
		{
			std::istringstream iss(line);

			// Move to '='
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '=');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('=' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			iss >> s.num_bs_types;
		}
		else if (boost::starts_with(line, "num_user_classes"))
		{
			std::istringstream iss(line);

			// Move to '='
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '=');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('=' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			iss >> s.num_user_classes;
		}
		else if (boost::starts_with(line, "topology.area_size"))
		{
			std::istringstream iss(line);

			// Move to '='
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '=');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('=' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			// Move to '['
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			iss >> s.topology_area_size.first;
			iss >> s.topology_area_size.second;
		}
		else if (boost::starts_with(line, "topology.bs_layout"))
		{
			std::istringstream iss(line);

			// Move to '='
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '=');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('=' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			std::string str;

			iss >> str;

			boost::to_lower(str);

			if (str == "center")
			{
				s.topology_bs_layout = center_bs_topology_layout;
			}
			else if (str == "diamond")
			{
				s.topology_bs_layout = diamond_bs_topology_layout;
			}
			else if (str == "random")
			{
				s.topology_bs_layout = random_bs_topology_layout;
			}
			else if (str == "square")
			{
				s.topology_bs_layout = square_bs_topology_layout;
			}
			else if (str == "polygon")
			{
				s.topology_bs_layout = polygon_bs_topology_layout;
			}
			else
			{
				DCS_EXCEPTION_THROW(std::runtime_error, "Unrecognized BS topology layout (at line " + ::detail::util::to_string(lineno) + ")");
			}
		}
		else if (boost::starts_with(line, "bs.heights"))
		{
			std::istringstream iss(line.substr(pos));

			// Move to '='
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '=');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('=' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			// Move to '['
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			s.bs_heights.resize(s.num_bs_types);
			for (std::size_t b = 0; b < s.num_bs_types; ++b)
			{
				iss >> s.bs_heights[b];
			}
		}
		else if (boost::starts_with(line, "bs.carrier_frequencies"))
		{
			std::istringstream iss(line.substr(pos));

			// Move to '='
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '=');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('=' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			// Move to '['
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			s.bs_carrier_frequencies.resize(s.num_bs_types);
			for (std::size_t b = 0; b < s.num_bs_types; ++b)
			{
				iss >> s.bs_carrier_frequencies[b];
			}
		}
//		else if (boost::starts_with(line, "bs.max_channel_capacities"))
//		{
//			std::istringstream iss(line.substr(pos));
//
//			// Move to '='
//			iss.ignore(std::numeric_limits<std::streamsize>::max(), '=');
//			DCS_ASSERT(iss.good(),
//					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('=' is missing at line " + ::detail::util::to_string(lineno) + ")"));
//
//			// Move to '['
//			iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
//			DCS_ASSERT(iss.good(),
//					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));
//
//			s.bs_max_channel_capacities.resize(s.num_bs_types);
//			for (std::size_t b = 0; b < s.num_bs_types; ++b)
//			{
//				iss >> s.bs_max_channel_capacities[b];
//			}
//		}
		else if (boost::starts_with(line, "bs.bandwidths"))
		{
			std::istringstream iss(line.substr(pos));

			// Move to '='
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '=');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('=' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			// Move to '['
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			s.bs_bandwidths.resize(s.num_bs_types);
			for (std::size_t b = 0; b < s.num_bs_types; ++b)
			{
				iss >> s.bs_bandwidths[b];
			}
		}
//		else if (boost::starts_with(line, "bs.num_subcarriers"))
//		{
//			std::istringstream iss(line.substr(pos));
//
//			// Move to '='
//			iss.ignore(std::numeric_limits<std::streamsize>::max(), '=');
//			DCS_ASSERT(iss.good(),
//					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('=' is missing at line " + ::detail::util::to_string(lineno) + ")"));
//
//			// Move to '['
//			iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
//			DCS_ASSERT(iss.good(),
//					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));
//
//			s.bs_num_subcarriers.resize(s.num_bs_types);
//			for (std::size_t b = 0; b < s.num_bs_types; ++b)
//			{
//				iss >> s.bs_num_subcarriers[b];
//			}
//		}
		else if (boost::starts_with(line, "bs.transmission_powers"))
		{
			std::istringstream iss(line.substr(pos));

			// Move to '='
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '=');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('=' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			// Move to '['
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			s.bs_transmission_powers.resize(s.num_bs_types);
			for (std::size_t b = 0; b < s.num_bs_types; ++b)
			{
				iss >> s.bs_transmission_powers[b];
			}
		}
		else if (boost::starts_with(line, "bs.idle_powers"))
		{
			std::istringstream iss(line.substr(pos));

			// Move to '='
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '=');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('=' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			// Move to '['
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			s.bs_idle_powers.resize(s.num_bs_types);
			for (std::size_t b = 0; b < s.num_bs_types; ++b)
			{
				iss >> s.bs_idle_powers[b];
			}

			// Move to ']'
			iss.ignore(std::numeric_limits<std::streamsize>::max(), ']');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file (']' is missing at line " + ::detail::util::to_string(lineno) + ")"));
		}
		else if (boost::starts_with(line, "bs.load_powers"))
		{
			std::istringstream iss(line.substr(pos));

			// Move to '='
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '=');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('=' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			// Move to '['
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			s.bs_load_powers.resize(s.num_bs_types);
			for (std::size_t b = 0; b < s.num_bs_types; ++b)
			{
				iss >> s.bs_load_powers[b];
			}

			//// Move to ']'
			//iss.ignore(std::numeric_limits<std::streamsize>::max(), ']');
			//DCS_ASSERT(iss.good(),
			//		   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file (']' is missing at line " + ::detail::util::to_string(lineno) + ")"));
		}
		else if (boost::starts_with(line, "provider.num_bss"))
		{
			std::istringstream iss(line.substr(pos));

			// Move to '='
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '=');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('=' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			// Move to '['
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			s.provider_num_bss.resize(s.num_providers);
			for (std::size_t p = 0; p < s.num_providers; ++p)
			{
				// Move to '['
				iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
				DCS_ASSERT(iss.good(),
						   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

				s.provider_num_bss[p].resize(s.num_bs_types);
				for (std::size_t b = 0; b < s.num_bs_types; ++b)
				{
					iss >> s.provider_num_bss[p][b];
				}

				// Move to ']'
				iss.ignore(std::numeric_limits<std::streamsize>::max(), ']');
				DCS_ASSERT(iss.good(),
						   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file (']' is missing at line " + ::detail::util::to_string(lineno) + ")"));
            }
		}
		else if (boost::starts_with(line, "provider.electricity_costs"))
		{
			std::istringstream iss(line.substr(pos));

			// Move to '='
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '=');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('=' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			// Move to '['
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			s.provider_electricity_costs.resize(s.num_providers);
			for (std::size_t p = 0; p < s.num_providers; ++p)
			{
				iss >> s.provider_electricity_costs[p];
			}
		}
		else if (boost::starts_with(line, "provider.alt_electricity_costs"))
		{
			std::istringstream iss(line.substr(pos));

			// Move to '='
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '=');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('=' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			// Move to '['
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			s.provider_alt_electricity_costs.resize(s.num_providers);
			for (std::size_t p = 0; p < s.num_providers; ++p)
			{
				iss >> s.provider_alt_electricity_costs[p];
			}
		}
		else if (boost::starts_with(line, "provider.alt_electricity_time_ranges"))
		{
			std::istringstream iss(line.substr(pos));

			// Move to '='
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '=');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('=' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			// Move to '['
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			s.provider_alt_electricity_time_ranges.resize(s.num_providers);
			for (std::size_t p = 0; p < s.num_providers; ++p)
			{
				// Move to '['
				iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
				DCS_ASSERT(iss.good(),
						   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

				RealT start(0);
				RealT stop(0);

				iss >> start >> stop;

				DCS_ASSERT(start <= stop,
						   DCS_EXCEPTION_THROW(std::logic_error, "Start time cannot be greater than stop time"));

				s.provider_alt_electricity_time_ranges[p] = std::make_pair(start, stop);

				// Move to ']'
				iss.ignore(std::numeric_limits<std::streamsize>::max(), ']');
				DCS_ASSERT(iss.good(),
						   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file (']' is missing at line " + ::detail::util::to_string(lineno) + ")"));
			}
		}
		else if (boost::starts_with(line, "provider.coalition_costs"))
		{
			std::istringstream iss(line.substr(pos));

			// Move to '='
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '=');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('=' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			// Move to '['
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			s.provider_coalition_costs.resize(s.num_providers);
			for (std::size_t p = 0; p < s.num_providers; ++p)
			{
				iss >> s.provider_coalition_costs[p];
			}
		}
//		else if (boost::starts_with(line, "provider.bs_coalition_change_costs"))
//		{
//			std::istringstream iss(line.substr(pos));
//
//			// Move to '='
//			iss.ignore(std::numeric_limits<std::streamsize>::max(), '=');
//			DCS_ASSERT(iss.good(),
//					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('=' is missing at line " + ::detail::util::to_string(lineno) + ")"));
//
//			// Move to '['
//			iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
//			DCS_ASSERT(iss.good(),
//					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));
//
//			s.provider_bs_coalition_change_costs.resize(s.num_providers);
//			for (std::size_t p = 0; p < s.num_providers; ++p)
//			{
//				iss >> s.provider_bs_coalition_change_costs[p];
//			}
//		}
		else if (boost::starts_with(line, "provider.bs_traffic_curves"))
		{
			std::istringstream iss(line.substr(pos));

			// Move to '='
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '=');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('=' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			// Move to '['
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			s.provider_bs_traffic_curves.resize(s.num_providers);
			for (std::size_t p = 0; p < s.num_providers; ++p)
			{
				// Move to '['
				iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
				DCS_ASSERT(iss.good(),
						   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

				s.provider_bs_traffic_curves[p].resize(s.num_bs_types);
				for (std::size_t b = 0; b < s.num_bs_types; ++b)
				{
					std::string str;

					while (iss && iss.peek() != ']')
					{
						str += iss.get();
					}
					//iss >> str;

					boost::to_lower(str);

					if (str == "constant")
					{
						s.provider_bs_traffic_curves[p][b] = constant_bs_traffic_curve;
					}
					else if (str == "linear")
					{
						s.provider_bs_traffic_curves[p][b] = linear_bs_traffic_curve;
					}
					else if (str == "cubic-spline")
					{
						s.provider_bs_traffic_curves[p][b] = cubic_spline_bs_traffic_curve;
					}
					else
					{
						DCS_EXCEPTION_THROW(std::runtime_error, "Unrecognized BS traffic curve category (at line " + ::detail::util::to_string(lineno) + ")");
					}
				}

				// Move to ']'
				iss.ignore(std::numeric_limits<std::streamsize>::max(), ']');
				DCS_ASSERT(iss.good(),
						   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file (']' is missing at line " + ::detail::util::to_string(lineno) + ")"));
			}
		}
		else if (boost::starts_with(line, "provider.bs_traffic_data_sizes"))
		{
			std::istringstream iss(line.substr(pos));

			// Move to '='
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '=');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('=' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			// Move to '['
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			provider_bs_traffic_data_sizes.resize(s.num_providers);
			for (std::size_t p = 0; p < s.num_providers; ++p)
			{
				// Move to '['
				iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
				DCS_ASSERT(iss.good(),
						   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

				provider_bs_traffic_data_sizes[p].resize(s.num_bs_types);
				for (std::size_t b = 0; b < s.num_bs_types; ++b)
				{
					iss >> provider_bs_traffic_data_sizes[p][b];
				}

				// Move to ']'
				iss.ignore(std::numeric_limits<std::streamsize>::max(), ']');
				DCS_ASSERT(iss.good(),
						   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file (']' is missing at line " + ::detail::util::to_string(lineno) + ")"));
			}
		}
		else if (boost::starts_with(line, "provider.bs_traffic_data"))
		{
			std::istringstream iss(line.substr(pos));

			// Move to '='
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '=');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('=' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			// Move to '['
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			s.provider_bs_traffic_data.resize(s.num_providers);
			for (std::size_t p = 0; p < s.num_providers; ++p)
			{
				// Move to '['
				iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
				DCS_ASSERT(iss.good(),
						   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

				s.provider_bs_traffic_data[p].resize(s.num_bs_types);
				for (std::size_t b = 0; b < s.num_bs_types; ++b)
				{
					// Move to '['
					iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
					DCS_ASSERT(iss.good(),
							   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

					s.provider_bs_traffic_data[p][b].resize(provider_bs_traffic_data_sizes[p][b]);
					for (std::size_t d = 0; d < provider_bs_traffic_data_sizes[p][b]; ++d)
					{
						// Move to '['
						iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
						DCS_ASSERT(iss.good(),
								   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

						RealT x(0);
						RealT y(0);

						iss >> x >> y;

						s.provider_bs_traffic_data[p][b][d] = std::make_pair(x, y);

						// Move to ']'
						iss.ignore(std::numeric_limits<std::streamsize>::max(), ']');
						DCS_ASSERT(iss.good(),
								   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file (']' is missing at line " + ::detail::util::to_string(lineno) + ")"));
					}

					// Move to ']'
					iss.ignore(std::numeric_limits<std::streamsize>::max(), ']');
					DCS_ASSERT(iss.good(),
							   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file (']' is missing at line " + ::detail::util::to_string(lineno) + ")"));
				}

				// Move to ']'
				iss.ignore(std::numeric_limits<std::streamsize>::max(), ']');
				DCS_ASSERT(iss.good(),
						   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file (']' is missing at line " + ::detail::util::to_string(lineno) + ")"));
			}
		}
		else if (boost::starts_with(line, "provider.bs_amortizing_costs"))
		{
			std::istringstream iss(line.substr(pos));

			// Move to '='
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '=');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('=' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			// Move to '['
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			s.provider_bs_amortizing_costs.resize(s.num_providers);
			for (std::size_t p = 0; p < s.num_providers; ++p)
			{
				// Move to '['
				iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
				DCS_ASSERT(iss.good(),
						   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

				s.provider_bs_amortizing_costs[p].resize(s.num_bs_types);
				for (std::size_t b = 0; b < s.num_bs_types; ++b)
				{
					iss >> s.provider_bs_amortizing_costs[p][b];
				}

				// Move to ']'
				iss.ignore(std::numeric_limits<std::streamsize>::max(), ']');
				DCS_ASSERT(iss.good(),
						   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file (']' is missing at line " + ::detail::util::to_string(lineno) + ")"));
            }
		}
		else if (boost::starts_with(line, "provider.bs_residual_amortizing_times"))
		{
			std::istringstream iss(line.substr(pos));

			// Move to '='
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '=');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('=' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			// Move to '['
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			s.provider_bs_residual_amortizing_times.resize(s.num_providers);
			for (std::size_t p = 0; p < s.num_providers; ++p)
			{
				// Move to '['
				iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
				DCS_ASSERT(iss.good(),
						   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

				s.provider_bs_residual_amortizing_times[p].resize(s.num_bs_types);
				for (std::size_t b = 0; b < s.num_bs_types; ++b)
				{
					iss >> s.provider_bs_residual_amortizing_times[p][b];
				}

				// Move to ']'
				iss.ignore(std::numeric_limits<std::streamsize>::max(), ']');
				DCS_ASSERT(iss.good(),
						   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file (']' is missing at line " + ::detail::util::to_string(lineno) + ")"));
            }
		}
		else if (boost::starts_with(line, "provider.bs_max_num_users"))
		{
			std::istringstream iss(line.substr(pos));

			// Move to '='
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '=');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('=' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			// Move to '['
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			s.provider_bs_max_num_users.resize(s.num_providers);
			for (std::size_t p = 0; p < s.num_providers; ++p)
			{
				// Move to '['
				iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
				DCS_ASSERT(iss.good(),
						   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

				s.provider_bs_max_num_users[p].resize(s.num_bs_types);
				for (std::size_t b = 0; b < s.num_bs_types; ++b)
				{
					iss >> s.provider_bs_max_num_users[p][b];
				}

				// Move to ']'
				iss.ignore(std::numeric_limits<std::streamsize>::max(), ']');
				DCS_ASSERT(iss.good(),
						   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file (']' is missing at line " + ::detail::util::to_string(lineno) + ")"));
			 }
		}
		else if (boost::starts_with(line, "provider.user_min_downlink_data_rates"))
		{
			std::istringstream iss(line.substr(pos));

			// Move to '='
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '=');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('=' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			// Move to '['
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			s.provider_user_min_downlink_data_rates.resize(s.num_providers);
			for (std::size_t p = 0; p < s.num_providers; ++p)
			{
				// Move to '['
				iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
				DCS_ASSERT(iss.good(),
						   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

				s.provider_user_min_downlink_data_rates[p].resize(s.num_user_classes);
				for (std::size_t u = 0; u < s.num_user_classes; ++u)
				{
					iss >> s.provider_user_min_downlink_data_rates[p][u];
				}

				// Move to ']'
				iss.ignore(std::numeric_limits<std::streamsize>::max(), ']');
				DCS_ASSERT(iss.good(),
						   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file (']' is missing at line " + ::detail::util::to_string(lineno) + ")"));
			}
		}
		else if (boost::starts_with(line, "provider.user_revenues"))
		{
			std::istringstream iss(line.substr(pos));

			// Move to '='
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '=');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('=' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			// Move to '['
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			s.provider_user_revenues.resize(s.num_providers);
			for (std::size_t p = 0; p < s.num_providers; ++p)
			{
				// Move to '['
				iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
				DCS_ASSERT(iss.good(),
						   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

				s.provider_user_revenues[p].resize(s.num_user_classes);
				for (std::size_t u = 0; u < s.num_user_classes; ++u)
				{
					iss >> s.provider_user_revenues[p][u];
				}

				// Move to ']'
				iss.ignore(std::numeric_limits<std::streamsize>::max(), ']');
				DCS_ASSERT(iss.good(),
						   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file (']' is missing at line " + ::detail::util::to_string(lineno) + ")"));
			}
		}
		else if (boost::starts_with(line, "provider.user_mixes"))
		{
			std::istringstream iss(line.substr(pos));

			// Move to '='
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '=');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('=' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			// Move to '['
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			s.provider_user_mixes.resize(s.num_providers);
			for (std::size_t p = 0; p < s.num_providers; ++p)
			{
				// Move to '['
				iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
				DCS_ASSERT(iss.good(),
						   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

				s.provider_user_mixes[p].resize(s.num_user_classes);
				RealT sum(0);
				for (std::size_t u = 0; u < s.num_user_classes; ++u)
				{
					iss >> s.provider_user_mixes[p][u];
					sum += s.provider_user_mixes[p][u];
				}
				// Normalize
				for (std::size_t u = 0; u < s.num_user_classes; ++u)
				{
					s.provider_user_mixes[p][u] /= sum;
				}

				// Move to ']'
				iss.ignore(std::numeric_limits<std::streamsize>::max(), ']');
				DCS_ASSERT(iss.good(),
						   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file (']' is missing at line " + ::detail::util::to_string(lineno) + ")"));
			}
		}
		else if (boost::starts_with(line, "user.downlink_data_rates"))
		{
			std::istringstream iss(line.substr(pos));

			// Move to '='
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '=');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('=' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			// Move to '['
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			s.user_downlink_data_rates.resize(s.num_user_classes);
			for (std::size_t u = 0; u < s.num_user_classes; ++u)
			{
				iss >> s.user_downlink_data_rates[u];
			}
		}
		else if (boost::starts_with(line, "user.thermal_noises"))
		{
			std::istringstream iss(line.substr(pos));

			// Move to '='
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '=');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('=' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			// Move to '['
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			s.user_thermal_noises.resize(s.num_user_classes);
			for (std::size_t u = 0; u < s.num_user_classes; ++u)
			{
				iss >> s.user_thermal_noises[u];
			}
		}
		else if (boost::starts_with(line, "user.heights"))
		{
			std::istringstream iss(line.substr(pos));

			// Move to '='
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '=');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('=' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			// Move to '['
			iss.ignore(std::numeric_limits<std::streamsize>::max(), '[');
			DCS_ASSERT(iss.good(),
					   DCS_EXCEPTION_THROW(std::runtime_error, "Malformed scenario file ('[' is missing at line " + ::detail::util::to_string(lineno) + ")"));

			s.user_heights.resize(s.num_user_classes);
			for (std::size_t u = 0; u < s.num_user_classes; ++u)
			{
				iss >> s.user_heights[u];
			}
		}
	}

	// Assign default values
	if (math::iszero(s.topology_area_size.first) || math::iszero(s.topology_area_size.second))
	{
		s.topology_area_size = default_topology_area_size;
	}
	if (s.bs_heights.size() == 0)
	{
		s.bs_heights.resize(s.num_bs_types);
		std::fill(s.bs_heights.begin(), s.bs_heights.end(), default_bs_height);
	}
	if (s.bs_carrier_frequencies.size() == 0)
	{
		s.bs_carrier_frequencies.resize(s.num_bs_types);
		std::fill(s.bs_carrier_frequencies.begin(), s.bs_carrier_frequencies.end(), default_bs_carrier_frequency);
	}
//	if (s.bs_max_channel_capacities.size() == 0)
//	{
//		s.bs_max_channel_capacities.resize(s.num_bs_types);
//		std::fill(s.bs_max_channel_capacities.begin(), s.bs_max_channel_capacities.end(), default_bs_max_channel_capacity);
//	}
	if (s.bs_bandwidths.size() == 0)
	{
		s.bs_bandwidths.resize(s.num_bs_types);
		std::fill(s.bs_bandwidths.begin(), s.bs_bandwidths.end(), default_bs_bandwidth);
	}
//	if (s.bs_num_subcarriers.size() == 0)
//	{
//		s.bs_num_subcarriers.resize(s.num_bs_types);
//		std::fill(s.bs_num_subcarriers.begin(), s.bs_num_subcarriers.end(), default_bs_num_subcarriers);
//	}
	if (s.bs_transmission_powers.size() == 0)
	{
		s.bs_transmission_powers.resize(s.num_bs_types);
		std::fill(s.bs_transmission_powers.begin(), s.bs_transmission_powers.end(), default_bs_transmission_power);
	}
	if (s.bs_idle_powers.size() == 0)
	{
		s.bs_idle_powers.resize(s.num_bs_types);
		std::fill(s.bs_idle_powers.begin(), s.bs_idle_powers.end(), default_bs_idle_power);
	}
	if (s.bs_load_powers.size() == 0)
	{
		s.bs_load_powers.resize(s.num_bs_types);
		std::fill(s.bs_load_powers.begin(), s.bs_load_powers.end(), default_bs_load_dependent_power);
	}
	if (s.provider_num_bss.size() == 0)
	{
		s.provider_num_bss.resize(s.num_providers);
		for (std::size_t p = 0; p < s.num_providers; ++p)
		{
			s.provider_num_bss[p].resize(s.num_bs_types);
			std::fill(s.provider_num_bss[p].begin(), s.provider_num_bss[p].end(), default_provider_num_bss);
		}
	}
	if (s.provider_electricity_costs.size() == 0)
	{
		s.provider_electricity_costs.resize(s.num_providers);
		std::fill(s.provider_electricity_costs.begin(), s.provider_electricity_costs.end(), default_provider_electricity_cost);
	}
	if (s.provider_alt_electricity_costs.size() == 0)
	{
		s.provider_alt_electricity_costs.resize(s.num_providers);
		std::fill(s.provider_alt_electricity_costs.begin(), s.provider_alt_electricity_costs.end(), default_provider_alt_electricity_cost);
	}
	if (s.provider_alt_electricity_time_ranges.size() == 0)
	{
		s.provider_alt_electricity_time_ranges.resize(s.num_providers);
		std::fill(s.provider_alt_electricity_time_ranges.begin(), s.provider_alt_electricity_time_ranges.end(), default_provider_alt_electricity_time_range);
	}
	if (s.provider_coalition_costs.size() == 0)
	{
		s.provider_coalition_costs.resize(s.num_providers);
		std::fill(s.provider_coalition_costs.begin(), s.provider_coalition_costs.end(), default_provider_coalition_cost);
	}
	if (s.provider_bs_traffic_curves.size() == 0)
	{
		s.provider_bs_traffic_curves.resize(s.num_providers);
		for (std::size_t p = 0; p < s.num_providers; ++p)
		{
			s.provider_bs_traffic_curves[p].resize(s.num_bs_types);
			std::fill(s.provider_bs_traffic_curves[p].begin(), s.provider_bs_traffic_curves[p].end(), default_provider_bs_traffic_curve);
		}
	}
	if (s.provider_bs_traffic_data.size() == 0)
	{
		s.provider_bs_traffic_data.resize(s.num_providers);
		for (std::size_t p = 0; p < s.num_providers; ++p)
		{
			s.provider_bs_traffic_data[p].resize(s.num_bs_types);
			std::fill(s.provider_bs_traffic_data[p].begin(), s.provider_bs_traffic_data[p].end(), default_provider_bs_traffic_data);
		}
	}
	if (s.provider_bs_amortizing_costs.size() == 0)
	{
		s.provider_bs_amortizing_costs.resize(s.num_providers);
		for (std::size_t p = 0; p < s.num_providers; ++p)
		{
			s.provider_bs_amortizing_costs[p].resize(s.num_bs_types);
			std::fill(s.provider_bs_amortizing_costs[p].begin(), s.provider_bs_amortizing_costs[p].end(), default_provider_bs_amortizing_cost);
		}
	}
	if (s.provider_bs_residual_amortizing_times.size() == 0)
	{
		s.provider_bs_residual_amortizing_times.resize(s.num_providers);
		for (std::size_t p = 0; p < s.num_providers; ++p)
		{
			s.provider_bs_residual_amortizing_times[p].resize(s.num_bs_types);
			std::fill(s.provider_bs_residual_amortizing_times[p].begin(), s.provider_bs_residual_amortizing_times[p].end(), default_provider_bs_residual_amortizing_time);
		}
	}
	if (s.provider_bs_max_num_users.size() == 0)
	{
		s.provider_bs_max_num_users.resize(s.num_providers);
		for (std::size_t p = 0; p < s.num_providers; ++p)
		{
			s.provider_bs_max_num_users[p].resize(s.num_bs_types);
			std::fill(s.provider_bs_max_num_users[p].begin(), s.provider_bs_max_num_users[p].end(), default_provider_bs_max_num_users);
		}
	}
	if (s.provider_user_min_downlink_data_rates.size() == 0)
	{
		s.provider_user_min_downlink_data_rates.resize(s.num_providers);
		for (std::size_t p = 0; p < s.num_providers; ++p)
		{
			s.provider_user_min_downlink_data_rates[p].resize(s.num_user_classes);
			std::fill(s.provider_user_min_downlink_data_rates[p].begin(), s.provider_user_min_downlink_data_rates[p].end(), default_provider_user_min_downlink_data_rate);
		}
	}
	if (s.provider_user_revenues.size() == 0)
	{
		s.provider_user_revenues.resize(s.num_providers);
		for (std::size_t p = 0; p < s.num_providers; ++p)
		{
			s.provider_user_revenues[p].resize(s.num_user_classes);
			std::fill(s.provider_user_revenues[p].begin(), s.provider_user_revenues[p].end(), default_provider_user_revenue);
		}
	}
	if (s.provider_user_mixes.size() == 0)
	{
		s.provider_user_mixes.resize(s.num_providers);
		for (std::size_t p = 0; p < s.num_providers; ++p)
		{
			s.provider_user_mixes[p].resize(s.num_user_classes);
			std::fill(s.provider_user_mixes[p].begin(), s.provider_user_mixes[p].end(), default_provider_user_mix);
		}
	}
	if (s.user_downlink_data_rates.size() == 0)
	{
		s.user_downlink_data_rates.resize(s.num_user_classes);
		std::fill(s.user_downlink_data_rates.begin(), s.user_downlink_data_rates.end(), default_user_downlink_data_rate);
	}
	if (s.user_thermal_noises.size() == 0)
	{
		s.user_thermal_noises.resize(s.num_user_classes);
		std::fill(s.user_thermal_noises.begin(), s.user_thermal_noises.end(), default_user_thermal_noise);
	}

	// Consistency checks
	DCS_ASSERT(s.num_providers > 0,
			   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of providers"));
	DCS_ASSERT(s.num_user_classes > 0,
			   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of user classes"));
	DCS_ASSERT(s.num_bs_types > 0,
			   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of BS types"));
	DCS_ASSERT(s.topology_area_size.first > 0 && s.topology_area_size.second > 0,
			   DCS_EXCEPTION_THROW(std::runtime_error, "Wrong area size"));
	DCS_ASSERT(s.num_bs_types == s.bs_heights.size(),
			   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of BS types in BS heights"));
	DCS_ASSERT(s.num_bs_types == s.bs_carrier_frequencies.size(),
			   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of BS types in BS carrier frequencies"));
//	DCS_ASSERT(s.num_bs_types == s.bs_max_channel_capacities.size(),
//			   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of BS types in BS max channel capacities"));
	DCS_ASSERT(s.num_bs_types == s.bs_bandwidths.size(),
			   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of BS types in BS bandwidths"));
//	DCS_ASSERT(s.num_bs_types == s.bs_num_subcarriers.size(),
//			   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of BS types in BS number of subcarriers"));
	DCS_ASSERT(s.num_bs_types == s.bs_transmission_powers.size(),
			   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of BS types in BS transmission powers"));
	DCS_ASSERT(s.num_bs_types == s.bs_idle_powers.size(),
			   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of BS types in BS idle power consumptions"));
	DCS_ASSERT(s.num_bs_types == s.bs_load_powers.size(),
			   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of BS types in BS load-dependent power consumptions"));
	DCS_ASSERT(s.num_providers == s.provider_num_bss.size(),
			   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of providers in number of BSs by providers"));
	if (s.num_providers > 0)
	{
		DCS_ASSERT(s.num_bs_types == s.provider_num_bss[0].size(),
				   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of BS types in number of BSs by providers"));
	}
	DCS_ASSERT(s.num_providers == s.provider_electricity_costs.size(),
			   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of providers in electricity costs by provider"));
	DCS_ASSERT(s.num_providers == s.provider_alt_electricity_costs.size(),
			   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of providers in alternative electricity costs by provider"));
	DCS_ASSERT(s.num_providers == s.provider_alt_electricity_time_ranges.size(),
			   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of providers in alternative electricity time ranges by provider"));
//	DCS_ASSERT(s.num_providers == s.provider_bs_coalition_change_costs.size(),
//			   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of coalition change costs, per provider"));
	DCS_ASSERT(s.num_providers == s.provider_coalition_costs.size(),
			   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of providers in coalition costs by provider"));
	DCS_ASSERT(s.num_providers == s.provider_bs_traffic_curves.size(),
			   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of providers in traffic curve types by provider"));
	if (s.num_providers > 0)
	{
		DCS_ASSERT(s.num_bs_types == s.provider_bs_traffic_curves[0].size(),
				   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of BS types in traffic curve types by provider"));
	}
//	DCS_ASSERT(s.num_providers == s.provider_bs_traffic_data_sizes.size(),
//			   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of providers in traffic data sizes by provider"));
//	if (s.num_providers > 0)
//	{
//		DCS_ASSERT(s.num_bs_types == s.provider_bs_traffic_data_sizes[0].size(),
//				   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of BS types in traffic data sizes by provider"));
//	}
	DCS_ASSERT(s.num_providers == s.provider_bs_traffic_data.size(),
			   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of providers in traffic data by provider"));
	if (s.num_providers > 0)
	{
		DCS_ASSERT(s.num_bs_types == s.provider_bs_traffic_data[0].size(),
				   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of BS types in traffic data by provider"));
	}
	DCS_ASSERT(s.num_providers == s.provider_bs_amortizing_costs.size(),
			   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of providers in amortizing costs by provider"));
	if (s.num_providers > 0)
	{
		DCS_ASSERT(s.num_bs_types == s.provider_bs_amortizing_costs[0].size(),
				   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of BS types in amortizing costs by provider"));
	}
	DCS_ASSERT(s.num_providers == s.provider_bs_residual_amortizing_times.size(),
			   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of providers in residual amortizing times by provider"));
	if (s.num_providers > 0)
	{
		DCS_ASSERT(s.num_bs_types == s.provider_bs_residual_amortizing_times[0].size(),
				   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of BS types in residual amortizing times by provider"));
	}
	DCS_ASSERT(s.num_providers == s.provider_bs_max_num_users.size(),
			   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of providers in BS max number of users by provider"));
	if (s.num_providers > 0)
	{
		DCS_ASSERT(s.num_bs_types == s.provider_bs_max_num_users[0].size(),
				   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of BS types in BS max number of users by provider"));
	}
	DCS_ASSERT(s.num_providers == s.provider_user_min_downlink_data_rates.size(),
			   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of providers in user min downlink data rates by provider"));
	if (s.num_providers > 0)
	{
		DCS_ASSERT(s.num_user_classes == s.provider_user_min_downlink_data_rates[0].size(),
				   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of user classes in user min downlink data rates by provider"));
	}
	DCS_ASSERT(s.num_providers == s.provider_user_revenues.size(),
			   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of providers in user revenues by provider"));
	if (s.num_providers > 0)
	{
		DCS_ASSERT(s.num_user_classes == s.provider_user_revenues[0].size(),
				   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of user classes in user revenues by provider"));
	}
	DCS_ASSERT(s.num_providers == s.provider_user_mixes.size(),
			   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of providers in user mixes by provider"));
	if (s.num_providers > 0)
	{
		DCS_ASSERT(s.num_user_classes == s.provider_user_mixes[0].size(),
				   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of user classes in user mixes by provider"));
	}
	DCS_ASSERT(s.num_user_classes == s.user_downlink_data_rates.size(),
			   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of user classes in desired downlink data rates by user class"));
	DCS_ASSERT(s.num_user_classes == s.user_thermal_noises.size(),
			   DCS_EXCEPTION_THROW(std::runtime_error, "Unexpected number of user classes in thermal noises by user class"));

	return s;
}

template <typename RealT, typename RNGT>
void run_experiment(scenario<RealT> const& scen, options<RealT> const& opts, RNGT& rng)
{
	boost::timer timer;

	std::cout << "- Scenario: " << scen << std::endl;
	std::cout << "- Options: " << opts << std::endl;

	if (opts.simulation_mode == reference_profit_estimation_simulation_mode
		|| opts.simulation_mode == both_simulation_mode)
	{
		timer.restart();

		//TODO: add a CLI option for the granularity
		//const RealT time_step = 30.0/60.0; // 30 minute
		//const RealT time_step = 15.0/60.0; // 15 minute
		//const RealT time_step = 5.0/60.0; // 5 minute
		const RealT time_step = 1.0/60.0; // 1 minute

		std::cout << "**** ESTIMATING REFERENCE ALONE PROFITS (granularity: " << time_step << " hours" << std::endl;

		std::vector<RealT> ref_alone_profits;
		std::vector<RealT> ref_alone_num_served_users;
		ref_alone_profits = estimate_alone_profits< weekly_traffic_pattern<RealT> >(time_step, scen, opts, rng, ref_alone_num_served_users);

		std::cout << "**** ELAPSED TIME: " << timer.elapsed() << std::endl;
		std::cout << "****************************************************************" << std::endl;
	}
	if (opts.simulation_mode == coalition_formation_simulation_mode
		|| opts.simulation_mode == both_simulation_mode)
	{
		for (std::size_t i = 0; i < opts.time_steps.size(); ++i)
		{
			timer.restart();

			std::cout << "**** PEAK EVERY " << opts.time_steps[i] << "h SCENARIO" << std::endl;

			if (opts.time_span <= 24.0)
			{
				peak_every_x_hours_scenario< daily_traffic_pattern<RealT> >(opts.time_steps[i], scen, opts, rng);
			}
			else
			{
				peak_every_x_hours_scenario< weekly_traffic_pattern<RealT> >(opts.time_steps[i], scen, opts, rng);
			}

			std::cout << "**** ELAPSED TIME: " << timer.elapsed() << std::endl;
			std::cout << "****************************************************************" << std::endl;
		}
	}
}

} // Namespace experiment

void usage(char const* progname)
{
	std::cerr << "Usage: " << progname << " [options]" << std::endl
			  << "Options:" << std::endl
			  << "--help" << std::endl
			  << "  Show this message." << std::endl
			  << "--assignment {'max-profit'|'min-cost'}" << std::endl
			  << "  User assignment strategy category, where:" << std::endl
			  << "  * 'max-profit' refers to the strategy based on profit maximization;" << std::endl
			  << "  * 'min-cost' refers to the strategy based on cost minimization." << std::endl
			  << "--ci-level <num>" << std::endl
			  << "  Level for the confidence intervals (must be a number in [0,1])." << std::endl
			  << "--ci-rel-precision <num>" << std::endl
			  << "  Relative precision for the half-width of the confidence intervals (must be a number in [0,1])." << std::endl
			  << "--find-all-parts" << std::endl
			  << "  For each time interval, find all possible stable partitions." << std::endl
			  << "--formation {'nash'|'pareto'|'social'}" << std::endl
			  << "  Coalition formation category, where:" << std::endl
			  << "  * 'nash' refers to the Nash-stable coalition formation;" << std::endl
			  << "  * 'pareto' refers to the Pareto Optimal coalition formation;" << std::endl
			  << "  * 'social' refers to the Social Optimum coalition formation;" << std::endl
			  << "--milp-relgap <num>" << std::endl
			  << "  A real number in [0,1] used to set the relative gap parameter of the MILP solver." << std::endl
			  << "--milp-tilim <num>" << std::endl
			  << "  A real positive number used to set the maximum number of seconds to wait for the termination of the MILP solver." << std::endl
			  << "--output-stats-file <file>" << std::endl
			  << "  The output file where writing statistics." << std::endl
			  << "--output-trace-file <file>" << std::endl
			  << "  The output file where writing run-trace information." << std::endl
			  << "--payoff {'banzhaf'|'chi'|'norm-banzhaf'|'shapley'}" << std::endl
			  << "  Payoff division category, where:" << std::endl
			  << "  * 'banzhaf' refers to the Banzhaf value;" << std::endl
			  << "  * 'chi' refers to the Chi value as defined by (Casajus,2009)." << std::endl
			  << "  * 'norm-banzhaf' refers to the normalized Banzhaf value;" << std::endl
			  << "  * 'shapley' refers to the Shapley value." << std::endl
//			  << "--partpref <str>: The type of preference relation to use for selecting the best CIPs partition." << std::endl
//			  << "   Possible values for <str> are:" << std::endl
//			  << "   - 'util', for utilitarian partition preference" << std::endl
//			  << "   - 'pareto', for Pareto partition preference" << std::endl
			  << "--rng-seed <num>" << std::endl
			  << "  Set the seed to use for random number generation." << std::endl
			  << "--scenario <file>" << std::endl
			  << "  The path to the file describing the scenario to use for the experiment." << std::endl
			  << "--sim-mode {'estimation'|'formation'}" << std::endl
			  << "  The simulation mode to use:" << std::endl
			  << "  * 'reference-profit': run the simulation to estimate the profits of each single provider;" << std::endl
			  << "  * 'coalition-formation': run the coalition formation simulation;" << std::endl
			  << "  * 'both': run both the simulations;" << std::endl
			  << "--traffic-discretization {'max'|'min'|'median'}" << std::endl
			  << "  The statistic to use for traffic load discretization:" << std::endl
			  << "  * 'max': use the maximum value on the curve;" << std::endl
			  << "  * 'min': use the minimum value on the curve;" << std::endl
			  << "  * 'median': use the median value on the curve;" << std::endl
			  << "--tstep <num>" << std::endl
			  << "  A real positive number representing the time interval for coalition formation computation." << std::endl
			  << "--tspan {'day'|'week'}" << std::endl
			  << "  The time interval along which the experiment spans, where:" << std::endl
			  << "  * hour: the experiment spans for 1 hour" << std::endl
			  << "  * day: the experiment spans for 1 day" << std::endl
			  << "  * week: the experiment spans for 1 week" << std::endl
			  << "--verbosity <num>" << std::endl
			  << "  An integer number in [0,9] representing the verbosity level (0 for 'minimum verbosity' and 9 for 'maximum verbosity)." << std::endl
			  << "--verbosity <num>" << std::endl
			  << "  An integer number in [0,9] representing the verbosity level (0 for 'minimum verbosity' and 9 for 'maximum verbosity)." << std::endl
			  << std::endl;
}

}} // Namespace <unnamed>::detail


int main(int argc, char* argv[])
{
	typedef double real_type;

	bool opt_help;
	bs::coalition_formation_category opt_coalition_formation;
	bs::coalition_value_division_category opt_coalition_value_division;
	detail::experiment::user_assignment_category opt_user_assignment;
	bs::partition_preference_category opt_partition_pref;
	real_type opt_milp_relative_gap;
	real_type opt_milp_time_limit;
	unsigned long opt_rng_seed;
	std::string opt_scenario_file;
	real_type opt_time_span;;
	std::vector<real_type> opt_time_steps;
	bool opt_find_all_best_partitions;
	std::string opt_out_stats_dat_file;
	std::string opt_out_trace_dat_file;
	real_type opt_ci_level;;
	real_type opt_ci_rel_precision;
	short opt_verbosity;
	detail::experiment::simulation_mode_category opt_sim_mode;
	detail::experiment::traffic_load_discretization_category opt_traffic_load_discretization;
	std::string opt_str;

	// Parse CLI options
	DCS_DEBUG_TRACE("Parse CLI options...");//XXX
	opt_help = cli::simple::get_option(argv, argv+argc, "--help");
	if (opt_help)
	{
		detail::usage(argv[0]);
		return 0;
	}
	opt_str = cli::simple::get_option<std::string>(argv, argv+argc, "--formation", "nash");
	if (opt_str == "nash")
	{
		opt_coalition_formation = bs::nash_stable_coalition_formation;
	}
	else if (opt_str == "pareto")
	{
		opt_coalition_formation = bs::pareto_optimal_coalition_formation;
	}
	else if (opt_str == "social")
	{
		opt_coalition_formation = bs::social_optimum_coalition_formation;
	}
	else
	{
		dcs::log_error(DCS_LOGGING_AT, "Unknown coalition formation category.");
		detail::usage(argv[0]);
		return 1;
	}
	opt_str = cli::simple::get_option<std::string>(argv, argv+argc, "--payoff", "shapley");
	if (opt_str == "shapley")
	{
		opt_coalition_value_division = bs::shapley_coalition_value_division;
	}
	else if (opt_str == "banzhaf")
	{
		opt_coalition_value_division = bs::banzhaf_coalition_value_division;
	}
	else if (opt_str == "norm-banzhaf")
	{
		opt_coalition_value_division = bs::normalized_banzhaf_coalition_value_division;
	}
	else if (opt_str == "chi")
	{
		opt_coalition_value_division = bs::chi_coalition_value_division;
	}
	else
	{
		dcs::log_error(DCS_LOGGING_AT, "Unknown coalition value division category.");
		detail::usage(argv[0]);
		return 1;
	}
	opt_str = cli::simple::get_option<std::string>(argv, argv+argc, "--assignment", "min-cost");
	if (opt_str == "max-profit")
	{
		opt_user_assignment = detail::experiment::max_profit_user_assignment;
	}
	else if (opt_str == "min-cost")
	{
		opt_user_assignment = detail::experiment::min_cost_user_assignment;
	}
	else
	{
		dcs::log_error(DCS_LOGGING_AT, "Unknown user assignment category.");
		detail::usage(argv[0]);
		return -1;
	}
	opt_str = cli::simple::get_option<std::string>(argv, argv+argc, "--partpref", "util");
	if (opt_str == "pareto")
	{
		opt_partition_pref = bs::pareto_partition_preference;
	}
	else if (opt_str == "util")
	{
		opt_partition_pref = bs::utilitarian_partition_preference;
	}
	else
	{
		dcs::log_error(DCS_LOGGING_AT, "Unknown partition preference category.");
		detail::usage(argv[0]);
		return -1;
	}
	opt_milp_relative_gap = cli::simple::get_option<real_type>(argv, argv+argc, "--milp-relgap", 0);
	opt_milp_time_limit = cli::simple::get_option<real_type>(argv, argv+argc, "--milp-tilim", -1);
	opt_rng_seed = cli::simple::get_option<unsigned long>(argv, argv+argc, "--rng-seed", 5489);
	opt_scenario_file = cli::simple::get_option<std::string>(argv, argv+argc, "--scenario");
	opt_str = cli::simple::get_option<std::string>(argv, argv+argc, "--tspan", "day");
	if (opt_str == "hour")
	{
		opt_time_span = 1;
	}
	else if (opt_str == "day")
	{
		opt_time_span = 24;
	}
	else if (opt_str == "week")
	{
		opt_time_span = 24*7;
	}
	else
	{
		dcs::log_error(DCS_LOGGING_AT, "Unknown time span.");
		detail::usage(argv[0]);
		return -1;
	}
	opt_time_steps = cli::simple::get_options<real_type>(argv, argv+argc, "--tstep", 1.0);
	opt_find_all_best_partitions = cli::simple::get_option(argv, argv+argc, "--find-all-parts");
	opt_out_stats_dat_file = cli::simple::get_option<std::string>(argv, argv+argc, "--out-stats-file");
	opt_out_trace_dat_file = cli::simple::get_option<std::string>(argv, argv+argc, "--out-trace-file");
	opt_ci_level = cli::simple::get_option<real_type>(argv, argv+argc, "--ci-level", 0.95);
	opt_ci_rel_precision = cli::simple::get_option<real_type>(argv, argv+argc, "--ci-rel-precision", 0.04);
	opt_verbosity = cli::simple::get_option<short>(argv, argv+argc, "--verbosity", VERBOSITY_NONE);
	if (opt_verbosity < 0)
	{
		opt_verbosity = 0;
	}
	else if (opt_verbosity > 9)
	{
		opt_verbosity = 9;
	}
	opt_str = cli::simple::get_option<std::string>(argv, argv+argc, "--sim-mode", "both");
	if (opt_str == "reference-profit")
	{
		opt_sim_mode = detail::experiment::reference_profit_estimation_simulation_mode;
	}
	else if (opt_str == "coalition-formation")
	{
		opt_sim_mode = detail::experiment::coalition_formation_simulation_mode;
	}
	else if (opt_str == "both")
	{
		opt_sim_mode = detail::experiment::both_simulation_mode;
	}
	else
	{
		dcs::log_error(DCS_LOGGING_AT, "Unknown simulation mode.");
		detail::usage(argv[0]);
		return -1;
	}
	opt_str = cli::simple::get_option<std::string>(argv, argv+argc, "--traffic-discretization", "max");
	if (opt_str == "max")
	{
		opt_traffic_load_discretization = detail::experiment::max_traffic_load_discretization;
	}
	else if (opt_str == "min")
	{
		opt_traffic_load_discretization = detail::experiment::min_traffic_load_discretization;
	}
	else if (opt_str == "median")
	{
		opt_traffic_load_discretization = detail::experiment::median_traffic_load_discretization;
	}
	else
	{
		dcs::log_error(DCS_LOGGING_AT, "Unknown traffic load discretization.");
		detail::usage(argv[0]);
		return -1;
	}

	// Check CLI options
	if (opt_scenario_file.empty())
	{
		std::cerr << "(E) Scenario file not specified" << std::endl;
		detail::usage(argv[0]);
		return -1;
	}

	// Prepare the experiment
	DCS_DEBUG_TRACE("Preparing the experiment...");//XXX
	detail::experiment::scenario<real_type> scenario;
	scenario = detail::experiment::make_scenario<real_type>(opt_scenario_file);
	detail::experiment::options<real_type> options;
	options.ci_level = opt_ci_level;
	options.ci_rel_precision = opt_ci_rel_precision;
	options.coalition_formation = opt_coalition_formation;
	options.coalition_value_division = opt_coalition_value_division;
	options.find_all_best_partitions = opt_find_all_best_partitions;
	options.milp_relative_gap = opt_milp_relative_gap;
	options.milp_time_limit = opt_milp_time_limit;
	options.output_stats_data_file = opt_out_stats_dat_file;
	options.output_trace_data_file = opt_out_trace_dat_file;
	options.partition_preference = opt_partition_pref;
	options.simulation_mode = opt_sim_mode;
	options.rng_seed = opt_rng_seed;
	options.time_span = opt_time_span;
	options.time_steps = opt_time_steps;
	options.traffic_load_discretization = opt_traffic_load_discretization;
	options.user_assignment = opt_user_assignment;
	options.verbosity = opt_verbosity;

	boost::random::mt19937 rng(options.rng_seed);

	// Run the experiment
	DCS_DEBUG_TRACE("Run the experiment...");//XXX
	detail::experiment::run_experiment(scenario, options, rng);
}
