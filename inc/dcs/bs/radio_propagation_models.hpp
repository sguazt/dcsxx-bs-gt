/**
 * \file inc/dcs/bs/path_loss_models.hpp
 *
 * \brief Models for radio propagation.
 *
 * The Okumura-Hata and COST-231 Hata models are taken from [1].
 *
 * References:
 * -# Saunders et al. "Antennas And Propagation For Wireless Communication Systems." 2nd Edition, Wiley, 2007.
 * .
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

#ifndef DCS_BS_RADIO_PROPAGATION_MODELS_HPP
#define DCS_BS_RADIO_PROPAGATION_MODELS_HPP


#include <cmath>


namespace dcs { namespace bs {

enum radio_propagation_area_category
{
	urban_small_radio_propagation_area,
	urban_medium_radio_propagation_area,
	urban_large_radio_propagation_area,
	suburban_radio_propagation_area,
	rural_open_radio_propagation_area,
	rural_quasi_open_radio_propagation_area
};

namespace detail { namespace /*<unnamed>*/ {

template <typename RealT>
RealT hata_correction_factor(RealT f, RealT hr, radio_propagation_area_category area)
{
	RealT C = 0;

	switch (area)
	{
		case urban_large_radio_propagation_area:
			if (f >= 150 && f < 300)
			{
				const RealT l = ::std::log10(1.54*hr);
				C = 8.29*l*l - 1.1;
			}
			else /* if (f <= 1500) */
			{
				const RealT l = ::std::log10(11.75*hr);
				C = 3.2*l*l - 4.97;
			}
			break;
		case urban_small_radio_propagation_area:
		case urban_medium_radio_propagation_area:
			{
				const RealT l = ::std::log10(f);
				C = (1.1*l-0.7)*hr - (1.56*l-0.8);
			}
			break;
		case suburban_radio_propagation_area:
			{
				const RealT l = ::std::log10(f/28.0);
				C = 2*l*l + 5.4;
			}
			break;
		case rural_open_radio_propagation_area:
		case rural_quasi_open_radio_propagation_area:
			{
				const RealT l = ::std::log10(f);
				C = 4.78*l*l - 18.33*l + 40.94;
			}
			break;
	}

	return C;
}

}} // Namespace detail::<unnamed>

/**
 * Compute the path loss according to a simple distance-based model.
 *
 * \tparam RealT The type for real numbers
 * \param e The path loss exponent
 * \param d The distance between the transmitter and the receiver (in km)
 * \return The path loss in dB
 */
template <typename RealT>
RealT distance_based_path_loss_model(RealT e, RealT d)
{
	return 10.0*e*::std::log10(d);
}

/**
 * Compute the path loss according to the Okumura-Hata model.
 *
 * \tparam RealT The type for real numbers
 * \param f The carrier frequency (in MHz)
 * \param hb The BS height (in m)
 * \param hr The receiver height (in m)
 * \param d The distance between the transmitter and the receiver (in km)
 * \return The path loss in dB
 */
template <typename RealT>
RealT okumura_hata_path_loss_model(RealT f, RealT hb, RealT hr, RealT d, radio_propagation_area_category area)
{
	const RealT lhb = ::std::log10(hb);
	const RealT C = detail::hata_correction_factor(f, hr, area);

	return 69.55 + 26.16*::std::log10(f) - 13.82*lhb + (44.9-6.55*lhb)*::std::log10(d) - C;
}

/**
 * Compute the path loss according to the COST-231 Hata model.
 *
 * Note, this model should only be used for urban area.
 *
 * \tparam RealT The type for real numbers
 * \param f The carrier frequency (in MHz)
 * \param hb The BS height (in m)
 * \param hr The receiver height (in m)
 * \param d The distance between the transmitter and the receiver (in km)
 * \param area The category of area where the radio signal propagates
 * \return The path loss in dB
 */
template <typename RealT>
RealT cost231_hata_path_loss_model(RealT f, RealT hb, RealT hr, RealT d, radio_propagation_area_category area)
{
	if (area == rural_open_radio_propagation_area
		|| area == rural_quasi_open_radio_propagation_area)
	{
		return okumura_hata_path_loss_model(f, hb, hr, d, area);
	}

	const RealT lhb = ::std::log10(hb);
	const RealT C = detail::hata_correction_factor(f, hr, urban_medium_radio_propagation_area);

	RealT Cm = 0;
	switch (area)
	{
		case urban_large_radio_propagation_area:
			Cm = 3;
			break;
		case urban_small_radio_propagation_area:
		case urban_medium_radio_propagation_area:
		case suburban_radio_propagation_area:
			Cm = 0;
			break;
		default:
		//case rural_open_radio_propagation_area:
		//case rural_quasi_open_radio_propagation_area:
			break;
	}

//DCS_DEBUG_TRACE("COST-231 Hata Model: f: " << f << ", hb: " << hb << ", hr: " << hr << ", d: " << d << ", area: " << area << ", C: " << C << ", Cm: " << Cm);///XXX

	//return 46.3 + 33.9*::std::log10(f) - 13.82*log10(hb) + (44.9-6.55*log10(hb))*::std::log10(d) - ((1.1*::std::log10(f)-0.7)*hr-(1.56*::std::log10(f)-0.8)) + 3;
	return 46.3 + 33.9*::std::log10(f) - 13.82*lhb + (44.9-6.55*lhb)*::std::log10(d) - C + Cm;
}

}} // Namespace dcs::bs

#endif // DCS_BS_RADIO_PROPAGATION_MODELS_HPP
