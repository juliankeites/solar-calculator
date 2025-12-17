import json
import warnings
from datetime import datetime, date, timedelta
import time as time_module  # Avoid naming conflict
from datetime import time as datetime_time  # Import time class from datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy import optimize

warnings.filterwarnings("ignore")

# --- YOUR LIVE SYSTEM DEFAULTS ---
LAT = 51.32
LON = -0.56
LIVE_PEAK_KW = 1.8

# Battery system defaults (Tesla Powerwall 3)
BATTERY_CAPACITY_KWH = 13.5  # Tesla Powerwall 3
BATTERY_EFFICIENCY = 0.95
BATTERY_MAX_CHARGE_RATE_KW = 5.0
BATTERY_MAX_DISCHARGE_RATE_KW = 5.0
BASELOAD_KW = 0.4
MIN_SOLAR_THRESHOLD = 20  # W/m² minimum for production

# Tariff defaults
EXPORT_TARIFF_GBP_PER_KWH = 0.15  # 15p/kWh
LOW_TARIFF_START = datetime_time(23, 30)  # 23:30
LOW_TARIFF_END = datetime_time(5, 30)    # 05:30
PEAK_FACTOR = 2.0  # Peak hours multiplier

# UK Power Price API
UK_POWER_PRICE_API_URL = "https://prices.fly.dev/prices"

arrays = [
    {"name": "NE", "n_modules": 7, "tilt": 45, "azimuth": 30, "loss_factor": 0.85},
    {"name": "East", "n_modules": 7, "tilt": 45, "azimuth": 115, "loss_factor": 0.85},
    {"name": "South", "n_modules": 5, "tilt": 45, "azimuth": 210, "loss_factor": 0.85},
    {"name": "West", "n_modules": 8, "tilt": 45, "azimuth": 296, "loss_factor": 0.85},
]

P_MODULE_WP = 440
TOTAL_KWP = 11.88
SYSTEM_EFFICIENCY = 0.85


class UKPowerPricePredictor:
    """Fetch and process UK power price data from the IOG API"""
    
    def __init__(self):
        self.base_url = UK_POWER_PRICE_API_URL
        self.timezone = "Europe/London"
        
    @st.cache_data(ttl=1800)  # Cache for 30 minutes (prices update every 30 min)
    def fetch_prices(_self, days_ahead=2):
        """Fetch power prices from the API for the next N days"""
        try:
            # First, get current date in UK time
            uk_time = datetime.now()
            
            # Make API request - the API returns last 48h actual + 24h forecast by default
            response = requests.get(_self.base_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data or 'data' not in data:
                st.warning("Price API returned empty response. Using simulated price data.")
                return _self._generate_simulated_prices(days_ahead)
            
            # Process the data
            prices_data = []
            
            # The API returns data in settlement periods (30-min intervals)
            # We need to convert to hourly and organize by date/time
            for entry in data.get('data', []):
                # Parse the entry - structure may vary
                if isinstance(entry, dict):
                    # Try to extract date and settlement period
                    if 'date' in entry and 'settlementPeriod' in entry:
                        date_str = entry['date']
                        settlement_period = entry['settlementPeriod']
                        price = entry.get('imbalancePrice', 0)
                        
                        # Convert settlement period to datetime
                        # Settlement period 1 = 00:00-00:30, 2 = 00:30-01:00, etc.
                        hour = (settlement_period - 1) // 2
                        minute = 0 if settlement_period % 2 == 1 else 30
                        
                        try:
                            timestamp = datetime.strptime(date_str, "%Y-%m-%d")
                            timestamp = timestamp.replace(hour=hour, minute=minute)
                            
                            # Check if this is forecast or actual
                            is_forecast = entry.get('forecast', True)
                            
                            prices_data.append({
                                'timestamp': timestamp,
                                'price_gbp_per_mwh': price,
                                'settlement_period': settlement_period,
                                'is_forecast': is_forecast,
                                'is_actual': not is_forecast
                            })
                        except ValueError as e:
                            continue
            
            # If no valid data found, generate simulated
            if not prices_data:
                return _self._generate_simulated_prices(days_ahead)
            
            # Create DataFrame
            df = pd.DataFrame(prices_data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Resample to hourly (average of two settlement periods)
            hourly_df = df.resample('H').agg({
                'price_gbp_per_mwh': 'mean',
                'is_forecast': lambda x: x.mode()[0] if len(x.mode()) > 0 else True,
                'is_actual': lambda x: x.mode()[0] if len(x.mode()) > 0 else False
            })
            
            # Ensure we have enough data points
            if len(hourly_df) < 24:
                # Pad with simulated data if needed
                return _self._generate_simulated_prices(days_ahead)
            
            # Convert £/MWh to £/kWh for easier comparison with solar
            hourly_df['price_gbp_per_kwh'] = hourly_df['price_gbp_per_mwh'] / 1000
            
            return hourly_df
            
        except requests.exceptions.RequestException as e:
            st.warning(f"Failed to fetch power prices: {str(e)}. Using simulated price data.")
            return _self._generate_simulated_prices(days_ahead)
        except Exception as e:
            st.warning(f"Error processing price data: {str(e)}. Using simulated price data.")
            return _self._generate_simulated_prices(days_ahead)
    
    def _generate_simulated_prices(self, days_ahead):
        """Generate realistic simulated price data when API fails"""
        # Generate time range for next N days
        now = datetime.now()
        start_time = datetime(now.year, now.month, now.day, 0, 0, 0)
        hours = days_ahead * 24
        
        timestamps = pd.date_range(
            start=start_time, 
            periods=hours, 
            freq='H',
            tz=self.timezone
        )
        
        # Create realistic price pattern based on time of day
        # Prices typically higher in morning (7-9) and evening (16-19)
        # Lower overnight and midday
        hour_of_day = np.array([t.hour for t in timestamps])
        
        # Base price pattern
        base_price = np.ones_like(hour_of_day, dtype=float) * 50  # Base £50/MWh
        
        # Morning peak (7-9)
        morning_mask = (hour_of_day >= 7) & (hour_of_day <= 9)
        base_price[morning_mask] += 30
        
        # Evening peak (16-19)
        evening_mask = (hour_of_day >= 16) & (hour_of_day <= 19)
        base_price[evening_mask] += 40
        
        # Overnight low (23-5)
        overnight_mask = (hour_of_day >= 23) | (hour_of_day <= 5)
        base_price[overnight_mask] -= 20
        
        # Add some randomness
        noise = np.random.normal(0, 10, len(base_price))
        base_price += noise
        
        # Ensure positive prices
        base_price = np.maximum(base_price, 10)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price_gbp_per_mwh': base_price,
            'price_gbp_per_kwh': base_price / 1000,
            'is_forecast': np.ones(len(base_price), dtype=bool),
            'is_actual': np.zeros(len(base_price), dtype=bool)
        })
        
        df.set_index('timestamp', inplace=True)
        
        # Mark first 24 hours as actual (if within current day)
        current_hour = now.hour
        if current_hour < 24:
            actual_mask = df.index.hour <= current_hour
            df.loc[actual_mask, 'is_forecast'] = False
            df.loc[actual_mask, 'is_actual'] = True
        
        return df
    
    def calculate_price_statistics(self, price_df):
        """Calculate statistics from price data"""
        if price_df.empty:
            return {}
        
        # Basic statistics
        stats = {
            'current_price': price_df['price_gbp_per_mwh'].iloc[0] if len(price_df) > 0 else 0,
            'avg_price_today': price_df['price_gbp_per_mwh'][:24].mean() if len(price_df) >= 24 else price_df['price_gbp_per_mwh'].mean(),
            'min_price_today': price_df['price_gbp_per_mwh'][:24].min() if len(price_df) >= 24 else price_df['price_gbp_per_mwh'].min(),
            'max_price_today': price_df['price_gbp_per_mwh'][:24].max() if len(price_df) >= 24 else price_df['price_gbp_per_mwh'].max(),
            'total_hours': len(price_df),
            'actual_hours': price_df['is_actual'].sum(),
            'forecast_hours': price_df['is_forecast'].sum(),
        }
        
        # Calculate price percentiles
        prices = price_df['price_gbp_per_mwh'].values
        stats['percentile_25'] = np.percentile(prices, 25)
        stats['percentile_50'] = np.percentile(prices, 50)  # Median
        stats['percentile_75'] = np.percentile(prices, 75)
        
        # Identify low price periods (< 25th percentile)
        low_price_threshold = stats['percentile_25']
        low_price_hours = price_df[price_df['price_gbp_per_mwh'] < low_price_threshold]
        stats['low_price_hours_count'] = len(low_price_hours)
        stats['low_price_threshold'] = low_price_threshold
        
        # Identify high price periods (> 75th percentile)
        high_price_threshold = stats['percentile_75']
        high_price_hours = price_df[price_df['price_gbp_per_mwh'] > high_price_threshold]
        stats['high_price_hours_count'] = len(high_price_hours)
        stats['high_price_threshold'] = high_price_threshold
        
        # Find best and worst price times
        if not price_df.empty:
            min_price_idx = price_df['price_gbp_per_mwh'].idxmin()
            max_price_idx = price_df['price_gbp_per_mwh'].idxmax()
            
            stats['best_time'] = min_price_idx.strftime('%H:%M')
            stats['best_price'] = price_df['price_gbp_per_mwh'].loc[min_price_idx]
            stats['worst_time'] = max_price_idx.strftime('%H:%M')
            stats['worst_price'] = price_df['price_gbp_per_mwh'].loc[max_price_idx]
        
        # Calculate price trend (current vs average)
        if stats['current_price'] > 0 and stats['avg_price_today'] > 0:
            stats['price_trend'] = (stats['current_price'] - stats['avg_price_today']) / stats['avg_price_today'] * 100
        else:
            stats['price_trend'] = 0
        
        return stats
    
    def identify_smart_charging_windows(self, price_df, solar_power_kw=None, battery_soc=None):
        """Identify optimal windows for smart charging based on prices"""
        if price_df.empty:
            return []
        
        # Get low price periods
        low_price_threshold = np.percentile(price_df['price_gbp_per_mwh'].values, 25)
        low_price_periods = price_df[price_df['price_gbp_per_mwh'] < low_price_threshold]
        
        # Group consecutive low price hours
        charging_windows = []
        current_window = []
        
        for idx, row in low_price_periods.iterrows():
            if not current_window:
                current_window = [idx]
            else:
                # Check if this hour is consecutive with the last
                last_time = current_window[-1]
                if (idx - last_time).total_seconds() <= 3600:  # 1 hour difference
                    current_window.append(idx)
                else:
                    # End of current window
                    if len(current_window) >= 1:  # Minimum 1 hour window
                        window_data = {
                            'start': current_window[0],
                            'end': current_window[-1],
                            'duration_hours': len(current_window),
                            'avg_price': price_df.loc[current_window, 'price_gbp_per_mwh'].mean(),
                            'min_price': price_df.loc[current_window, 'price_gbp_per_mwh'].min(),
                            'max_price': price_df.loc[current_window, 'price_gbp_per_mwh'].max(),
                        }
                        charging_windows.append(window_data)
                    current_window = [idx]
        
        # Add the last window if it exists
        if current_window and len(current_window) >= 1:
            window_data = {
                'start': current_window[0],
                'end': current_window[-1],
                'duration_hours': len(current_window),
                'avg_price': price_df.loc[current_window, 'price_gbp_per_mwh'].mean(),
                'min_price': price_df.loc[current_window, 'price_gbp_per_mwh'].min(),
                'max_price': price_df.loc[current_window, 'price_gbp_per_mwh'].max(),
            }
            charging_windows.append(window_data)
        
        return charging_windows


class EnhancedSolarEstimator:
    """Enhanced solar production estimator with multiple methods"""

    def __init__(self, lat, lon, arrays):
        self.lat = lat
        self.lon = lon
        self.arrays = arrays
        self.total_kwp = sum(a["n_modules"] * P_MODULE_WP / 1000 for a in arrays)

    @st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
    def get_comprehensive_weather_data(_self, days=3):
        """Get comprehensive weather data with multiple parameters"""
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={_self.lat}&longitude={_self.lon}"
            f"&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,"
            f"surface_pressure,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,"
            f"direct_radiation,diffuse_radiation,global_tilted_irradiance,"
            f"direct_normal_irradiance,shortwave_radiation,precipitation,"
            f"wind_speed_10m,wind_direction_10m,is_day"
            f"&timezone=Europe%2FLondon&forecast_days={days}"
        )

        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            hourly = data.get("hourly", {})

            times = pd.to_datetime(hourly["time"])

            # Basic irradiance data
            direct_rad = np.array(hourly.get("direct_radiation", [0] * len(times)))
            diffuse_rad = np.array(hourly.get("diffuse_radiation", [0] * len(times)))
            dni = np.array(hourly.get("direct_normal_irradiance", [0] * len(times)))
            gti = np.array(hourly.get("global_tilted_irradiance", [0] * len(times)))
            
            # Day/night indicator
            is_day = np.array(hourly.get("is_day", [1] * len(times)))

            # Weather parameters
            cloud_cover = np.array(hourly.get("cloud_cover", [0] * len(times)))
            temperature = np.array(hourly.get("temperature_2m", [15] * len(times)))
            humidity = np.array(hourly.get("relative_humidity_2m", [60] * len(times)))
            wind_speed = np.array(hourly.get("wind_speed_10m", [2] * len(times)))
            pressure = np.array(hourly.get("surface_pressure", [1013] * len(times)))

            # Calculate GHI - ensure night hours are zero
            ghi = np.maximum(0, direct_rad + diffuse_rad)
            # Set GHI to zero for night hours
            ghi = ghi * is_day

            return {
                "times": times,
                "ghi": ghi,
                "dni": dni,
                "dhi": diffuse_rad,
                "gti": gti,
                "cloud_cover": cloud_cover,
                "temperature": temperature,
                "humidity": humidity,
                "wind_speed": wind_speed,
                "pressure": pressure,
                "direct_rad": direct_rad,
                "diffuse_rad": diffuse_rad,
                "is_day": is_day,
            }

        except Exception as e:
            st.warning(f"Weather API failed: {str(e)}. Using simulated data.")
            return _self._generate_simulated_data(days)

    def _generate_simulated_data(self, days):
        """Generate realistic simulated data"""
        n_points = days * 24
        now = datetime.now()
        start_time = datetime(now.year, now.month, now.day, now.hour)
        times = pd.date_range(start=start_time, periods=n_points, freq="H")

        # Generate day/night cycle based on solar position
        solar_pos = self.calculate_solar_position(times)
        zenith = solar_pos["zenith"]
        is_day = zenith < 90

        t = np.linspace(0, days * 2 * np.pi, n_points)
        
        # Solar pattern based on time of day
        hour_of_day = times.hour.values
        day_pattern = np.zeros_like(hour_of_day, dtype=float)
        for hour in range(24):
            hour_mask = hour_of_day == hour
            if 6 <= hour <= 18:
                intensity = np.sin(np.pi * (hour - 6) / 12) ** 2
                day_pattern[hour_mask] = intensity
        
        if days > 1:
            day_pattern = np.tile(day_pattern[:24], days)[:n_points]

        cloud_noise = 0.3 * np.sin(t / 3) + 0.2 * np.sin(t / 7) + 0.1 * np.random.randn(n_points)
        cloud_cover = np.clip(30 + 40 * (cloud_noise + 1) / 2, 0, 100)

        ghi_clear = 1000 * day_pattern * np.clip(1 - 0.1 * np.sin(t / 10), 0.7, 1)
        cloud_factor = 1 - 0.8 * (cloud_cover / 100) ** 1.5
        ghi = ghi_clear * cloud_factor
        ghi = np.where(is_day, ghi, 0)

        temperature = 15 + 10 * day_pattern + 5 * np.sin(t / 10)

        return {
            "times": times,
            "ghi": ghi,
            "dni": np.where(is_day, ghi * 0.7, 0),
            "dhi": np.where(is_day, ghi * 0.3, 0),
            "gti": np.where(is_day, ghi * 1.1, 0),
            "cloud_cover": cloud_cover,
            "temperature": temperature,
            "humidity": 60 + 20 * np.sin(t / 5),
            "wind_speed": 2 + 3 * day_pattern,
            "pressure": 1013 * np.ones_like(ghi),
            "direct_rad": np.where(is_day, ghi * 0.6, 0),
            "diffuse_rad": np.where(is_day, ghi * 0.4, 0),
            "is_day": is_day.astype(int),
        }

    def calculate_solar_position(self, times):
        """Calculate accurate solar position"""
        # Convert to numpy array of timestamps
        timestamps = pd.to_datetime(times)
        
        # Day of year
        doy = timestamps.dayofyear.values
        
        # Solar declination (Spencer 1971)
        B = 2 * np.pi * (doy - 1) / 365
        delta = 0.006918 - 0.399912 * np.cos(B) + 0.070257 * np.sin(B) \
                - 0.006758 * np.cos(2*B) + 0.000907 * np.sin(2*B) \
                - 0.002697 * np.cos(3*B) + 0.001480 * np.sin(3*B)
        delta = np.degrees(delta)
        
        # Equation of time (degrees)
        E = 229.2 * (0.000075 + 0.001868 * np.cos(B) - 0.032077 * np.sin(B) \
                    - 0.014615 * np.cos(2*B) - 0.040849 * np.sin(2*B))
        
        # Local solar time (hours)
        lst = timestamps.hour.values + timestamps.minute.values/60
        t_corr = 4 * (self.lon - 0) + E  # Simplified timezone correction
        solar_time = lst + t_corr/60
        
        # Hour angle (degrees)
        omega = 15 * (solar_time - 12)
        
        # Convert to radians
        lat_rad = np.radians(self.lat)
        delta_rad = np.radians(delta)
        omega_rad = np.radians(omega)
        
        # Solar zenith angle
        cos_theta_z = (np.sin(lat_rad) * np.sin(delta_rad) + 
                      np.cos(lat_rad) * np.cos(delta_rad) * np.cos(omega_rad))
        cos_theta_z = np.clip(cos_theta_z, 0, 1)
        theta_z = np.degrees(np.arccos(cos_theta_z))
        
        # Solar azimuth angle
        sin_alpha = np.cos(delta_rad) * np.sin(omega_rad) / (np.sin(np.radians(theta_z)) + 1e-6)
        sin_alpha = np.clip(sin_alpha, -1, 1)
        cos_alpha = (np.sin(delta_rad) * np.cos(lat_rad) - 
                    np.cos(delta_rad) * np.sin(lat_rad) * np.cos(omega_rad)) / (np.sin(np.radians(theta_z)) + 1e-6)
        
        alpha = np.degrees(np.arctan2(sin_alpha, cos_alpha))
        alpha = np.where(alpha < 0, alpha + 360, alpha)
        
        # Extraterrestrial radiation
        G_sc = 1367  # Solar constant (W/m²)
        G_on = G_sc * (1 + 0.033 * np.cos(np.radians(360 * doy / 365)))
        
        return {
            "zenith": theta_z,
            "azimuth": alpha,
            "cos_zenith": cos_theta_z,
            "declination": delta,
            "hour_angle": omega,
            "extraterrestrial": G_on * cos_theta_z,
        }

    def clearsky_model_ineichen(self, solar_pos, pressure=1013):
        """Ineichen-Perez clearsky model (more accurate than Hottel)"""
        zenith = solar_pos["zenith"]
        cos_zenith = solar_pos["cos_zenith"]
        G_on = solar_pos["extraterrestrial"] / cos_zenith
        
        # Air mass (Kasten-Young formula)
        air_mass = 1 / (cos_zenith + 0.50572 * (96.07995 - zenith)**-1.6364)
        air_mass = np.clip(air_mass, 1, 10)
        
        # Pressure correction
        air_mass *= pressure / 1013
        
        # Clear sky model coefficients (for mid-latitude climate)
        fh1 = np.exp(-pressure / 8000)
        fh2 = np.exp(-pressure / 1250)
        
        # Direct normal irradiance
        c = 0.056 * air_mass * fh1
        b = 0.4 * fh2
        dni_clear = G_on * np.exp(-0.09 * air_mass * (fh1 + fh2))
        
        # Diffuse horizontal irradiance
        dhi_clear = G_on * (0.006 + 0.045 * (1 - np.exp(-air_mass * c)))
        
        # Global horizontal irradiance
        ghi_clear = dni_clear * cos_zenith + dhi_clear
        
        # Clip nighttime values
        mask = zenith < 90
        ghi_clear = np.where(mask, ghi_clear, 0)
        dni_clear = np.where(mask, dni_clear, 0)
        dhi_clear = np.where(mask, dhi_clear, 0)
        
        return {
            "ghi": ghi_clear,
            "dni": dni_clear,
            "dhi": dhi_clear,
            "air_mass": air_mass
        }

    def cloud_transmission_model(self, ghi_clear, cloud_cover, solar_pos):
        """Advanced cloud transmission model"""
        zenith = solar_pos["zenith"]
        cos_zenith = solar_pos["cos_zenith"]
        
        # Cloud optical depth parameterization
        # Different cloud types have different extinction coefficients
        cloud_optical_depth = 0.1 + 10 * (cloud_cover/100)**2
        
        # Transmission through clouds (Beer-Lambert law)
        transmission = np.exp(-cloud_optical_depth / cos_zenith)
        
        # Minimum transmission (even thick clouds transmit some diffuse)
        min_transmission = 0.1 + 0.2 * (1 - cloud_cover/100)
        transmission = np.clip(transmission, min_transmission, 1)
        
        # Apply to clearsky GHI
        ghi_cloudy = ghi_clear * transmission
        
        # Enhance diffuse component under clouds (forward scattering)
        diffuse_enhancement = 1 + 0.5 * (cloud_cover/100)
        ghi_cloudy = np.where(cloud_cover > 50, 
                             ghi_cloudy * (1 + 0.1 * (cloud_cover-50)/50), 
                             ghi_cloudy)
        
        return ghi_cloudy, transmission

    def pvwatts_model(self, ghi, temperature, wind_speed, solar_pos):
        """NREL PVWatts v8 model implementation"""
        # Reference conditions
        T_ref = 25  # °C
        G_ref = 1000  # W/m²
        gamma = -0.004  # Temperature coefficient (%/°C)
        
        # Module temperature model (Sandia)
        T_module = temperature + ghi * np.exp(-3.47 - 0.0594 * wind_speed) / 1000
        
        # Power output
        power_dc = self.total_kwp * 1000 * (ghi / G_ref) * \
                  (1 + gamma * (T_module - T_ref))
        
        # Inverter efficiency (CEC model)
        power_dc_per_unit = power_dc / (self.total_kwp * 1000)
        inv_eff = 0.96 - 0.05 * power_dc_per_unit + 0.04 * power_dc_per_unit**2
        inv_eff = np.clip(inv_eff, 0.9, 0.965)
        
        power_ac = power_dc * inv_eff * SYSTEM_EFFICIENCY
        
        # Clip to realistic values
        power_ac = np.clip(power_ac, 0, self.total_kwp * 1000 * 1.1)
        
        return power_ac, T_module, inv_eff

    def perez_model_poa(self, weather_data, solar_pos):
        """Perez model for plane-of-array irradiance"""
        ghi = weather_data["ghi"]
        dni = weather_data["dni"]
        dhi = weather_data["dhi"]
        
        zenith = solar_pos["zenith"]
        azimuth = solar_pos["azimuth"]
        cos_zenith = solar_pos["cos_zenith"]
        
        total_poa = np.zeros_like(ghi)
        
        for array in self.arrays:
            tilt = array["tilt"]
            array_azimuth = array["azimuth"]
            
            # Calculate angles
            tilt_rad = np.radians(tilt)
            zenith_rad = np.radians(zenith)
            
            # Angle of incidence
            cos_aoi = (np.cos(zenith_rad) * np.cos(tilt_rad) + 
                      np.sin(zenith_rad) * np.sin(tilt_rad) * 
                      np.cos(np.radians(azimuth - array_azimuth)))
            cos_aoi = np.clip(cos_aoi, 0, 1)
            
            # Sky view factor
            f_sky = (1 + np.cos(tilt_rad)) / 2
            f_ground = (1 - np.cos(tilt_rad)) / 2
            
            # Ground albedo (typical)
            albedo = 0.2
            
            # POA irradiance components
            poa_beam = dni * cos_aoi
            poa_diffuse = dhi * f_sky
            poa_ground = ghi * albedo * f_ground
            
            # Total POA for this array
            poa_total = poa_beam + poa_diffuse + poa_ground
            
            # Convert to power for this array
            array_kwp = array["n_modules"] * P_MODULE_WP / 1000
            array_power = poa_total / 1000 * array_kwp * array["loss_factor"] * 0.96
            
            total_poa += array_power * 1000  # Convert to watts
        
        return total_poa

    def machine_learning_correction(self, predicted_power, historical_ratio=None):
        """Simple ML-like correction based on patterns"""
        # This is a simplified version - real implementation would use scikit-learn
        
        # Pattern-based correction
        n = len(predicted_power)
        hour_of_day = np.arange(n) % 24
        
        # Morning bias (often overestimated)
        morning_mask = (hour_of_day >= 8) & (hour_of_day <= 11)
        predicted_power[morning_mask] *= 0.95
        
        # Afternoon bias (often underestimated)
        afternoon_mask = (hour_of_day >= 13) & (hour_of_day <= 16)
        predicted_power[afternoon_mask] *= 1.05
        
        # Apply historical performance ratio if available
        if historical_ratio is not None:
            predicted_power *= historical_ratio
        
        return predicted_power

    def calculate_sunrise_sunset(self, times):
        """Calculate sunrise and sunset times for each day"""
        solar_pos = self.calculate_solar_position(times)
        zenith = solar_pos["zenith"]
        
        # Find hours where sun is above horizon (zenith < 90)
        sun_up = zenith < 90
        
        # Group by day
        df = pd.DataFrame({
            'time': times,
            'sun_up': sun_up,
            'zenith': zenith
        })
        df['date'] = df['time'].dt.date
        
        # Calculate sunrise/sunset for each day
        sunrise_sunset = {}
        for date_val, group in df.groupby('date'):
            day_hours = group[group['sun_up']]
            if len(day_hours) > 0:
                sunrise = day_hours['time'].min()
                sunset = day_hours['time'].max()
                sunrise_sunset[date_val] = {
                    'sunrise': sunrise,
                    'sunset': sunset,
                    'daylight_hours': len(day_hours),
                    'avg_zenith': day_hours['zenith'].mean()
                }
        
        return sunrise_sunset

    def estimate_production(self, weather_data, method='combined'):
        """Main estimation method with multiple approaches"""
        
        # Calculate solar position
        solar_pos = self.calculate_solar_position(weather_data['times'])
        
        # Method 1: PVWatts model (simplest)
        p_pvwatts, T_module, inv_eff = self.pvwatts_model(
            weather_data['ghi'], 
            weather_data['temperature'],
            weather_data['wind_speed'],
            solar_pos
        )
        
        # Method 2: Perez POA model (most accurate for tilted arrays)
        p_perez = self.perez_model_poa(weather_data, solar_pos)
        
        # Method 3: Cloud-adjusted clearsky
        clearsky = self.clearsky_model_ineichen(solar_pos, weather_data['pressure'].mean())
        ghi_cloudy, cloud_trans = self.cloud_transmission_model(
            clearsky['ghi'], 
            weather_data['cloud_cover'],
            solar_pos
        )
        p_cloud_adj, _, _ = self.pvwatts_model(
            ghi_cloudy,
            weather_data['temperature'],
            weather_data['wind_speed'],
            solar_pos
        )
        
        # Combine methods based on selected approach
        if method == 'pvwatts':
            final_power = p_pvwatts
        elif method == 'perez':
            final_power = p_perez
        elif method == 'cloud':
            final_power = p_cloud_adj
        elif method == 'combined':
            # Weighted combination (adapt based on conditions)
            weights = {
                'pvwatts': 0.3,
                'perez': 0.4,
                'cloud': 0.3
            }
            final_power = (weights['pvwatts'] * p_pvwatts +
                          weights['perez'] * p_perez +
                          weights['cloud'] * p_cloud_adj)
        else:
            final_power = p_pvwatts
        
        # Apply ML correction
        final_power = self.machine_learning_correction(final_power)
        
        # Day mask - only produce when sun is above horizon
        zenith = solar_pos["zenith"]
        day_mask = zenith < 90
        
        # Final zero-check for night and clip to system limits
        final_power = np.where(day_mask, final_power, 0)
        final_power = np.clip(final_power, 0, self.total_kwp * 1000)
        
        return {
            'power': final_power,
            'pvwatts': np.where(day_mask, p_pvwatts, 0),
            'perez': np.where(day_mask, p_perez, 0),
            'cloud_adj': np.where(day_mask, p_cloud_adj, 0),
            'temperature': T_module,
            'inverter_eff': inv_eff,
            'cloud_transmission': cloud_trans,
            'solar_zenith': solar_pos['zenith'],
            'day_mask': day_mask,
        }

    def calculate_expected_max(self, days=7):
        """Clear-sky theoretical maximum over a horizon, with daily series"""
        now = datetime.now()
        start_time = datetime(now.year, now.month, now.day, 0, 0, 0)
        times = pd.date_range(start=start_time, periods=days * 24, freq="H")

        # Generate ideal weather data for clearsky
        ideal_weather = {
            "times": times,
            "ghi": np.zeros(len(times)),
            "temperature": 20 * np.ones(len(times)),
            "cloud_cover": np.zeros(len(times)),
            "wind_speed": 2 * np.ones(len(times)),
            "pressure": 1013 * np.ones(len(times)),
            "dni": np.zeros(len(times)),
            "dhi": np.zeros(len(times))
        }

        # Calculate solar position
        solar_pos = self.calculate_solar_position(times)
        
        # Get clearsky irradiance
        clearsky = self.clearsky_model_ineichen(solar_pos)
        
        # Use ideal GHI, DNI, DHI from clearsky model
        ideal_weather["ghi"] = clearsky['ghi']
        ideal_weather["dni"] = clearsky['dni']
        ideal_weather["dhi"] = clearsky['dhi']

        max_estimation = self.estimate_production(ideal_weather, method="perez")

        df = pd.DataFrame(
            {"power": max_estimation["power"], "ghi": ideal_weather["ghi"]},
            index=times,
        )

        daily_max_kw = df["power"].resample("D").max() / 1000.0
        daily_energy_kwh = df["power"].resample("D").sum() / 1000.0

        return {
            "daily_max_kw": daily_max_kw,
            "daily_energy_kwh": daily_energy_kwh,
            "total_max_kw": daily_max_kw.max(),
            "avg_daily_energy": daily_energy_kwh.mean(),
        }

    def simulate_battery_self_powered(self, solar_power_kw, baseload_kw, 
                                    battery_capacity_kwh, initial_soc_percent=50,
                                    max_charge_rate_kw=5.0, max_discharge_rate_kw=5.0,
                                    reserve_percent=0):
        """
        Simulate Tesla Powerwall Self-Powered mode
        Maximizes self-consumption, doesn't charge from grid
        """
        n = len(solar_power_kw)
        battery_soc = np.zeros(n)
        grid_import = np.zeros(n)
        grid_export = np.zeros(n)
        self_consumption = np.zeros(n)
        
        # Convert initial SOC to kWh
        current_soc_kwh = battery_capacity_kwh * initial_soc_percent / 100
        reserve_kwh = battery_capacity_kwh * reserve_percent / 100
        
        for i in range(n):
            solar_gen = solar_power_kw[i]
            net_load = baseload_kw - solar_gen
            
            if net_load < 0:  # Excess solar
                excess = -net_load
                
                # Charge battery with excess solar (up to max charge rate)
                charge_possible = min(
                    excess,
                    max_charge_rate_kw,
                    (battery_capacity_kwh - current_soc_kwh) / 1.0  # 1 hour timestep
                )
                charge_energy = charge_possible * BATTERY_EFFICIENCY
                
                current_soc_kwh += charge_energy
                self_consumption[i] = solar_gen
                
                # Remaining excess goes to grid
                grid_export[i] = excess - charge_possible
                grid_import[i] = 0
                
            else:  # Solar deficit
                deficit = net_load
                
                # Try to discharge battery (but keep reserve)
                discharge_possible = min(
                    deficit,
                    max_discharge_rate_kw,
                    (current_soc_kwh - reserve_kwh) / 1.0  # Don't discharge below reserve
                )
                
                if discharge_possible > 0:
                    current_soc_kwh -= discharge_possible
                    deficit -= discharge_possible
                    self_consumption[i] = solar_gen
                else:
                    self_consumption[i] = solar_gen
                
                # Remaining deficit comes from grid
                grid_import[i] = deficit
                grid_export[i] = 0
            
            # Ensure SOC stays within bounds
            current_soc_kwh = np.clip(current_soc_kwh, 0, battery_capacity_kwh)
            battery_soc[i] = current_soc_kwh
            
        return {
            'battery_soc_kwh': battery_soc,
            'battery_soc_percent': battery_soc / battery_capacity_kwh * 100,
            'grid_import_kw': grid_import,
            'grid_export_kw': grid_export,
            'self_consumption_kw': self_consumption,
            'solar_to_battery_kwh': np.sum(np.maximum(0, solar_power_kw - baseload_kw)),
            'total_grid_import_kwh': np.sum(grid_import),
            'total_grid_export_kwh': np.sum(grid_export),
            'self_sufficiency_percent': np.sum(self_consumption) / (np.sum(solar_power_kw) + 1e-6) * 100
        }

    def simulate_battery_time_based(self, solar_power_kw, baseload_kw, 
                                  battery_capacity_kwh, initial_soc_percent=50,
                                  max_charge_rate_kw=5.0, max_discharge_rate_kw=5.0,
                                  reserve_percent=10, low_tariff_start=LOW_TARIFF_START,
                                  low_tariff_end=LOW_TARIFF_END, times=None):
        """
        Simulate Tesla Powerwall Time-Based Control mode
        Charges from grid during low tariff, optimizes for next day's solar
        """
        n = len(solar_power_kw)
        battery_soc = np.zeros(n)
        grid_import = np.zeros(n)
        grid_export = np.zeros(n)
        self_consumption = np.zeros(n)
        grid_charge_kw = np.zeros(n)
        
        # Convert initial SOC to kWh
        current_soc_kwh = battery_capacity_kwh * initial_soc_percent / 100
        reserve_kwh = battery_capacity_kwh * reserve_percent / 100
        
        # Create time masks
        time_of_day = pd.Series(times).dt.time
        low_tariff_mask = ((time_of_day >= low_tariff_start) | (time_of_day <= low_tariff_end)).values
        
        # Calculate expected solar for next day to determine overnight charge target
        # Group by date
        df_solar = pd.DataFrame({
            'time': times,
            'solar_kw': solar_power_kw,
            'date': pd.Series(times).dt.date
        })
        
        # Calculate daily solar generation
        daily_solar = df_solar.groupby('date')['solar_kw'].sum().reset_index()
        daily_solar.columns = ['date', 'daily_solar_kwh']
        
        # Create a mapping of date to next day's expected solar
        date_to_solar = dict(zip(daily_solar['date'], daily_solar['daily_solar_kwh']))
        
        # Calculate optimal overnight charge target
        # Target = Expected house demand until next low tariff - Expected solar
        hours_until_next_night = 24  # Simplification
        expected_demand = baseload_kw * hours_until_next_night
        
        for i in range(n):
            solar_gen = solar_power_kw[i]
            current_time = times[i]
            current_date = current_time.date()
            
            # Get expected solar for next day
            next_date = current_date + timedelta(days=1)
            expected_next_day_solar = date_to_solar.get(next_date, 0)
            
            # Calculate target SOC for overnight charging
            # We want enough battery to cover expected demand minus expected solar
            target_energy_kwh = max(0, expected_demand - expected_next_day_solar)
            target_energy_kwh = min(target_energy_kwh, battery_capacity_kwh)
            target_soc_percent = (target_energy_kwh / battery_capacity_kwh) * 100
            
            net_load = baseload_kw - solar_gen
            
            if low_tariff_mask[i]:
                # Low tariff period - can charge from grid
                if net_load < 0:  # Excess solar
                    excess = -net_load
                    
                    # Charge battery with excess solar first
                    charge_from_solar = min(
                        excess,
                        max_charge_rate_kw,
                        (battery_capacity_kwh - current_soc_kwh) / 1.0
                    )
                    charge_energy_solar = charge_from_solar * BATTERY_EFFICIENCY
                    current_soc_kwh += charge_energy_solar
                    excess -= charge_from_solar
                    
                    # Then charge from grid to reach target if needed
                    charge_needed = max(0, target_energy_kwh - current_soc_kwh)
                    charge_from_grid = min(
                        charge_needed,
                        max_charge_rate_kw - charge_from_solar,
                        (battery_capacity_kwh - current_soc_kwh) / 1.0
                    )
                    grid_charge_kw[i] = charge_from_grid
                    current_soc_kwh += charge_from_grid
                    
                    # Any remaining excess goes to grid
                    grid_export[i] = excess
                    grid_import[i] = 0
                    self_consumption[i] = solar_gen
                    
                else:  # Solar deficit during low tariff
                    deficit = net_load
                    
                    # Try to discharge battery (but keep reserve)
                    discharge_possible = min(
                        deficit,
                        max_discharge_rate_kw,
                        (current_soc_kwh - reserve_kwh) / 1.0
                    )
                    
                    if discharge_possible > 0:
                        current_soc_kwh -= discharge_possible
                        deficit -= discharge_possible
                        self_consumption[i] = solar_gen
                    else:
                        self_consumption[i] = solar_gen
                    
                    # Import deficit from grid
                    grid_import[i] = deficit
                    grid_export[i] = 0
                    
                    # Also charge from grid to target if needed
                    charge_needed = max(0, target_energy_kwh - current_soc_kwh)
                    charge_from_grid = min(
                        charge_needed,
                        max_charge_rate_kw,
                        (battery_capacity_kwh - current_soc_kwh) / 1.0
                    )
                    grid_charge_kw[i] = charge_from_grid
                    current_soc_kwh += charge_from_grid
                    grid_import[i] += charge_from_grid
                    
            else:
                # High tariff period - minimize grid import
                if net_load < 0:  # Excess solar
                    excess = -net_load
                    
                    # Charge battery with excess solar
                    charge_possible = min(
                        excess,
                        max_charge_rate_kw,
                        (battery_capacity_kwh - current_soc_kwh) / 1.0
                    )
                    charge_energy = charge_possible * BATTERY_EFFICIENCY
                    current_soc_kwh += charge_energy
                    self_consumption[i] = solar_gen
                    
                    # Remaining excess goes to grid
                    grid_export[i] = excess - charge_possible
                    grid_import[i] = 0
                    
                else:  # Solar deficit during high tariff
                    deficit = net_load
                    
                    # Try to discharge battery (but keep reserve)
                    discharge_possible = min(
                        deficit,
                        max_discharge_rate_kw,
                        (current_soc_kwh - reserve_kwh) / 1.0
                    )
                    
                    if discharge_possible > 0:
                        current_soc_kwh -= discharge_possible
                        deficit -= discharge_possible
                        self_consumption[i] = solar_gen
                    else:
                        self_consumption[i] = solar_gen
                    
                    # Remaining deficit comes from grid
                    grid_import[i] = deficit
                    grid_export[i] = 0
            
            # Ensure SOC stays within bounds
            current_soc_kwh = np.clip(current_soc_kwh, 0, battery_capacity_kwh)
            battery_soc[i] = current_soc_kwh
        
        total_export_revenue = np.sum(grid_export) * EXPORT_TARIFF_GBP_PER_KWH
        
        return {
            'battery_soc_kwh': battery_soc,
            'battery_soc_percent': battery_soc / battery_capacity_kwh * 100,
            'grid_import_kw': grid_import,
            'grid_export_kw': grid_export,
            'grid_charge_kw': grid_charge_kw,
            'self_consumption_kw': self_consumption,
            'solar_to_battery_kwh': np.sum(np.maximum(0, solar_power_kw - baseload_kw)),
            'total_grid_import_kwh': np.sum(grid_import),
            'total_grid_export_kwh': np.sum(grid_export),
            'total_grid_charge_kwh': np.sum(grid_charge_kw),
            'total_export_revenue_gbp': total_export_revenue,
            'self_sufficiency_percent': np.sum(self_consumption) / (np.sum(solar_power_kw) + 1e-6) * 100,
            'overnight_charge_target_percent': target_soc_percent
        }

    def calculate_export_potential(self, solar_power_kw, baseload_kw, peak_factor=2.0):
        """
        Calculate potential export power and energy
        Includes consideration of peak house demand
        """
        n = len(solar_power_kw)
        
        # Calculate net export potential
        # During peak hours, house demand = baseload * peak_factor
        hour_of_day = np.arange(n) % 24
        peak_hours = (hour_of_day >= 16) & (hour_of_day <= 19)  # 4-7 PM typical peak
        
        house_demand = np.ones(n) * baseload_kw
        house_demand[peak_hours] = baseload_kw * peak_factor
        
        # Calculate net export
        net_export = np.maximum(0, solar_power_kw - house_demand)
        
        # Calculate statistics
        max_export_kw = np.max(net_export)
        total_export_kwh = np.sum(net_export)
        export_revenue = total_export_kwh * EXPORT_TARIFF_GBP_PER_KWH
        
        return {
            'export_power_kw': net_export,
            'max_export_kw': max_export_kw,
            'total_export_kwh': total_export_kwh,
            'export_revenue_gbp': export_revenue,
            'export_hours': np.sum(net_export > 0),
            'house_demand_kw': house_demand,
            'peak_hours_mask': peak_hours
        }


def run_app():
    st.set_page_config(
        page_title="Enhanced Solar Production Estimator",
        layout="wide",
    )

    st.title("🏠 Enhanced Solar Production with Tesla Powerwall & Smart Charging")

    with st.sidebar:
        st.header("⚙️ Configuration")
        lat = st.number_input("Latitude", value=LAT, format="%.4f")
        lon = st.number_input("Longitude", value=LON, format="%.4f")
        
        # Historical performance ratio input
        historical_ratio = st.slider(
            "Historical Performance Ratio",
            min_value=0.5,
            max_value=1.2,
            value=1.0,
            step=0.01,
            help="Adjust based on historical system performance (1.0 = nominal)"
        )
        
        # Power Price Settings
        st.subheader("💡 UK Power Prices")
        show_prices = st.checkbox(
            "Show Power Price Predictions",
            value=True,
            help="Display UK imbalance price predictions for smart charging"
        )
        
        price_days_ahead = st.slider(
            "Price Forecast Days",
            min_value=1,
            max_value=3,
            value=2,
            help="Number of days ahead for price predictions"
        )
        
        price_threshold_percentile = st.slider(
            "Low Price Threshold (Percentile)",
            min_value=5,
            max_value=40,
            value=25,
            help="Prices below this percentile are considered 'low' for smart charging"
        )
        
        st.subheader("🔋 Tesla Powerwall Settings")
        
        # Powerwall mode selection
        powerwall_mode = st.radio(
            "Powerwall Mode",
            ["Self-Powered", "Time-Based Control", "Smart Charging"],
            index=2,
            help="Self-Powered: Maximize self-consumption. Time-Based Control: Charge from grid during low tariff. Smart Charging: Optimize based on power prices."
        )
        
        battery_capacity = st.number_input(
            "Battery capacity (kWh)", 
            value=BATTERY_CAPACITY_KWH,
            min_value=0.0,
            max_value=100.0,
            step=0.1
        )
        
        initial_soc = st.slider(
            "Initial State of Charge (%)", 
            0, 100, 50
        )
        
        max_charge_rate = st.number_input(
            "Max charge rate (kW)", 
            value=BATTERY_MAX_CHARGE_RATE_KW,
            min_value=0.0,
            max_value=20.0,
            step=0.1
        )
        
        max_discharge_rate = st.number_input(
            "Max discharge rate (kW)", 
            value=BATTERY_MAX_DISCHARGE_RATE_KW,
            min_value=0.0,
            max_value=20.0,
            step=0.1
        )
        
        # Initialize time-based control variables with defaults
        reserve_percent = 10  # Default reserve for Time-Based Control
        low_tariff_start = LOW_TARIFF_START
        low_tariff_end = LOW_TARIFF_END
        
        if powerwall_mode == "Time-Based Control":
            reserve_percent = st.slider(
                "Reserve Level (%)",
                0, 50, 10,
                help="Minimum battery level that won't be discharged"
            )
            
            st.subheader("⏰ Time-Based Control Settings")
            
            col1, col2 = st.columns(2)
            with col1:
                low_start_hour = st.number_input("Low Tariff Start Hour", 0, 23, 23)
                low_start_min = st.number_input("Low Start Minute", 0, 59, 30)
            with col2:
                low_end_hour = st.number_input("Low Tariff End Hour", 0, 23, 5)
                low_end_min = st.number_input("Low End Minute", 0, 59, 30)
            
            # Use datetime_time (imported as datetime_time from datetime)
            low_tariff_start = datetime_time(low_start_hour, low_start_min)
            low_tariff_end = datetime_time(low_end_hour, low_end_min)
            
            st.info(f"Low tariff period: {low_tariff_start.strftime('%H:%M')} to {low_tariff_end.strftime('%H:%M')}")
        
        st.subheader("🏠 House Load Settings")
        baseload = st.number_input(
            "Baseload (kW)", 
            value=BASELOAD_KW,
            min_value=0.0,
            max_value=10.0,
            step=0.1,
            help="Continuous house power consumption"
        )
        
        peak_factor = st.number_input(
            "Peak Load Factor", 
            value=PEAK_FACTOR,
            min_value=1.0,
            max_value=5.0,
            step=0.1,
            help="Multiplier for baseload during peak hours (4-7 PM)"
        )
        
        st.subheader("💰 Export Settings")
        export_tariff = st.number_input(
            "Export Tariff (£/kWh)",
            value=EXPORT_TARIFF_GBP_PER_KWH,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            format="%.3f",
            help="Price you receive for exported electricity"
        )
        
        st.subheader("📈 Forecast Settings")
        days = st.slider("Forecast days", min_value=1, max_value=7, value=3)
        show_night = st.checkbox("Show night hours", value=False)
        
        method = st.selectbox(
            "Solar estimation method",
            options=["combined", "pvwatts", "perez", "cloud"],
            index=0,
            help="combined: Weighted average of all methods, pvwatts: NREL model, perez: Tilted array model, cloud: Cloud-adjusted clearsky"
        )

    # Initialize estimators
    estimator = EnhancedSolarEstimator(lat, lon, arrays)
    price_predictor = UKPowerPricePredictor()

    st.subheader("📊 Weather data and predictions")
    with st.spinner("Fetching weather data and running models..."):
        weather_data = estimator.get_comprehensive_weather_data(days=days)
        max_calc = estimator.calculate_expected_max(days=days)

        methods = ["pvwatts", "perez", "cloud", "combined"]
        results = {}
        for m in methods:
            results[m] = estimator.estimate_production(weather_data, method=m)
            # Apply historical ratio correction
            results[m]["power"] = estimator.machine_learning_correction(
                results[m]["power"], 
                historical_ratio
            )

    # Fetch power prices if enabled
    price_data = None
    price_stats = {}
    charging_windows = []
    
    if show_prices:
        with st.spinner("Fetching UK power prices..."):
            price_data = price_predictor.fetch_prices(days_ahead=price_days_ahead)
            if price_data is not None and not price_data.empty:
                price_stats = price_predictor.calculate_price_statistics(price_data)
                charging_windows = price_predictor.identify_smart_charging_windows(price_data)

    # Calculate sunrise/sunset times
    sunrise_sunset = estimator.calculate_sunrise_sunset(weather_data["times"])
    
    # Use as much of the horizon as available
    times = weather_data["times"]
    df_comparison = pd.DataFrame(index=times)

    for m in methods:
        df_comparison[f"power_{m}"] = results[m]["power"]

    df_comparison["ghi"] = weather_data["ghi"]
    df_comparison["cloud_cover"] = weather_data["cloud_cover"]
    df_comparison["temperature"] = weather_data["temperature"]
    df_comparison["is_day"] = weather_data["is_day"]
    df_comparison["zenith"] = results["combined"]["solar_zenith"]
    df_comparison["day_mask"] = results["combined"]["day_mask"]
    df_comparison["cloud_transmission"] = results["combined"]["cloud_transmission"]
    
    # Filter for daylight hours if not showing night
    if not show_night:
        df_daylight = df_comparison[df_comparison["day_mask"]].copy()
    else:
        df_daylight = df_comparison.copy()

    # Hourly energy (kWh)
    for m in methods:
        df_comparison[f"energy_{m}"] = df_comparison[f"power_{m}"] / 1000.0
        df_daylight[f"energy_{m}"] = df_daylight[f"power_{m}"] / 1000.0

    # Get solar power for battery simulation
    solar_power_kw = results[method]["power"] / 1000  # Convert to kW
    
    # Calculate export potential
    export_results = estimator.calculate_export_potential(
        solar_power_kw=solar_power_kw,
        baseload_kw=baseload,
        peak_factor=peak_factor
    )
    
    # Run battery simulation based on mode
    if powerwall_mode == "Self-Powered":
        battery_results = estimator.simulate_battery_self_powered(
            solar_power_kw=solar_power_kw,
            baseload_kw=baseload,
            battery_capacity_kwh=battery_capacity,
            initial_soc_percent=initial_soc,
            max_charge_rate_kw=max_charge_rate,
            max_discharge_rate_kw=max_discharge_rate,
            reserve_percent=0
        )
    else:  # Time-Based Control or Smart Charging
        battery_results = estimator.simulate_battery_time_based(
            solar_power_kw=solar_power_kw,
            baseload_kw=baseload,
            battery_capacity_kwh=battery_capacity,
            initial_soc_percent=initial_soc,
            max_charge_rate_kw=max_charge_rate,
            max_discharge_rate_kw=max_discharge_rate,
            reserve_percent=reserve_percent,
            low_tariff_start=low_tariff_start,
            low_tariff_end=low_tariff_end,
            times=times
        )
    
    # Add battery and export results to dataframe
    df_comparison["battery_soc_percent"] = battery_results["battery_soc_percent"]
    df_comparison["grid_import_kw"] = battery_results["grid_import_kw"]
    df_comparison["grid_export_kw"] = battery_results["grid_export_kw"]
    df_comparison["self_consumption_kw"] = battery_results["self_consumption_kw"]
    df_comparison["export_potential_kw"] = export_results['export_power_kw']
    df_comparison["house_demand_kw"] = export_results['house_demand_kw']
    df_comparison["peak_hours"] = export_results['peak_hours_mask']
    df_comparison["net_load_kw"] = baseload - solar_power_kw
    
    if powerwall_mode == "Time-Based Control" or powerwall_mode == "Smart Charging":
        df_comparison["grid_charge_kw"] = battery_results["grid_charge_kw"]
        # Create low tariff mask
        time_of_day = pd.Series(times).dt.time
        df_comparison["low_tariff"] = (
            (time_of_day >= low_tariff_start) | 
            (time_of_day <= low_tariff_end)
        ).values

    # --- POWER PRICE DISPLAY SECTION ---
    if show_prices and price_data is not None and not price_data.empty:
        st.markdown("### ⚡ UK Power Price Predictions")
        
        # Price Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            trend_icon = "🔼" if price_stats.get('price_trend', 0) > 0 else "🔽"
            trend_color = "red" if price_stats.get('price_trend', 0) > 0 else "green"
            st.metric(
                "Current Price", 
                f"£{price_stats.get('current_price', 0):.1f}/MWh",
                f"{trend_icon} {abs(price_stats.get('price_trend', 0)):.1f}%",
                delta_color=trend_color
            )
        
        with col2:
            st.metric(
                "Today's Average", 
                f"£{price_stats.get('avg_price_today', 0):.1f}/MWh",
                f"£{price_stats.get('min_price_today', 0):.0f}-{price_stats.get('max_price_today', 0):.0f}"
            )
        
        with col3:
            low_hours = price_stats.get('low_price_hours_count', 0)
            total_hours = min(24, len(price_data))
            st.metric(
                "Low Price Hours", 
                f"{low_hours}/{total_hours}",
                f"{price_stats.get('best_time', 'N/A')} @ £{price_stats.get('best_price', 0):.0f}"
            )
        
        with col4:
            st.metric(
                "Smart Charging Windows",
                f"{len(charging_windows)}",
                f"Best: {charging_windows[0]['duration_hours']}h" if charging_windows else "None found"
            )
        
        # Create price visualization
        fig_price, ax_price = plt.subplots(figsize=(12, 6))
        
        # Separate actual and forecast prices
        actual_prices = price_data[price_data['is_actual'] == True]
        forecast_prices = price_data[price_data['is_forecast'] == True]
        
        # Plot actual prices
        if not actual_prices.empty:
            ax_price.plot(
                actual_prices.index, 
                actual_prices['price_gbp_per_mwh'],
                'o-', 
                color='blue', 
                linewidth=2, 
                markersize=4,
                label='Actual Prices'
            )
        
        # Plot forecast prices
        if not forecast_prices.empty:
            ax_price.plot(
                forecast_prices.index, 
                forecast_prices['price_gbp_per_mwh'],
                '--', 
                color='orange', 
                linewidth=2,
                label='Forecast Prices'
            )
        
        # Add price thresholds
        low_threshold = price_stats.get('low_price_threshold', 0)
        high_threshold = price_stats.get('high_price_threshold', 0)
        
        ax_price.axhline(y=low_threshold, color='green', linestyle=':', alpha=0.5, label='Low Price Threshold')
        ax_price.axhline(y=high_threshold, color='red', linestyle=':', alpha=0.5, label='High Price Threshold')
        
        # Fill between thresholds
        ax_price.fill_between(
            price_data.index, 
            low_threshold, 
            high_threshold,
            alpha=0.1, 
            color='yellow',
            label='Medium Price Zone'
        )
        
        # Highlight low price windows
        for window in charging_windows[:3]:  # Show top 3 windows
            start_time = window['start']
            end_time = window['end'] + timedelta(hours=1)  # Add 1 hour for display
            ax_price.axvspan(
                start_time, 
                end_time, 
                alpha=0.2, 
                color='green',
                label='Smart Charging Window' if window == charging_windows[0] else ""
            )
        
        # Add current time marker
        current_time = datetime.now()
        ax_price.axvline(x=current_time, color='black', linestyle='--', alpha=0.7, label='Current Time')
        
        ax_price.set_xlabel('Time')
        ax_price.set_ylabel('Price (£/MWh)')
        ax_price.set_title('UK Power Prices - Actual & Forecast')
        ax_price.legend(loc='upper right', fontsize=9)
        ax_price.grid(True, alpha=0.3)
        
        # Format x-axis
        ax_price.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M\n%d/%m'))
        
        plt.tight_layout()
        st.pyplot(fig_price)
        
        # Smart Charging Recommendations
        if charging_windows:
            st.markdown("#### 🔋 Smart Charging Recommendations")
            
            cols = st.columns(min(len(charging_windows), 3))
            for idx, window in enumerate(charging_windows[:3]):
                with cols[idx % 3]:
                    start_str = window['start'].strftime('%H:%M')
                    end_str = (window['end'] + timedelta(hours=1)).strftime('%H:%M')
                    
                    st.metric(
                        f"Window {idx+1}",
                        f"{start_str} - {end_str}",
                        f"£{window['avg_price']:.0f}/MWh"
                    )
                    st.caption(f"Duration: {window['duration_hours']}h | Best: £{window['min_price']:.0f}/MWh")
        
        # Price Data Table
        with st.expander("📋 View Detailed Price Data"):
            price_display = price_data.copy()
            price_display['Time'] = price_display.index.strftime('%Y-%m-%d %H:%M')
            price_display['Price (£/MWh)'] = price_display['price_gbp_per_mwh'].round(1)
            price_display['Price (£/kWh)'] = price_display['price_gbp_per_kwh'].round(3)
            price_display['Type'] = price_display['is_actual'].apply(lambda x: 'Actual' if x else 'Forecast')
            
            display_cols = ['Time', 'Price (£/MWh)', 'Price (£/kWh)', 'Type']
            st.dataframe(price_display[display_cols], use_container_width=True)

    # Summary statistics
    summary_data = []
    for m in methods:
        peak_kw = df_comparison[f"power_{m}"].max() / 1000
        energy_kwh_today = df_comparison[f"energy_{m}"].resample("D").sum().iloc[0]
        summary_data.append(
            {
                "Method": m.upper(),
                "Peak (kW)": round(peak_kw, 2),
                "Today (kWh)": round(energy_kwh_today, 1),
                "Diff from Max (%)": round(
                    peak_kw / max_calc["total_max_kw"] * 100, 1
                ),
            }
        )
    summary_df = pd.DataFrame(summary_data)

    # Display metrics in columns
    st.markdown("### 📈 System Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("#### Solar Production")
        st.metric("System capacity (kWp)", f"{TOTAL_KWP:.2f}")
        st.metric("Theoretical max (kW)", f"{max_calc['total_max_kw']:.2f}")
        st.metric("Today's estimated solar", f"{summary_df.loc[summary_df['Method'] == method.upper(), 'Today (kWh)'].iloc[0]:.1f} kWh")

    with col2:
        st.markdown("#### Battery Performance")
        final_soc = battery_results["battery_soc_percent"][-1]
        st.metric("Final SOC", f"{final_soc:.1f}%")
        st.metric("Self-sufficiency", f"{battery_results['self_sufficiency_percent']:.1f}%")
        if powerwall_mode == "Time-Based Control" or powerwall_mode == "Smart Charging":
            st.metric("Overnight charge target", f"{battery_results.get('overnight_charge_target_percent', 0):.0f}%")

    with col3:
        st.markdown("#### Financials")
        st.metric("Export revenue", f"£{export_results['export_revenue_gbp']:.2f}")
        st.metric("Total export", f"{export_results['total_export_kwh']:.1f} kWh")
        st.metric("Max export power", f"{export_results['max_export_kw']:.2f} kW")

    with col4:
        st.markdown("#### Performance")
        current_method_peak = summary_df.loc[summary_df['Method'] == method.upper(), 'Peak (kW)'].iloc[0]
        system_efficiency = (current_method_peak / max_calc['total_max_kw']) * 100
        st.metric("System Efficiency", f"{system_efficiency:.1f}%")
        st.metric("Performance Ratio", f"{historical_ratio:.2f}")
        st.metric("Selected Method", method.upper())

    # --- Sunrise/Sunset Information ---
    st.markdown("### 🌅 Sunrise/Sunset Times")
    sunrise_cols = st.columns(min(len(sunrise_sunset), 4))
    for idx, (date_val, info) in enumerate(list(sunrise_sunset.items())[:4]):
        with sunrise_cols[idx % 4]:
            st.metric(
                f"{date_val}",
                f"{info['sunrise'].strftime('%H:%M')} - {info['sunset'].strftime('%H:%M')}",
                f"{info['daylight_hours']}h daylight"
            )

    # --- METHOD COMPARISON PLOTS ---
    st.markdown("### 📊 Method Comparison Analysis")
    
    # Create comparison figure
    fig_comparison, axes_comparison = plt.subplots(2, 2, figsize=(14, 10))
    fig_comparison.suptitle('Enhanced Solar Estimation - Method Comparison', 
                           fontsize=16, fontweight='bold')
    
    # Plot 1: Power comparison
    ax1 = axes_comparison[0, 0]
    colors = {'pvwatts': 'blue', 'perez': 'green', 'cloud': 'orange', 'combined': 'red'}
    for m in methods:
        ax1.plot(df_daylight.index, df_daylight[f'power_{m}']/1000, 
                color=colors[m], label=f'{m.upper()}', alpha=0.8, linewidth=2)
    ax1.axhline(LIVE_PEAK_KW, color='purple', linestyle='--', label='Your Live Peak')
    ax1.axhline(max_calc['total_max_kw'], color='black', linestyle=':', 
               label=f'Theoretical Max ({max_calc["total_max_kw"]:.1f}kW)')
    ax1.set_ylabel('Power (kW)', fontweight='bold')
    ax1.set_title('Power Production - Method Comparison')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Energy accumulation
    ax2 = axes_comparison[0, 1]
    for m in methods:
        cumulative_energy = df_daylight[f'energy_{m}'].cumsum()
        ax2.plot(df_daylight.index, cumulative_energy, 
                color=colors[m], label=f'{m.upper()}: {cumulative_energy.iloc[-1]:.1f}kWh', 
                linewidth=2)
    ax2.set_ylabel('Cumulative Energy (kWh)', fontweight='bold')
    ax2.set_title('Daily Energy Accumulation')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Method differences from combined
    ax3 = axes_comparison[1, 0]
    base_power = df_daylight['power_combined'] / 1000
    for m in [m for m in methods if m != 'combined']:
        diff = (df_daylight[f'power_{m}']/1000 - base_power)
        ax3.plot(df_daylight.index, diff, color=colors[m], 
                label=f'{m.upper()} - Combined', alpha=0.7)
    ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_ylabel('Difference from Combined (kW)', fontweight='bold')
    ax3.set_title('Method Differences')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance ratio
    ax4 = axes_comparison[1, 1]
    for m in methods:
        performance_ratio = (df_daylight[f'power_{m}']/1000) / max_calc['total_max_kw'] * 100
        ax4.plot(df_daylight.index, performance_ratio, 
                color=colors[m], label=f'{m.upper()}', alpha=0.7)
    ax4.set_ylabel('Performance Ratio (%)', fontweight='bold')
    ax4.set_title('Performance Ratio vs Theoretical Max')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 100)
    
    # Format x-axes
    for ax in axes_comparison.flat:
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M\n%d/%m'))
    
    plt.tight_layout()
    st.pyplot(fig_comparison)

    # --- SYSTEM PERFORMANCE PLOTS ---
    st.markdown("### 📈 Power, Energy and Battery Analysis")

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(
        f"Tesla Powerwall: {powerwall_mode} Mode - {datetime.now().strftime('%d %b %Y')}",
        fontsize=16,
        fontweight="bold",
    )

    # Plot 1: Solar Production and House Demand
    ax1 = axes[0, 0]
    # Plot solar production
    ax1.plot(
        df_daylight.index,
        solar_power_kw[df_comparison["day_mask"]],
        color='gold',
        linewidth=2,
        label='Solar Production'
    )
    # Plot house demand
    ax1.fill_between(
        df_comparison.index,
        0,
        df_comparison["house_demand_kw"],
        alpha=0.3,
        color='red',
        label='House Demand'
    )
    # Highlight peak hours
    peak_mask = df_comparison["peak_hours"]
    if peak_mask.any():
        ax1.fill_between(
            df_comparison.index[peak_mask],
            0,
            df_comparison["house_demand_kw"][peak_mask],
            alpha=0.5,
            color='darkred',
            label='Peak Hours'
        )
    
    ax1.set_ylabel("Power (kW)", fontweight="bold")
    ax1.set_title("Solar Production vs House Demand")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Battery State of Charge
    ax2 = axes[0, 1]
    ax2.plot(df_comparison.index, df_comparison["battery_soc_percent"], 
             color='green', linewidth=2, label='Battery SOC')
    ax2.fill_between(df_comparison.index, 0, df_comparison["battery_soc_percent"], 
                     alpha=0.3, color='green')
    
    # Add reserve level line for Time-Based Control
    if powerwall_mode == "Time-Based Control":
        ax2.axhline(y=reserve_percent, color='red', linestyle='--', alpha=0.7, 
                   label=f'Reserve ({reserve_percent}%)')
    
    ax2.set_ylabel("Battery SOC (%)", fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.set_title("Battery State of Charge")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Grid Import/Export
    ax3 = axes[1, 0]
    # Stack import and export
    import_mask = df_comparison["grid_import_kw"] > 0
    export_mask = df_comparison["grid_export_kw"] > 0
    
    if import_mask.any():
        ax3.bar(df_comparison.index[import_mask], df_comparison["grid_import_kw"][import_mask], 
                width=0.03, color='red', alpha=0.6, label='Grid Import')
    if export_mask.any():
        ax3.bar(df_comparison.index[export_mask], -df_comparison["grid_export_kw"][export_mask], 
                width=0.03, color='green', alpha=0.6, label='Grid Export')
    
    ax3.axhline(0, color='black', linewidth=0.5)
    ax3.set_ylabel("Grid Power (kW)", fontweight="bold")
    ax3.set_title("Grid Import/Export")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Export Potential
    ax4 = axes[1, 1]
    ax4.plot(df_comparison.index, df_comparison["export_potential_kw"], 
             color='orange', linewidth=2, label='Export Potential')
    ax4.fill_between(df_comparison.index, 0, df_comparison["export_potential_kw"], 
                     alpha=0.3, color='orange')
    
    # Add cumulative export value
    ax4_twin = ax4.twinx()
    cumulative_export = np.cumsum(df_comparison["export_potential_kw"]) * export_tariff
    ax4_twin.plot(df_comparison.index, cumulative_export, 
                  color='purple', linestyle='--', label='Cumulative Export Value')
    ax4_twin.set_ylabel("Export Value (£)", color='purple')
    ax4_twin.tick_params(axis='y', labelcolor='purple')
    
    ax4.set_ylabel("Export Power (kW)", fontweight="bold")
    ax4.set_title("Export Potential and Value")
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Time-Based Control Features (if applicable)
    ax5 = axes[2, 0]
    if powerwall_mode == "Time-Based Control" or powerwall_mode == "Smart Charging":
        # Plot low tariff periods
        low_tariff_mask = df_comparison.get("low_tariff", np.zeros(len(df_comparison), dtype=bool))
        if low_tariff_mask.any():
            for start_time in df_comparison.index[low_tariff_mask]:
                ax5.axvspan(start_time, start_time + timedelta(hours=1), 
                           alpha=0.2, color='blue', label='Low Tariff' if start_time == df_comparison.index[low_tariff_mask][0] else "")
        
        # Plot grid charging
        grid_charge_mask = df_comparison.get("grid_charge_kw", np.zeros(len(df_comparison))) > 0
        if grid_charge_mask.any():
            ax5.bar(df_comparison.index[grid_charge_mask], 
                   df_comparison["grid_charge_kw"][grid_charge_mask], 
                   width=0.03, color='cyan', alpha=0.7, label='Grid Charging')
        
        ax5.set_ylabel("Power (kW)", fontweight="bold")
        ax5.set_title("Time-Based Control: Low Tariff & Grid Charging")
        ax5.legend()
    else:
        # Plot cloud transmission for Self-Powered mode
        ax5.plot(df_daylight.index, df_daylight["cloud_transmission"], 
                color='blue', linewidth=2, label='Cloud Transmission')
        ax5.fill_between(df_daylight.index, 0, df_daylight["cloud_transmission"], 
                        alpha=0.3, color='blue')
        ax5.set_ylabel("Transmission Factor", fontweight="bold")
        ax5.set_title("Cloud Transmission")
        ax5.set_ylim(0, 1)
        ax5.legend()
    
    ax5.grid(True, alpha=0.3)

    # Plot 6: Net Load Profile
    ax6 = axes[2, 1]
    net_load = df_comparison["net_load_kw"]
    positive_mask = (net_load >= 0) & df_comparison["day_mask"]
    negative_mask = (net_load < 0) & df_comparison["day_mask"]
    
    if positive_mask.any():
        ax6.bar(df_comparison.index[positive_mask], net_load[positive_mask], 
                width=0.03, color='red', alpha=0.6, label='Grid Import Needed')
    if negative_mask.any():
        ax6.bar(df_comparison.index[negative_mask], net_load[negative_mask], 
                width=0.03, color='green', alpha=0.6, label='Excess Solar')
    
    ax6.axhline(0, color='black', linewidth=0.5)
    ax6.set_ylabel("Net Load (kW)", fontweight="bold")
    ax6.set_xlabel("Time")
    ax6.set_title("Net Load Profile (House Demand - Solar)")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # --- DAILY SUMMARY ---
    st.markdown("### 📅 Daily Energy Summary")
    
    daily_summary = pd.DataFrame()
    for m in methods:
        daily_summary[f'{m}_kwh'] = df_comparison[f'energy_{m}'].resample('D').sum()
    
    daily_summary['grid_import_kwh'] = df_comparison['grid_import_kw'].resample('D').sum()
    daily_summary['grid_export_kwh'] = df_comparison['grid_export_kw'].resample('D').sum()
    daily_summary['self_consumption_kwh'] = df_comparison['self_consumption_kw'].resample('D').sum()
    daily_summary['export_potential_kwh'] = df_comparison['export_potential_kw'].resample('D').sum()
    
    # Calculate financials
    daily_summary['export_value_gbp'] = daily_summary['export_potential_kwh'] * export_tariff
    daily_summary['self_sufficiency_%'] = (daily_summary['self_consumption_kwh'] / 
                                          (daily_summary[f'{method}_kwh'] + 1e-6) * 100)
    
    st.dataframe(daily_summary.round(1), use_container_width=True)

    # --- RECOMMENDATIONS ---
    st.markdown("### 💡 Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔋 Battery Optimization")
        
        # Find best charging times
        excess_solar_mask = (solar_power_kw > baseload) & df_comparison["day_mask"]
        if excess_solar_mask.any():
            best_charge_hours = pd.Series(solar_power_kw - baseload)[excess_solar_mask]
            best_charge_times = best_charge_hours.nlargest(3).index
            
            st.write("**Best times for solar charging:**")
            for time in best_charge_times:
                excess = solar_power_kw[time] - baseload
                st.write(f"- {time.strftime('%H:%M')}: {excess:.2f} kW excess solar")
        
        if powerwall_mode == "Time-Based Control":
            st.write("**Time-Based Control Strategy:**")
            st.write(f"- Reserve level: {reserve_percent}%")
            st.write(f"- Low tariff charging: {low_tariff_start.strftime('%H:%M')} to {low_tariff_end.strftime('%H:%M')}")
            if 'overnight_charge_target_percent' in battery_results:
                st.write(f"- Target overnight charge: {battery_results['overnight_charge_target_percent']:.0f}%")
        
        if show_prices and charging_windows:
            st.write("**Smart Charging Opportunities:**")
            for i, window in enumerate(charging_windows[:2]):
                start_str = window['start'].strftime('%H:%M')
                end_str = (window['end'] + timedelta(hours=1)).strftime('%H:%M')
                st.write(f"- {start_str}-{end_str}: £{window['avg_price']:.0f}/MWh ({window['duration_hours']}h)")
    
    with col2:
        st.markdown("#### ⚡ System Performance")
        
        # Method recommendation
        avg_powers = {m: df_comparison[f'power_{m}'].mean()/1000 for m in methods}
        recommended_method = max(avg_powers.items(), key=lambda x: x[1])[0]
        
        st.write("**Estimation Method Analysis:**")
        st.write(f"- Recommended method: **{recommended_method.upper()}**")
        st.write(f"- Selected method: **{method.upper()}**")
        st.write(f"- System efficiency: {system_efficiency:.1f}% of theoretical max")
        
        # Improvement potential
        improvement_potential = max_calc['total_max_kw'] - current_method_peak
        if improvement_potential > 0:
            st.write(f"- Potential improvement: {improvement_potential:.2f} kW")
        
        # Tomorrow's forecast
        if len(df_comparison) > 24:
            tomorrow_power = df_comparison['power_combined'][24:48].max() / 1000
            tomorrow_energy = df_comparison['energy_combined'][24:48].sum()
            st.write("**Tomorrow's Forecast:**")
            st.write(f"- Expected peak: {tomorrow_power:.2f} kW")
            st.write(f"- Expected energy: {tomorrow_energy:.1f} kWh")
        
        # Price-based recommendations
        if show_prices and price_stats:
            st.write("**Price Strategy:**")
            if price_stats.get('price_trend', 0) > 5:
                st.write("- ⚠️ Prices trending higher - consider charging now")
            elif price_stats.get('price_trend', 0) < -5:
                st.write("- ✅ Prices trending lower - can wait for better rates")
            
            if price_stats.get('low_price_hours_count', 0) >= 6:
                st.write("- ✅ Good opportunity for overnight charging")

    # Export results
    results_summary = {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "total_kwp": TOTAL_KWP,
            "battery_capacity_kwh": float(battery_capacity),
            "baseload_kw": float(baseload),
            "powerwall_mode": powerwall_mode,
            "estimation_method": method,
            "historical_ratio": float(historical_ratio),
        },
        "performance": {
            "system_efficiency_percent": float(system_efficiency),
            "improvement_potential_kw": float(improvement_potential),
            "recommended_method": recommended_method,
        },
        "price_data": {
            "current_price_gbp_per_mwh": float(price_stats.get('current_price', 0)) if price_stats else 0,
            "avg_price_today_gbp_per_mwh": float(price_stats.get('avg_price_today', 0)) if price_stats else 0,
            "price_trend_percent": float(price_stats.get('price_trend', 0)) if price_stats else 0,
            "smart_charging_windows": charging_windows if show_prices else [],
        },
        "battery_performance": {
            "final_soc_percent": float(battery_results["battery_soc_percent"][-1]),
            "self_sufficiency_percent": float(battery_results["self_sufficiency_percent"]),
            "total_grid_import_kwh": float(battery_results["total_grid_import_kwh"]),
            "total_grid_export_kwh": float(battery_results["total_grid_export_kwh"]),
        },
        "export_potential": {
            "total_export_kwh": float(export_results['total_export_kwh']),
            "max_export_kw": float(export_results['max_export_kw']),
            "export_revenue_gbp": float(export_results['export_revenue_gbp']),
            "export_tariff_gbp_per_kwh": float(export_tariff),
        },
        "method_predictions": {
            m: {
                "peak_kw": float(df_comparison[f'power_{m}'].max()/1000),
                "today_kwh": float(df_comparison[f'energy_{m}'][:24].sum()),
            } for m in methods
        },
        "daily_summary": daily_summary.to_dict(orient="index"),
    }

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download JSON results",
            data=json.dumps(results_summary, indent=2),
            file_name=f"powerwall_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
        )
    
    with col2:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()


if __name__ == "__main__":
    run_app()
