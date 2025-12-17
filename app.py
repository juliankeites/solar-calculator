def identify_smart_charging_windows(self, price_df, solar_power_kw=None, battery_soc=None):
    """Identify optimal windows for smart charging based on prices"""
    if price_df.empty or 'price_gbp_per_mwh' not in price_df.columns:
        return []
    
    # Calculate low price threshold (25th percentile)
    try:
        low_price_threshold = np.percentile(price_df['price_gbp_per_mwh'].values, 25)
    except:
        low_price_threshold = price_df['price_gbp_per_mwh'].min()
    
    # Get low price periods
    low_price_periods = price_df[price_df['price_gbp_per_mwh'] < low_price_threshold]
    
    if low_price_periods.empty:
        return []
    
    # Group consecutive low price hours
    charging_windows = []
    current_window = []
    
    # Sort by index
    low_price_periods = low_price_periods.sort_index()
    
    for idx, row in low_price_periods.iterrows():
        if not current_window:
            current_window = [idx]
        else:
            # Check if this hour is consecutive with the last
            last_time = current_window[-1]
            time_diff = (idx - last_time).total_seconds()
            
            if time_diff <= 3600:  # 1 hour difference (within 1 hour)
                current_window.append(idx)
            else:
                # End of current window
                if len(current_window) >= 1:  # Minimum 1 hour window
                    window_times = price_df.loc[current_window]
                    window_data = {
                        'start': current_window[0],
                        'end': current_window[-1],
                        'duration_hours': len(current_window),
                        'avg_price': float(window_times['price_gbp_per_mwh'].mean()),
                        'min_price': float(window_times['price_gbp_per_mwh'].min()),
                        'max_price': float(window_times['price_gbp_per_mwh'].max()),
                    }
                    charging_windows.append(window_data)
                current_window = [idx]
    
    # Add the last window if it exists
    if current_window and len(current_window) >= 1:
        window_times = price_df.loc[current_window]
        window_data = {
            'start': current_window[0],
            'end': current_window[-1],
            'duration_hours': len(current_window),
            'avg_price': float(window_times['price_gbp_per_mwh'].mean()),
            'min_price': float(window_times['price_gbp_per_mwh'].min()),
            'max_price': float(window_times['price_gbp_per_mwh'].max()),
        }
        charging_windows.append(window_data)
    
    # Sort windows by duration (longest first) or by lowest average price
    charging_windows.sort(key=lambda x: x['duration_hours'], reverse=True)
    
    return charging_windows
