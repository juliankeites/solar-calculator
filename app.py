import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Solar Production Calculator",
    page_icon="☀️",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
<style>
.stButton>button {
    background-color: #FF4B4B;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    padding: 10px 24px;
}
.stButton>button:hover {
    background-color: #FF3333;
    color: white;
}
.css-1d391kg {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# Title and header
st.title("☀️ Free Solar Production Calculator")
st.markdown("**Estimate your solar system's monthly and annual production**")
st.markdown("---")

# Sidebar for inputs
with st.sidebar:
    st.header("📍 Location Settings")
    
    # Location input with two methods
    method = st.radio("Location Input Method:", ["Coordinates", "Address Lookup"])
    
    if method == "Coordinates":
        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Latitude", value=51.32, min_value=-90.0, max_value=90.0, step=0.01, help="Positive for North, Negative for South")
        with col2:
            lon = st.number_input("Longitude", value=-0.56, min_value=-180.0, max_value=180.0, step=0.01, help="Positive for East, Negative for West")
    else:
        address = st.text_input("Enter Address:", "London, UK")
        if st.button("Geocode Address"):
            # Simplified geocoding - in real app, use geopy or similar
            st.info("Using London coordinates for demo. Real app would convert address.")
            lat = 51.32
            lon = -0.56
        else:
            lat = 51.32
            lon = -0.56
    
    st.header("🔋 System Configuration")
    total_kwp = st.number_input("Total System Size (kWp)", value=11.88, min_value=0.1, max_value=100.0, step=0.1, help="Total kilowatt-peak of your system")
    module_wattage = st.slider("Module Wattage (W)", 300, 500, 440, step=10)
    live_peak = st.number_input("Your Actual Peak (kW)", value=1.8, min_value=0.1, max_value=100.0, step=0.1, help="Peak power you've actually measured")
    
    st.header("📐 Solar Arrays Configuration")
    num_arrays = st.slider("Number of Arrays", 1, 6, 4)
    
    arrays = []
    for i in range(num_arrays):
        st.subheader(f"Array {i+1}")
        
        col_name, col_modules = st.columns(2)
        with col_name:
            name = st.text_input(f"Name", value=f"Array {i+1}", key=f"name_{i}")
        with col_modules:
            modules = st.number_input(f"Modules", value=7 if i<4 else 0, min_value=0, max_value=100, key=f"modules_{i}")
        
        col_tilt, col_azimuth = st.columns(2)
        with col_tilt:
            tilt = st.slider(f"Tilt (°)", 0, 90, 45, key=f"tilt_{i}")
        with col_azimuth:
            # Default azimuths for common orientations
            azimuth_options = {"South": 180, "East": 90, "West": 270, "North": 0, "SE": 135, "SW": 225, "NE": 45, "NW": 315}
            azimuth_choice = st.selectbox(f"Orientation", list(azimuth_options.keys()), 
                                         index=min(i, 3) if i<4 else 0, key=f"azimuth_choice_{i}")
            azimuth = azimuth_options[azimuth_choice]
        
        loss = st.slider(f"Loss Factor", 0.7, 1.0, 0.85, key=f"loss_{i}")
        
        if modules > 0:
            arrays.append({
                'name': name,
                'n_modules': modules,
                'tilt': tilt,
                'azimuth': azimuth,
                'loss_factor': loss
            })
    
    st.header("📅 Analysis Settings")
    year = st.selectbox("Year for Analysis", range(2023, 2031), index=2)
    
    # Weather model selection
    weather_model = st.selectbox("Weather Model", 
                                ["UK Typical", "Optimistic", "Conservative", "Custom"])
    
    if weather_model == "Custom":
        cloud_factor = st.slider("Average Cloud Cover (%)", 0, 100, 60)
    
    calculate = st.button("🚀 Calculate Solar Production", type="primary", use_container_width=True)

# Main content area
if calculate:
    with st.spinner("Calculating solar production... This may take a few seconds"):
        
        # Show system summary
        st.header("📊 System Summary")
        
        total_modules = sum(a['n_modules'] for a in arrays)
        calculated_kwp = total_modules * module_wattage / 1000
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Arrays", num_arrays)
        with col2:
            st.metric("Total Modules", total_modules)
        with col3:
            st.metric("Calculated kWp", f"{calculated_kwp:.2f}")
        with col4:
            st.metric("Input kWp", f"{total_kwp:.2f}")
        
        # Calculate monthly production (simplified model)
        # In a real app, this would use proper solar calculations
        
        # UK monthly adjustment factors (typical weather)
        uk_monthly_factors = {
            1: 0.20, 2: 0.25, 3: 0.35, 4: 0.45, 5: 0.50,
            6: 0.55, 7: 0.60, 8: 0.55, 9: 0.45, 10: 0.35,
            11: 0.25, 12: 0.20
        }
        
        # Adjust based on selected weather model
        if weather_model == "Optimistic":
            uk_monthly_factors = {k: v * 1.2 for k, v in uk_monthly_factors.items()}
        elif weather_model == "Conservative":
            uk_monthly_factors = {k: v * 0.8 for k, v in uk_monthly_factors.items()}
        elif weather_model == "Custom":
            base_factor = 1 - (cloud_factor / 100)
            uk_monthly_factors = {k: base_factor for k in uk_monthly_factors.keys()}
        
        monthly_stats = {}
        
        for month in range(1, 13):
            # Simplified solar calculation based on latitude and month
            # Base production varies with season
            day_of_year = (month - 1) * 30 + 15
            declination = 23.45 * np.sin(np.radians(360 * (day_of_year - 81) / 365))
            
            # Hour angle at solar noon
            hour_angle = 0
            
            # Solar zenith angle
            lat_rad = np.radians(lat)
            dec_rad = np.radians(declination)
            cos_zenith = (np.sin(lat_rad) * np.sin(dec_rad) + 
                        np.cos(lat_rad) * np.cos(dec_rad) * np.cos(np.radians(hour_angle)))
            cos_zenith = max(0, cos_zenith)
            
            # Clearsky production
            clearsky_factor = cos_zenith * (1 + 0.5 * (1 - abs(lat)/90))
            base_production = total_kwp * 4 * clearsky_factor
            
            # Adjust for tilt and orientation (simplified)
            orientation_factor = 1.0
            for array in arrays:
                if array['tilt'] > 0:
                    # Simple tilt adjustment
                    tilt_factor = 1 + (array['tilt'] / 90) * 0.3
                    orientation_factor *= tilt_factor
            
            base_production *= orientation_factor / len(arrays) if arrays else 1
            
            # Apply UK weather factors
            uk_factor = uk_monthly_factors[month]
            
            monthly_stats[month] = {
                'month_name': calendar.month_name[month],
                'energy_clearsky': base_production,
                'energy_uk': base_production * uk_factor,
                'peak_clearsky': total_kwp * 0.8 * clearsky_factor,
                'peak_uk': total_kwp * 0.8 * clearsky_factor * uk_factor,
                'sun_hours': base_production * uk_factor / total_kwp if total_kwp > 0 else 0
            }
        
        # Display results
        st.success("✅ Calculation complete!")
        
        # Annual Summary Cards
        st.header("📈 Annual Summary")
        
        annual_clearsky = sum(stats['energy_clearsky'] * 30.5 for stats in monthly_stats.values())
        annual_uk = sum(stats['energy_uk'] * 30.5 for stats in monthly_stats.values())
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Annual Production", f"{annual_uk:,.0f} kWh", 
                     f"{(annual_uk/annual_clearsky*100-100):+.0f}% vs clearsky")
        with col2:
            avg_daily = annual_uk / 365
            st.metric("Average Daily", f"{avg_daily:.1f} kWh")
        with col3:
            capacity_factor = (annual_uk / (total_kwp * 24 * 365)) * 100
            st.metric("Capacity Factor", f"{capacity_factor:.1f}%")
        with col4:
            performance_ratio = (live_peak / total_kwp) * 100 if total_kwp > 0 else 0
            st.metric("Performance Ratio", f"{performance_ratio:.0f}%")
        
        # Monthly Production Chart
        st.header("📅 Monthly Production Forecast")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        months = list(monthly_stats.keys())
        month_names = [calendar.month_abbr[m] for m in months]
        
        x = np.arange(len(months))
        width = 0.35
        
        clearsky = [monthly_stats[m]['energy_clearsky'] for m in months]
        uk_actual = [monthly_stats[m]['energy_uk'] for m in months]
        
        bars1 = ax.bar(x - width/2, clearsky, width, label='Clearsky Maximum', 
                      color='#FFA500', alpha=0.8)
        bars2 = ax.bar(x + width/2, uk_actual, width, label=f'UK {weather_model}', 
                      color='#1E90FF', alpha=0.8)
        
        ax.set_xlabel('Month', fontweight='bold', fontsize=12)
        ax.set_ylabel('Daily Energy Production (kWh)', fontweight='bold', fontsize=12)
        ax.set_title(f'Monthly Average Daily Production - {year}', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(month_names, fontsize=11)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on top of bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                if height > 0.1:
                    ax.text(bar.get_x() + bar.get_width()/2, height,
                            f'{height:.1f}', ha='center', va='bottom', 
                            fontsize=9, fontweight='bold')
        
        autolabel(bars1)
        autolabel(bars2)
        
        st.pyplot(fig)
        
        # Monthly Data Table
        st.header("📋 Monthly Breakdown")
        
        table_data = []
        for month, stats in monthly_stats.items():
            table_data.append({
                'Month': stats['month_name'],
                'Clearsky (kWh/day)': f"{stats['energy_clearsky']:.1f}",
                f'UK {weather_model} (kWh/day)': f"{stats['energy_uk']:.1f}",
                'UK Peak (kW)': f"{stats['peak_uk']:.1f}",
                'Sun Hours': f"{stats['sun_hours']:.1f}",
                'Efficiency': f"{(stats['energy_uk']/stats['energy_clearsky']*100 if stats['energy_clearsky']>0 else 0):.0f}%"
            })
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, height=400)
        
        # Performance Analysis
        st.header("⚡ Performance Analysis")
        
        expected_peak = max(stats['peak_uk'] for stats in monthly_stats.values())
        expected_annual = annual_uk
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Expected Annual", f"{expected_annual:,.0f} kWh")
        with col2:
            st.metric("Expected Peak", f"{expected_peak:.1f} kW")
        with col3:
            diff_percent = ((live_peak - expected_peak) / expected_peak) * 100
            st.metric("Your Actual Peak", f"{live_peak:.1f} kW", 
                     f"{diff_percent:+.1f}% vs expected")
        
        # Financial Estimates
        st.header("💰 Financial Estimates")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            electricity_price = st.number_input("Electricity Price (£/kWh)", 
                                              value=0.34, min_value=0.01, 
                                              max_value=1.0, step=0.01)
        with col2:
            feed_in_tariff = st.number_input("Export Tariff (£/kWh)", 
                                            value=0.15, min_value=0.0, 
                                            max_value=1.0, step=0.01)
        with col3:
            self_consumption = st.slider("Self-Consumption (%)", 0, 100, 50)
        
        savings = (annual_uk * self_consumption/100 * electricity_price +
                  annual_uk * (100-self_consumption)/100 * feed_in_tariff)
        
        st.info(f"**Estimated Annual Savings:** £{savings:,.0f}")
        
        # Export Options
        st.header("💾 Export Results")
        
        # Create downloadable CSV
        csv = df.to_csv(index=False).encode('utf-8')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"solar_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        with col2:
            # Generate report summary
            report = f"""
            SOLAR PRODUCTION ANALYSIS REPORT
            ================================
            Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            Location: {lat}°N, {lon}°E
            System Size: {total_kwp} kWp
            Number of Arrays: {num_arrays}
            
            ANNUAL SUMMARY:
            - Annual Production: {annual_uk:,.0f} kWh
            - Average Daily: {avg_daily:.1f} kWh
            - Capacity Factor: {capacity_factor:.1f}%
            - Estimated Savings: £{savings:,.0f}/year
            
            PERFORMANCE:
            - Expected Peak: {expected_peak:.1f} kW
            - Your Actual Peak: {live_peak:.1f} kW
            - Performance: {performance_ratio:.0f}%
            
            Best Month: {monthly_stats[np.argmax([m['energy_uk'] for m in monthly_stats.values()])+1]['month_name']}
            Worst Month: {monthly_stats[np.argmin([m['energy_uk'] for m in monthly_stats.values()])+1]['month_name']}
            """
            
            st.download_button(
                label="📄 Download Report",
                data=report,
                file_name=f"solar_report_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
        with col3:
            if st.button("🖨️ Print Summary"):
                st.balloons()
                st.success("Print dialog should open (if supported by browser)")

else:
    # Show welcome/instructions
    st.header("Welcome to the Solar Production Calculator!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 What This Tool Does:
        
        1. **Estimates** your solar system's production
        2. **Calculates** monthly and annual energy output
        3. **Compares** your actual vs expected performance
        4. **Provides** financial estimates
        5. **Exports** reports for planning
        
        ### 📊 What You'll Get:
        
        - Monthly production charts
        - Annual energy totals
        - Peak power expectations
        - Financial savings estimates
        - Performance analysis
        """)
    
    with col2:
        st.markdown("""
        ### 🛠️ How to Use:
        
        1. **Enter Location** (coordinates or address)
        2. **Configure System** (size, modules, etc.)
        3. **Set Up Arrays** (tilt, orientation, losses)
        4. **Click Calculate** to see results
        5. **Export** reports if needed
        
        ### 💡 Tips for Best Results:
        
        - Use accurate location coordinates
        - Input real system specifications
        - Consider actual weather patterns
        - Compare with your actual data
        
        ### ⚡ Quick Start:
        
        Just use the default values first to see how it works!
        """)
    
    # Quick example
    st.markdown("---")
    if st.button("🚀 Try Example Configuration", type="secondary"):
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    Made with ❤️ using Streamlit • 
    This tool provides estimates only • 
    Actual production may vary based on local conditions
    </div>
    """, 
    unsafe_allow_html=True
)
