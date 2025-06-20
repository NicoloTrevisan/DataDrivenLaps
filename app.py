#!/usr/bin/env python3
"""
F1 Data Driven Laps - Interactive Streamlit Application

Copyright (c) 2025 Nico - F1 Data Driven Laps
Licensed under the MIT License - see LICENSE file for details.

This application creates interactive F1 telemetry visualizations using FastF1 data.
"""

import streamlit as st
import fastf1
import fastf1.plotting
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from scipy.interpolate import interp1d
import tempfile
import requests
from pathlib import Path
import io
import base64
import warnings

# Suppress deprecation warnings to keep the UI clean
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Page configuration
st.set_page_config(
    page_title="F1 Data Driven Laps",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF0000;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .stSelectbox > div > div {
            font-size: 14px;
        }
        .stButton > button {
            font-size: 16px;
            padding: 0.75rem 1rem;
        }
        .stMetric {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    
    /* Desktop optimizations */
    @media (min-width: 769px) {
        .stSidebar {
            width: 350px;
        }
        .main .block-container {
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 1200px;
            margin: 0 auto;
        }
    }
    
    /* Image container improvements */
    .stImage > img {
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    
    /* Improved button styling */
    .stButton > button {
        background: linear-gradient(90deg, #FF0000, #DC0000);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #DC0000, #B71C1C);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Improved metrics styling */
    .stMetric {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Better info boxes */
    .stInfo {
        border-left: 4px solid #17a2b8;
    }
    
    /* Loading spinner improvements */
    .stSpinner > div {
        color: #FF0000 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_drivers' not in st.session_state:
    st.session_state.selected_drivers = []
if 'session_loaded' not in st.session_state:
    st.session_state.session_loaded = False
if 'image_generated' not in st.session_state:
    st.session_state.image_generated = False

# Team colors mapping (from original code)
TEAM_COLORS_MAPPING = {
    'Red Bull Racing': '#0600EF',
    'Mercedes': '#00D2BE',
    'Ferrari': '#DC0000',
    'McLaren': '#FF8700',
    'Aston Martin': '#006F62',
    'Alpine': '#0090FF',
    'Williams': '#005AFF',
    'AlphaTauri': '#2B4562',
    'Kick Sauber': '#00E676',
    'Haas F1 Team': '#B6BABD'
}

DEFAULT_COLORS = ['#FF0000', '#0000FF', '#00FF00', '#FFA500', '#800080', '#FFFF00']

def format_lap_time(seconds):
    """Format seconds to MM:SS.mmm"""
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes}:{remaining_seconds:06.3f}"

def setup_environment():
    """Setup matplotlib and FastF1 cache"""
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 150
    
    try:
        fastf1.plotting.setup_mpl(color_scheme='fastf1', misc_mpl_mods=False)
    except TypeError:
        fastf1.plotting.setup_mpl(misc_mpl_mods=False)
    
    # Setup cache
    cache_path = os.path.expanduser('~/cache_fastf1_streamlit')
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    fastf1.Cache.enable_cache(cache_path)

def adjust_color_luminance(color_hex, factor):
    """Adjust the luminance of a color by a factor"""
    try:
        color_hex = color_hex.lstrip('#')
        r = int(color_hex[0:2], 16) / 255.0
        g = int(color_hex[2:4], 16) / 255.0
        b = int(color_hex[4:6], 16) / 255.0
        
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        diff = max_val - min_val
        
        lightness = (max_val + min_val) / 2
        
        if diff == 0:
            hue = saturation = 0
        else:
            saturation = diff / (2 - max_val - min_val) if lightness > 0.5 else diff / (max_val + min_val)
            
            if max_val == r:
                hue = (g - b) / diff + (6 if g < b else 0)
            elif max_val == g:
                hue = (b - r) / diff + 2
            else:
                hue = (r - g) / diff + 4
            hue /= 6
        
        new_lightness = min(0.9, max(0.1, lightness * factor))
        
        def hsl_to_rgb_component(p, q, t):
            if t < 0: t += 1
            if t > 1: t -= 1
            if t < 1/6: return p + (q - p) * 6 * t
            if t < 1/2: return q
            if t < 2/3: return p + (q - p) * (2/3 - t) * 6
            return p
        
        if saturation == 0:
            new_r = new_g = new_b = new_lightness
        else:
            q = new_lightness * (1 + saturation) if new_lightness < 0.5 else new_lightness + saturation - new_lightness * saturation
            p = 2 * new_lightness - q
            new_r = hsl_to_rgb_component(p, q, hue + 1/3)
            new_g = hsl_to_rgb_component(p, q, hue)
            new_b = hsl_to_rgb_component(p, q, hue - 1/3)
        
        new_r = int(new_r * 255)
        new_g = int(new_g * 255)
        new_b = int(new_b * 255)
        
        return f"#{new_r:02x}{new_g:02x}{new_b:02x}"
        
    except Exception:
        return color_hex

def get_team_color(driver_code, session, drivers_to_plot, is_secondary=False):
    """Get the team color for a driver"""
    team_name = None
    base_color = None
    
    try:
        driver_info = session.get_driver(driver_code)
        if driver_info is not None and not driver_info.empty:
            team_name = driver_info['TeamName']
            
            if team_name in TEAM_COLORS_MAPPING:
                base_color = TEAM_COLORS_MAPPING[team_name]
                
                if is_secondary:
                    return adjust_color_luminance(base_color, 1.4)
                else:
                    return adjust_color_luminance(base_color, 0.85)
    except Exception:
        pass

    try:
        color = fastf1.plotting.get_driver_color(driver_code, session=session)
        generic_colors = ['#000000', '#ffffff', '#808080', '#B6BABD']
        
        if driver_info is not None and not driver_info.empty:
            if color.upper() in generic_colors and driver_info['TeamName'] != 'Haas F1 Team':
                pass
            else:
                if is_secondary and base_color is None:
                    return adjust_color_luminance(color, 1.3)
                else:
                    return color
        else:
            if color.upper() not in generic_colors:
                if is_secondary:
                    return adjust_color_luminance(color, 1.3)
                else:
                    return color
    except Exception:
        pass
    
    driver_index = drivers_to_plot.index(driver_code) if driver_code in drivers_to_plot else 0
    fallback_color = DEFAULT_COLORS[driver_index % len(DEFAULT_COLORS)]
    
    if is_secondary:
        return adjust_color_luminance(fallback_color, 1.4)
    else:
        return adjust_color_luminance(fallback_color, 0.9)

@st.cache_data(show_spinner=False)
def get_available_gps(year):
    """Get available Grand Prix for a given year"""
    try:
        schedule = fastf1.get_event_schedule(year)
        gp_names = schedule['EventName'].tolist()
        return gp_names
    except Exception as e:
        st.error(f"Error loading Grand Prix list: {str(e)}")
        return []

@st.cache_data(show_spinner=False)
def get_available_sessions(year, gp_name):
    """Get available sessions for a given year and GP"""
    try:
        schedule = fastf1.get_event_schedule(year)
        event_data = schedule[schedule['EventName'] == gp_name].iloc[0]
        
        # Standard session names
        sessions = []
        session_mapping = {
            'Practice 1': 'FP1',
            'Practice 2': 'FP2', 
            'Practice 3': 'FP3',
            'Qualifying': 'Qualifying',
            'Sprint': 'Sprint',
            'Sprint Qualifying': 'Sprint Qualifying',
            'Race': 'Race'
        }
        
        for session_name in ['Practice 1', 'Practice 2', 'Practice 3', 'Sprint Qualifying', 'Sprint', 'Qualifying', 'Race']:
            try:
                session = fastf1.get_session(year, gp_name, session_name)
                if session is not None:
                    display_name = session_mapping.get(session_name, session_name)
                    sessions.append((session_name, display_name))
            except:
                continue
        
        return sessions
    except Exception as e:
        st.error(f"Error loading sessions: {str(e)}")
        return []

@st.cache_resource(show_spinner=False)
def load_session_data(year, gp_name, session_name):
    """Load session data with caching"""
    try:
        session = fastf1.get_session(year, gp_name, session_name)
        session.load()
        return session
    except Exception as e:
        raise Exception(f"Failed to load session data: {str(e)}")

@st.cache_data(show_spinner=False)
def get_session_drivers_with_times(_session, year, gp_name, session_name):
    """
    Get available drivers from the session, enriched with their fastest lap times.
    This function is now more robust and relies on lap data as the source of truth.
    Updated to include session parameters for proper cache invalidation.
    """
    try:
        laps = _session.laps
        if laps.empty:
            st.error("No lap data available for this session.")
            return []

        # Get all unique drivers from the laps data
        driver_codes = laps['Driver'].unique()
        
        driver_info_list = []
        for code in driver_codes:
            try:
                # Get driver's fastest lap
                driver_laps = laps.pick_driver(code)
                fastest_lap = driver_laps.pick_fastest()

                if fastest_lap is None or pd.isna(fastest_lap['LapTime']):
                    continue

                # Get driver details
                driver_details = _session.get_driver(code)
                team_name = driver_details.get('TeamName', 'Unknown Team')
                full_name = driver_details.get('FullName', code)

                driver_info_list.append({
                    'code': code,
                    'name': full_name,
                    'team': team_name,
                    'lap_time': fastest_lap.LapTime.total_seconds(),
                    'lap_time_formatted': format_lap_time(fastest_lap.LapTime.total_seconds())
                })
            except Exception:
                # Ignore drivers if they have inconsistent data
                continue
        
        if not driver_info_list:
            st.error("Could not retrieve any valid driver lap times for this session.")
            return []

        # Sort final list by lap time
        driver_info_list.sort(key=lambda x: x['lap_time'])
        return driver_info_list

    except Exception as e:
        st.error(f"A critical error occurred while getting the driver list: {str(e)}")
        # Adding a specific hint for a common FastF1 issue
        if "The data you are trying to access has not been loaded yet" in str(e):
             st.warning("Hint: This might be an issue with loading session data. Try refreshing or selecting a different session.")
        return []

# Compatibility function for existing code
def get_session_drivers(_session):
    """Compatibility wrapper for get_session_drivers_with_times"""
    return get_session_drivers_with_times(_session)

@st.cache_data(show_spinner=False)
def prepare_driver_data(_session, drivers_to_plot, driver_selection_mode):
    """Prepare driver data for analysis"""
    try:
        progress_placeholder = st.empty()
        
        with progress_placeholder.container():
            st.info("🔄 Preparing driver data...")
        
        # Get fastest laps for selected drivers
        driver_data_dict = {}
        
        for driver_code in drivers_to_plot:
            progress_placeholder.info(f"🔄 Loading data for {driver_code}...")
            
            # Try to get driver laps using abbreviation first, then fallback to car number
            driver_laps = _session.laps.pick_driver(driver_code)
            
            # If no laps found with abbreviation, try to find by car number
            if driver_laps.empty:
                # Look for the car number corresponding to this abbreviation
                try:
                    for car_num in _session.drivers:
                        driver_info = _session.get_driver(car_num)
                        if (driver_info is not None and not driver_info.empty and 
                            driver_info.get('Abbreviation') == driver_code):
                            driver_laps = _session.laps.pick_driver(car_num)
                            break
                except:
                    pass
            
            if driver_laps.empty:
                raise Exception(f"No lap data found for driver {driver_code}")
            
            fastest_lap = driver_laps.pick_fastest()
            if fastest_lap.empty:
                raise Exception(f"No valid fastest lap found for driver {driver_code}")
            
            telemetry = fastest_lap.get_telemetry()
            if telemetry.empty:
                raise Exception(f"No telemetry data found for driver {driver_code}")
            
            driver_data_dict[driver_code] = {
                'lap': fastest_lap,
                'telemetry': telemetry,
                'lap_time': fastest_lap['LapTime'].total_seconds(),
                'compound': fastest_lap['Compound']
            }
        
        progress_placeholder.empty()
        return driver_data_dict
        
    except Exception as e:
        raise Exception(f"Failed to prepare driver data: {str(e)}")

def create_data_driven_lap_image(driver_data_dict, _session, drivers_to_plot, year, gp_name, session_display):
    """Create a single PNG image showing data-driven lap visualization"""
    try:
        # Get driver data
        p1_code, p2_code = drivers_to_plot[0], drivers_to_plot[1]
        
        # Get driver colors
        p1_color = get_team_color(p1_code, _session, drivers_to_plot, is_secondary=False)
        p2_color = get_team_color(p2_code, _session, drivers_to_plot, is_secondary=True)
        
        # Create figure (mobile layout for social media)
        fig = plt.figure(figsize=(12, 21), facecolor='#000000')
        gs = fig.add_gridspec(5, 1, height_ratios=[0.4, 6.8, 1.2, 0.8, 0.8], hspace=0.15)
        
        ax_title = fig.add_subplot(gs[0, 0])
        ax_track = fig.add_subplot(gs[1, 0])
        ax_gear_speed = fig.add_subplot(gs[2, 0])
        ax_tyre_info = fig.add_subplot(gs[3, 0])
        ax_timer = fig.add_subplot(gs[4, 0])
        
        # Style setup
        clean_green = '#00FF00'
        clean_white = '#FFFFFF'
        clean_gray = '#1a1a1a'
        border_color = '#333333'
        
        panel_style = {'facecolor': clean_gray, 'edgecolor': border_color, 'linewidth': 0.5, 'alpha': 0.9}
        for ax in [ax_gear_speed, ax_tyre_info, ax_timer]:
            ax.set_facecolor(panel_style['facecolor'])
            for spine in ax.spines.values():
                spine.set_color(panel_style['edgecolor'])
                spine.set_linewidth(panel_style['linewidth'])
        ax_track.set_facecolor('#0A0A0A')
        
        # Get telemetry data
        p1_tel = driver_data_dict[p1_code]['telemetry']
        p2_tel = driver_data_dict[p2_code]['telemetry']
        
        # ORIGINAL TRACK COLORING LOGIC: Compare speeds at each point
        # Use P1 telemetry as reference for track layout
        track_segments_x = p1_tel['X'].to_numpy()
        track_segments_y = p1_tel['Y'].to_numpy()
        
        # Create track segments
        points = np.array([track_segments_x, track_segments_y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Get P1 speeds
        p1_speeds = p1_tel['Speed'].to_numpy()
        
        # Align P2 speeds to P1 distance for proper comparison
        p1_dist = p1_tel['Distance'].to_numpy()
        p2_dist = p2_tel['Distance'].to_numpy()
        p2_speed_interp = interp1d(p2_dist, p2_tel['Speed'].to_numpy(), 
                                  kind='linear', bounds_error=False, fill_value='extrapolate')
        p2_speeds_aligned = p2_speed_interp(p1_dist)
        
        # Create segment colors based on who is faster at each point
        p1_color_rgba = mcolors.to_rgba(p1_color)
        p2_color_rgba = mcolors.to_rgba(p2_color)
        
        segment_colors = []
        for i in range(len(segments)):
            if i < len(p1_speeds) and i < len(p2_speeds_aligned):
                if p1_speeds[i] >= p2_speeds_aligned[i]:
                    segment_colors.append(p1_color_rgba)
                else:
                    segment_colors.append(p2_color_rgba)
            else:
                # Default to neutral color for edge cases
                segment_colors.append(mcolors.to_rgba('#404040'))
        
        # Create and add track with speed-based coloring
        lc = LineCollection(segments, colors=segment_colors, linewidths=10, alpha=0.9, capstyle='round')
        ax_track.add_collection(lc)
        
        # Add car positions at mid-lap points
        mid_point_p1 = len(p1_tel) // 2
        mid_point_p2 = len(p2_tel) // 2
        
        # Car dots (without borders) - positioned at mid-lap
        car_radius = np.mean([track_segments_x.max() - track_segments_x.min(), 
                             track_segments_y.max() - track_segments_y.min()]) * 0.008
        
        car1_circle = Circle((track_segments_x[mid_point_p1], track_segments_y[mid_point_p1]), 
                           radius=car_radius, facecolor=p1_color, edgecolor='none', zorder=10)
        car2_circle = Circle((track_segments_x[mid_point_p2], track_segments_y[mid_point_p2]), 
                           radius=car_radius, facecolor=p2_color, edgecolor='none', zorder=10)
        ax_track.add_patch(car1_circle)
        ax_track.add_patch(car2_circle)
        
        ax_track.set_xlim(track_segments_x.min() - 200, track_segments_x.max() + 200)
        ax_track.set_ylim(track_segments_y.min() - 200, track_segments_y.max() + 200)
        ax_track.set_aspect('equal')
        ax_track.axis('off')
        
        # Add legend for track coloring - positioned above the track
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=p1_color, 
                      markeredgecolor='none', markersize=12, label=f"{p1_code}"),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=p2_color, 
                      markeredgecolor='none', markersize=12, label=f"{p2_code}")
        ]
        ax_track.legend(handles=legend_elements, loc='upper center', ncol=2, 
                       facecolor=clean_gray, edgecolor=border_color, labelcolor=clean_white, 
                       fontsize=14, bbox_to_anchor=(0.5, 1.08))
        
        # Title
        ax_title.text(0.5, 0.5, f'{year} {gp_name} {session_display}', 
                     ha='center', va='center', fontsize=24, color=clean_white, fontweight='bold')
        ax_title.axis('off')
        
        # Speed comparison only (removed brake/throttle)
        ax_gear_speed.plot(p1_tel['Speed'], color=p1_color, linewidth=3, label=f'{p1_code} Speed', alpha=0.8)
        
        # Align P2 speed to P1 length for comparison
        p2_speed_aligned_plot = np.interp(np.linspace(0, len(p2_tel)-1, len(p1_tel)), 
                                         range(len(p2_tel)), p2_tel['Speed'])
        ax_gear_speed.plot(p2_speed_aligned_plot, color=p2_color, linewidth=3, label=f'{p2_code} Speed', alpha=0.8)
        
        ax_gear_speed.set_ylabel('Speed (km/h)', color=clean_white, fontsize=14, fontweight='bold')
        ax_gear_speed.tick_params(colors=clean_white, labelsize=12)
        ax_gear_speed.legend(loc='upper right', facecolor=clean_gray, edgecolor=border_color, labelcolor=clean_white)
        ax_gear_speed.grid(True, alpha=0.15, color=clean_white, linestyle='-', linewidth=0.5)
        
        # Tire information with life
        p1_compound = driver_data_dict[p1_code]['compound'] if driver_data_dict[p1_code]['compound'] else 'UNKNOWN'
        p2_compound = driver_data_dict[p2_code]['compound'] if driver_data_dict[p2_code]['compound'] else 'UNKNOWN'
        
        # Get tire life from lap data
        p1_tyre_life = driver_data_dict[p1_code]['lap'].get('TyreLife', 'N/A')
        p2_tyre_life = driver_data_dict[p2_code]['lap'].get('TyreLife', 'N/A')
        
        # Handle NaN values
        if pd.isna(p1_tyre_life):
            p1_tyre_life = 'N/A'
        else:
            p1_tyre_life = int(p1_tyre_life)
            
        if pd.isna(p2_tyre_life):
            p2_tyre_life = 'N/A'
        else:
            p2_tyre_life = int(p2_tyre_life)
        
        # Tire compound colors
        compound_colors = {
            'SOFT': '#FF0000',     # Red
            'MEDIUM': '#FFFF00',   # Yellow  
            'HARD': '#FFFFFF',     # White
            'INTERMEDIATE': '#00FF00', # Green
            'WET': '#0000FF'       # Blue
        }
        
        p1_compound_color = compound_colors.get(p1_compound, clean_white)
        p2_compound_color = compound_colors.get(p2_compound, clean_white)
        
        # Tire panel
        ax_tyre_info.text(0.25, 0.8, f"{p1_code}", fontsize=16, color=p1_color, fontweight='bold', ha='center')
        ax_tyre_info.text(0.75, 0.8, f"{p2_code}", fontsize=16, color=p2_color, fontweight='bold', ha='center')
        
        # Compound with colored background
        ax_tyre_info.text(0.25, 0.5, p1_compound, fontsize=14, fontweight='bold', color='black', ha='center',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor=p1_compound_color, edgecolor=p1_color, linewidth=1.5))
        ax_tyre_info.text(0.75, 0.5, p2_compound, fontsize=14, fontweight='bold', color='black', ha='center',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor=p2_compound_color, edgecolor=p2_color, linewidth=1.5))
        
        # Tire life
        p1_life_text = f"Life: {p1_tyre_life}" if p1_tyre_life != 'N/A' else "Life: N/A"
        p2_life_text = f"Life: {p2_tyre_life}" if p2_tyre_life != 'N/A' else "Life: N/A"
        ax_tyre_info.text(0.25, 0.2, p1_life_text, fontsize=12, color=clean_white, ha='center')
        ax_tyre_info.text(0.75, 0.2, p2_life_text, fontsize=12, color=clean_white, ha='center')
        ax_tyre_info.axis('off')
        
        # Final info
        ax_timer.text(0.5, 0.5, '🏁 FASTEST LAP COMPARISON', 
                     ha='center', va='center', fontsize=20, color=clean_white, fontweight='bold')
        ax_timer.axis('off')
        
        # Add watermark
        fig.text(0.95, 0.98, '@datadrivenlaps', fontsize=26, color='#FFFFFF', 
                alpha=0.7, ha='right', va='top', fontweight='bold', 
                transform=fig.transFigure, rotation=0)
        
        # Save to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='black')
        buf.seek(0)
        
        plt.close(fig)
        
        return buf
        
    except Exception as e:
        plt.close('all')
        raise Exception(f"Failed to create image: {str(e)}")

# Main App Layout
def main():
    setup_environment()
    
    # Header
    st.markdown('<h1 class="main-header">🏎️ F1 Data Driven Laps</h1>', unsafe_allow_html=True)
    st.markdown("### Compare driver performance with interactive track visualizations")
    
    # Quick guide
    st.info("🚀 **Quick Guide:** Select any two F1 drivers and instantly see which sections of the track each driver is faster on. The track is color-coded to show speed differences - perfect for analyzing racing lines, braking points, and acceleration zones. Generate high-quality images ready for download and sharing!")
    
    # Simplified layout - use sidebar for controls
    with st.sidebar:
        st.header("🏁 Race Selection")
        
        # Year selection
        year = st.number_input("Year", min_value=2018, max_value=2025, value=2025, step=1)
        
        # GP selection
        gp_options = get_available_gps(year)
        if not gp_options:
            st.error("No Grand Prix data available for selected year")
            return
        
        # Try to default to a proper race instead of testing
        default_gp_index = 0
        if gp_options:
            # Look for a proper race (avoid testing sessions)
            for i, gp in enumerate(gp_options):
                if 'testing' not in gp.lower() and 'test' not in gp.lower():
                    default_gp_index = i
                    break
        gp_name = st.selectbox("Grand Prix", gp_options, index=default_gp_index)
        
        # Session selection
        session_options = get_available_sessions(year, gp_name)
        if not session_options:
            st.error("No session data available for selected Grand Prix")
            return
        
        session_display = st.selectbox("Session", [s[1] for s in session_options], index=0)
        session_name = next(s[0] for s in session_options if s[1] == session_display)
        
        st.header("👥 Driver Selection")
        
        # Driver selection mode
        driver_mode = st.radio(
            "Selection Mode",
            ["Specific Drivers", "Teammates", "P1 vs P2"],
            index=2
        )
    
    # Load session for driver selection (common for both layouts)
    # Create a unique session key to avoid cached/stale data
    session_key = f"{year}_{gp_name}_{session_name}"
    if 'current_session_key' not in st.session_state:
        st.session_state.current_session_key = ""
    
    # Reset driver selection if session changed
    if st.session_state.current_session_key != session_key:
        st.session_state.current_session_key = session_key
        st.session_state.session_loaded = False
        
    try:
        with st.spinner("Loading session data..."):
            session = load_session_data(year, gp_name, session_name)
            driver_info = get_session_drivers_with_times(session, year, gp_name, session_name)
            st.session_state.session_loaded = True
            
            # Check if we have driver data
            if not driver_info:
                st.error(f"❌ No driver data available for {year} {gp_name} {session_display}")
                st.info("💡 **Suggestions:**")
                st.info("• Try a different year (2018-2024 have complete data)")
                st.info("• Try a different Grand Prix")
                st.info("• Try a different session (FP1, FP2, Race)")
                st.info("• 2025 data may not be available yet")
                return
                
    except Exception as e:
        st.error(f"Error loading session: {str(e)}")
        st.info("💡 **Suggestions:**")
        st.info("• Try a different year (2018-2024 have complete data)")
        st.info("• Try a different Grand Prix or session")
        return
    
    # Driver selection logic - continue in sidebar
    drivers_to_plot = []
    selection_status = ""
    
    with st.sidebar:
        if driver_mode == "Specific Drivers":
            selected_drivers = st.multiselect(
                "Select 2 drivers",
                [f"{d['code']} - {d['name']} ({d['lap_time_formatted']})" for d in driver_info],
                max_selections=2
            )
            drivers_to_plot = [d.split(' - ')[0] for d in selected_drivers]
            if len(drivers_to_plot) == 2:
                selection_status = f"✅ Selected: {' vs '.join(drivers_to_plot)}"
            elif len(drivers_to_plot) == 1:
                selection_status = f"⚠️ Please select one more driver (currently: {drivers_to_plot[0]})"
            else:
                selection_status = "⚠️ Please select 2 drivers to compare"
            
        elif driver_mode == "Teammates":
            # Get all teams, but be more lenient with filtering
            all_teams = [d['team'] for d in driver_info if d['team']]
            teams = list(set([team for team in all_teams if team and team.strip() != '' and team != 'Unknown']))
            
            if teams:
                selected_team = st.selectbox("Select Team", teams, index=0)
                teammates = [d['code'] for d in driver_info if d['team'] == selected_team]
                if len(teammates) >= 2:
                    drivers_to_plot = teammates[:2]
                    selection_status = f"🤝 Selected: {' vs '.join(drivers_to_plot)}"
                else:
                    selection_status = f"⚠️ Only {len(teammates)} driver(s) found for {selected_team}"
            else:
                # Fallback: show all available teams even if they might be problematic
                all_teams_fallback = list(set([d['team'] for d in driver_info if d['team']]))
                if all_teams_fallback:
                    selected_team = st.selectbox("Select Team", all_teams_fallback, index=0)
                    teammates = [d['code'] for d in driver_info if d['team'] == selected_team]
                    if len(teammates) >= 2:
                        drivers_to_plot = teammates[:2]
                        selection_status = f"🤝 Selected: {' vs '.join(drivers_to_plot)}"
                    else:
                        selection_status = f"⚠️ Only {len(teammates)} driver(s) found for {selected_team}"
                else:
                    selection_status = "⚠️ No teams found"
                
        elif driver_mode == "P1 vs P2":
            # Only auto-select if we have valid driver info and session data
            if driver_info and len(driver_info) >= 2:
                try:
                    # First try using session results for proper P1 vs P2
                    results = session.results
                    if not results.empty:
                        p1_driver = results.iloc[0]['Abbreviation']
                        p2_driver = results.iloc[1]['Abbreviation']
                        # Verify these drivers are in our driver_info list
                        driver_codes = [d['code'] for d in driver_info]
                        if p1_driver in driver_codes and p2_driver in driver_codes:
                            drivers_to_plot = [p1_driver, p2_driver]
                            selection_status = f"🏆 Selected: {p1_driver} (P1) vs {p2_driver} (P2)"
                        else:
                            # Fallback to fastest two from our list
                            p1_driver = driver_info[0]['code']
                            p2_driver = driver_info[1]['code']
                            drivers_to_plot = [p1_driver, p2_driver]
                            selection_status = f"🏆 Selected: {p1_driver} (P1) vs {p2_driver} (P2)"
                    else:
                        # Fallback to fastest two drivers from our sorted list
                        p1_driver = driver_info[0]['code']
                        p2_driver = driver_info[1]['code']
                        drivers_to_plot = [p1_driver, p2_driver]
                        selection_status = f"🏆 Selected: {p1_driver} (P1) vs {p2_driver} (P2)"
                except Exception as e:
                    # Fallback to fastest two drivers from our sorted list
                    try:
                        p1_driver = driver_info[0]['code']
                        p2_driver = driver_info[1]['code']
                        drivers_to_plot = [p1_driver, p2_driver]
                        selection_status = f"🏆 Selected: {p1_driver} (P1) vs {p2_driver} (P2)"
                    except:
                        selection_status = "⚠️ Could not determine P1 vs P2"
            else:
                selection_status = "⚠️ Please wait for session data to load..."
    
    # Display selection status
    if selection_status:
        if "✅" in selection_status or "🏆" in selection_status or "🤝" in selection_status:
            st.success(selection_status)
        else:
            st.warning(selection_status)
    
    # Debug cache clearing button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Clear Cache & Refresh"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.clear()
        st.rerun()
    
    # Main content area
    if len(drivers_to_plot) == 2:
        st.markdown(f'<div class="status-box success-box">✅ Ready to analyze: <strong>{drivers_to_plot[0]} vs {drivers_to_plot[1]}</strong></div>', unsafe_allow_html=True)
        
        # Single column layout for clean design
        if st.button("🖼️ Generate Data Driven Lap Image", type="primary", use_container_width=True):
            try:
                # Prepare data
                with st.spinner("🏎️ Preparing driver data..."):
                    driver_data = prepare_driver_data(session, drivers_to_plot, driver_mode)
                
                # Generate image
                with st.spinner("🏎️ Creating data-driven lap image..."):
                    image_buffer = create_data_driven_lap_image(
                        driver_data, session, drivers_to_plot, year, gp_name, session_display
                    )
                
                # Display image with responsive sizing
                st.subheader("🏁 Your Data Driven Lap Image")
                
                # Add responsive image sizing with CSS
                st.markdown("""
                <style>
                .responsive-image {
                    max-width: 600px;
                    width: 100%;
                    height: auto;
                    margin: 0 auto;
                    display: block;
                }
                @media (max-width: 768px) {
                    .responsive-image {
                        max-width: 100%;
                    }
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Display image with responsive container
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(image_buffer, use_container_width=True)
                
                # Download button
                st.download_button(
                    label="📥 Download High-Resolution Image",
                    data=image_buffer.getvalue(),
                    file_name=f"Data_Driven_Lap_{year}_{gp_name}_{session_display}_{drivers_to_plot[0]}_vs_{drivers_to_plot[1]}.png",
                    mime="image/png",
                    use_container_width=True
                )
                    
            except Exception as e:
                st.error(f"❌ Error generating image: {str(e)}")
    
    else:
        st.markdown('<div class="status-box info-box">ℹ️ Please select exactly 2 drivers to generate a data driven lap image.</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("Made using [FastF1](https://github.com/theOehrly/Fast-F1) and [Streamlit](https://streamlit.io)")

if __name__ == "__main__":
    main() 
