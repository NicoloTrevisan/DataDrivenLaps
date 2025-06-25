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
    }
    
    /* Desktop optimizations */
    @media (min-width: 769px) {
        .stSidebar {
            width: 350px;
        }
        .main .block-container {
            padding-left: 2rem;
            padding-right: 2rem;
        }
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
    # Input validation
    if not color_hex or not isinstance(color_hex, str):
        return '#FF0000'  # Emergency fallback
    
    if not isinstance(factor, (int, float)) or factor <= 0:
        return color_hex  # Return original if factor is invalid
    
    try:
        # Clean the hex color string
        color_hex = str(color_hex).strip().lstrip('#')
        if len(color_hex) != 6:
            return '#FF0000'  # Invalid hex format
            
        # Convert hex to RGB
        r = int(color_hex[0:2], 16) / 255.0
        g = int(color_hex[2:4], 16) / 255.0
        b = int(color_hex[4:6], 16) / 255.0
        
        # Validate RGB values
        if not all(0 <= val <= 1 for val in [r, g, b]):
            return color_hex
        
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
        
        # Ensure values are in valid range
        new_r = max(0, min(1, new_r))
        new_g = max(0, min(1, new_g))
        new_b = max(0, min(1, new_b))
        
        new_r = int(new_r * 255)
        new_g = int(new_g * 255)
        new_b = int(new_b * 255)
        
        return f"#{new_r:02x}{new_g:02x}{new_b:02x}"
        
    except (ValueError, TypeError, RecursionError) as e:
        # Handle specific recursion errors
        if isinstance(e, RecursionError):
            return '#FF0000'  # Emergency red for recursion
        return color_hex  # Return original for other errors
    except Exception:
        return color_hex  # General fallback

def get_team_color(driver_code, session, drivers_to_plot, is_secondary=False, _recursion_depth=0):
    """Get the team color for a driver"""
    # Recursion protection
    if _recursion_depth > 5:
        # Emergency fallback to prevent infinite recursion
        driver_index = 0
        try:
            driver_index = drivers_to_plot.index(driver_code) if driver_code in drivers_to_plot else 0
        except:
            pass
        fallback_color = DEFAULT_COLORS[driver_index % len(DEFAULT_COLORS)]
        return fallback_color
    
    team_name = None
    base_color = None
    driver_info = None  # Initialize to None
    
    # First attempt: Get team color from mapping
    try:
        driver_info = session.get_driver(driver_code)
        if driver_info is not None and not driver_info.empty:
            team_name = driver_info.get('TeamName', '')
            
            if team_name in TEAM_COLORS_MAPPING:
                base_color = TEAM_COLORS_MAPPING[team_name]
                
                if is_secondary:
                    return adjust_color_luminance(base_color, 1.4)
                else:
                    return adjust_color_luminance(base_color, 0.85)
    except Exception as e:
        # Log the error but continue to fallback methods
        driver_info = None

    # Second attempt: Get FastF1 driver color
    try:
        color = fastf1.plotting.get_driver_color(driver_code, session=session)
        if not color:
            raise ValueError("No color returned")
            
        generic_colors = ['#000000', '#ffffff', '#808080', '#B6BABD']
        
        # Only use driver_info if it was successfully retrieved
        if driver_info is not None and not driver_info.empty:
            team_name_check = driver_info.get('TeamName', '')
            if color.upper() in [c.upper() for c in generic_colors] and team_name_check != 'Haas F1 Team':
                # Skip generic colors unless it's Haas (which actually uses grey/silver)
                pass
            else:
                if is_secondary and base_color is None:
                    return adjust_color_luminance(color, 1.3)
                else:
                    return color
        else:
            # No driver info available, use color if it's not generic
            if color.upper() not in [c.upper() for c in generic_colors]:
                if is_secondary:
                    return adjust_color_luminance(color, 1.3)
                else:
                    return color
    except Exception as e:
        # FastF1 color failed, continue to final fallback
        pass
    
    # Final fallback: Use default colors
    try:
        driver_index = drivers_to_plot.index(driver_code) if driver_code in drivers_to_plot else 0
        fallback_color = DEFAULT_COLORS[driver_index % len(DEFAULT_COLORS)]
        
        if is_secondary:
            return adjust_color_luminance(fallback_color, 1.4)
        else:
            return adjust_color_luminance(fallback_color, 0.9)
    except Exception as e:
        # Ultimate emergency fallback
        return '#FF0000' if not is_secondary else '#800000'

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

def get_session_drivers_with_times(_session):
    """Get available drivers from session with their fastest lap times, sorted by performance"""
    try:
        drivers = _session.drivers
        driver_info = []
        
        # Debug info
        st.write(f"DEBUG: Found {len(drivers)} drivers in session: {drivers[:5]}...")
        
        for driver in drivers:
            try:
                # Get driver info
                info = _session.get_driver(driver)
                if info is not None and not info.empty:
                    # Get fastest lap time for this driver in this session
                    driver_laps = _session.laps.pick_drivers(driver)
                    if not driver_laps.empty:
                        fastest_lap = driver_laps.pick_fastest()
                        if not fastest_lap.empty and pd.notna(fastest_lap['LapTime']):
                            # Use Abbreviation if available, otherwise use the driver identifier
                            driver_abbrev = info.get('Abbreviation', driver)
                            if pd.isna(driver_abbrev) or driver_abbrev == '':
                                driver_abbrev = driver
                            
                            driver_info.append({
                                'code': driver_abbrev,  # Use abbreviation instead of car number
                                'name': f"{info['FirstName']} {info['LastName']}",
                                'team': info['TeamName'],
                                'lap_time': fastest_lap['LapTime'].total_seconds(),
                                'lap_time_formatted': format_lap_time(fastest_lap['LapTime'].total_seconds()),
                                'color': get_team_color(driver_abbrev, _session, [driver_abbrev], is_secondary=False)
                            })
            except Exception as driver_error:
                # Debug: show what went wrong with this driver
                st.write(f"DEBUG: Error processing driver {driver}: {driver_error}")
                # Fallback for drivers without complete data
                driver_info.append({
                    'code': driver,
                    'name': driver,
                    'team': 'Unknown',
                    'lap_time': float('inf'),
                    'lap_time_formatted': 'No Time',
                    'color': '#808080'
                })
        
        # Debug: show results
        st.write(f"DEBUG: Processed {len(driver_info)} drivers successfully")
        if driver_info:
            st.write(f"DEBUG: First driver: {driver_info[0]}")
            teams = list(set([d['team'] for d in driver_info if d['team'] != 'Unknown']))
            st.write(f"DEBUG: Found {len(teams)} teams: {teams[:3]}...")
        
        # Sort by lap time (fastest first)
        driver_info.sort(key=lambda x: x['lap_time'])
        return driver_info
    except Exception as e:
        st.error(f"ERROR in get_session_drivers_with_times: {str(e)}")
        raise Exception(f"Failed to get driver list with times: {str(e)}")

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
            driver_laps = _session.laps.pick_drivers(driver_code)
            
            # If no laps found with abbreviation, try to find by car number
            if driver_laps.empty:
                # Look for the car number corresponding to this abbreviation
                try:
                    for car_num in _session.drivers:
                        driver_info = _session.get_driver(car_num)
                        if (driver_info is not None and not driver_info.empty and 
                            driver_info.get('Abbreviation') == driver_code):
                            driver_laps = _session.laps.pick_drivers(car_num)
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

def create_data_driven_lap_plot(driver_data_dict, _session, drivers_to_plot, year, gp_name, session_display):
    """Create and return a matplotlib figure showing data-driven lap visualization"""
    # Get driver data
    p1_code, p2_code = drivers_to_plot[0], drivers_to_plot[1]
    
    # Get driver colors with simple fallbacks
    try:
        p1_color = get_team_color(p1_code, _session, drivers_to_plot, is_secondary=False, _recursion_depth=0)
    except:
        p1_color = '#FF0000'  # Red fallback
    
    try:
        p2_color = get_team_color(p2_code, _session, drivers_to_plot, is_secondary=True, _recursion_depth=0)
    except:
        p2_color = '#0000FF'  # Blue fallback
    
    # Create figure (mobile layout for social media)
    fig = plt.figure(figsize=(12, 21), facecolor='#000000')
    gs = fig.add_gridspec(6, 1, height_ratios=[0.4, 5.6, 1.0, 0.8, 1.0, 0.8], hspace=0.15)
    
    ax_title = fig.add_subplot(gs[0, 0])
    ax_track = fig.add_subplot(gs[1, 0])
    ax_gear_speed = fig.add_subplot(gs[2, 0])
    ax_tyre_info = fig.add_subplot(gs[3, 0])
    ax_sectors = fig.add_subplot(gs[4, 0])
    ax_timer = fig.add_subplot(gs[5, 0])
    
    # Style setup
    clean_green = '#00FF00'
    clean_white = '#FFFFFF'
    clean_gray = '#1a1a1a'
    border_color = '#333333'
    
    panel_style = {'facecolor': clean_gray, 'edgecolor': border_color, 'linewidth': 0.5, 'alpha': 0.9}
    for ax in [ax_gear_speed, ax_tyre_info, ax_sectors, ax_timer]:
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
    
    # Lap times comparison with proper formatting
    lap_time_1 = driver_data_dict[p1_code]['lap_time']
    lap_time_2 = driver_data_dict[p2_code]['lap_time']
    delta = abs(lap_time_1 - lap_time_2)
    
    ax_sectors.text(0.1, 0.7, f'{p1_code}:', fontsize=16, color=p1_color, fontweight='bold')
    ax_sectors.text(0.35, 0.7, format_lap_time(lap_time_1), fontsize=16, color=clean_white, family='monospace')
    ax_sectors.text(0.1, 0.3, f'{p2_code}:', fontsize=16, color=p2_color, fontweight='bold')
    ax_sectors.text(0.35, 0.3, format_lap_time(lap_time_2), fontsize=16, color=clean_white, family='monospace')
    
    gap_color = clean_green if delta < 0.1 else ('#FFA500' if delta < 0.5 else '#FF0000')
    ax_sectors.text(0.7, 0.5, f'⏱️ GAP: {delta:.3f}s', fontsize=16, color=gap_color, fontweight='bold')
    ax_sectors.axis('off')
    
    # Final info
    ax_timer.text(0.5, 0.5, '🏁 FASTEST LAP COMPARISON', 
                 ha='center', va='center', fontsize=20, color=clean_white, fontweight='bold')
    ax_timer.axis('off')
    
    # Add watermark
    fig.text(0.95, 0.98, '@datadrivenlaps', fontsize=26, color='#FFFFFF', 
            alpha=0.7, ha='right', va='top', fontweight='bold', 
            transform=fig.transFigure, rotation=0)
    
    return fig

# Main App Layout
def main():
    setup_environment()
    
    # Header
    st.markdown('<h1 class="main-header">🏎️ F1 Data Driven Laps</h1>', unsafe_allow_html=True)
    st.markdown("### Create stunning F1 data-driven lap visualizations from telemetry data")
    
    # Responsive layout - use full width on mobile, sidebar on desktop
    if st.session_state.get('mobile_view', False):
        # Mobile layout - everything in main area
        st.header("🏁 Race Selection")
        
        # Mobile-friendly input layout
        col_year, col_gp = st.columns([1, 2])
        with col_year:
            year = st.number_input("Year", min_value=2018, max_value=2025, value=2025, step=1)
        with col_gp:
            gp_options = get_available_gps(year)
            if not gp_options:
                st.error("No Grand Prix data available for selected year")
                return
            gp_name = st.selectbox("Grand Prix", gp_options, index=0)
        
        # Session selection
        session_options = get_available_sessions(year, gp_name)
        if not session_options:
            st.error("No session data available for selected Grand Prix")
            return
        session_display = st.selectbox("Session", [s[1] for s in session_options], index=0)
        session_name = next(s[0] for s in session_options if s[1] == session_display)
        
        st.header("👥 Driver Selection")
        driver_mode = st.selectbox("Selection Mode", ["Specific Drivers", "Teammates", "P1 vs P2"], index=2)
        
    else:
        # Desktop layout - use sidebar
        with st.sidebar:
            st.header("🏁 Race Selection")
            
            # Year selection
            year = st.number_input("Year", min_value=2018, max_value=2025, value=2025, step=1)
            
            # GP selection
            gp_options = get_available_gps(year)
            if not gp_options:
                st.error("No Grand Prix data available for selected year")
                return
            
            gp_name = st.selectbox("Grand Prix", gp_options, index=0)
            
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
    try:
        with st.spinner("Loading session data..."):
            session = load_session_data(year, gp_name, session_name)
            driver_info = get_session_drivers_with_times(session)
            st.session_state.session_loaded = True
    except Exception as e:
        st.error(f"Error loading session: {str(e)}")
        return
    
    # Driver selection logic (common for both layouts)
    drivers_to_plot = []
    
    # Prepare team list for teammate selection
    teams = list(set([d['team'] for d in driver_info if d['team'] != 'Unknown']))
    
    # Mobile/Desktop responsive driver selection
    if not st.session_state.get('mobile_view', False):
        # Desktop - continue in sidebar
        with st.sidebar:
            if driver_mode == "Specific Drivers":
                # Show driver selection with formatted options
                driver_options = [f"{d['code']} - {d['name']} ({d['lap_time_formatted']})" for d in driver_info]
                selected_drivers = st.multiselect(
                    "Select 2 drivers",
                    driver_options,
                    max_selections=2,
                    help="Choose any two drivers from the session to compare their fastest laps"
                )
                drivers_to_plot = [d.split(' - ')[0] for d in selected_drivers]
                
                if len(selected_drivers) == 2:
                    st.success(f"🏎️ Selected: {drivers_to_plot[0]} vs {drivers_to_plot[1]}")
                elif len(selected_drivers) == 1:
                    st.info(f"Selected: {drivers_to_plot[0]} - Please select one more driver")
                
            elif driver_mode == "Teammates":
                if teams:
                    selected_team = st.selectbox(
                        "Select Team", 
                        teams,
                        help="Choose a team to compare their drivers"
                    )
                    teammates = [d['code'] for d in driver_info if d['team'] == selected_team]
                    if len(teammates) >= 2:
                        drivers_to_plot = teammates[:2]
                        # Show which drivers were selected
                        teammate_names = [d['name'].split()[-1] for d in driver_info if d['code'] in drivers_to_plot]
                        st.success(f"🤝 Selected: {drivers_to_plot[0]} vs {drivers_to_plot[1]}")
                        st.info(f"Comparing teammates: {teammate_names[0]} vs {teammate_names[1]}")
                    else:
                        st.warning(f"Only {len(teammates)} driver(s) found for {selected_team}")
                        if len(teammates) == 1:
                            st.info(f"Available: {teammates[0]}")
                else:
                    st.error("No teams found with valid data in this session")
                    
            elif driver_mode == "P1 vs P2":
                try:
                    # First try using session results for proper P1 vs P2
                    results = session.results
                    if not results.empty and len(results) >= 2:
                        p1_driver = results.iloc[0]['Abbreviation'] 
                        p2_driver = results.iloc[1]['Abbreviation']
                        drivers_to_plot = [p1_driver, p2_driver]
                        
                        # Get driver names for display
                        p1_name = next((d['name'].split()[-1] for d in driver_info if d['code'] == p1_driver), p1_driver)
                        p2_name = next((d['name'].split()[-1] for d in driver_info if d['code'] == p2_driver), p2_driver)
                        
                        st.success(f"🏆 P1: {p1_driver} ({p1_name}) vs P2: {p2_driver} ({p2_name})")
                        st.info("Using official session results for P1 vs P2")
                    else:
                        # Fallback to fastest two drivers from our sorted list
                        if len(driver_info) >= 2:
                            p1_driver = driver_info[0]['code']
                            p2_driver = driver_info[1]['code']
                            drivers_to_plot = [p1_driver, p2_driver]
                            
                            p1_time = driver_info[0]['lap_time_formatted']
                            p2_time = driver_info[1]['lap_time_formatted']
                            
                            st.success(f"🏆 Fastest: {p1_driver} ({p1_time}) vs {p2_driver} ({p2_time})")
                            st.info("Using fastest lap times (no official results available)")
                        else:
                            st.warning("Not enough drivers with lap times available")
                except Exception as e:
                    # Fallback to fastest two drivers from our sorted list
                    try:
                        if len(driver_info) >= 2:
                            p1_driver = driver_info[0]['code']
                            p2_driver = driver_info[1]['code']
                            drivers_to_plot = [p1_driver, p2_driver]
                            st.success(f"🏆 Selected: {p1_driver} vs {p2_driver} (fastest times)")
                        else:
                            st.warning("Not enough drivers with lap times available")
                    except:
                        st.warning("Could not determine P1 vs P2")
    else:
        # Mobile - in main area with enhanced UI
        if driver_mode == "Specific Drivers":
            # Show driver selection with formatted options
            driver_options = [f"{d['code']} - {d['name']} ({d['lap_time_formatted']})" for d in driver_info]
            selected_drivers = st.multiselect(
                "Select 2 drivers",
                driver_options,
                max_selections=2,
                help="Choose any two drivers from the session to compare their fastest laps"
            )
            drivers_to_plot = [d.split(' - ')[0] for d in selected_drivers]
            
            if len(selected_drivers) == 2:
                st.success(f"🏎️ Selected: {drivers_to_plot[0]} vs {drivers_to_plot[1]}")
            elif len(selected_drivers) == 1:
                st.info(f"Selected: {drivers_to_plot[0]} - Please select one more driver")
            
        elif driver_mode == "Teammates":
            if teams:
                selected_team = st.selectbox(
                    "Select Team", 
                    teams,
                    help="Choose a team to compare their drivers"
                )
                teammates = [d['code'] for d in driver_info if d['team'] == selected_team]
                if len(teammates) >= 2:
                    drivers_to_plot = teammates[:2]
                    # Show which drivers were selected
                    teammate_names = [d['name'].split()[-1] for d in driver_info if d['code'] in drivers_to_plot]
                    st.success(f"🤝 Selected: {drivers_to_plot[0]} vs {drivers_to_plot[1]}")
                    st.info(f"Comparing teammates: {teammate_names[0]} vs {teammate_names[1]}")
                else:
                    st.warning(f"Only {len(teammates)} driver(s) found for {selected_team}")
                    if len(teammates) == 1:
                        st.info(f"Available: {teammates[0]}")
            else:
                st.error("No teams found with valid data in this session")
                
        elif driver_mode == "P1 vs P2":
            try:
                # First try using session results for proper P1 vs P2
                results = session.results
                if not results.empty and len(results) >= 2:
                    p1_driver = results.iloc[0]['Abbreviation']
                    p2_driver = results.iloc[1]['Abbreviation']
                    drivers_to_plot = [p1_driver, p2_driver]
                    
                    # Get driver names for display
                    p1_name = next((d['name'].split()[-1] for d in driver_info if d['code'] == p1_driver), p1_driver)
                    p2_name = next((d['name'].split()[-1] for d in driver_info if d['code'] == p2_driver), p2_driver)
                    
                    st.success(f"🏆 P1: {p1_driver} ({p1_name}) vs P2: {p2_driver} ({p2_name})")
                    st.info("Using official session results for P1 vs P2")
                else:
                    # Fallback to fastest two drivers from our sorted list
                    if len(driver_info) >= 2:
                        p1_driver = driver_info[0]['code']
                        p2_driver = driver_info[1]['code']
                        drivers_to_plot = [p1_driver, p2_driver]
                        
                        p1_time = driver_info[0]['lap_time_formatted']
                        p2_time = driver_info[1]['lap_time_formatted']
                        
                        st.success(f"🏆 Fastest: {p1_driver} ({p1_time}) vs {p2_driver} ({p2_time})")
                        st.info("Using fastest lap times (no official results available)")
                    else:
                        st.warning("Not enough drivers with lap times available")
            except Exception as e:
                # Fallback to fastest two drivers from our sorted list
                try:
                    if len(driver_info) >= 2:
                        p1_driver = driver_info[0]['code']
                        p2_driver = driver_info[1]['code']
                        drivers_to_plot = [p1_driver, p2_driver]
                        st.success(f"🏆 Selected: {p1_driver} vs {p2_driver} (fastest times)")
                    else:
                        st.warning("Not enough drivers with lap times available")
                except:
                    st.warning("Could not determine P1 vs P2")
    
    # Mobile view toggle
    st.sidebar.markdown("---")
    if st.sidebar.button("📱 Toggle Mobile View"):
        st.session_state.mobile_view = not st.session_state.get('mobile_view', False)
        st.rerun()
    
    # Main content area (responsive)
    if len(drivers_to_plot) == 2:
        st.markdown(f'<div class="status-box success-box">✅ Ready to analyze: <strong>{drivers_to_plot[0]} vs {drivers_to_plot[1]}</strong></div>', unsafe_allow_html=True)
        
        if st.session_state.get('mobile_view', False):
            # Mobile: single column layout
            if st.button("🏎️ Generate Data Driven Lap Visualization", type="primary", use_container_width=True):
                try:
                    # Prepare data
                    with st.spinner("🏎️ Preparing driver data..."):
                        driver_data = prepare_driver_data(session, drivers_to_plot, driver_mode)
                    
                    # Display lap time comparison with proper formatting
                    st.subheader("📊 Lap Time Comparison")
                    
                    lap_time_1 = driver_data[drivers_to_plot[0]]['lap_time']
                    lap_time_2 = driver_data[drivers_to_plot[1]]['lap_time']
                    delta = abs(lap_time_1 - lap_time_2)
                    
                    # Mobile-friendly metrics layout
                    st.metric(
                        f"{drivers_to_plot[0]} (Fastest Lap)",
                        format_lap_time(lap_time_1),
                        f"{driver_data[drivers_to_plot[0]]['compound']}"
                    )
                    st.metric(
                        f"{drivers_to_plot[1]} (Fastest Lap)",
                        format_lap_time(lap_time_2),
                        f"{driver_data[drivers_to_plot[1]]['compound']}"
                    )
                    
                    st.info(f"⏱️ Gap: {delta:.3f} seconds")
                    
                    # Generate image
                    st.subheader("🎨 Image Generation")
                    
                    # Generate and display plot
                    st.subheader("🏁 Your Data Driven Lap Visualization")
                    
                    with st.spinner("🏎️ Creating data-driven lap visualization..."):
                        fig = create_data_driven_lap_plot(
                            driver_data, session, drivers_to_plot, year, gp_name, session_display
                        )
                    
                    # Display plot directly
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)  # Clean up
                        
                except Exception as e:
                    st.error(f"❌ Error generating visualization: {str(e)}")
            
            # Mobile tips
            st.info("💡 **Mobile Tips:**\n🏎️ Visualization generates in just a few seconds\n🏎️ Optimized for mobile viewing\n🏎️ Interactive matplotlib display\n🏎️ Perfect for analysis")
            
        else:
            # Desktop: two-column layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if st.button("🏎️ Generate Data Driven Lap Visualization", type="primary", use_container_width=True):
                    try:
                        # Prepare data
                        with st.spinner("🏎️ Preparing driver data..."):
                            driver_data = prepare_driver_data(session, drivers_to_plot, driver_mode)
                        
                        # Display lap time comparison with proper formatting
                        st.subheader("📊 Lap Time Comparison")
                        
                        lap_time_1 = driver_data[drivers_to_plot[0]]['lap_time']
                        lap_time_2 = driver_data[drivers_to_plot[1]]['lap_time']
                        delta = abs(lap_time_1 - lap_time_2)
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric(
                                f"{drivers_to_plot[0]} (Fastest Lap)",
                                format_lap_time(lap_time_1),
                                f"{driver_data[drivers_to_plot[0]]['compound']}"
                            )
                        with col_b:
                            st.metric(
                                f"{drivers_to_plot[1]} (Fastest Lap)",
                                format_lap_time(lap_time_2),
                                f"{driver_data[drivers_to_plot[1]]['compound']}"
                            )
                        
                        st.info(f"⏱️ Gap: {delta:.3f} seconds")
                        
                        # Generate and display plot
                        st.subheader("🏁 Your Data Driven Lap Visualization")
                        
                        with st.spinner("🏎️ Creating data-driven lap visualization..."):
                            fig = create_data_driven_lap_plot(
                                driver_data, session, drivers_to_plot, year, gp_name, session_display
                            )
                        
                        # Display plot directly
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)  # Clean up
                            
                    except Exception as e:
                        st.error(f"❌ Error generating visualization: {str(e)}")
            
            with col2:
                st.info("💡 **Tips:**\n🏎️ Visualization generates in just a few seconds\n🏎️ Interactive matplotlib display\n🏎️ Perfect for detailed analysis\n🏎️ Mobile responsive design")
    
    else:
        st.markdown('<div class="status-box info-box">ℹ️ Please select exactly 2 drivers to generate a data driven lap image.</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("Made using [FastF1](https://github.com/theOehrly/Fast-F1) and [Streamlit](https://streamlit.io)")

if __name__ == "__main__":
    main() 