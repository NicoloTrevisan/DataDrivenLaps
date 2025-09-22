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
import sys
import importlib.util
import importlib.machinery
import contextlib

# Page configuration
st.set_page_config(
    page_title="F1 Data Driven Laps",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Quieter UI: hide error details and toolbar
st.set_option("client.showErrorDetails", False)
st.set_option("client.toolbarMode", "minimal")

# Suppress glyph and plotting warnings in UI
warnings.filterwarnings("ignore", message=r"Glyph .* missing from current font")
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")
warnings.filterwarnings("ignore", category=FutureWarning, module="fastf1.plotting")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF0000;
        margin-bottom: 0.25rem;
    }
    .tagline {
        text-align: center;
        color: #e0e0e0;
        margin-bottom: 1.25rem;
        font-size: 1.05rem;
    }
    /* Center content and limit width */
    .main .block-container {
        max-width: 1100px;
        margin-left: auto;
        margin-right: auto;
        padding-top: 0.75rem;
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
        .main-header { font-size: 2rem; }
        .stSelectbox > div > div { font-size: 14px; }
        .stButton > button { font-size: 16px; padding: 0.75rem 1rem; }
        .stMetric { background-color: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; }
    }
    
    /* Desktop optimizations */
    @media (min-width: 769px) {
        .stSidebar { width: 350px; }
        .main .block-container { padding-left: 2rem; padding-right: 2rem; }
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
    .stMetric { background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 1rem; }
    
    /* Better info boxes */
    .stInfo { border-left: 4px solid #17a2b8; }
    
    /* Loading spinner improvements */
    .stSpinner > div { color: #FF0000 !important; }
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

# === New helpers ===

def convert_speed_units(speed_kmh_series, units):
    """Convert speed series from km/h to requested units ('km/h' or 'mph')."""
    if units == 'mph':
        return speed_kmh_series * 0.621371
    return speed_kmh_series


def _first_valid_event_with_sessions(year: int) -> tuple:
    """Return (gp_name, session_display) for the first GP in a year with available sessions.
    Skips testing/pre-season events and returns ('', '') if none found.
    """
    try:
        gps = get_available_gps(year)
        if not gps:
            return ('', '')
        for gp in gps:
            label = (gp or '').lower()
            if 'test' in label or 'pre-season' in label or 'testing' in label:
                continue
            sessions = get_available_sessions(year, gp)
            if not sessions:
                continue
            # Prefer Qualifying, else Race, else first available
            displays = [s[1] for s in sessions]
            if 'Qualifying' in displays:
                return (gp, 'Qualifying')
            if 'Race' in displays:
                return (gp, 'Race')
            return (gp, displays[0])
        return ('', '')
    except Exception:
        return ('', '')


def _random_valid_selection(years: list) -> tuple:
    """Try to find a random (year, gp_name, session_display) with available sessions.
    Attempts up to 15 random picks across provided years. Returns ('', '', '') if none.
    """
    import random
    tries = 15
    for _ in range(tries):
        try:
            year = random.choice(years)
            gps = get_available_gps(year)
            if not gps:
                continue
            # Filter out testing-like events
            gps = [g for g in gps if g and ('test' not in g.lower()) and ('pre-season' not in g.lower()) and ('testing' not in g.lower())]
            if not gps:
                continue
            gp = random.choice(gps)
            sessions = get_available_sessions(year, gp)
            if not sessions:
                continue
            displays = [s[1] for s in sessions]
            sess_disp = 'Qualifying' if 'Qualifying' in displays else ('Race' if 'Race' in displays else random.choice(displays))
            return (year, gp, sess_disp)
        except Exception:
            continue
    return ('', '', '')


def compute_time_delta_by_distance(p1_tel: pd.DataFrame, p2_tel: pd.DataFrame) -> tuple:
    """Compute cumulative time delta (s) along distance where negative => P1 ahead.
    Returns (distance_m, delta_time_s).
    """
    # Align on P1 distance grid
    p1_dist = p1_tel['Distance'].to_numpy()
    p2_dist = p2_tel['Distance'].to_numpy()
    p2_speed_interp = interp1d(p2_dist, p2_tel['Speed'].to_numpy(), kind='linear', bounds_error=False, fill_value='extrapolate')
    p1_speed_kmh = p1_tel['Speed'].to_numpy()
    p2_speed_kmh_aligned = p2_speed_interp(p1_dist)

    # Convert to m/s
    v1 = np.maximum(p1_speed_kmh, 1e-3) / 3.6
    v2 = np.maximum(p2_speed_kmh_aligned, 1e-3) / 3.6

    ds = np.diff(p1_dist, prepend=p1_dist[0])
    dt1 = ds / np.maximum(v1, 1e-6)
    dt2 = ds / np.maximum(v2, 1e-6)
    delta_time = np.cumsum(dt2 - dt1)
    return p1_dist, delta_time


def create_time_delta_plot(distance_m: np.ndarray, delta_time_s: np.ndarray, p1_code: str, p2_code: str) -> plt.Figure:
    """Create a compact time delta vs distance plot."""
    fig, ax = plt.subplots(figsize=(10, 2.8), facecolor='#000000')
    ax.set_facecolor('#111111')
    ax.plot(distance_m, delta_time_s, color='#FFFFFF', linewidth=1.8)
    ax.axhline(0, color='#666666', linewidth=1, linestyle='--', alpha=0.7)
    ax.set_xlabel('Distance (m)', color='#FFFFFF')
    ax.set_ylabel('Δ time (s)', color='#FFFFFF')
    ax.tick_params(colors='#FFFFFF')
    ax.grid(True, color='#FFFFFF', alpha=0.12)
    ax.set_title(f'Time Delta: {p1_code} vs {p2_code}', color='#FFFFFF')
    return fig


def generate_story_highlights(p1_code: str, p2_code: str, p1_tel: pd.DataFrame, p2_tel: pd.DataFrame) -> list:
    """Generate 2-3 simple story highlights from telemetry comparisons."""
    dist, delta_t = compute_time_delta_by_distance(p1_tel, p2_tel)

    # Smooth delta for stability
    if len(delta_t) > 10:
        window = max(5, len(delta_t)//200)
        kernel = np.ones(window)/window
        delta_smooth = np.convolve(delta_t, kernel, mode='same')
    else:
        delta_smooth = delta_t

    # Compute local changes over windows
    step = max(10, len(delta_smooth)//100)
    changes = np.diff(delta_smooth, n=1)
    # Most negative change (p1 big gain), most positive (p2 big gain)
    idx_gain_p1 = int(np.argmin(changes)) if len(changes) else 0
    idx_gain_p2 = int(np.argmax(changes)) if len(changes) else 0

    highlights = []
    if len(dist) > 1:
        def fmt_segment(idx):
            i0 = max(0, idx - step)
            i1 = min(len(dist)-1, idx + step)
            return f"{int(dist[i0])}–{int(dist[i1])} m"
        # Magnitudes over small windows
        gain_p1 = float(np.min(changes)) if len(changes) else 0.0
        gain_p2 = float(np.max(changes)) if len(changes) else 0.0
        if gain_p1 < 0:
            highlights.append(f"{p1_code} gains about {abs(gain_p1):.2f}s over {fmt_segment(idx_gain_p1)}")
        if gain_p2 > 0:
            highlights.append(f"{p2_code} gains about {abs(gain_p2):.2f}s over {fmt_segment(idx_gain_p2)}")

    # Top speed difference highlight
    try:
        p1_speed_max = float(np.nanmax(p1_tel['Speed']))
        p2_speed_max = float(np.nanmax(p2_tel['Speed']))
        diff = p1_speed_max - p2_speed_max
        faster = p1_code if diff >= 0 else p2_code
        highlights.append(f"Highest top speed: {faster} by {abs(diff):.0f} km/h")
    except Exception:
        pass

    # Keep to 3 items
    return highlights[:3]

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
        st.info("We couldn't load the Grand Prix list right now. Please try another year.")
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
        st.info("We couldn't load the sessions for this Grand Prix. Try a different one.")
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
        
        for driver in drivers:
            try:
                info = _session.get_driver(driver)
                if info is not None and not info.empty:
                    driver_laps = _session.laps.pick_drivers(driver)
                    if not driver_laps.empty:
                        fastest_lap = driver_laps.pick_fastest()
                        if not fastest_lap.empty and pd.notna(fastest_lap['LapTime']):
                            driver_abbrev = info.get('Abbreviation', driver)
                            if pd.isna(driver_abbrev) or driver_abbrev == '':
                                driver_abbrev = driver
                            driver_info.append({
                                'code': driver_abbrev,
                                'name': f"{info['FirstName']} {info['LastName']}",
                                'team': info['TeamName'],
                                'lap_time': fastest_lap['LapTime'].total_seconds(),
                                'lap_time_formatted': format_lap_time(fastest_lap['LapTime'].total_seconds()),
                                'color': get_team_color(driver_abbrev, _session, [driver_abbrev], is_secondary=False)
                            })
            except Exception:
                driver_info.append({
                    'code': driver,
                    'name': driver,
                    'team': 'Unknown',
                    'lap_time': float('inf'),
                    'lap_time_formatted': 'No Time',
                    'color': '#808080'
                })
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
    """Prepare driver data for analysis with status updates."""
    try:
        driver_data_dict = {}
        for driver_code in drivers_to_plot:
            driver_laps = _session.laps.pick_drivers(driver_code)
            if driver_laps.empty:
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
        return driver_data_dict
    except Exception as e:
        raise Exception(f"Failed to prepare driver data: {str(e)}")

def create_data_driven_lap_plot(driver_data_dict, _session, drivers_to_plot, year, gp_name, session_display, watermark_text='@datadrivenlaps', units='km/h', aspect_ratio='Story 9:16'):
    """Create and return a matplotlib figure showing data-driven lap visualization"""
    # Aspect ratio handling
    aspect_map = {
        'Story 9:16': (12, 21),
        'Post 1:1': (18, 18),
        'Widescreen 16:9': (21, 12)
    }
    figsize = aspect_map.get(aspect_ratio, (12, 21))

    # Get driver data
    p1_code, p2_code = drivers_to_plot[0], drivers_to_plot[1]
    
    # Colors
    try:
        p1_color = get_team_color(p1_code, _session, drivers_to_plot, is_secondary=False, _recursion_depth=0)
    except:
        p1_color = '#FF0000'
    try:
        p2_color = get_team_color(p2_code, _session, drivers_to_plot, is_secondary=True, _recursion_depth=0)
    except:
        p2_color = '#0000FF'
    
    # Figure
    fig = plt.figure(figsize=figsize, facecolor='#000000')
    # Removed tyre panel for a cleaner layout (5 rows)
    gs = fig.add_gridspec(5, 1, height_ratios=[0.5, 6.0, 1.2, 1.0, 0.8], hspace=0.15)
    ax_title = fig.add_subplot(gs[0, 0])
    ax_track = fig.add_subplot(gs[1, 0])
    ax_gear_speed = fig.add_subplot(gs[2, 0])
    ax_sectors = fig.add_subplot(gs[3, 0])
    ax_timer = fig.add_subplot(gs[4, 0])

    clean_green = '#00FF00'
    clean_white = '#FFFFFF'
    clean_gray = '#1a1a1a'
    border_color = '#333333'
    
    panel_style = {'facecolor': clean_gray, 'edgecolor': border_color, 'linewidth': 0.5, 'alpha': 0.9}
    for ax in [ax_gear_speed, ax_sectors, ax_timer]:
        ax.set_facecolor(panel_style['facecolor'])
        for spine in ax.spines.values():
            spine.set_color(panel_style['edgecolor'])
            spine.set_linewidth(panel_style['linewidth'])
    ax_track.set_facecolor('#0A0A0A')

    # Telemetry
    p1_tel = driver_data_dict[p1_code]['telemetry']
    p2_tel = driver_data_dict[p2_code]['telemetry']

    # Track coloring by who is faster at each point
    track_segments_x = p1_tel['X'].to_numpy()
    track_segments_y = p1_tel['Y'].to_numpy()
    points = np.array([track_segments_x, track_segments_y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    p1_speeds = p1_tel['Speed'].to_numpy()
    p1_dist = p1_tel['Distance'].to_numpy()
    p2_dist = p2_tel['Distance'].to_numpy()
    p2_speed_interp = interp1d(p2_dist, p2_tel['Speed'].to_numpy(), kind='linear', bounds_error=False, fill_value='extrapolate')
    p2_speeds_aligned = p2_speed_interp(p1_dist)

    # Constant colors across the track
    p1_color_rgba = mcolors.to_rgba(p1_color)
    p2_color_rgba = mcolors.to_rgba(p2_color)
    segment_colors = []
    for i in range(len(segments)):
        if i < len(p1_speeds) and i < len(p2_speeds_aligned):
            seg_col = p1_color_rgba if p1_speeds[i] >= p2_speeds_aligned[i] else p2_color_rgba
            segment_colors.append(seg_col)
        else:
            segment_colors.append(mcolors.to_rgba('#404040'))
    lc = LineCollection(segments, colors=segment_colors, linewidths=10, capstyle='round', alpha=0.9)
    ax_track.add_collection(lc)

    # Car markers
    mid_point_p1 = len(p1_tel) // 2
    mid_point_p2 = len(p2_tel) // 2
    car_radius = np.mean([track_segments_x.max() - track_segments_x.min(), 
                         track_segments_y.max() - track_segments_y.min()]) * 0.008
    car1_circle = Circle((track_segments_x[mid_point_p1], track_segments_y[mid_point_p1]), radius=car_radius, facecolor=p1_color, edgecolor='none', zorder=10)
    car2_circle = Circle((track_segments_x[mid_point_p2], track_segments_y[mid_point_p2]), radius=car_radius, facecolor=p2_color, edgecolor='none', zorder=10)
    ax_track.add_patch(car1_circle)
    ax_track.add_patch(car2_circle)
    ax_track.set_xlim(track_segments_x.min() - 200, track_segments_x.max() + 200)
    ax_track.set_ylim(track_segments_y.min() - 200, track_segments_y.max() + 200)
    ax_track.set_aspect('equal')
    ax_track.axis('off')

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=p1_color, markeredgecolor='none', markersize=12, label=f"{p1_code}"),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=p2_color, markeredgecolor='none', markersize=12, label=f"{p2_code}")
    ]
    ax_track.legend(handles=legend_elements, loc='upper center', ncol=2, facecolor=clean_gray, edgecolor=border_color, labelcolor=clean_white, fontsize=14, bbox_to_anchor=(0.5, 1.08))

    # Title
    ax_title.text(0.5, 0.5, f'{year} {gp_name} {session_display}', 
                 ha='center', va='center', fontsize=24, color=clean_white, fontweight='bold')
    ax_title.axis('off')

    # Speed comparison with units toggle
    p1_speed_units = convert_speed_units(p1_tel['Speed'], units)
    p2_speed_units = convert_speed_units(p2_tel['Speed'], units)
    ax_gear_speed.plot(p1_speed_units, color=p1_color, linewidth=3, label=f'{p1_code} Speed', alpha=0.8)
    p2_speed_aligned_plot = np.interp(np.linspace(0, len(p2_tel)-1, len(p1_tel)), 
                                     range(len(p2_tel)), p2_speed_units)
    ax_gear_speed.plot(p2_speed_aligned_plot, color=p2_color, linewidth=3, label=f'{p2_code} Speed', alpha=0.8)
    ax_gear_speed.set_ylabel(f'Speed ({units})', color=clean_white, fontsize=14, fontweight='bold')
    ax_gear_speed.tick_params(colors=clean_white, labelsize=12)
    ax_gear_speed.legend(loc='upper right', facecolor=clean_gray, edgecolor=border_color, labelcolor=clean_white)
    ax_gear_speed.grid(True, alpha=0.15, color=clean_white, linestyle='-', linewidth=0.5)

    # Lap times and gap
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

    # Footer panel
    ax_timer.text(0.5, 0.5, '🏁 FASTEST LAP COMPARISON', 
                 ha='center', va='center', fontsize=20, color=clean_white, fontweight='bold')
    ax_timer.axis('off')

    # Watermark
    fig.text(0.95, 0.98, watermark_text, fontsize=26, color='#FFFFFF', 
            alpha=0.7, ha='right', va='top', fontweight='bold', transform=fig.transFigure, rotation=0)
    return fig

# --- Optional animated outputs (GIF/MP4) integration ---
def _map_session_display_to_code(session_display: str) -> str:
    """Map human-readable session names to FastF1 short codes used in the animation generator."""
    mapping = {
        'Practice 1': 'FP1',
        'Practice 2': 'FP2',
        'Practice 3': 'FP3',
        'Qualifying': 'Q',
        'Race': 'R',
        'Sprint Qualifying': 'SQ',
        'Sprint': 'SR'
    }
    return mapping.get(session_display, session_display)

def _generate_gif_mp4_outputs(year: int, gp_name: str, session_display: str, drivers_to_plot: list, make_gif: bool, make_mp4: bool, progress_placeholder=None, text_placeholder=None, mp4_fps: int = 10) -> dict:
    """Use scripts/interactive_f1_ghost_racing.py to create GIF/MP4 for the current selection.
    Returns dict with optional keys 'gif_path' and 'mp4_path'.
    """
    results: dict = {}
    try:
        log_entries = []
        def _log(msg: str):
            log_entries.append(msg)
            # Add to persistent debug log
            if 'debug_log' not in st.session_state:
                st.session_state.debug_log = []
            st.session_state.debug_log.append(f"[MP4 Gen] {msg}")
            
            if text_placeholder is not None:
                # Update a single terminal-like block
                try:
                    text_placeholder.code("\n".join(log_entries))
                except Exception:
                    text_placeholder.write(msg)
        # Try multiple locations to import the generator module
        if1 = None
        module_name = 'interactive_f1_ghost_racing'
        # First: sibling file inside deployment_ready (if present)
        candidate1 = Path(__file__).resolve().parent / f'{module_name}.py'
        # Second: repo-level scripts directory
        candidate2 = Path(__file__).resolve().parents[1] / 'scripts' / f'{module_name}.py'

        def _load_module_from_path(name, path):
            loader = importlib.machinery.SourceFileLoader(name, str(path))
            spec = importlib.util.spec_from_loader(loader.name, loader)
            mod = importlib.util.module_from_spec(spec)
            loader.exec_module(mod)  # type: ignore
            return mod

        if candidate1.exists():
            _log(f"Using local module: {candidate1}")
            if1 = _load_module_from_path(module_name, candidate1)
        elif candidate2.exists():
            _log(f"Using repo scripts module: {candidate2}")
            if1 = _load_module_from_path(module_name, candidate2)
        else:
            # Last attempt: sys.path import if repository layout is available at runtime
            try:
                scripts_dir = Path(__file__).resolve().parents[1] / 'scripts'
                if str(scripts_dir) not in sys.path:
                    sys.path.append(str(scripts_dir))
                import interactive_f1_ghost_racing as if1  # type: ignore
                _log("Imported interactive_f1_ghost_racing via sys.path")
            except Exception as _e:
                raise ImportError("Could not locate interactive_f1_ghost_racing.py. Ensure 'scripts/' is deployed alongside the app or copy the file into 'deployment_ready/'.")

        # Create a Streamlit-backed progress bar proxy to replace the module's ProgressBar
        class _StreamlitProgressBar:
            def __init__(self, total: int, desc: str = "Progress"):
                self.total = max(1, int(total))
                self.current = 0
                self.desc = desc
                self._prog = None
                self._txt = None
                if progress_placeholder is not None:
                    self._prog = progress_placeholder.progress(0)
                if text_placeholder is not None:
                    self._txt = text_placeholder
                    self._txt.write(f"{self.desc}: 0% (0/{self.total})")

            def update(self, n: int = 1):
                self.current = min(self.total, self.current + n)
                percent = int(self.current * 100 / self.total)
                if self._prog is not None:
                    self._prog.progress(percent)
                if self._txt is not None:
                    self._txt.write(f"{self.desc}: {percent}% ({self.current}/{self.total})")

            def close(self):
                if self._prog is not None:
                    self._prog.progress(100)
                if self._txt is not None:
                    self._txt.write(f"{self.desc}: 100% ({self.total}/{self.total})")

        # Swap the generator's ProgressBar with our Streamlit proxy
        if1.ProgressBar = _StreamlitProgressBar
        from interactive_f1_ghost_racing import InteractiveDataDrivenLaps  # type: ignore

        gen = InteractiveDataDrivenLaps()
        gen.year = int(year)
        gen.gp_name = gp_name
        gen.session_name = _map_session_display_to_code(session_display)
        # Force specific drivers mode using the two selected abbreviations
        gen.driver_selection_mode = 'SpecificDrivers'
        gen.drivers_to_plot = [drivers_to_plot[0], drivers_to_plot[1]]

        # Load and prepare data
        _log("Loading session data...")
        stdout_buf = io.StringIO()
        with contextlib.redirect_stdout(stdout_buf):
            if not gen.load_session_data():
                _log("Failed to load session data")
                raise RuntimeError('Failed to load session data for animation.')
            _log("Preparing driver data...")
            if not gen.prepare_driver_data():
                _log("Failed to prepare driver data")
                raise RuntimeError('Failed to prepare driver data for animation.')
            _log("Preparing animation frames...")
            if not gen.prepare_cinematic_animation_data():
                _log("Failed to prepare animation frames")
                raise RuntimeError('Failed to prepare animation frames.')
            _log("Calculating sector times...")
            gen.calculate_sector_times()

        # Generate outputs
        if make_gif:
            gif_path = gen.create_gif()
            if gif_path:
                results['gif_path'] = gif_path
        if make_mp4:
            # Fast path: call the internal animation with custom FPS (lower = faster)
            try:
                _log(f"Creating MP4 at {int(mp4_fps)} FPS...")
                mp4_path = gen._create_animation(output_fps=int(mp4_fps), output_extension='mp4', writer_class=if1.FFMpegWriter, writer_options={'bitrate': 6000})
            except Exception:
                # Fallback to default 60 FPS method if internal call signature changes
                _log("Fast MP4 path failed, falling back to default method (60 FPS)...")
                mp4_path = gen.create_mp4()
            if mp4_path:
                results['mp4_path'] = mp4_path
            else:
                _log("MP4 generation returned no path (None)")
                raise RuntimeError('MP4 generation returned no file path (None).')
        results['stdout'] = stdout_buf.getvalue()
        # Persist logs to file
        try:
            outputs_dir = Path(tempfile.gettempdir()) / 'ddlaps_logs'
            outputs_dir.mkdir(parents=True, exist_ok=True)
            log_path = outputs_dir / f"animation_{year}_{gp_name.replace(' ','_')}_{drivers_to_plot[0]}_vs_{drivers_to_plot[1]}.log"
            log_path.write_text("\n".join(log_entries) + "\n\nSTDOUT:\n" + results.get('stdout',''))
            results['log_path'] = str(log_path)
        except Exception:
            pass
        results['log_text'] = "\n".join(log_entries)
        return results
    except Exception as e:
        # Try to persist what we have from this scope
        try:
            outputs_dir = Path(tempfile.gettempdir()) / 'ddlaps_logs'
            outputs_dir.mkdir(parents=True, exist_ok=True)
            log_path = outputs_dir / f"animation_error_{year}_{gp_name.replace(' ','_')}_{drivers_to_plot[0]}_vs_{drivers_to_plot[1]}.log"
            try:
                captured = stdout_buf.getvalue()
            except Exception:
                captured = ''
            try:
                log_text = "\n".join(log_entries)
            except Exception:
                log_text = ''
            log_path.write_text(log_text + "\n\nSTDOUT:\n" + captured)
            raise Exception(f"Animated output generation failed: {e} | Log saved to {log_path}")
        except Exception:
            # Surface a friendly error up the stack
            raise Exception(f"Animated output generation failed: {e}")

# Main App Layout
def main():
    setup_environment()
    
    # Header
    st.markdown('<h1 class="main-header">🏎️ F1 Data Driven Laps</h1>', unsafe_allow_html=True)
    st.markdown('<div class="tagline">Who\'s faster where, and why? Watch a lap unfold with real telemetry.</div>', unsafe_allow_html=True)
    st.markdown("### Create stunning F1 data-driven lap visualizations from telemetry data")
    
    # Ensure common state defaults
    st.session_state.setdefault('mobile_view', False)
    st.session_state.setdefault('units', 'km/h')
    st.session_state.setdefault('aspect_ratio', 'Story 9:16')

    # Handle pending preset before any widgets are created
    if st.session_state.get('pending_preset', False):
        # Clear widget-bound keys so we can set them safely
        for key in ['year_input', 'gp_select', 'session_select', 'driver_mode']:
            if key in st.session_state:
                try:
                    del st.session_state[key]
                except Exception:
                    pass
        # Apply preset values
        st.session_state['year_input'] = st.session_state.get('preset_year', 2025)
        st.session_state['gp_select'] = st.session_state.get('preset_gp', 'Monaco Grand Prix')
        st.session_state['session_select'] = st.session_state.get('preset_session', 'Qualifying')
        st.session_state['driver_mode'] = 'P1 vs P2'
        st.session_state['auto_generate'] = True
        st.session_state['pending_preset'] = False
        st.rerun()

    # Top-level demo button (before any selection widgets)
    if not st.session_state.get('hide_demo', False):
        demo_col1, demo_col2 = st.columns([1, 3])
        with demo_col1:
            if st.button("🎬 Try a demo"):
                st.session_state['preset_year'] = 2025
                st.session_state['preset_gp'] = 'Monaco Grand Prix'
                st.session_state['preset_session'] = 'Qualifying'
                st.session_state['pending_preset'] = True
                st.session_state['hide_demo'] = True
                st.rerun()
    
    # Sidebar / Mobile controls
    if st.session_state.get('mobile_view', False):
        st.header("🏁 Race Selection")
        col_year, col_gp = st.columns([1, 2])
        with col_year:
            year = st.number_input("Year", min_value=2018, max_value=2025, value=st.session_state.get('year_input', 2025), step=1, key='year_input')
        with col_gp:
            gp_options = get_available_gps(year)
            if not gp_options:
                st.error("No Grand Prix data available for selected year")
                return
            default_gp = st.session_state.get('gp_select', gp_options[0]) if gp_options else None
            gp_name = st.selectbox("Grand Prix", gp_options, index=gp_options.index(default_gp) if default_gp in gp_options else 0, key='gp_select')
        session_options = get_available_sessions(year, gp_name)
        if not session_options:
            st.error("No session data available for selected Grand Prix")
            return
        default_session = st.session_state.get('session_select', session_options[0][1])
        session_display = st.selectbox("Session", [s[1] for s in session_options], index=[s[1] for s in session_options].index(default_session) if default_session in [s[1] for s in session_options] else 0, key='session_select')
        session_name = next(s[0] for s in session_options if s[1] == session_display)
        st.header("👥 Driver Selection")
        driver_mode = st.selectbox("Selection Mode", ["Specific Drivers", "Teammates", "P1 vs P2"], index=2, key='driver_mode')
    else:
        with st.sidebar:
            st.header("🏁 Race Selection")
            year = st.number_input("Year", min_value=2018, max_value=2025, value=st.session_state.get('year_input', 2025), step=1, key='year_input')
            gp_options = get_available_gps(year)
            if not gp_options:
                st.error("No Grand Prix data available for selected year")
                return
            default_gp = st.session_state.get('gp_select', gp_options[0]) if gp_options else None
            gp_name = st.selectbox("Grand Prix", gp_options, index=gp_options.index(default_gp) if default_gp in gp_options else 0, key='gp_select')
            session_options = get_available_sessions(year, gp_name)
            if not session_options:
                st.error("No session data available for selected Grand Prix")
                return
            default_session = st.session_state.get('session_select', session_options[0][1])
            session_display = st.selectbox("Session", [s[1] for s in session_options], index=[s[1] for s in session_options].index(default_session) if default_session in [s[1] for s in session_options] else 0, key='session_select')
            session_name = next(s[0] for s in session_options if s[1] == session_display)
            st.header("👥 Driver Selection")
            driver_mode = st.radio("Selection Mode", ["Specific Drivers", "Teammates", "P1 vs P2"], index=2, key='driver_mode')


    # Load session and driver list with status
    try:
        session = load_session_data(year, gp_name, session_name)
        driver_info = get_session_drivers_with_times(session)
        st.session_state.session_loaded = True
    except Exception as e:
        st.info("We couldn't load data for this selection. Please try another session.")
        return

    drivers_to_plot = []
    teams = list(set([d['team'] for d in driver_info if d['team'] != 'Unknown']))

    # Driver selection UI (unchanged logic, but without debug)
    if not st.session_state.get('mobile_view', False):
        with st.sidebar:
            if driver_mode == "Specific Drivers":
                driver_options = [f"{d['code']} - {d['name']} ({d['lap_time_formatted']})" for d in driver_info]
                selected_drivers = st.multiselect("Select 2 drivers", driver_options, max_selections=2, help="Choose any two drivers from the session to compare their fastest laps")
                drivers_to_plot = [d.split(' - ')[0] for d in selected_drivers]
                if len(selected_drivers) == 2:
                    st.success(f"🏎️ Selected: {drivers_to_plot[0]} vs {drivers_to_plot[1]}")
                elif len(selected_drivers) == 1:
                    st.info(f"Selected: {drivers_to_plot[0]} - Please select one more driver")
            elif driver_mode == "Teammates":
                if teams:
                    selected_team = st.selectbox("Select Team", teams, help="Choose a team to compare their drivers")
                    teammates = [d['code'] for d in driver_info if d['team'] == selected_team]
                    if len(teammates) >= 2:
                        drivers_to_plot = teammates[:2]
                        teammate_names = [d['name'].split()[-1] for d in driver_info if d['code'] in drivers_to_plot]
                        st.success(f"🤝 Selected: {drivers_to_plot[0]} vs {drivers_to_plot[1]}")
                        st.info(f"Comparing teammates: {teammate_names[0]} vs {teammate_names[1]}")
                    else:
                        st.info(f"Only {len(teammates)} driver(s) found for {selected_team}")
                        if len(teammates) == 1:
                            st.info(f"Available: {teammates[0]}")
                else:
                    st.info("No teams found with valid data in this session.")
            elif driver_mode == "P1 vs P2":
                try:
                    results = session.results
                    if not results.empty and len(results) >= 2:
                        p1_driver = results.iloc[0]['Abbreviation'] 
                        p2_driver = results.iloc[1]['Abbreviation']
                        drivers_to_plot = [p1_driver, p2_driver]
                        p1_name = next((d['name'].split()[-1] for d in driver_info if d['code'] == p1_driver), p1_driver)
                        p2_name = next((d['name'].split()[-1] for d in driver_info if d['code'] == p2_driver), p2_driver)
                        st.success(f"🏆 P1: {p1_driver} ({p1_name}) vs P2: {p2_driver} ({p2_name})")
                        st.info("Using official session results for P1 vs P2")
                    else:
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
                except Exception:
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
            # Central animated outputs block under driver selection (desktop)
            st.markdown("---")
            st.subheader("🎬 Animated Output (Fast MP4)")
            st.caption("Lower FPS MP4 for quicker results. Output will appear next to the image.")
            fast_mp4_fps = st.slider("MP4 FPS (lower = faster)", min_value=5, max_value=30, value=5, step=1, help="Lower FPS reduces render time and file size.")
            gen_btn_center = st.button("🚀 Generate MP4", key='gen_anim_desktop')
            if gen_btn_center:
                if len(drivers_to_plot) != 2:
                    st.info("Select exactly two drivers first.")
                else:
                    with st.spinner("Generating animated outputs. This can take several minutes, please wait..."):
                        debug_exp = st.expander("Debug console", expanded=True)
                        prog_area = debug_exp.progress(0)
                        txt_area = debug_exp.empty()
                        try:
                            st.session_state['mp4_in_progress'] = True
                            outputs = _generate_gif_mp4_outputs(year, gp_name, session_display, drivers_to_plot, make_gif=False, make_mp4=True, progress_placeholder=prog_area, text_placeholder=txt_area, mp4_fps=fast_mp4_fps)
                            if outputs.get('mp4_path'):
                                st.session_state['latest_mp4_path'] = outputs['mp4_path']
                                st.success(f"MP4 created: {outputs['mp4_path']}")
                            # If no outputs, let exceptions surface; don't show a premature 'no outputs' message
                        except Exception as e:
                            # Add to debug log
                            if 'debug_log' not in st.session_state:
                                st.session_state.debug_log = []
                            st.session_state.debug_log.append(f"[ERROR] Desktop MP4 Generation failed: {str(e)}")
                            st.info(str(e))
                        finally:
                            st.session_state['mp4_in_progress'] = False

            # Auto-generate when 2 drivers are selected (desktop)
            if len(drivers_to_plot) == 2:
                st.session_state['hide_demo'] = True
            # Optional settings after generate button
            st.markdown("---")
            st.subheader("⚙️ Settings")
            st.session_state.units = st.selectbox("Speed units", ["km/h", "mph"], index=["km/h","mph"].index(st.session_state.get('units','km/h')))
            st.session_state.aspect_ratio = st.selectbox("Aspect ratio", ["Story 9:16","Post 1:1","Widescreen 16:9"], index=["Story 9:16","Post 1:1","Widescreen 16:9"].index(st.session_state.get('aspect_ratio','Story 9:16')))
            if st.button("📱 Toggle Mobile View"):
                st.session_state.mobile_view = not st.session_state.get('mobile_view', False)
                st.rerun()
            
    else:
        if driver_mode == "Specific Drivers":
            driver_options = [f"{d['code']} - {d['name']} ({d['lap_time_formatted']})" for d in driver_info]
            selected_drivers = st.multiselect("Select 2 drivers", driver_options, max_selections=2, help="Choose any two drivers from the session to compare their fastest laps")
            drivers_to_plot = [d.split(' - ')[0] for d in selected_drivers]
            if len(selected_drivers) == 2:
                st.success(f"🏎️ Selected: {drivers_to_plot[0]} vs {drivers_to_plot[1]}")
            elif len(selected_drivers) == 1:
                st.info(f"Selected: {drivers_to_plot[0]} - Please select one more driver")
        elif driver_mode == "Teammates":
            if teams:
                selected_team = st.selectbox("Select Team", teams, help="Choose a team to compare their drivers")
                teammates = [d['code'] for d in driver_info if d['team'] == selected_team]
                if len(teammates) >= 2:
                    drivers_to_plot = teammates[:2]
                    teammate_names = [d['name'].split()[-1] for d in driver_info if d['code'] in drivers_to_plot]
                    st.success(f"🤝 Selected: {drivers_to_plot[0]} vs {drivers_to_plot[1]}")
                    st.info(f"Comparing teammates: {teammate_names[0]} vs {teammate_names[1]}")
                else:
                    st.info(f"Only {len(teammates)} driver(s) found for {selected_team}")
                    if len(teammates) == 1:
                        st.info(f"Available: {teammates[0]}")
            else:
                st.info("No teams found with valid data in this session.")
        elif driver_mode == "P1 vs P2":
            try:
                results = session.results
                if not results.empty and len(results) >= 2:
                    p1_driver = results.iloc[0]['Abbreviation']
                    p2_driver = results.iloc[1]['Abbreviation']
                    drivers_to_plot = [p1_driver, p2_driver]
                    p1_name = next((d['name'].split()[-1] for d in driver_info if d['code'] == p1_driver), p1_driver)
                    p2_name = next((d['name'].split()[-1] for d in driver_info if d['code'] == p2_driver), p2_driver)
                    st.success(f"🏆 P1: {p1_driver} ({p1_name}) vs P2: {p2_driver} ({p2_name})")
                    st.info("Using official session results for P1 vs P2")
                else:
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
            except Exception:
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
        # Central animated outputs block for mobile under selection
        st.markdown("---")
        st.subheader("🎬 Animated Output (Fast MP4)")
        st.caption("Lower FPS MP4 for quicker results. Output will appear next to the image.")
        fast_mp4_fps_m = st.slider("MP4 FPS (lower = faster)", min_value=5, max_value=30, value=5, step=1, help="Lower FPS reduces render time and file size.", key='mp4_fps_mobile')
        gen_btn_center_m = st.button("🚀 Generate MP4", key='gen_anim_mobile')
        if gen_btn_center_m:
            if len(drivers_to_plot) != 2:
                st.info("Select exactly two drivers first.")
            else:
                with st.spinner("Generating animated outputs. This can take several minutes, please wait..."):
                    debug_exp = st.expander("Debug console", expanded=True)
                    prog_area = debug_exp.progress(0)
                    txt_area = debug_exp.empty()
                    try:
                        st.session_state['mp4_in_progress'] = True
                        outputs = _generate_gif_mp4_outputs(year, gp_name, session_display, drivers_to_plot, make_gif=False, make_mp4=True, progress_placeholder=prog_area, text_placeholder=txt_area, mp4_fps=fast_mp4_fps_m)
                        if outputs.get('mp4_path'):
                            st.session_state['latest_mp4_path'] = outputs['mp4_path']
                            st.success(f"MP4 created: {outputs['mp4_path']}")
                        # If no outputs, let exceptions surface; don't show a premature 'no outputs' message
                    except Exception as e:
                        # Add to debug log
                        if 'debug_log' not in st.session_state:
                            st.session_state.debug_log = []
                        st.session_state.debug_log.append(f"[ERROR] Mobile MP4 Generation failed: {str(e)}")
                        st.info(str(e))
                    finally:
                        st.session_state['mp4_in_progress'] = False

        # Auto-generate when 2 drivers are selected (mobile)
        if len(drivers_to_plot) == 2:
            st.session_state['hide_demo'] = True
        # Optional settings for mobile at bottom
        st.markdown("---")
        st.subheader("⚙️ Settings")
        st.session_state.units = st.selectbox("Speed units", ["km/h", "mph"], index=["km/h","mph"].index(st.session_state.get('units','km/h')))
        st.session_state.aspect_ratio = st.selectbox("Aspect ratio", ["Story 9:16","Post 1:1","Widescreen 16:9"], index=["Story 9:16","Post 1:1","Widescreen 16:9"].index(st.session_state.get('aspect_ratio','Story 9:16')))
        st.markdown("---")
        st.subheader("🎬 Optional: Animated Outputs (may take time)")
        make_gif = st.checkbox("Generate GIF (animated)", value=False, help="Creates an animated GIF; may take longer and produce a large file.")
        make_mp4 = st.checkbox("Generate MP4 (video)", value=False, help="Creates a 60 FPS MP4; requires ffmpeg and can take a while.")



    # Main content area - Auto-generate when 2 drivers are selected
    if len(drivers_to_plot) == 2:
        st.caption(f"Ready to analyze: {drivers_to_plot[0]} vs {drivers_to_plot[1]}")
        
        # Auto-generate visualization immediately
        try:
            driver_data = prepare_driver_data(session, drivers_to_plot, driver_mode)
            # Build main figure
            fig = create_data_driven_lap_plot(
                driver_data, session, drivers_to_plot, year, gp_name, session_display,
                watermark_text='@datadrivenlaps',
                units=st.session_state.get('units', 'km/h'),
                aspect_ratio=st.session_state.get('aspect_ratio', 'Story 9:16')
            )
            # Decide layout: centered image normally; two columns if MP4 generating or available
            mp4_active = st.session_state.get('mp4_in_progress', False) or bool(st.session_state.get('latest_mp4_path'))
            if mp4_active:
                col_left, col_right = st.columns([1, 1])
                with col_left:
                    st.subheader("Lap visualization")
                    st.pyplot(fig, use_container_width=True)
            else:
                st.subheader("Lap visualization")
                center_left, center, center_right = st.columns([1, 2.2, 1])
                with center:
                    st.pyplot(fig, use_container_width=True)

            # Story highlights
            p1_tel = driver_data[drivers_to_plot[0]]['telemetry']
            p2_tel = driver_data[drivers_to_plot[1]]['telemetry']
            highlights = generate_story_highlights(drivers_to_plot[0], drivers_to_plot[1], p1_tel, p2_tel)
            if highlights:
                st.subheader("📌 Highlights")
                for h in highlights:
                    st.markdown(f"- {h}")

            # Download buttons for multiple aspect ratios
            st.subheader("⬇️ Download")
            from io import BytesIO
            def fig_bytes(ar_label: str) -> BytesIO:
                f = create_data_driven_lap_plot(
                    driver_data, session, drivers_to_plot, year, gp_name, session_display,
                    watermark_text='@datadrivenlaps',
                    units=st.session_state.get('units', 'km/h'),
                    aspect_ratio=ar_label
                )
                buf = BytesIO()
                f.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)
                plt.close(f)
                return buf
            col_d1, col_d2, col_d3 = st.columns([1,1,1])
            filename_base = f"{year}_{gp_name.replace(' ', '_')}_{session_display.replace(' ', '')}_{drivers_to_plot[0]}_vs_{drivers_to_plot[1]}"
            with col_d1:
                st.download_button("Story 9:16 (PNG)", data=fig_bytes('Story 9:16'), file_name=f"{filename_base}_story.png", mime="image/png")
            with col_d2:
                st.download_button("Post 1:1 (PNG)", data=fig_bytes('Post 1:1'), file_name=f"{filename_base}_post.png", mime="image/png")
            with col_d3:
                st.download_button("Widescreen 16:9 (PNG)", data=fig_bytes('Widescreen 16:9'), file_name=f"{filename_base}_16x9.png", mime="image/png")

            plt.close(fig)

            # If we have a recent MP4, show it aligned with the image and add download button in Download section
            if st.session_state.get('latest_mp4_path'):
                # Place the video at the same vertical height as the image by rendering in the right column when present
                try:
                    if 'col_right' in locals():
                        with col_right:
                            st.subheader("Lap movie")
                            with open(st.session_state['latest_mp4_path'], 'rb') as f:
                                st.video(f.read())
                except Exception:
                    pass

            # MP4 download button in the main Download section (only when available)
            if st.session_state.get('latest_mp4_path'):
                try:
                    with open(st.session_state['latest_mp4_path'], 'rb') as f:
                        st.download_button("Download MP4", data=f, file_name=Path(st.session_state['latest_mp4_path']).name, mime="video/mp4", key='dl_mp4_main')
                except Exception:
                    pass

            # Optional animated outputs block
            if st.session_state.get('mobile_view', False):
                st.markdown("---")
                st.subheader("🎬 Animated Outputs (Optional)")
                make_gif = st.checkbox("Generate GIF (animated)", value=False, help="Creates an animated GIF; may take longer and produce a large file.")
                make_mp4 = st.checkbox("Generate MP4 (video)", value=False, help="Creates a 60 FPS MP4; requires ffmpeg and can take a while.")
                if (make_gif or make_mp4) and st.button("🚀 Generate GIF/MP4"):
                    with st.spinner("Generating animated outputs. This can take several minutes, please wait..."):
                        prog_area = st.empty()
                        txt_area = st.empty()
                        st.caption("Logs:")
                        try:
                            outputs = _generate_gif_mp4_outputs(year, gp_name, session_display, drivers_to_plot, make_gif, make_mp4, progress_placeholder=prog_area, text_placeholder=txt_area)
                            if outputs.get('gif_path'):
                                st.success(f"GIF created: {outputs['gif_path']}")
                                with open(outputs['gif_path'], 'rb') as f:
                                    st.download_button("Download GIF", data=f, file_name=Path(outputs['gif_path']).name, mime="image/gif")
                            if outputs.get('mp4_path'):
                                st.success(f"MP4 created: {outputs['mp4_path']}")
                                with open(outputs['mp4_path'], 'rb') as f:
                                    st.download_button("Download MP4", data=f, file_name=Path(outputs['mp4_path']).name, mime="video/mp4")
                            if not outputs:
                                st.info("No animated outputs were generated.")
                        except Exception as e:
                            st.info(str(e))
        except Exception as e:
            st.info("Something went wrong while generating the visualization. Please try again.")
    else:
        st.caption("Select exactly two drivers to generate a data-driven lap image.")
    
    # Persistent Debug Console at bottom
    st.markdown("---")
    st.subheader("🔧 Debug Console")
    
    # Initialize debug log in session state
    if 'debug_log' not in st.session_state:
        st.session_state.debug_log = []
    
    # Add current status to debug log
    current_status = []
    current_status.append(f"Session loaded: {st.session_state.get('session_loaded', False)}")
    current_status.append(f"Drivers selected: {len(drivers_to_plot) if 'drivers_to_plot' in locals() else 0}")
    current_status.append(f"MP4 in progress: {st.session_state.get('mp4_in_progress', False)}")
    current_status.append(f"Latest MP4 path: {st.session_state.get('latest_mp4_path', 'None')}")
    
    # Show debug info
    with st.expander("Live Debug Info", expanded=False):
        st.text("Current Status:")
        for status in current_status:
            st.text(f"  • {status}")
        
        if st.session_state.debug_log:
            st.text("\nRecent Debug Messages:")
            for i, log_entry in enumerate(st.session_state.debug_log[-10:]):  # Show last 10 entries
                st.text(f"  {i+1}. {log_entry}")
        
        if st.button("Clear Debug Log"):
            st.session_state.debug_log = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("Made using [FastF1](https://github.com/theOehrly/Fast-F1) and [Streamlit](https://streamlit.io)")

if __name__ == "__main__":
    main() 