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
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import sys
import requests

# Try to import tqdm for progress bar, fallback to simple progress if not available
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Simple fallback, no attempt to install in this interactive version for now
    print("⚠️ tqdm not found. Progress bars will be simpler. Consider installing tqdm (`pip install tqdm`) for a better experience.")

class SimpleProgressBar:
    """Simple progress bar fallback if tqdm is not available"""
    def __init__(self, total, desc="Progress"):
        self.total = total
        self.current = 0
        self.desc = desc
        self.bar_length = 40
        if self.total == 0: # Avoid division by zero if total is 0
            self.total = 1

    def update(self, n=1):
        self.current += n
        self.display()

    def display(self):
        percent = (self.current / self.total) * 100
        filled_length = int(self.bar_length * self.current // self.total)
        bar = '█' * filled_length + '-' * (self.bar_length - filled_length)
        sys.stdout.write(f'\r{self.desc}: |{bar}| {percent:.1f}% ({self.current}/{self.total})')
        sys.stdout.flush()

    def close(self):
        sys.stdout.write('\n') # New line when complete
        sys.stdout.flush()

# Use tqdm if available, otherwise use the simple progress bar
ProgressBar = tqdm if TQDM_AVAILABLE else SimpleProgressBar

class InteractiveDataDrivenLaps:
    def __init__(self):
        self.year = None
        self.gp_name = None
        self.session_name = None
        self.driver_selection_mode = None # 'P1P2', 'SpecificDrivers', 'Teammates'
        self.drivers_to_plot = [] # List of driver codes/abbreviations
        self.team_to_plot = None 
        self.output_formats = [] # ['png', 'gif', 'mp4']

        self.team_colors_mapping = {
            # F1 2023/2024/2025 approximate colors - needs refinement and expansion
            'Red Bull Racing': '#0600EF', # Dark Blue
            'Mercedes': '#00D2BE',        # Teal
            'Ferrari': '#DC0000',         # Red
            'McLaren': '#FF8700',         # Orange
            'Aston Martin': '#006F62',    # Green
            'Alpine': '#0090FF',          # Blue
            'Williams': '#005AFF',        # Lighter Blue
            'AlphaTauri': '#2B4562',      # Navy (Visa Cash App RB) / RB F1 Team
            'Kick Sauber': '#00E676',      # Bright Green (Stake F1 Team / Audi)
            'Haas F1 Team': '#B6BABD'     # Light Gray/White
        }
        self.default_colors = ['#FF0000', '#0000FF', '#00FF00', '#FFA500', '#800080', '#FFFF00'] # Fallback colors

        self.track_segments_x = None # To store P1 X-coordinates for track segments
        self.track_segments_y = None # To store P1 Y-coordinates for track segments
        self.track_segment_colors_current = None # To store the dynamically updated colors of segments
        self.d1_speeds_orig = None # Store original P1 speeds for comparison
        self.d2_speeds_aligned_orig = None # Store original P2 speeds (aligned to P1 dist) for comparison
        self.d1_full_tel_distance = None # Store original P1 telemetry distances for segment mapping
        self.p1_color_obj = None # Store P1 matplotlib color object
        self.p2_color_obj = None # Store P2 matplotlib color object

        self.setup_matplotlib()
        self.setup_fastf1_cache()

    def setup_matplotlib(self):
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 150
        
        try:
            fastf1.plotting.setup_mpl(color_scheme='fastf1', misc_mpl_mods=False)
        except TypeError: # older fastf1 version
            fastf1.plotting.setup_mpl(misc_mpl_mods=False)

    def setup_fastf1_cache(self):
        cache_path = os.path.expanduser('~/cache_fastf1_interactive')
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        fastf1.Cache.enable_cache(cache_path)
        print(f"🏎️ FastF1 cache enabled at: {cache_path}")

    def adjust_color_luminance(self, color_hex, factor):
        """
        Adjust the luminance of a color by a factor.
        factor > 1.0 makes it lighter, factor < 1.0 makes it darker
        """
        try:
            # Remove # if present
            color_hex = color_hex.lstrip('#')
            
            # Convert hex to RGB
            r = int(color_hex[0:2], 16) / 255.0
            g = int(color_hex[2:4], 16) / 255.0
            b = int(color_hex[4:6], 16) / 255.0
            
            # Convert RGB to HSL for luminance adjustment
            max_val = max(r, g, b)
            min_val = min(r, g, b)
            diff = max_val - min_val
            
            # Calculate lightness
            lightness = (max_val + min_val) / 2
            
            if diff == 0:
                hue = saturation = 0  # achromatic
            else:
                saturation = diff / (2 - max_val - min_val) if lightness > 0.5 else diff / (max_val + min_val)
                
                if max_val == r:
                    hue = (g - b) / diff + (6 if g < b else 0)
                elif max_val == g:
                    hue = (b - r) / diff + 2
                else:
                    hue = (r - g) / diff + 4
                hue /= 6
            
            # Adjust lightness
            new_lightness = min(0.9, max(0.1, lightness * factor))  # Keep within reasonable bounds
            
            # Convert back to RGB
            def hsl_to_rgb_component(p, q, t):
                if t < 0: t += 1
                if t > 1: t -= 1
                if t < 1/6: return p + (q - p) * 6 * t
                if t < 1/2: return q
                if t < 2/3: return p + (q - p) * (2/3 - t) * 6
                return p
            
            if saturation == 0:
                new_r = new_g = new_b = new_lightness  # achromatic
            else:
                q = new_lightness * (1 + saturation) if new_lightness < 0.5 else new_lightness + saturation - new_lightness * saturation
                p = 2 * new_lightness - q
                new_r = hsl_to_rgb_component(p, q, hue + 1/3)
                new_g = hsl_to_rgb_component(p, q, hue)
                new_b = hsl_to_rgb_component(p, q, hue - 1/3)
            
            # Convert back to hex
            new_r = int(new_r * 255)
            new_g = int(new_g * 255)
            new_b = int(new_b * 255)
            
            return f"#{new_r:02x}{new_g:02x}{new_b:02x}"
            
        except Exception:
            # Fallback: simple RGB adjustment if HSL conversion fails
            try:
                color_hex = color_hex.lstrip('#')
                r = int(color_hex[0:2], 16)
                g = int(color_hex[2:4], 16)
                b = int(color_hex[4:6], 16)
                
                if factor > 1.0:  # Lighten
                    r = min(255, int(r + (255 - r) * (factor - 1.0)))
                    g = min(255, int(g + (255 - g) * (factor - 1.0)))
                    b = min(255, int(b + (255 - b) * (factor - 1.0)))
                else:  # Darken
                    r = max(0, int(r * factor))
                    g = max(0, int(g * factor))
                    b = max(0, int(b * factor))
                
                return f"#{r:02x}{g:02x}{b:02x}"
            except Exception:
                return color_hex  # Return original color if all fails

    def get_team_color(self, driver_code, session, is_secondary=False):
        """Get the team color for a driver. Uses official FastF1 color if possible, then mapping."""
        team_name = None
        base_color = None
        
        try:
            driver_info = session.get_driver(driver_code)
            if driver_info is not None and not driver_info.empty:
                team_name = driver_info['TeamName']
                
                # For teammates, use base team color with luminance adjustment
                if team_name in self.team_colors_mapping:
                    base_color = self.team_colors_mapping[team_name]
                    
                    if is_secondary:
                        # Make the secondary driver's color lighter (more luminous)
                        return self.adjust_color_luminance(base_color, 1.4)  # 40% lighter
                    else:
                        # Keep primary driver's color as is, or slightly darker for contrast
                        return self.adjust_color_luminance(base_color, 0.85)  # 15% darker
        except Exception:
            pass

        # Fallback to FastF1's plotting color if available
        try:
            color = fastf1.plotting.get_driver_color(driver_code, session=session)
            generic_colors = ['#000000', '#ffffff', '#808080', '#B6BABD']
            
            if driver_info is not None and not driver_info.empty:
                 if color.upper() in generic_colors and driver_info['TeamName'] != 'Haas F1 Team':
                     pass # Let it fall through to defaults
                 else:
                    if is_secondary and base_color is None:
                        # If we're using FastF1 color for secondary, make it lighter
                        return self.adjust_color_luminance(color, 1.3)
                    else:
                        return color
            else:
                if color.upper() not in generic_colors:
                    if is_secondary:
                        return self.adjust_color_luminance(color, 1.3)
                    else:
                        return color

        except Exception:
            pass
        
        # Final fallback to default colors with luminance adjustment
        driver_index = self.drivers_to_plot.index(driver_code) if driver_code in self.drivers_to_plot else 0
        fallback_color = self.default_colors[driver_index % len(self.default_colors)]
        
        if is_secondary:
            return self.adjust_color_luminance(fallback_color, 1.4)
        else:
            return self.adjust_color_luminance(fallback_color, 0.9)

    def run(self):
        """Main function to run the interactive selection and plotting."""
        print("Welcome to the Interactive F1 Data Driven Laps Generator!")
        print("=========================================================")
        if not self.collect_user_inputs():
            return

        # --- Confirmation Screen ---
        print("\n📋 === SUMMARY OF YOUR CHOICES ===")
        print(f"Year: {self.year}")
        print(f"Grand Prix: {self.gp_name}")
        print(f"Session: {self.session_name}")
        if self.driver_selection_mode == 'P1P2':
            print("Driver Comparison: P1 vs P2")
        elif self.driver_selection_mode == 'SpecificDrivers':
            print(f"Driver Comparison: {self.drivers_to_plot[0]} vs {self.drivers_to_plot[1]}")
        elif self.driver_selection_mode == 'Teammates':
            print(f"Driver Comparison: Teammates from {self.team_to_plot}")
        
        print(f"Output Formats: {', '.join(self.output_formats).upper() if self.output_formats else 'None'}")
        print("=================================")
        
        while True:
            proceed = input("Proceed with these settings? (y/n): ").strip().lower()
            if proceed == 'y':
                break
            elif proceed == 'n':
                print("❌ Operation cancelled by user.")
                return
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
        # --------------------------

        if not self.load_session_data(): 
            return
        if not self.prepare_driver_data():
            return
            
        self.prepare_cinematic_animation_data()
        # self.prepare_track_features() # placeholder for track line coloring if re-added
        self.calculate_sector_times()

        if 'png' in self.output_formats:
            self.create_preview_image()
        if 'gif' in self.output_formats:
            self.create_gif()
        if 'mp4' in self.output_formats:
            self.create_mp4()
            
        print("\n🎉 All requested outputs generated successfully!")

    def collect_user_inputs(self):
        """Collect all necessary inputs from the user via terminal prompts."""
        print("Welcome to the Interactive F1 Data Driven Laps Generator!")
        print("=========================================================")
        
        # Get Year
        while True:
            try:
                year_input = input("🗓️ Enter the F1 season year (press Enter for 2025): ").strip()
                if not year_input:  # Default to current year if no input
                    self.year = 2025
                    print(f"📅 Using default year: {self.year}")
                    break
                else:
                    self.year = int(year_input)
                    if 1950 <= self.year <= pd.Timestamp.now().year + 1: # Allow one year in future for pre-season
                        break
                    else:
                        print(f"Invalid year. Please enter a year between 1950 and {pd.Timestamp.now().year + 1}.")
            except ValueError:
                print("Invalid input. Please enter a number for the year.")

        # Get GP Name
        while True:
            self.gp_name = input(f"🌍 Enter the Grand Prix name for {self.year} (e.g., Monaco, Bahrain, or 'list' to see available GPs): ").strip()
            if self.gp_name.lower() == 'list':
                self.list_available_gps()
                continue
            if len(self.gp_name) > 2: # Basic validation
                break
            else:
                print("GP name seems too short. Please enter a valid GP name or 'list'.")
        
        # Get Session
        # Check if this is a sprint weekend (simplified approach)
        # Sprint weekends for 2024-2025 are typically: China, Miami, Austria, USA (COTA), Brazil, Qatar
        sprint_weekends_2024_2025 = [
            'china', 'chinese', 'miami', 'austria', 'austrian', 'usa', 'united states', 
            'brazil', 'brazilian', 'qatar', 'las vegas'  # Common sprint weekend locations
        ]
        
        is_sprint_weekend = any(location in self.gp_name.lower() for location in sprint_weekends_2024_2025)
        
        # Build the valid sessions list
        regular_sessions = ['FP1', 'FP2', 'FP3', 'Q', 'R']  # Simplified to just Q for qualifying
        sprint_sessions = ['SQ', 'SR'] if is_sprint_weekend else []  # SQ = Sprint Qualifying, SR = Sprint Race
        valid_sessions = regular_sessions + sprint_sessions
        
        # Create display string
        regular_display = "FP1/FP2/FP3/Q/R"
        if sprint_sessions:
            sprint_display = '/'.join(sprint_sessions)
            session_display = f"{regular_display} | Sprint: {sprint_display}"
        else:
            session_display = regular_display

        while True:
            self.session_name = input(f"⏱️ Enter the session for {self.year} {self.gp_name} GP ({session_display}): ").strip().upper()
            
            # Handle legacy session names for backward compatibility
            if self.session_name == 'S' and is_sprint_weekend:
                self.session_name = 'SR'  # Old 'S' (Sprint) -> 'SR' (Sprint Race)
            elif self.session_name in ['Q1', 'Q2', 'Q3']:  # Convert Q1/Q2/Q3 to Q
                self.session_name = 'Q'
                print(f"📝 Note: Using 'Q' (full qualifying session) instead of {self.session_name}")
            
            if self.session_name in valid_sessions:
                break
            else:
                if sprint_sessions:
                    print(f"Invalid session. Please choose from: {', '.join(regular_sessions)} | Sprint: {', '.join(sprint_sessions)}")
                else:
                    print(f"Invalid session. Please choose from: {', '.join(regular_sessions)}")

        # --- Load session here to list drivers/teams before asking for specific ones ---
        # Temporarily load session to get driver/team list if needed for choices 2 or 3
        temp_session_loaded_for_info = False
        if self.driver_selection_mode in ['2', '3']: #提前加载session的判断条件
            print("\n⏳ Attempting to load session data to list available drivers/teams...")
            # Use a temporary object or a flag to indicate this specific loading purpose
            # We are essentially doing a part of self.load_session_data() here early
            try:
                temp_session = fastf1.get_session(self.year, self.gp_name, self.session_name)
                temp_session.load(laps=True, telemetry=False, weather=False, messages=False) # Only need laps for names
                if temp_session.laps is not None and not temp_session.laps.empty:
                    temp_session_loaded_for_info = True
                    print("✅ Session data loaded for driver/team listing.")
                else:
                    print("⚠️ Could not load lap data to list drivers/teams. You'll need to enter them manually.")
            except Exception as e:
                print(f"⚠️ Error loading session for driver/team list: {e}. You'll need to enter them manually.")
        # -----------------------------------------------------------------------------

        # Get Driver Selection Mode
        print("\n📊 How would you like to select drivers for comparison?")
        print("1. P1 vs P2 (Fastest two drivers of the session)")
        print("2. Specific Drivers (Enter two driver codes, e.g., VER, HAM)")
        print("3. Teammates (Enter a team name, e.g., Ferrari)")
        
        # Store the choice from above to avoid asking again if we already loaded session for it.
        # This choice_for_driver_mode variable will be used to decide if we need to prompt for driver/team names.

        while True:
            if 'choice_for_driver_mode' not in locals(): # If not pre-selected due to temp session load
                 choice_for_driver_mode = input("Enter your choice (1, 2, or 3): ").strip()

            if choice_for_driver_mode == '1':
                self.driver_selection_mode = 'P1P2'
                break
            elif choice_for_driver_mode == '2':
                self.driver_selection_mode = 'SpecificDrivers'
                if temp_session_loaded_for_info:
                    available_drivers = sorted(temp_session.laps['Driver'].unique())
                    print("\nAvailable driver codes in this session:")
                    # Print in columns for better readability
                    max_len = max(len(d) for d in available_drivers) if available_drivers else 0
                    cols = 5 
                    for i in range(0, len(available_drivers), cols):
                        print("  ".join(f"{d:<{max_len}}" for d in available_drivers[i:i+cols]))
                    print("")
                
                d1 = input("Enter Driver 1 code (e.g., VER): ").strip().upper()
                d2 = input("Enter Driver 2 code (e.g., HAM): ").strip().upper()
                if d1 and d2 and d1 != d2:
                    self.drivers_to_plot = [d1, d2]
                    break
                else:
                    print("Invalid driver codes. Please enter two different, non-empty codes.")
                    if 'choice_for_driver_mode' in locals(): del choice_for_driver_mode # Reset to re-ask mode

            elif choice_for_driver_mode == '3':
                self.driver_selection_mode = 'Teammates'
                if temp_session_loaded_for_info:
                    available_teams = sorted(temp_session.laps['TeamName'].dropna().unique())
                    print("\nAvailable teams in this session:")
                    for team_name_item in available_teams:
                        print(f"- {team_name_item}")
                    print("")

                self.team_to_plot = input("Enter Team Name (e.g., Red Bull Racing): ").strip()
                if self.team_to_plot:
                    break
                else:
                    print("Team name cannot be empty.")
                    if 'choice_for_driver_mode' in locals(): del choice_for_driver_mode # Reset to re-ask mode
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
                if 'choice_for_driver_mode' in locals(): del choice_for_driver_mode # Reset to re-ask mode
            
            # Clear pre-loaded choice if we loop back due to error
            if 'choice_for_driver_mode' in locals() and self.driver_selection_mode is None:
                del choice_for_driver_mode

        # Get Output Formats
        print("\n🖼️ Which output formats would you like?")
        print("   (You can choose multiple, e.g., 'png,gif')")
        print("1. PNG (Still preview image)")
        print("2. GIF (Animated)")
        print("3. MP4 (Video)")
        print("4. All (PNG, GIF, and MP4)")
        output_options = {'1': 'png', '2': 'gif', '3': 'mp4'}
        while True:
            choices_str = input("Enter your choice(s) (comma-separated, e.g., 1,2 or 4 for all): ").strip()
            if choices_str == '4':
                self.output_formats = ['png', 'gif', 'mp4']
                break
            
            selected_formats = set()
            valid_choice = True
            for ch in choices_str.split(','):
                ch = ch.strip()
                if ch in output_options:
                    selected_formats.add(output_options[ch])
                elif ch: # if it's not empty and not a valid choice
                    valid_choice = False
                    break
            
            if valid_choice and selected_formats:
                self.output_formats = list(selected_formats)
                break
            elif not selected_formats and not choices_str: # User pressed enter for no formats
                 print("No output format selected. Exiting.")
                 return False
            else:
                print("Invalid choice. Please use numbers 1-3 (comma-separated) or 4 for all.")
        
        print("\n👍 Inputs collected. Starting data processing...")
        return True

    def list_available_gps(self):
        """Lists available GPs for the selected year using FastF1."""
        try:
            schedule = fastf1.get_event_schedule(self.year, include_testing=False)
            if schedule.empty:
                print(f"No schedule data found for {self.year}.")
                return
            
            print(f"\n📅 Available GPs for {self.year}:")
            for index, event in schedule.iterrows():
                print(f"- {event['EventName']} (Round {event['RoundNumber']})")
            print("") # Extra line for readability
        except Exception as e:
            print(f"Could not fetch event schedule for {self.year}: {e}")
            print("Please ensure the year is correct and you have an internet connection.")

    def load_session_data(self):
        """Loads the F1 session data using FastF1."""
        print(f"\n🎬 Loading data for: {self.year} {self.gp_name} GP - {self.session_name}")
        
        try:
            self.session_obj = fastf1.get_session(self.year, self.gp_name, self.session_name)
            self.session_obj.load(telemetry=True, weather=False, messages=True)
            print(f"✅ Successfully loaded: {self.session_obj.event['EventName']} - {self.session_obj.name}")
            
            if self.session_obj.laps is None or self.session_obj.laps.empty:
                print("❌ No lap data available for this session.")
                return False 
            
            return True
        except ValueError as e_value:
            if "invalid session type" in str(e_value).lower():
                print(f"❌ Could not find session '{self.session_name}' for {self.year} {self.gp_name}")
                print("   Try using: FP1, FP2, FP3, Q, R")
            else:
                print(f"❌ Value Error: {e_value}")
            return False
        except Exception as e:
            print(f"❌ Error loading session data: {e}")
            return False

    def prepare_driver_data(self):
        """Prepares telemetry and lap data for the selected drivers."""
        if self.session_obj is None:
            print("❌ Session data not loaded. Cannot prepare driver data.")
            return False

        if self.driver_selection_mode == 'P1P2':
            # Get the two fastest drivers from the session
            all_laps_sorted = self.session_obj.laps.sort_values(by='LapTime')
            fastest_unique_drivers = all_laps_sorted.drop_duplicates(subset=['Driver'])
            
            print(f"\n🔍 Top drivers in {self.session_name}:")
            for i, (idx, row) in enumerate(fastest_unique_drivers.head(10).iterrows()):
                print(f"   {i+1:2d}. {row['Driver']} - {row['LapTime']} (Lap {row['LapNumber']})")
            
            if len(fastest_unique_drivers) < 2:
                print("❌ Not enough unique drivers with fastest laps to compare P1 and P2.")
                return False
                
            self.drivers_to_plot = [fastest_unique_drivers.iloc[0]['Driver'], fastest_unique_drivers.iloc[1]['Driver']]
            print(f"🏆 Selected P1 ({self.drivers_to_plot[0]}) and P2 ({self.drivers_to_plot[1]}) from {self.session_name}")

        elif self.driver_selection_mode == 'Teammates':
            # First, try to get team drivers from session driver info (more reliable)
            team_drivers_from_session = []
            matched_team_name = None
            
            try:
                # Get all drivers in the session and find teammates
                for driver_number in self.session_obj.drivers:
                    driver_details = self.session_obj.get_driver(driver_number)
                    if driver_details is not None and not driver_details.empty:
                        team_name_from_driver = driver_details.get('TeamName', '')
                        if self.team_to_plot.lower() in team_name_from_driver.lower():
                            team_drivers_from_session.append(driver_details['Abbreviation'])
                            if matched_team_name is None:
                                matched_team_name = team_name_from_driver
                
                if len(team_drivers_from_session) >= 2:
                    self.drivers_to_plot = list(team_drivers_from_session[:2])
                    print(f"🏆 Found teammates from session info: {self.drivers_to_plot[0]} vs {self.drivers_to_plot[1]} ({matched_team_name})")
                else:
                    print(f"ℹ️ Could not find two drivers for team '{self.team_to_plot}' in session driver info. Found: {team_drivers_from_session}")
                    
                    # Fallback: Check if TeamName column exists in lap data
                    if 'TeamName' not in self.session_obj.laps.columns:
                        print(f"❌ Error: 'TeamName' column not found in lap data for {self.session_obj.event['EventName']} - {self.session_obj.name}.")
                        print("   Cannot perform teammate comparison for this session as team data is missing from both session info and laps.")
                        print("   This can sometimes happen with certain session types. Try P1/P2 or specific drivers instead.")
                        return False

                    # Try to find teammates from lap data
                    drivers_in_team_from_laps = self.session_obj.laps[self.session_obj.laps['TeamName'].str.contains(self.team_to_plot, case=False, na=False)]['Driver'].unique()
                    
                    if len(drivers_in_team_from_laps) < 2:
                        print(f"❌ Could not find two distinct drivers for team '{self.team_to_plot}' in lap data either. Found: {drivers_in_team_from_laps}")
                        
                        # Last attempt: try exact team name matching from available teams
                        if 'TeamName' in self.session_obj.laps.columns:
                            all_teams_in_session = self.session_obj.laps['TeamName'].dropna().unique()
                            print(f"Available teams in lap data: {list(all_teams_in_session)}")
                        return False
                    else:
                        self.drivers_to_plot = list(drivers_in_team_from_laps[:2])
                        print(f"🏆 Comparing teammates from lap data: {self.drivers_to_plot[0]} vs {self.drivers_to_plot[1]}")
                        
            except Exception as e_session_info:
                print(f"❌ Error getting team info from session drivers: {e_session_info}")
                return False
        
        self.driver_data_dict = {}
        temp_lap_data_for_sorting = []

        for i, driver_code_from_list in enumerate(self.drivers_to_plot):
            # Use standard FastF1 method to get driver's fastest lap
            lap = self.session_obj.laps.pick_drivers(driver_code_from_list).pick_fastest()
            
            if lap is None or pd.isna(lap['LapTime']):
                print(f"⚠️ No fastest lap data found for driver {driver_code_from_list}. Skipping this driver.")
                continue
            
            # Use the Driver Abbreviation from the lap data for consistency as key
            actual_driver_code = lap['Driver'] 

            telemetry = lap.get_telemetry().add_distance()
            if telemetry.empty:
                print(f"⚠️ No telemetry data for {actual_driver_code}'s fastest lap. Skipping.")
                continue

            driver_info_session = self.session_obj.get_driver(actual_driver_code) 
            
            # Get base color first (without secondary modification)
            assigned_color = self.get_team_color(actual_driver_code, self.session_obj, is_secondary=False)

            current_driver_lap_data = {
                'lap_info': lap,
                'telemetry': telemetry,
                'driver_name': driver_info_session['FullName'] if driver_info_session is not None else actual_driver_code,
                'team_name': driver_info_session['TeamName'] if driver_info_session is not None else 'N/A',
                'color': assigned_color,
                'lap_time_td': lap['LapTime'],
                'lap_time_seconds': lap['LapTime'].total_seconds(),
                'compound': lap['Compound'],
                'tyre_life': lap['TyreLife'] if 'TyreLife' in lap and pd.notna(lap['TyreLife']) else 'N/A',
                'lap_number': lap['LapNumber']
            }
            self.driver_data_dict[actual_driver_code] = current_driver_lap_data
            temp_lap_data_for_sorting.append(current_driver_lap_data)

        if len(self.driver_data_dict) < 2:
            print("❌ Not enough valid driver data to proceed (need 2 with telemetry).")
            return False

        temp_lap_data_for_sorting.sort(key=lambda x: x['lap_time_seconds'])
        # Update self.drivers_to_plot to reflect the actual sorted order of drivers with data
        self.drivers_to_plot = [data['lap_info']['Driver'] for data in temp_lap_data_for_sorting][:2] 

        if len(self.drivers_to_plot) < 2:
             print("❌ Error after sorting and filtering: Not enough drivers with valid data.")
             return False

        # Check if final drivers are teammates and adjust colors accordingly
        d1_team = self.driver_data_dict[self.drivers_to_plot[0]]['team_name']
        d2_team = self.driver_data_dict[self.drivers_to_plot[1]]['team_name']
        
        are_teammates = (d1_team != 'N/A' and d2_team != 'N/A' and d1_team == d2_team)
        
        if are_teammates:
            print(f"🤝 Detected teammates: {self.drivers_to_plot[0]} and {self.drivers_to_plot[1]} both from {d1_team}")
            # Apply teammate color differentiation
            base_color = self.get_team_color(self.drivers_to_plot[0], self.session_obj, is_secondary=False)
            
            # P1 gets darker color, P2 gets lighter color
            self.driver_data_dict[self.drivers_to_plot[0]]['color'] = self.adjust_color_luminance(base_color, 0.85)  # 15% darker
            self.driver_data_dict[self.drivers_to_plot[1]]['color'] = self.adjust_color_luminance(base_color, 1.4)   # 40% lighter
        
        # Enhanced console output with tyre information
        for driver_code in self.drivers_to_plot:
            driver_data = self.driver_data_dict[driver_code]
            lap = driver_data['lap_info']
            tyre_info_str = f"{lap['Compound']}"
            if 'TyreLife' in lap and pd.notna(lap['TyreLife']):
                tyre_info_str += f" (Life: {int(lap['TyreLife'])})"
            
            teammate_suffix = " (Teammate colors applied)" if are_teammates else ""
            print(f"🏎️ Loaded {driver_code} ({driver_data['driver_name']}) - {lap['LapTime']} "
                  f"(Lap {lap['LapNumber']}, {tyre_info_str}) - Color: {driver_data['color']} (Team: {driver_data['team_name']}){teammate_suffix}")

        self.p1_code = self.drivers_to_plot[0]
        self.p2_code = self.drivers_to_plot[1]

        d1_data = self.driver_data_dict[self.p1_code]
        d2_data = self.driver_data_dict[self.p2_code]

        self.time_delta_val = d2_data['lap_time_seconds'] - d1_data['lap_time_seconds']
        self.animation_duration = min(d1_data['lap_time_seconds'], d2_data['lap_time_seconds'])
        
        print(f"\n📊 Final Comparison: {self.p1_code} ({d1_data['driver_name']}) vs {self.p2_code} ({d2_data['driver_name']})")
        print(f"   P1 Lap Time ({self.p1_code}): {d1_data['lap_time_td']}")
        print(f"   P2 Lap Time ({self.p2_code}): {d2_data['lap_time_td']}")
        print(f"   Delta (P2 - P1): {self.time_delta_val:+.3f}s")
        print(f"🎬 Animation duration set to fastest lap: {self.animation_duration:.3f}s (Driver {self.p1_code if self.animation_duration == d1_data['lap_time_seconds'] else self.p2_code})")

        # Track coloring based on who is faster at each segment - INITIAL SETUP
        d1_full_tel = self.driver_data_dict[self.p1_code]['telemetry']

        # Store for dynamic update
        self.track_segments_x = d1_full_tel['X'].to_numpy()
        self.track_segments_y = d1_full_tel['Y'].to_numpy()
        
        points = np.array([self.track_segments_x, self.track_segments_y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Initialize segment colors to a neutral grey
        neutral_track_color = mcolors.to_rgba('#404040') # Dark grey
        self.track_segment_colors_current = np.full((len(segments), 4), neutral_track_color)

        # Store driver colors and speeds needed for dynamic update
        self.d1_speeds_orig = d1_full_tel['Speed'].to_numpy()
        self.d1_full_tel_distance = d1_full_tel['Distance'].to_numpy()

        d2_full_tel = self.driver_data_dict[self.p2_code]['telemetry']
        d1_dist_orig = d1_full_tel['Distance'].to_numpy()
        d2_dist_orig = d2_full_tel['Distance'].to_numpy()
        d2_speed_on_d1_dist_func = interp1d(d2_dist_orig, d2_full_tel['Speed'].to_numpy(), kind='linear', fill_value="extrapolate")
        self.d2_speeds_aligned_orig = d2_speed_on_d1_dist_func(self.d1_full_tel_distance)

        self.p1_color_obj = mcolors.to_rgba(self.driver_data_dict[self.p1_code]['color'])
        self.p2_color_obj = mcolors.to_rgba(self.driver_data_dict[self.p2_code]['color'])
        
        # Create the LineCollection object for the track (geometry and initial colors)
        # This will be added to ax_track in _create_plot_layout
        points = np.array([self.track_segments_x, self.track_segments_y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # Ensure track_segment_colors_current is initialized if not already (it should be by now)
        if self.track_segment_colors_current is None or len(self.track_segment_colors_current) != len(segments):
            neutral_track_color = mcolors.to_rgba('#404040') # Dark grey
            self.track_segment_colors_current = np.full((len(segments), 4), neutral_track_color)

        self.lc_track_segments = LineCollection(segments, colors=self.track_segment_colors_current, linewidths=10, zorder=1, capstyle='round')

        return True

    def prepare_cinematic_animation_data(self):
        """Prepares interpolated 60 FPS animation data for smooth visuals."""
        target_fps = 60 # For MP4 and base calculations
        self.animation_total_frames = int(self.animation_duration * target_fps)
        if self.animation_total_frames <=0: # safety for very short/zero duration
            print("⚠️ Animation duration is too short. Setting minimum frames.")
            self.animation_total_frames = target_fps # 1 second min
            self.animation_duration = 1.0
            
        self.animation_time_points = np.linspace(0, self.animation_duration, self.animation_total_frames)
        
        print(f"🎞️ Preparing animation: {self.animation_total_frames} frames at {target_fps} FPS for {self.animation_duration:.3f}s")
        
        self.animation_frame_data = [] 
        
        for driver_idx, driver_code_key in enumerate(self.drivers_to_plot):
            tel_data = self.driver_data_dict[driver_code_key]['telemetry'].copy()
            tel_data['Time'] = (tel_data['Time'] - tel_data['Time'].min()).dt.total_seconds()
            tel_data = tel_data.drop_duplicates(subset=['Time'], keep='first').sort_values('Time')

            driver_frames_dict = {}
            params_to_interpolate = ['X', 'Y', 'Speed', 'Distance', 'Throttle', 'nGear', 'DRS']
            
            for param in params_to_interpolate:
                if param in tel_data.columns and not tel_data[param].empty:
                    if param in ['nGear', 'DRS']:
                        # Step interpolation for discrete values
                        from scipy.interpolate import interp1d
                        interp_func = interp1d(tel_data['Time'], tel_data[param], kind='previous', 
                                             bounds_error=False, fill_value=(tel_data[param].iloc[0], tel_data[param].iloc[-1]))
                        resampled_values = interp_func(self.animation_time_points)
                        
                        if param == 'DRS': 
                            resampled_values = (resampled_values > 8).astype(int)
                        elif param == 'nGear':
                            resampled_values = resampled_values.astype(int)
                    else:
                        # Linear interpolation for continuous values
                        resampled_values = np.interp(self.animation_time_points, tel_data['Time'], tel_data[param])
                    
                    driver_frames_dict[param] = resampled_values
                else: 
                     if param == 'nGear': 
                         driver_frames_dict[param] = np.full(self.animation_total_frames, 8)
                     else:
                         driver_frames_dict[param] = np.zeros(self.animation_total_frames)

            # Handle brake data
            if 'Brake' in tel_data.columns and not tel_data['Brake'].empty:
                brake_values = tel_data['Brake'].astype(float)
                resampled_brake = np.interp(self.animation_time_points, tel_data['Time'], brake_values)
                driver_frames_dict['Brake'] = (resampled_brake > 0).astype(int)
            else:
                driver_frames_dict['Brake'] = np.zeros(self.animation_total_frames, dtype=int)
            
            driver_frames_dict['Time'] = self.animation_time_points
            self.animation_frame_data.append(driver_frames_dict)
        
        if not self.animation_frame_data or len(self.animation_frame_data) < 2:
            print("❌ Failed to prepare animation frame data for both drivers.")
            return False
            
        print("✅ Animation frame data prepared.")
        return True

    def calculate_sector_times(self):
        """Calculates sector times for the two drivers being plotted."""
        self.sector_times_data = {}
        if not hasattr(self, 'driver_data_dict') or not self.driver_data_dict:
            print("⚠️ Driver data not available for sector time calculation.")
            return
        if not hasattr(self, 'p1_code') or not hasattr(self, 'p2_code'):
            print("⚠️ P1/P2 codes not set for sector time calculation.")
            return

        for driver_code_key in [self.p1_code, self.p2_code]:
            if driver_code_key not in self.driver_data_dict:
                print(f"⚠️ Data for driver {driver_code_key} not found in driver_data_dict for sector times.")
                self.sector_times_data[driver_code_key] = {'S1': 0, 'S2': 0, 'S3': 0}
                continue
                
            lap_info = self.driver_data_dict[driver_code_key]['lap_info']
            sectors = {}
            sectors['S1'] = lap_info['Sector1Time'].total_seconds() if pd.notna(lap_info['Sector1Time']) else 0
            sectors['S2'] = lap_info['Sector2Time'].total_seconds() if pd.notna(lap_info['Sector2Time']) else 0
            sectors['S3'] = lap_info['Sector3Time'].total_seconds() if pd.notna(lap_info['Sector3Time']) else 0
            self.sector_times_data[driver_code_key] = sectors

        p1_sectors = self.sector_times_data[self.p1_code]
        p2_sectors = self.sector_times_data[self.p2_code]
        self.sector_deltas_display = {
            'S1': p2_sectors.get('S1',0) - p1_sectors.get('S1',0),
            'S2': p2_sectors.get('S2',0) - p1_sectors.get('S2',0),
            'S3': p2_sectors.get('S3',0) - p1_sectors.get('S3',0)
        }
        
        print(f"\n📊 Sector Analysis ({self.p1_code} vs {self.p2_code}):")
        for sector_label in ['S1', 'S2', 'S3']:
            p1_t = p1_sectors.get(sector_label, 0)
            p2_t = p2_sectors.get(sector_label, 0)
            delta_t = self.sector_deltas_display.get(sector_label,0)
            print(f"   {sector_label}: {self.p1_code} {p1_t:.3f}s | {self.p2_code} {p2_t:.3f}s | Δ{delta_t:+.3f}s")

    def _create_plot_layout(self, for_preview=False):
        # Increase figure size for better mobile viewing
        fig = plt.figure(figsize=(12, 21), facecolor='#000000')
        gs = fig.add_gridspec(7, 1, height_ratios=[0.4, 5.6, 1.2, 1.0, 0.8, 1.0, 0.8], hspace=0.15)
        
        ax_title = fig.add_subplot(gs[0, 0])
        ax_track = fig.add_subplot(gs[1, 0])
        ax_brake_throttle = fig.add_subplot(gs[2, 0])
        ax_gear_speed = fig.add_subplot(gs[3, 0])
        ax_tyre_info = fig.add_subplot(gs[4, 0])
        ax_sectors = fig.add_subplot(gs[5, 0])
        ax_timer = fig.add_subplot(gs[6, 0])
        
        # Add watermark handle in the right corner
        fig.text(0.95, 0.98, '@datadrivenlaps', fontsize=26, color='#FFFFFF', 
                alpha=0.7, ha='right', va='top', fontweight='bold', 
                transform=fig.transFigure, rotation=0)
        
        clean_green = '#00FF00'
        clean_white = '#FFFFFF'
        clean_black = '#000000'
        clean_gray = '#1a1a1a'
        clean_silver = '#CCCCCC'
        border_color = '#333333' # Darker border

        panel_style = {'facecolor': clean_gray, 'edgecolor': border_color, 'linewidth': 0.5, 'alpha': 0.9}
        for ax in [ax_brake_throttle, ax_gear_speed, ax_tyre_info, ax_sectors, ax_timer]:
            ax.set_facecolor(panel_style['facecolor'])
            for spine in ax.spines.values():
                spine.set_color(panel_style['edgecolor'])
                spine.set_linewidth(panel_style['linewidth'])
        ax_track.set_facecolor('#0A0A0A') # Very dark gray for track background

        gp_event_name = self.session_obj.event['OfficialEventName'] if hasattr(self.session_obj.event, 'OfficialEventName') else self.session_obj.event['EventName']
        session_display_name = self.session_obj.name
        
        # Simplify session names
        session_name_mapping = {
            'Practice 1': 'FP1',
            'Practice 2': 'FP2', 
            'Practice 3': 'FP3',
            'Qualifying': 'Qualifying',
            'Sprint Qualifying': 'Sprint Qualifying',
            'Sprint': 'Sprint Race',
            'Race': 'Race'
        }
        simplified_session_name = session_name_mapping.get(session_display_name, session_display_name)
        
        # Increase title font sizes with better spacing
        ax_title.text(0.5, 0.75, gp_event_name.upper(), transform=ax_title.transAxes, fontweight='bold', fontsize=22, color=clean_white, ha='center', va='center')
        ax_title.text(0.5, 0.25, simplified_session_name.upper(), transform=ax_title.transAxes, fontweight='normal', fontsize=18, color=clean_silver, ha='center', va='center')
        ax_title.axis('off')

        # Add the pre-calculated LineCollection for the track (initially grey, dynamically colored)
        if hasattr(self, 'track_segments_x') and self.track_segments_x is not None:
            # Create a new LineCollection for this figure (can't reuse across multiple figures)
            points = np.array([self.track_segments_x, self.track_segments_y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # ALWAYS start with neutral grey for each new figure - track should reveal progressively
            neutral_track_color = mcolors.to_rgba('#404040')  # Dark grey
            neutral_colors = np.full((len(segments), 4), neutral_track_color)
            
            # Create new LineCollection for this figure with neutral colors
            lc_track_for_this_fig = LineCollection(segments, colors=neutral_colors, linewidths=10, zorder=1, capstyle='round')
            ax_track.add_collection(lc_track_for_this_fig)
            
            # Store reference to this figure's LineCollection for updates
            self.lc_track_segments = lc_track_for_this_fig
            # Reset track colors to neutral for this figure
            self.track_segment_colors_current = neutral_colors.copy()
            
            ax_track.set_aspect('equal')
            # Use self.track_segments_x and self.track_segments_y which were populated in prepare_driver_data
            if self.track_segments_x is not None and self.track_segments_y is not None and \
               len(self.track_segments_x) > 0 and len(self.track_segments_y) > 0:
                x_min, x_max = self.track_segments_x.min(), self.track_segments_x.max()
                y_min, y_max = self.track_segments_y.min(), self.track_segments_y.max()
                x_range, y_range = x_max - x_min, y_max - y_min
                margin = max(x_range, y_range) * 0.08 
                ax_track.set_xlim(x_min - margin, x_max + margin)
                ax_track.set_ylim(y_min - margin, y_max + margin)
            else:
                # Fallback if track segment data isn't ready, though it should be
                ax_track.set_xlim(-1, 1)
                ax_track.set_ylim(-1, 1)
            ax_track.axis('off')
        else:
            print("⚠️ Warning: track_segments_x not found or not initialized. Track will not be drawn.")

        d1_data = self.driver_data_dict[self.p1_code]
        d2_data = self.driver_data_dict[self.p2_code]
        d1_color = d1_data['color'] # This is the overall P1 color for their dot, etc.
        d2_color = d2_data['color'] # Overall P2 color for their dot
        car_radius = np.mean([x_range, y_range]) * 0.008 
        self.car1_dot = Circle((0,0), radius=car_radius, facecolor=d1_color, edgecolor='#FFFFFF', linewidth=0.5, zorder=10)
        self.car2_dot = Circle((0,0), radius=car_radius, facecolor=d2_color, edgecolor='#FFFFFF', linewidth=0.5, zorder=10)
        ax_track.add_patch(self.car1_dot)
        ax_track.add_patch(self.car2_dot)
        
        d1_name_short = d1_data['lap_info']['Driver'] 
        d2_name_short = d2_data['lap_info']['Driver']
        
        # Create legend with appropriate labels based on selection mode
        if self.driver_selection_mode == 'Teammates':
            # For teammates, show team name and indicate which is darker/lighter
            team_name = d1_data.get('team_name', 'Team')
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=d1_color, markeredgecolor='#FFFFFF', markeredgewidth=0.5, markersize=10, label=f"{d1_name_short} ({team_name})"),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=d2_color, markeredgecolor='#FFFFFF', markeredgewidth=0.5, markersize=10, label=f"{d2_name_short} ({team_name})")
            ]
        else:
            # For P1/P2 or specific drivers, use P1/P2 designation
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=d1_color, markeredgecolor='#FFFFFF', markeredgewidth=0.5, markersize=10, label=f"P1 {d1_name_short}"),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=d2_color, markeredgecolor='#FFFFFF', markeredgewidth=0.5, markersize=10, label=f"P2 {d2_name_short}")
            ]
        
        ax_track.legend(handles=legend_elements, loc='upper center', ncol=2, facecolor=clean_gray, edgecolor=border_color, labelcolor=clean_white, fontsize=14).get_frame().set_linewidth(0.5)

        bar_positions = [0.8, 1.8, 3.2, 4.2]
        bar_width = 0.6
        brake_color_hex = '#D32F2F' 
        throttle_color_hex = '#4CAF50' 
        # Increase brake/throttle panel font sizes
        ax_brake_throttle.text((bar_positions[0]+bar_positions[1])/2 , 106, d1_name_short, ha='center', va='bottom', color=d1_color, fontsize=14, fontweight='bold')
        ax_brake_throttle.text((bar_positions[2]+bar_positions[3])/2 , 106, d2_name_short, ha='center', va='bottom', color=d2_color, fontsize=14, fontweight='bold')
        self.brake_bars_patches = ax_brake_throttle.bar([bar_positions[0], bar_positions[2]], [0,0], width=bar_width, color=brake_color_hex, edgecolor=clean_white, linewidth=0.5)
        self.throttle_bars_patches = ax_brake_throttle.bar([bar_positions[1], bar_positions[3]], [0,0], width=bar_width, color=throttle_color_hex, edgecolor=clean_white, linewidth=0.5)
        ax_brake_throttle.set_xticks(bar_positions)
        ax_brake_throttle.set_xticklabels(['BRK', 'THR', 'BRK', 'THR'], fontsize=13, color=clean_white)
        ax_brake_throttle.set_ylim(0, 100)
        ax_brake_throttle.set_ylabel('%', fontsize=14, color=clean_white, fontweight='bold')
        ax_brake_throttle.grid(True, axis='y', alpha=0.15, color=clean_silver, linestyle='-', linewidth=0.5)
        ax_brake_throttle.tick_params(colors=clean_white, labelsize=12)

        gear_color_hex = '#FFC107' 
        speed_color_hex = '#03A9F4'
        # Increase gear/speed panel font sizes
        ax_gear_speed.text((bar_positions[0]+bar_positions[1])/2 , 106, d1_name_short, ha='center', va='bottom', color=d1_color, fontsize=14, fontweight='bold')
        ax_gear_speed.text((bar_positions[2]+bar_positions[3])/2 , 106, d2_name_short, ha='center', va='bottom', color=d2_color, fontsize=14, fontweight='bold')
        self.gear_bars_patches = ax_gear_speed.bar([bar_positions[0], bar_positions[2]], [0,0], width=bar_width, color=gear_color_hex, edgecolor=clean_white, linewidth=0.5)
        self.speed_bars_patches = ax_gear_speed.bar([bar_positions[1], bar_positions[3]], [0,0], width=bar_width, color=speed_color_hex, edgecolor=clean_white, linewidth=0.5)
        ax_gear_speed.set_xticks(bar_positions)
        ax_gear_speed.set_xticklabels(['GEAR', 'SPEED', 'GEAR', 'SPEED'], fontsize=13, color=clean_white)
        ax_gear_speed.set_ylim(0, 100) 
        ax_gear_speed.set_ylabel('%', fontsize=14, color=clean_white, fontweight='bold')
        ax_gear_speed.grid(True, axis='y', alpha=0.15, color=clean_silver, linestyle='-', linewidth=0.5)
        ax_gear_speed.tick_params(colors=clean_white, labelsize=12)
        # Increase gear/speed value text sizes and position at base of bars
        self.gear_value_texts = [ax_gear_speed.text(bar_positions[0], 5, "N", ha='center', va='bottom', fontsize=14, fontweight='bold', color=clean_black),
                           ax_gear_speed.text(bar_positions[2], 5, "N", ha='center', va='bottom', fontsize=14, fontweight='bold', color=clean_black)]
        self.speed_value_texts = [ax_gear_speed.text(bar_positions[1], 5, "0", ha='center', va='bottom', fontsize=14, fontweight='bold', color=clean_black),
                            ax_gear_speed.text(bar_positions[3], 5, "0", ha='center', va='bottom', fontsize=14, fontweight='bold', color=clean_black)]

        # Tyre Information Panel
        d1_compound = d1_data['compound'] if d1_data['compound'] else 'UNKNOWN'
        d2_compound = d2_data['compound'] if d2_data['compound'] else 'UNKNOWN'
        d1_tyre_life = d1_data['tyre_life'] if d1_data['tyre_life'] != 'N/A' else 'N/A'
        d2_tyre_life = d2_data['tyre_life'] if d2_data['tyre_life'] != 'N/A' else 'N/A'
        
        # Tyre compound colors (F1 2024/2025 compounds)
        compound_colors = {
            'SOFT': '#FF0000',     # Red
            'MEDIUM': '#FFFF00',   # Yellow  
            'HARD': '#FFFFFF',     # White
            'INTERMEDIATE': '#00FF00', # Green
            'WET': '#0000FF'       # Blue
        }
        
        d1_compound_color = compound_colors.get(d1_compound, clean_silver)
        d2_compound_color = compound_colors.get(d2_compound, clean_silver)
        
        # Tyre panel headers
        ax_tyre_info.text(0.25, 0.85, f"{d1_name_short}", transform=ax_tyre_info.transAxes, fontsize=14, fontweight='bold', color=d1_color, ha='center')
        ax_tyre_info.text(0.75, 0.85, f"{d2_name_short}", transform=ax_tyre_info.transAxes, fontsize=14, fontweight='bold', color=d2_color, ha='center')
        ax_tyre_info.text(0.5, 0.85, "TYRE INFO", transform=ax_tyre_info.transAxes, fontsize=14, fontweight='bold', color=clean_silver, ha='center')
        
        # Compound information
        ax_tyre_info.text(0.25, 0.50, d1_compound, transform=ax_tyre_info.transAxes, fontsize=15, fontweight='bold', color=d1_compound_color, ha='center', 
                         bbox=dict(boxstyle="round,pad=0.3", facecolor=clean_black, edgecolor=d1_compound_color, linewidth=1.5))
        ax_tyre_info.text(0.75, 0.50, d2_compound, transform=ax_tyre_info.transAxes, fontsize=15, fontweight='bold', color=d2_compound_color, ha='center',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor=clean_black, edgecolor=d2_compound_color, linewidth=1.5))
        
        # Tyre life information
        d1_tyre_life_display = f"Tyre Life: {int(d1_tyre_life)}" if d1_tyre_life != 'N/A' else "Tyre Life: N/A"
        d2_tyre_life_display = f"Tyre Life: {int(d2_tyre_life)}" if d2_tyre_life != 'N/A' else "Tyre Life: N/A"
        ax_tyre_info.text(0.25, 0.15, d1_tyre_life_display, transform=ax_tyre_info.transAxes, fontsize=12, color=clean_white, ha='center')
        ax_tyre_info.text(0.75, 0.15, d2_tyre_life_display, transform=ax_tyre_info.transAxes, fontsize=12, color=clean_white, ha='center')
        ax_tyre_info.axis('off')

        # Increase sector panel font sizes
        ax_sectors.text(0.15, 0.82, f"P1 {self.p1_code}", transform=ax_sectors.transAxes, fontsize=14, fontweight='bold', color=d1_color, ha='center')
        ax_sectors.text(0.85, 0.82, f"P2 {self.p2_code}", transform=ax_sectors.transAxes, fontsize=14, fontweight='bold', color=d2_color, ha='center')
        ax_sectors.text(0.5, 0.82, "SECTOR DELTA", transform=ax_sectors.transAxes, fontsize=14, fontweight='bold', color=clean_silver, ha='center')
        for i, sector_label in enumerate(['S1', 'S2', 'S3']):
            y_pos = 0.60 - i * 0.22
            s1_t = self.sector_times_data[self.p1_code].get(sector_label, 0)
            s2_t = self.sector_times_data[self.p2_code].get(sector_label, 0)
            delta_t_val = self.sector_deltas_display.get(sector_label,0)
            ax_sectors.text(0.15, y_pos, f"{s1_t:.3f}", transform=ax_sectors.transAxes, fontsize=14, color=clean_white, ha='center')
            ax_sectors.text(0.85, y_pos, f"{s2_t:.3f}", transform=ax_sectors.transAxes, fontsize=14, color=clean_white, ha='center')
            delta_col_s = clean_white if abs(delta_t_val) < 0.001 else ('#4CAF50' if delta_t_val < 0 else '#D32F2F')
            ax_sectors.text(0.5, y_pos, f"{delta_t_val:+.3f}s", transform=ax_sectors.transAxes, fontsize=14, color=delta_col_s, fontweight='bold', ha='center')
        ax_sectors.axis('off')

        # Increase timer panel font sizes
        self.live_timer_text = ax_timer.text(0.5, 0.75, "00:00.000", transform=ax_timer.transAxes, fontsize=24, fontweight='bold', color=clean_white, ha='center', va='center', family='monospace')
        p1_final_str = str(d1_data['lap_time_td']).split(' ')[-1][:-3]
        p2_final_str = str(d2_data['lap_time_td']).split(' ')[-1][:-3]
        ax_timer.text(0.22, 0.35, f"{self.p1_code} {p1_final_str}", transform=ax_timer.transAxes, fontsize=15, color=d1_color, ha='center', fontweight='bold')
        ax_timer.text(0.78, 0.35, f"{self.p2_code} {p2_final_str}", transform=ax_timer.transAxes, fontsize=15, color=d2_color, ha='center', fontweight='bold')
        delta_col_final_lap = '#4CAF50' if self.time_delta_val < 0 else ('#D32F2F' if self.time_delta_val > 0 else clean_white)
        ax_timer.text(0.5, 0.15, f"Gap: {self.time_delta_val:+.3f}s", transform=ax_timer.transAxes, fontsize=16, fontweight='bold', color=delta_col_final_lap, ha='center', va='center')
        ax_timer.axis('off')

        try:
            plt.tight_layout()
        except UserWarning:
            pass # Can sometimes warn with gridspec, usually fine
        return fig

    def _update_frame_elements(self, frame_idx, current_time_at_frame):
        # Access driver data using the correct structure: animation_frame_data[driver_idx][param][frame_idx]
        d1_anim_data = self.animation_frame_data[0]
        d2_anim_data = self.animation_frame_data[1]

        # Update car positions on track (animation data is numpy arrays, use direct indexing)
        self.car1_dot.set_center((d1_anim_data['X'][frame_idx], d1_anim_data['Y'][frame_idx]))
        self.car2_dot.set_center((d2_anim_data['X'][frame_idx], d2_anim_data['Y'][frame_idx]))

        # REVERTED: Simple track coloring that works (progressive reveal)
        d1_dist_anim = d1_anim_data['Distance'][frame_idx]
        d2_dist_anim = d2_anim_data['Distance'][frame_idx]
        max_dist_covered_anim = max(d1_dist_anim, d2_dist_anim)

        # Update track segment colors based on who is faster at each segment, up to current position
        for i in range(len(self.track_segment_colors_current)):
            if i < len(self.d1_full_tel_distance) - 1:
                segment_end_dist = self.d1_full_tel_distance[i + 1]
                if segment_end_dist <= max_dist_covered_anim:
                    # Color based on who is faster at this segment
                    if self.d1_speeds_orig[i] >= self.d2_speeds_aligned_orig[i]:
                        self.track_segment_colors_current[i] = self.p1_color_obj
                    else:
                        self.track_segment_colors_current[i] = self.p2_color_obj
        
        # Update the LineCollection with new colors
        self.lc_track_segments.set_colors(self.track_segment_colors_current)

        # OPTIMIZED: Pre-compute values to avoid repeated calculations
        d1_brake_height = d1_anim_data['Brake'][frame_idx] * 100
        d1_throttle_height = d1_anim_data['Throttle'][frame_idx] * 100
        d2_brake_height = d2_anim_data['Brake'][frame_idx] * 100
        d2_throttle_height = d2_anim_data['Throttle'][frame_idx] * 100
        
        d1_gear = int(d1_anim_data['nGear'][frame_idx])
        d2_gear = int(d2_anim_data['nGear'][frame_idx])
        d1_speed = d1_anim_data['Speed'][frame_idx]
        d2_speed = d2_anim_data['Speed'][frame_idx]
        
        # Pre-compute gear and speed heights
        max_s_bar = 370.0
        d1_gear_height = (d1_gear / 8.0) * 100 if d1_gear > 0 else 0
        d2_gear_height = (d2_gear / 8.0) * 100 if d2_gear > 0 else 0
        d1_speed_height = min(100, (d1_speed / max_s_bar) * 100)
        d2_speed_height = min(100, (d2_speed / max_s_bar) * 100)
        
        # Update brake and throttle bars
        self.brake_bars_patches[0].set_height(d1_brake_height)
        self.brake_bars_patches[1].set_height(d2_brake_height)
        self.throttle_bars_patches[0].set_height(d1_throttle_height)
        self.throttle_bars_patches[1].set_height(d2_throttle_height)

        # Update gear and speed bars
        self.gear_bars_patches[0].set_height(d1_gear_height)
        self.gear_bars_patches[1].set_height(d2_gear_height)
        self.speed_bars_patches[0].set_height(d1_speed_height)
        self.speed_bars_patches[1].set_height(d2_speed_height)
        
        # Update gear and speed text values (fixed position at base)
        d1_gear_text = f"G{d1_gear if d1_gear > 0 else 'N'}"
        d2_gear_text = f"G{d2_gear if d2_gear > 0 else 'N'}"
        d1_speed_text = f"{d1_speed:.0f}"
        d2_speed_text = f"{d2_speed:.0f}"
        
        self.gear_value_texts[0].set_text(d1_gear_text)
        self.gear_value_texts[1].set_text(d2_gear_text)
        self.speed_value_texts[0].set_text(d1_speed_text)
        self.speed_value_texts[1].set_text(d2_speed_text)
        
        # Update timer display
        minutes = int(current_time_at_frame // 60)
        seconds = current_time_at_frame % 60
        self.live_timer_text.set_text(f"{minutes:02d}:{seconds:06.3f}")
        
        # Return updated artists for blitting
        artists = [
            self.car1_dot, self.car2_dot, self.lc_track_segments,
            self.live_timer_text
        ]
        artists.extend(self.brake_bars_patches)
        artists.extend(self.throttle_bars_patches)
        artists.extend(self.gear_bars_patches)
        artists.extend(self.speed_bars_patches)
        artists.extend(self.gear_value_texts)
        artists.extend(self.speed_value_texts)
        
        return tuple(artists)

    def create_preview_image(self):
        print("\n🖼️ Creating Preview Image...")
        if not self.animation_frame_data or len(self.animation_frame_data) < 2 or 'X' not in self.animation_frame_data[0] or self.animation_frame_data[0]['X'] is None:
            print("❌ Animation data not ready for preview.")
            return None
        fig = self._create_plot_layout(for_preview=True)
        final_anim_frame_idx = len(self.animation_frame_data[0]['X']) -1 
        final_time = self.animation_time_points[final_anim_frame_idx]
        self._update_frame_elements(final_anim_frame_idx, final_time)
        filename_base = f"Preview_{self.year}_{self.gp_name.replace(' ', '_').replace('-','_')}_{self.session_name}_{self.p1_code}_vs_{self.p2_code}"
        outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
        pngs_dir = os.path.join(outputs_dir, 'PNGs')
        if not os.path.exists(pngs_dir): os.makedirs(pngs_dir)
        full_path = os.path.join(pngs_dir, f"{filename_base}.png")
        try:
            fig.savefig(full_path, dpi=200, bbox_inches='tight', facecolor='black') # Lower DPI for faster preview
            plt.close(fig)
            print(f"✅ Preview image saved: {full_path} ({os.path.getsize(full_path)/(1024*1024):.1f} MB)")
            return full_path
        except Exception as e:
            print(f"❌ Error saving preview image: {e}")
            plt.close(fig)
            return None

    def _create_animation(self, output_fps, output_extension, writer_class, writer_options=None):
        if not self.animation_frame_data or len(self.animation_frame_data) < 2 or 'X' not in self.animation_frame_data[0] or self.animation_frame_data[0]['X'] is None:
            print(f"❌ Animation data not ready for {output_extension.upper()}.")
            return None
            
        fig = self._create_plot_layout() 
        static_end_seconds = 3
        static_frames = static_end_seconds * output_fps
        live_duration_output_frames = int(self.animation_duration * output_fps)
        total_output_video_frames = live_duration_output_frames + static_frames

        # Map output frames to the master animation_total_frames (source 60FPS data)
        source_data_indices = np.linspace(0, self.animation_total_frames - 1, live_duration_output_frames, endpoint=True).astype(int)
        last_live_source_data_idx = self.animation_total_frames - 1
        source_data_indices_incl_static = np.concatenate([
            source_data_indices,
            np.full(static_frames, last_live_source_data_idx, dtype=int)
        ])
        
        time_points_for_display_output = np.concatenate([
            np.linspace(0, self.animation_duration, live_duration_output_frames, endpoint=True),
            np.full(static_frames, self.animation_duration)
        ])

        desc = f"{output_extension.upper()} Rendering"
        progress_bar_anim = ProgressBar(total=total_output_video_frames, desc=desc)

        def animate_single_frame(output_frame_idx_iterator):
            current_source_data_idx = source_data_indices_incl_static[output_frame_idx_iterator]
            current_display_time_for_timer = time_points_for_display_output[output_frame_idx_iterator]
            updated_artists_tuple = self._update_frame_elements(current_source_data_idx, current_display_time_for_timer)
            progress_bar_anim.update()
            return updated_artists_tuple

        anim_obj = FuncAnimation(fig, animate_single_frame, frames=total_output_video_frames, 
                                 interval=1000/output_fps, blit=True, repeat=False)

        filename_base = f"{self.year}_{self.gp_name.replace(' ', '_').replace('-','_')}_{self.session_name}_{self.p1_code}_vs_{self.p2_code}"
        outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
        
        # Create appropriate subfolder based on output format
        if output_extension == 'gif':
            subfolder_dir = os.path.join(outputs_dir, 'GIFs')
        elif output_extension == 'mp4':
            subfolder_dir = os.path.join(outputs_dir, 'MP4s')
        else:
            subfolder_dir = outputs_dir  # fallback
            
        if not os.path.exists(subfolder_dir): os.makedirs(subfolder_dir)
        full_path_output = os.path.join(subfolder_dir, f"{filename_base}.{output_extension}")

        try:
            writer_obj = writer_class(fps=output_fps, **(writer_options or {}))
            # Save with tight bbox to match PNG output coverage
            anim_obj.save(full_path_output, writer=writer_obj, 
                         savefig_kwargs={'bbox_inches': 'tight', 'pad_inches': 0.1, 'facecolor': '#000000'})
            progress_bar_anim.close()
            print(f"✅ {output_extension.upper()} saved: {full_path_output} ({os.path.getsize(full_path_output)/(1024*1024):.1f} MB)")
            return full_path_output
        except Exception as e_anim:
            progress_bar_anim.close()
            print(f"❌ {output_extension.upper()} creation failed: {e_anim}")
            if "FFMpegWriter" in str(writer_class) and ("ffmpeg" in str(e_anim).lower() or "not found" in str(e_anim).lower()):
                print("    -> Ensure ffmpeg is installed and in your system's PATH.")
                print("    -> You can typically install it via: 'sudo apt install ffmpeg' (Linux) or 'brew install ffmpeg' (macOS).")
            return None
        finally:
            plt.close(fig)

    def create_gif(self):
        print("\n🎬 Creating GIF...")
        return self._create_animation(output_fps=25, output_extension='gif', writer_class=PillowWriter) # 25 FPS for smaller GIF

    def create_mp4(self):
        print("\n🎥 Creating MP4...")
        return self._create_animation(output_fps=60, output_extension='mp4', writer_class=FFMpegWriter, writer_options={'bitrate': 8000})

if __name__ == "__main__":
    # Only run the CLI version if this script is executed directly
    generator = InteractiveDataDrivenLaps()
    generator.run()
