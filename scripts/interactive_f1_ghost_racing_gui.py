import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for GUI
import matplotlib.pyplot as plt

# Import the existing InteractiveDataDrivenLaps class
from interactive_f1_ghost_racing import InteractiveDataDrivenLaps
import fastf1

class F1DataDrivenLapsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🏎️ F1 Data Driven Laps Generator")
        self.root.geometry("1000x900")  # Increased size for better usability
        self.root.configure(bg='#1a1a1a')
        
        # Configure proper window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Configure style for dark theme
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_styles()
        
        # Initialize variables
        self.available_gps = []
        self.available_drivers = []
        self.available_teams = []
        self.session_loaded = False
        self.current_session_obj = None  # Store loaded session for dropdown population
        self.is_generating = False  # Track if generation is in progress
        
        # Create the main interface
        self.create_widgets()
        
        # Initialize the racing generator
        self.racing_generator = InteractiveDataDrivenLaps()
        
    def configure_styles(self):
        """Configure dark theme styles"""
        # Configure colors
        bg_color = '#1a1a1a'
        fg_color = '#ffffff'
        select_color = '#333333'
        button_color = '#0600EF'
        accent_color = '#FF8700'  # McLaren orange for accents
        
        self.style.configure('TFrame', background=bg_color)
        self.style.configure('TLabel', background=bg_color, foreground=fg_color, font=('Arial', 12))
        self.style.configure('Title.TLabel', background=bg_color, foreground=fg_color, font=('Arial', 20, 'bold'))
        self.style.configure('Heading.TLabel', background=bg_color, foreground=accent_color, font=('Arial', 14, 'bold'))
        self.style.configure('TButton', font=('Arial', 11))
        self.style.configure('Generate.TButton', font=('Arial', 14, 'bold'))
        self.style.configure('TCombobox', fieldbackground=select_color, background=select_color, font=('Arial', 11))
        self.style.configure('TEntry', fieldbackground=select_color, font=('Arial', 11))
        self.style.configure('TCheckbutton', background=bg_color, foreground=fg_color, font=('Arial', 11))
        self.style.configure('TRadiobutton', background=bg_color, foreground=fg_color, font=('Arial', 11))
        
    def create_widgets(self):
        """Create the main GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="25")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🏎️ F1 Data Driven Laps Generator", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 30))
        
        row = 1
        
        # Session Selection Section
        session_heading = ttk.Label(main_frame, text="📅 Session Selection", style='Heading.TLabel')
        session_heading.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 15))
        row += 1
        
        # Year selection
        ttk.Label(main_frame, text="Year:").grid(row=row, column=0, sticky=tk.W, pady=8)
        self.year_var = tk.StringVar(value="2025")
        year_entry = ttk.Entry(main_frame, textvariable=self.year_var, width=12)
        year_entry.grid(row=row, column=1, sticky=tk.W, pady=8)
        year_entry.bind('<KeyRelease>', self.on_year_change)
        row += 1
        
        # GP selection
        ttk.Label(main_frame, text="Grand Prix:").grid(row=row, column=0, sticky=tk.W, pady=8)
        self.gp_var = tk.StringVar()
        self.gp_combo = ttk.Combobox(main_frame, textvariable=self.gp_var, width=35)
        self.gp_combo.grid(row=row, column=1, sticky=tk.W, pady=8)
        self.gp_combo.bind('<<ComboboxSelected>>', self.on_gp_change)
        refresh_gp_btn = ttk.Button(main_frame, text="🔄 Refresh GPs", command=self.load_gps)
        refresh_gp_btn.grid(row=row, column=2, padx=(15, 0), pady=8)
        row += 1
        
        # Session selection
        ttk.Label(main_frame, text="Session:").grid(row=row, column=0, sticky=tk.W, pady=8)
        self.session_var = tk.StringVar()
        self.session_combo = ttk.Combobox(main_frame, textvariable=self.session_var, width=25)
        self.session_combo['values'] = ('FP1', 'FP2', 'FP3', 'Q', 'R')
        self.session_combo.set('Q')
        self.session_combo.grid(row=row, column=1, sticky=tk.W, pady=8)
        self.session_combo.bind('<<ComboboxSelected>>', self.on_session_change)
        row += 1
        
        # Separator
        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=20)
        row += 1
        
        # Driver selection mode
        driver_heading = ttk.Label(main_frame, text="🏁 Driver Selection", style='Heading.TLabel')
        driver_heading.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 15))
        row += 1
        
        self.selection_mode = tk.StringVar(value="P1P2")
        
        mode_frame = ttk.Frame(main_frame)
        mode_frame.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=8)
        
        ttk.Radiobutton(mode_frame, text="🥇 P1 vs P2 (Fastest drivers)", variable=self.selection_mode, 
                       value="P1P2", command=self.on_mode_change).grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Radiobutton(mode_frame, text="👥 Specific Drivers", variable=self.selection_mode, 
                       value="SpecificDrivers", command=self.on_mode_change).grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Radiobutton(mode_frame, text="🤝 Teammates", variable=self.selection_mode, 
                       value="Teammates", command=self.on_mode_change).grid(row=2, column=0, sticky=tk.W, pady=5)
        row += 1
        
        # Driver/Team selection (dynamic based on mode)
        self.selection_frame = ttk.Frame(main_frame)
        self.selection_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=15)
        self.selection_frame.columnconfigure(1, weight=1)
        row += 1
        
        # Separator
        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=20)
        row += 1
        
        # Output format selection
        output_heading = ttk.Label(main_frame, text="🎬 Output Formats", style='Heading.TLabel')
        output_heading.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 15))
        row += 1
        
        format_frame = ttk.Frame(main_frame)
        format_frame.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=8)
        
        self.png_var = tk.BooleanVar(value=True)
        self.gif_var = tk.BooleanVar(value=False)
        self.mp4_var = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(format_frame, text="📸 PNG (Preview)", variable=self.png_var).grid(row=0, column=0, sticky=tk.W, padx=(0, 25), pady=3)
        ttk.Checkbutton(format_frame, text="🎞️ GIF (Animated)", variable=self.gif_var).grid(row=0, column=1, sticky=tk.W, padx=(0, 25), pady=3)
        ttk.Checkbutton(format_frame, text="🎥 MP4 (Video)", variable=self.mp4_var).grid(row=0, column=2, sticky=tk.W, pady=3)
        row += 1
        
        # Separator
        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=25)
        row += 1
        
        # Generate button
        self.generate_btn = ttk.Button(main_frame, text="🚀 Generate Data Driven Laps", 
                                      command=self.generate_racing, style='Generate.TButton')
        self.generate_btn.grid(row=row, column=0, columnspan=3, pady=15)
        row += 1
        
        # Progress section
        progress_heading = ttk.Label(main_frame, text="📊 Progress", style='Heading.TLabel')
        progress_heading.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        row += 1
        
        # Progress bar
        self.progress_var = tk.StringVar(value="Ready to generate")
        ttk.Label(main_frame, textvariable=self.progress_var).grid(row=row, column=0, columnspan=3, pady=8)
        row += 1
        
        self.progress_bar = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress_bar.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=8)
        row += 1
        
        # Status/Log area
        log_heading = ttk.Label(main_frame, text="📋 Generation Log", style='Heading.TLabel')
        log_heading.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(15, 10))
        row += 1
        
        log_frame = ttk.Frame(main_frame)
        log_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(row, weight=1)
        
        self.log_text = tk.Text(log_frame, height=10, bg='#2a2a2a', fg='#ffffff', font=('Consolas', 10))
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Initialize
        self.create_selection_widgets()
        self.load_gps()
        
    def create_selection_widgets(self):
        """Create driver/team selection widgets based on current mode"""
        # Clear existing widgets
        for widget in self.selection_frame.winfo_children():
            widget.destroy()
            
        mode = self.selection_mode.get()
        
        if mode == "P1P2":
            info_label = ttk.Label(self.selection_frame, text="🏆 Will automatically select the two fastest drivers from the session")
            info_label.grid(row=0, column=0, columnspan=2, sticky=tk.W)
            
        elif mode == "SpecificDrivers":
            ttk.Label(self.selection_frame, text="Driver 1:").grid(row=0, column=0, sticky=tk.W, pady=5)
            self.driver1_var = tk.StringVar()
            self.driver1_combo = ttk.Combobox(self.selection_frame, textvariable=self.driver1_var, width=20)
            self.driver1_combo.grid(row=0, column=1, sticky=tk.W, pady=5, padx=(15, 0))
            
            ttk.Label(self.selection_frame, text="Driver 2:").grid(row=1, column=0, sticky=tk.W, pady=5)
            self.driver2_var = tk.StringVar()
            self.driver2_combo = ttk.Combobox(self.selection_frame, textvariable=self.driver2_var, width=20)
            self.driver2_combo.grid(row=1, column=1, sticky=tk.W, pady=5, padx=(15, 0))
            
            # Load button for driver data
            load_drivers_btn = ttk.Button(self.selection_frame, text="📊 Load Drivers", command=self.load_drivers_for_selection)
            load_drivers_btn.grid(row=0, column=2, padx=(15, 0), pady=5)
            
            # Status label
            self.driver_status_label = ttk.Label(self.selection_frame, text="Click 'Load Drivers' to populate the lists")
            self.driver_status_label.grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=5)
            
            # Update driver lists if session is already loaded
            if self.session_loaded and self.available_drivers:
                self.update_driver_lists()
                
        elif mode == "Teammates":
            ttk.Label(self.selection_frame, text="Team:").grid(row=0, column=0, sticky=tk.W, pady=5)
            self.team_var = tk.StringVar()
            self.team_combo = ttk.Combobox(self.selection_frame, textvariable=self.team_var, width=30)
            self.team_combo.grid(row=0, column=1, sticky=tk.W, pady=5, padx=(15, 0))
            
            # Load button for team data
            load_teams_btn = ttk.Button(self.selection_frame, text="🏁 Load Teams", command=self.load_teams_for_selection)
            load_teams_btn.grid(row=0, column=2, padx=(15, 0), pady=5)
            
            # Status label
            self.team_status_label = ttk.Label(self.selection_frame, text="Click 'Load Teams' to populate the list")
            self.team_status_label.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=5)
            
            # Update team list if session is already loaded
            if self.session_loaded and self.available_teams:
                self.update_team_list()
    
    def log_message(self, message):
        """Add a message to the log area"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def on_year_change(self, event=None):
        """Handle year change"""
        self.load_gps()
    
    def on_gp_change(self, event=None):
        """Handle GP change"""
        self.session_loaded = False
        self.create_selection_widgets()
        # Load available sessions for this specific GP
        self.load_available_sessions()
    
    def on_session_change(self, event=None):
        """Handle session change"""
        self.session_loaded = False
        self.create_selection_widgets()
    
    def on_mode_change(self):
        """Handle selection mode change"""
        self.create_selection_widgets()
    
    def load_gps(self):
        """Load available GPs for the selected year"""
        try:
            year = int(self.year_var.get())
            self.log_message(f"Loading GPs for {year}...")
            
            # Load in a separate thread to avoid blocking UI
            def load_gps_thread():
                try:
                    schedule = fastf1.get_event_schedule(year, include_testing=False)
                    if not schedule.empty:
                        gp_names = schedule['EventName'].tolist()
                        self.root.after(0, lambda: self.update_gp_list(gp_names))
                    else:
                        self.root.after(0, lambda: self.log_message(f"No GPs found for {year}"))
                except Exception as e:
                    self.root.after(0, lambda: self.log_message(f"Error loading GPs: {e}"))
            
            threading.Thread(target=load_gps_thread, daemon=True).start()
            
        except ValueError:
            self.log_message("Invalid year format")
    
    def update_gp_list(self, gp_names):
        """Update the GP combobox with loaded GPs"""
        self.gp_combo['values'] = gp_names
        if gp_names:
            self.log_message(f"Loaded {len(gp_names)} GPs")
            if "Monaco" in gp_names:
                self.gp_var.set("Monaco")
            else:
                self.gp_var.set(gp_names[0])
    
    def load_session_data(self):
        """Load session data for driver/team lists"""
        if self.session_loaded:
            return True
            
        try:
            year = int(self.year_var.get())
            gp = self.gp_var.get()
            session = self.session_var.get()
            
            if not gp:
                self.log_message("Please select a Grand Prix")
                return False
            
            self.log_message(f"Loading session data: {year} {gp} {session}...")
            
            # Load session data
            session_obj = fastf1.get_session(year, gp, session)
            session_obj.load(telemetry=False, weather=False, messages=False)
            
            if session_obj.laps is None or session_obj.laps.empty:
                self.log_message("No lap data available for this session")
                return False
            
            # Extract driver and team information
            self.available_drivers = sorted(session_obj.laps['Driver'].unique())
            
            # Get teams from session driver info
            teams = set()
            for driver_num in session_obj.drivers:
                driver_info = session_obj.get_driver(driver_num)
                if driver_info is not None and not driver_info.empty:
                    team_name = driver_info.get('TeamName', '')
                    if team_name:
                        teams.add(team_name)
            
            self.available_teams = sorted(list(teams))
            self.session_loaded = True
            
            self.log_message(f"Loaded session data: {len(self.available_drivers)} drivers, {len(self.available_teams)} teams")
            
            # Update the selection widgets
            if self.selection_mode.get() == "SpecificDrivers":
                self.update_driver_lists()
            elif self.selection_mode.get() == "Teammates":
                self.update_team_list()
            
            return True
            
        except Exception as e:
            self.log_message(f"Error loading session data: {e}")
            return False
    
    def update_driver_lists(self):
        """Update driver comboboxes"""
        if hasattr(self, 'driver1_combo') and hasattr(self, 'driver2_combo'):
            self.driver1_combo['values'] = self.available_drivers
            self.driver2_combo['values'] = self.available_drivers
            
            # Set default values with some popular combinations
            if len(self.available_drivers) >= 2:
                # Try to set some interesting default combinations
                default_drivers = ['VER', 'HAM', 'LEC', 'NOR', 'PIA', 'RUS', 'SAI', 'PER']
                available_set = set(self.available_drivers)
                
                driver1_set = False
                driver2_set = False
                
                for driver in default_drivers:
                    if driver in available_set:
                        if not driver1_set:
                            self.driver1_var.set(driver)
                            driver1_set = True
                        elif not driver2_set and driver != self.driver1_var.get():
                            self.driver2_var.set(driver)
                            driver2_set = True
                            break
                
                # Fallback to first two drivers if defaults not found
                if not driver1_set:
                    self.driver1_var.set(self.available_drivers[0])
                if not driver2_set:
                    self.driver2_var.set(self.available_drivers[1] if len(self.available_drivers) > 1 else self.available_drivers[0])
            
            if hasattr(self, 'driver_status_label'):
                self.driver_status_label.config(text=f"✅ {len(self.available_drivers)} drivers available")
    
    def update_team_list(self):
        """Update team combobox"""
        if hasattr(self, 'team_combo'):
            self.team_combo['values'] = self.available_teams
            
            # Set default to popular teams
            priority_teams = ["Ferrari", "Red Bull Racing", "Mercedes", "McLaren", "Aston Martin"]
            
            for team in priority_teams:
                if team in self.available_teams:
                    self.team_var.set(team)
                    break
            else:
                # Fallback to first team if priority teams not found
                if self.available_teams:
                    self.team_var.set(self.available_teams[0])
            
            if hasattr(self, 'team_status_label'):
                self.team_status_label.config(text=f"✅ {len(self.available_teams)} teams available")
    
    def load_drivers_for_selection(self):
        """Load drivers specifically for the driver selection dropdown"""
        if not self.gp_var.get() or not self.session_var.get():
            messagebox.showwarning("Missing Information", "Please select a Grand Prix and Session first")
            return
            
        self.driver_status_label.config(text="⏳ Loading drivers...")
        self.root.update_idletasks()
        
        def load_thread():
            success = self.load_session_data()
            if success:
                self.root.after(0, lambda: self.update_driver_lists())
                self.root.after(0, lambda: self.driver_status_label.config(text=f"✅ Loaded {len(self.available_drivers)} drivers"))
            else:
                self.root.after(0, lambda: self.driver_status_label.config(text="❌ Failed to load drivers"))
        
        threading.Thread(target=load_thread, daemon=True).start()
    
    def load_teams_for_selection(self):
        """Load teams specifically for the team selection dropdown"""
        if not self.gp_var.get() or not self.session_var.get():
            messagebox.showwarning("Missing Information", "Please select a Grand Prix and Session first")
            return
            
        self.team_status_label.config(text="⏳ Loading teams...")
        self.root.update_idletasks()
        
        def load_thread():
            success = self.load_session_data()
            if success:
                self.root.after(0, lambda: self.update_team_list())
                self.root.after(0, lambda: self.team_status_label.config(text=f"✅ Loaded {len(self.available_teams)} teams"))
            else:
                self.root.after(0, lambda: self.team_status_label.config(text="❌ Failed to load teams"))
        
        threading.Thread(target=load_thread, daemon=True).start()
    
    def validate_inputs(self):
        """Validate all inputs before generation"""
        # Check basic inputs
        try:
            year = int(self.year_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid year")
            return False
        
        if not self.gp_var.get():
            messagebox.showerror("Invalid Input", "Please select a Grand Prix")
            return False
        
        if not self.session_var.get():
            messagebox.showerror("Invalid Input", "Please select a session")
            return False
        
        # Check output formats
        if not (self.png_var.get() or self.gif_var.get() or self.mp4_var.get()):
            messagebox.showerror("Invalid Input", "Please select at least one output format")
            return False
        
        # Check driver/team selection based on mode
        mode = self.selection_mode.get()
        
        if mode == "SpecificDrivers":
            if not self.driver1_var.get() or not self.driver2_var.get():
                messagebox.showerror("Invalid Input", "Please select both drivers")
                return False
            if self.driver1_var.get() == self.driver2_var.get():
                messagebox.showerror("Invalid Input", "Please select two different drivers")
                return False
        
        elif mode == "Teammates":
            if not self.team_var.get():
                messagebox.showerror("Invalid Input", "Please select a team")
                return False
        
        return True
    
    def generate_racing(self):
        """Generate the ghost racing visualization"""
        if not self.validate_inputs():
            return
        
        # Load session data if needed
        if self.selection_mode.get() in ["SpecificDrivers", "Teammates"] and not self.session_loaded:
            if not self.load_session_data():
                return
        
        # Set generation state
        self.is_generating = True
        
        # Disable generate button and start progress
        self.generate_btn.config(state='disabled')
        self.progress_bar.start()
        self.progress_var.set("Generating...")
        
        # Clear log
        self.log_text.delete(1.0, tk.END)
        
        # Run generation in separate thread
        def generate_thread():
            try:
                # Set up the racing generator
                self.racing_generator.year = int(self.year_var.get())
                self.racing_generator.gp_name = self.gp_var.get()
                self.racing_generator.session_name = self.session_var.get()
                
                # Set driver/team selection
                if self.selection_mode.get() == "SpecificDrivers":
                    self.racing_generator.drivers_to_plot = [self.driver1_var.get(), self.driver2_var.get()]
                    self.racing_generator.driver_selection_mode = 'SpecificDrivers'
                elif self.selection_mode.get() == "Teammates":
                    self.racing_generator.team_to_plot = self.team_var.get()
                    self.racing_generator.driver_selection_mode = 'Teammates'
                else:  # P1P2 mode
                    self.racing_generator.driver_selection_mode = 'P1P2'
                
                # Set output formats
                formats = []
                if self.png_var.get():
                    formats.append('png')
                if self.gif_var.get():
                    formats.append('gif')
                if self.mp4_var.get():
                    formats.append('mp4')
                self.racing_generator.output_formats = formats
                
                # Redirect output to GUI
                original_print = print
                def gui_print(*args, **kwargs):
                    message = ' '.join(str(arg) for arg in args)
                    self.root.after(0, lambda m=message: self.log_message(m))
                
                # Temporarily replace print
                import builtins
                builtins.print = gui_print
                
                try:
                    # Load session data
                    if not self.racing_generator.load_session_data():
                        raise Exception("Failed to load session data")
                    
                    # Prepare driver data
                    if not self.racing_generator.prepare_driver_data():
                        raise Exception("Failed to prepare driver data")
                    
                    # Prepare animation data
                    self.racing_generator.prepare_cinematic_animation_data()
                    self.racing_generator.calculate_sector_times()
                    
                    # Generate outputs
                    if 'png' in formats:
                        self.racing_generator.create_preview_image()
                    if 'gif' in formats:
                        self.racing_generator.create_gif()
                    if 'mp4' in formats:
                        self.racing_generator.create_mp4()
                    
                    self.root.after(0, lambda: self.generation_complete(True))
                    
                finally:
                    # Restore original print
                    builtins.print = original_print
                    
            except Exception as e:
                self.root.after(0, lambda: self.generation_complete(False, str(e)))
        
        threading.Thread(target=generate_thread, daemon=True).start()
    
    def generation_complete(self, success, error_msg=None):
        """Handle generation completion"""
        self.is_generating = False
        self.generate_btn.config(state='normal')
        self.progress_bar.stop()
        
        # Clean up matplotlib resources
        plt.close('all')
        
        if success:
            self.progress_var.set("Generation completed successfully!")
            self.log_message("\n🎉 All requested outputs generated successfully!")
            
            # Ask if user wants to open output folder
            if messagebox.askyesno("Generation Complete", "Generation completed successfully!\n\nWould you like to open the output folder?"):
                self.open_output_folder()
        else:
            self.progress_var.set("Generation failed")
            self.log_message(f"\n❌ Generation failed: {error_msg}")
            messagebox.showerror("Generation Failed", f"Generation failed:\n{error_msg}")
    
    def open_output_folder(self):
        """Open the output folder"""
        try:
            outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
            if os.path.exists(outputs_dir):
                if sys.platform == "win32":
                    os.startfile(outputs_dir)
                elif sys.platform == "darwin":
                    os.system(f"open '{outputs_dir}'")
                else:
                    os.system(f"xdg-open '{outputs_dir}'")
            else:
                messagebox.showwarning("Folder Not Found", "Output folder not found")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open output folder: {e}")
    
    def load_available_sessions(self):
        """Load available sessions for the selected GP"""
        if not self.gp_var.get() or not self.year_var.get():
            return
            
        try:
            year = int(self.year_var.get())
            gp = self.gp_var.get()
            
            self.log_message(f"Loading available sessions for {gp} {year}...")
            
            def load_sessions_thread():
                try:
                    # Get event schedule to find this specific event
                    schedule = fastf1.get_event_schedule(year, include_testing=False)
                    event_info = schedule[schedule['EventName'] == gp]
                    
                    if event_info.empty:
                        self.root.after(0, lambda: self.log_message(f"Could not find event info for {gp}"))
                        return
                    
                    # Try to load each possible session type to see what's available
                    possible_sessions = ['FP1', 'FP2', 'FP3', 'SQ', 'Q', 'SR', 'R']
                    available_sessions = []
                    
                    for session_type in possible_sessions:
                        try:
                            session = fastf1.get_session(year, gp, session_type)
                            # Just check if the session exists without loading full data
                            if session is not None:
                                available_sessions.append(session_type)
                        except:
                            # Session doesn't exist, skip it
                            continue
                    
                    # Update the session dropdown in the main thread
                    self.root.after(0, lambda: self.update_session_dropdown(available_sessions))
                    self.root.after(0, lambda: self.log_message(f"Available sessions for {gp}: {', '.join(available_sessions)}"))
                    
                except Exception as e:
                    self.root.after(0, lambda: self.log_message(f"Error loading sessions: {e}"))
            
            threading.Thread(target=load_sessions_thread, daemon=True).start()
            
        except ValueError:
            self.log_message("Invalid year format")
    
    def update_session_dropdown(self, available_sessions):
        """Update the session dropdown with only available sessions"""
        if available_sessions:
            self.session_combo['values'] = available_sessions
            
            # Set a smart default based on what's available
            if 'Q' in available_sessions:
                self.session_combo.set('Q')  # Qualifying is usually most interesting
            elif 'R' in available_sessions:
                self.session_combo.set('R')  # Race as second choice
            elif 'SQ' in available_sessions:
                self.session_combo.set('SQ')  # Sprint Qualifying for sprint weekends
            elif available_sessions:
                self.session_combo.set(available_sessions[0])  # Any available session
        else:
            # Fallback to default sessions if detection fails
            self.session_combo['values'] = ('FP1', 'FP2', 'FP3', 'Q', 'R')
            self.session_combo.set('Q')

    def on_closing(self):
        """Handle window closing"""
        if self.is_generating:
            if messagebox.askokcancel("Quit", "Generation is in progress. Do you want to quit anyway?"):
                # Force cleanup
                plt.close('all')
                self.root.quit()
                self.root.destroy()
        else:
            # Clean shutdown
            plt.close('all')
            self.root.quit()
            self.root.destroy()

def main():
    # Ensure proper cleanup on exit
    import atexit
    atexit.register(lambda: plt.close('all'))
    
    try:
        root = tk.Tk()
        app = F1DataDrivenLapsGUI(root)
        
        # Start the GUI main loop
        root.mainloop()
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Application error: {e}")
    finally:
        # Ensure matplotlib cleanup
        plt.close('all')
        # Force exit to prevent hanging
        os._exit(0)

if __name__ == "__main__":
    main() 