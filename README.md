# 🏎️ F1 Ghost Racing Streamlit App

A responsive web application for creating F1 ghost racing images using official telemetry data from FastF1.

Example of a full video output: https://www.youtube.com/shorts/U7LXbB6u0EM (current web version is simpliefied due to resource limitations)

## ✨ Features

### 📱 Mobile-First Design
- **Responsive UI**: Seamlessly adapts between mobile and desktop layouts
- **Touch-Friendly**: Large buttons and intuitive mobile controls
- **Session-Based Selection**: Only shows drivers present in the selected session
- **Enhanced Dropdowns**: Driver selection with lap times and performance data

### 🏁 Driver Selection Modes
1. **🎯 Specific Drivers**: Choose any 2 drivers from session participants
   - Sorted by fastest lap time (P1, P2, P3...)
   - Shows driver codes, names, and formatted lap times
   - Mobile & Desktop: Enhanced multiselect dropdown with performance data

2. **👥 Teammates**: Compare drivers from the same team
   - Automatically finds teammates in the session
   - Applies team color differentiation for clarity

3. **🏆 P1 vs P2**: Compare the two fastest drivers
   - Automatically selects fastest lap times from the session
   - Perfect for qualifying and practice session analysis

### 🎨 Visual Enhancements
- **Session-Based Driver Lists**: Only shows drivers present in the selected session
- **Team Color Scheme**: Uses official F1 team colors throughout the interface
- **Racing Car Icons**: 🏎️ Racing-themed emojis instead of generic icons
- **F1-Authentic Styling**: Modern dark theme with F1 red gradients

### 📊 Data Analysis
- **Real-Time Track Coloring**: Track segments colored by speed comparison
- **Professional Time Format**: MM:SS.mmm format (e.g., "1:11.278")
- **Tire Information**: Compound type and tire life data
- **Mobile-Optimized Output**: 9:16 aspect ratio perfect for social media

## 🚀 Quick Start

### Prerequisites
```bash
pip install streamlit fastf1 matplotlib numpy pandas scipy
```

### Running the App
```bash
# Simple launch
streamlit run app.py

# Or use the launcher script
python run_local.py

# Custom port
streamlit run app.py --server.port 8501
```

### Testing
```bash
python test_features.py
```

## 📱 Mobile Experience

The app automatically detects mobile devices and provides:

### Enhanced Driver Selection
- **Performance Data**: Driver dropdowns include lap times and session data
- **Session Filtering**: Only shows drivers who participated in the selected session
- **Touch-Friendly**: Large, easy-to-use dropdown menus
- **Visual Feedback**: Clear driver abbreviations and formatted times

### Responsive Layout
- **Single Column**: All controls in main area for easy scrolling
- **Stacked Metrics**: Mobile-friendly metric display
- **Full-Width Buttons**: Easy thumb access
- **Optimized Spacing**: Comfortable touch targets

## 🖥️ Desktop Experience

Full-featured interface with:
- **Sidebar Controls**: Traditional sidebar layout for settings
- **Two-Column Display**: Metrics and controls side-by-side
- **Keyboard Navigation**: Full keyboard support
- **Advanced Options**: Extended configuration options

## 🔧 Technical Details

### Session Data
- **Live Driver Lists**: Only shows drivers who participated in the selected session
- **Performance Sorting**: Drivers ordered by fastest lap time
- **Real-Time Updates**: Session data cached for performance

### Color System
- **Official Team Colors**: Uses 2024/2025 F1 team color palette
- **Teammate Differentiation**: Automatic color adjustment for same-team drivers
- **Accessibility**: High contrast ratios for readability

### Image Generation
- **High Resolution**: 300 DPI PNG output
- **Mobile Optimized**: 9:16 aspect ratio for social media
- **Fast Generation**: Optimized for quick turnaround
- **Professional Quality**: Broadcasting-ready visual style

## 🌐 Deployment

### Streamlit Community Cloud
1. Fork this repository
2. Connect to Streamlit Community Cloud
3. Deploy directly from GitHub
4. Access your app at: `https://your-app.streamlit.app`

### Local Development
```bash
git clone <repository>
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

## 📊 Supported Sessions
- **Practice Sessions**: FP1, FP2, FP3
- **Qualifying**: Q1, Q2, Q3 (combined as 'Q')
- **Sprint Weekends**: Sprint Qualifying, Sprint Race
- **Race**: Main race sessions
- **Year Range**: 2018-2025 (defaults to 2025)

## 🤝 Contributing

Feel free to:
- Report bugs and issues
- Suggest new features
- Submit pull requests
- Improve mobile responsiveness

## 📄 License

Educational and personal use. Please respect F1 data usage terms.

## 🙏 Acknowledgments

- **FastF1**: Official F1 telemetry data library
- **Streamlit**: Reactive web framework
- **F1 Community**: For inspiration and feedback

---

## 🎯 Perfect For

- 📱 **Social Media**: Mobile-optimized images for Instagram, TikTok
- 📊 **Analysis**: Professional F1 data visualization
- 🏁 **Racing Fans**: Easy-to-use F1 ghost racing creator
- 📈 **Content Creators**: High-quality F1 content generation

**Ready to create amazing F1 content? Start racing! 🏁** 
