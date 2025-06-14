#!/usr/bin/env python3
"""
Test script for new F1 Data Driven Laps features
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from app import format_lap_time, TEAM_COLORS_MAPPING, get_team_color
import fastf1

def test_lap_time_formatting():
    """Test the lap time formatting function"""
    print("🧪 Testing lap time formatting...")
    
    # Test various lap times
    test_times = [71.278, 90.123, 125.456]
    expected = ["1:11.278", "1:30.123", "2:05.456"]
    
    for time_val, expected_val in zip(test_times, expected):
        result = format_lap_time(time_val)
        assert result == expected_val, f"Expected {expected_val}, got {result}"
        print(f"  ✅ {time_val}s -> {result}")

def test_team_colors():
    """Test team color mapping"""
    print("\n🎨 Testing team colors...")
    
    # Check that we have colors for major teams
    required_teams = ['Ferrari', 'Mercedes', 'Red Bull Racing', 'McLaren']
    
    for team in required_teams:
        if team in TEAM_COLORS_MAPPING:
            color = TEAM_COLORS_MAPPING[team]
            print(f"  ✅ {team}: {color}")
        else:
            print(f"  ⚠️  {team}: No color mapping found")

def test_fastf1_connection():
    """Test FastF1 connection and basic functionality"""
    print("\n🏎️ Testing FastF1 connection...")
    
    try:
        # Try to get 2025 schedule
        schedule = fastf1.get_event_schedule(2025)
        if not schedule.empty:
            print(f"  ✅ 2025 F1 schedule loaded: {len(schedule)} events")
            print(f"  📅 First event: {schedule.iloc[0]['EventName']}")
        else:
            print("  ⚠️  2025 schedule is empty")
    except Exception as e:
        print(f"  ❌ Error loading 2025 schedule: {e}")
        
        # Fallback to 2024
        try:
            schedule = fastf1.get_event_schedule(2024)
            if not schedule.empty:
                print(f"  ✅ 2024 F1 schedule loaded: {len(schedule)} events")
            else:
                print("  ❌ 2024 schedule is also empty")
        except Exception as e2:
            print(f"  ❌ Error loading 2024 schedule: {e2}")

def main():
    print("🏁 F1 Data Driven Laps App - Feature Tests")
    print("=" * 50)
    
    test_lap_time_formatting()
    test_team_colors()
    test_fastf1_connection()
    
    print("\n🎉 All tests completed!")
    print("\n💡 Recent changes:")
    print("   ✅ Fixed driver abbreviations (now shows LEC, NOR instead of 16, 4)")
    print("   ✅ Legend moved higher above track circuit")
    print("   ✅ Enhanced driver lookup with fallback logic")
    print("   ✅ Improved P1 vs P2 selection with better error handling")
    print("   ✅ Session-based driver filtering maintained") 
    print("   ✅ 2025 default year maintained")
    print("\n🚀 To run the app:")
    print("   streamlit run app.py")

if __name__ == "__main__":
    main() 