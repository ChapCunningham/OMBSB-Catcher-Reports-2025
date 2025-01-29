import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define constants
rulebook_left = -0.83083
rulebook_right = 0.83083
rulebook_bottom = 1.5
rulebook_top = 3.3775

# Define expanded strike zone (25% larger)
expand_x = (rulebook_right - rulebook_left) * 0.25
expand_y = (rulebook_top - rulebook_bottom) * 0.25
expanded_left = rulebook_left - expand_x
expanded_right = rulebook_right + expand_x
expanded_bottom = rulebook_bottom - expand_y
expanded_top = rulebook_top + expand_y

# Define 9 even zones inside the original strike zone
x_splits = np.linspace(rulebook_left, rulebook_right, 4)
y_splits = np.linspace(rulebook_bottom, rulebook_top, 4)

# Define shadow zones
shadow_zones = {
    "10": [(expanded_left, rulebook_left), (2.545, rulebook_top)],  # Upper Left Shadow
    "11": [(rulebook_right, expanded_right), (2.545, rulebook_top)],  # Upper Right Shadow
    "12": [(expanded_left, rulebook_left), (expanded_bottom, 2.545)],  # Lower Left Shadow
    "13": [(rulebook_right, expanded_right), (expanded_bottom, 2.545)]  # Lower Right Shadow
}

# Combine strike zones and shadow zones
zones = {}
zone_id = 1
for i in range(3):
    for j in range(3):
        zones[str(zone_id)] = [(x_splits[j], x_splits[j + 1]), (y_splits[i], y_splits[i + 1])]
        zone_id += 1
zones.update(shadow_zones)

# Load datasets
sec_csv_path = "/content/drive/MyDrive/Catching Model OMBSB/SEC_Pitching_pbp.csv"
fawley_csv_path = "/content/drive/MyDrive/CLASS+ (trained with D1 Data)/Spring Intrasquads MASTER.csv"
columns_needed = ['Batter', 'BatterSide', 'Pitcher', 'PitcherThrows',
                  'Catcher', 'PitchCall', 'TaggedPitchType',
                  'PlateLocSide', 'PlateLocHeight', 'Date']
df_sec = pd.read_csv(sec_csv_path, usecols=columns_needed)
df_fawley = pd.read_csv(fawley_csv_path, usecols=columns_needed)

# Combine datasets
df_combined = pd.concat([df_sec, df_fawley])
df_combined['Date'] = pd.to_datetime(df_combined['Date'], errors='coerce')

# Streamlit UI
st.title("Strike Zone Comparison Tool")

# Catcher selection
catcher_options = ["Spring Preseason All"] + list(df_combined['Catcher'].dropna().unique())
selected_catcher = st.selectbox("Select Catcher:", catcher_options)

# Date selection
date_options = df_combined['Date'].dropna().sort_values().unique()
date_range = st.date_input("Select Date Range:", [date_options[0], date_options[-1]])

# Filter data based on selection
filtered_df = df_combined.copy()

if selected_catcher != "Spring Preseason All":
    filtered_df = filtered_df[filtered_df['Catcher'] == selected_catcher]

if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[(filtered_df['Date'] >= pd.Timestamp(start_date)) &
                              (filtered_df['Date'] <= pd.Timestamp(end_date))]

# Function to calculate strike ratios
def calculate_strike_ratios(df):
    strike_ratios = {}
    for zone, ((x_min, x_max), (y_min, y_max)) in zones.items():
        zone_df = df[(df['PlateLocSide'] >= x_min) & (df['PlateLocSide'] < x_max) &
                     (df['PlateLocHeight'] >= y_min) & (df['PlateLocHeight'] < y_max)]

        total_pitches = len(zone_df)
        strikes = len(zone_df[zone_df['PitchCall'] == 'StrikeCalled'])

        strike_ratio = strikes / total_pitches if total_pitches > 0 else 0
        strike_ratios[zone] = strike_ratio
    return strike_ratios

# Calculate strike ratios
strike_ratios = calculate_strike_ratios(filtered_df)

# Plot the strike zone
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-3, 3)
ax.set_ylim(0, 5)

# Draw the strike zone and shadow zones
dx = (rulebook_right - rulebook_left) / 3
dy = (rulebook_top - rulebook_bottom) / 3
for i in range(4):
    ax.plot([rulebook_left + i * dx, rulebook_left + i * dx], [rulebook_bottom, rulebook_top], 'k-', linewidth=1)
    ax.plot([rulebook_left, rulebook_right], [rulebook_bottom + i * dy, rulebook_bottom + i * dy], 'k-', linewidth=1)

ax.plot([expanded_left, expanded_right], [expanded_bottom, expanded_bottom], 'b--', linewidth=2)
ax.plot([expanded_left, expanded_right], [expanded_top, expanded_top], 'b--', linewidth=2)
ax.plot([expanded_left, expanded_left], [expanded_bottom, expanded_top], 'b--', linewidth=2)
ax.plot([expanded_right, expanded_right], [expanded_bottom, expanded_top], 'b--', linewidth=2)

# Label strike ratios
for zone, ((x_min, x_max), (y_min, y_max)) in zones.items():
    text_x = (x_min + x_max) / 2
    text_y = (y_min + y_max) / 2
    ax.text(text_x, text_y, f"{strike_ratios[zone]:.2f}", ha='center', va='center', fontsize=12, color='red')

# Customize plot
title = f"Strike Zone Comparison: {selected_catcher} ({date_range[0]} to {date_range[1]})"
ax.set_title(title)
ax.set_xlabel("Horizontal Location (PlateLocSide)")
ax.set_ylabel("Vertical Location (PlateLocHeight)")

# Show the plot in Streamlit
st.pyplot(fig)
