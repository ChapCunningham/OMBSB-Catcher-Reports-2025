import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define constants for the strike zone
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

# File paths
sec_csv_path = "SEC_Pitching_pbp_cleaned_for_catchers.csv"
fawley_csv_path = "Spring Intrasquads MASTER.csv"

# Load datasets
columns_needed = ['Batter', 'BatterSide', 'Pitcher', 'PitcherThrows',
                  'Catcher', 'PitchCall', 'TaggedPitchType',
                  'PlateLocSide', 'PlateLocHeight', 'Date']
df_sec = pd.read_csv(sec_csv_path, usecols=columns_needed)
df_fawley = pd.read_csv(fawley_csv_path, usecols=columns_needed)

# Filter for relevant PitchCalls
df_sec = df_sec[df_sec['PitchCall'].isin(['StrikeCalled', 'BallCalled'])]
df_fawley = df_fawley[df_fawley['PitchCall'].isin(['StrikeCalled', 'BallCalled'])]

# Streamlit UI
st.title("Catcher Strike Zone Comparison")

# Catcher selection
catcher_options = df_fawley['Catcher'].dropna().unique()
selected_catcher = st.selectbox("Select a Catcher:", catcher_options)

# Date selection
date_options = pd.to_datetime(df_fawley['Date']).dropna().unique()
date_range = st.date_input("Select Date Range:", [date_options.min(), date_options.max()])

# Filter data
filtered_fawley = df_fawley[df_fawley['Catcher'] == selected_catcher]
filtered_fawley = filtered_fawley[
    (pd.to_datetime(filtered_fawley['Date']) >= pd.Timestamp(date_range[0])) &
    (pd.to_datetime(filtered_fawley['Date']) <= pd.Timestamp(date_range[1]))
]

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

# Calculate strike ratios for the selected catcher and SEC averages
fawley_strike_ratios = calculate_strike_ratios(filtered_fawley)
sec_strike_ratios = calculate_strike_ratios(df_sec)

# Calculate differences
strike_diff = {zone: fawley_strike_ratios[zone] - sec_strike_ratios[zone] for zone in zones}

# Plot the strike zone comparison
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

ax.plot([expanded_left, rulebook_left], [strike_zone_middle_y, strike_zone_middle_y], 'b--', linewidth=1)
ax.plot([expanded_right, rulebook_right], [strike_zone_middle_y, strike_zone_middle_y], 'b--', linewidth=1)
ax.plot([strike_zone_middle_x, strike_zone_middle_x], [expanded_bottom, rulebook_bottom], 'b--', linewidth=1)
ax.plot([strike_zone_middle_x, strike_zone_middle_x], [rulebook_top, expanded_top], 'b--', linewidth=1)
# Label strike differences
for zone, ((x_min, x_max), (y_min, y_max)) in zones.items():
    text_x = (x_min + x_max) / 2
    text_y = (y_min + y_max) / 2
    ax.text(text_x, text_y, f"{strike_diff[zone]:.2f}", ha='center', va='center', fontsize=12, color='red')

# Customize plot
title = f"Strike Zone Comparison: {selected_catcher} vs SEC Averages\n({date_range[0]} to {date_range[1]})"
ax.set_title(title)
ax.set_xlabel("Horizontal Location (PlateLocSide)")
ax.set_ylabel("Vertical Location (PlateLocHeight)")

# Show the plot in Streamlit
st.pyplot(fig)
