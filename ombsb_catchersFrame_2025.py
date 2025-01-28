import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

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

# Define middle of the strike zone
strike_zone_middle_x = (rulebook_left + rulebook_right) / 2
strike_zone_middle_y = (rulebook_bottom + rulebook_top) / 2

# Define 9 even zones inside the original strike zone
x_splits = np.linspace(rulebook_left, rulebook_right, 4)
y_splits = np.linspace(rulebook_bottom, rulebook_top, 4)

# Define shadow zone boundaries
shadow_zones = {
    "10": [(expanded_left, rulebook_left), (strike_zone_middle_y, rulebook_top)],
    "11": [(rulebook_right, expanded_right), (strike_zone_middle_y, rulebook_top)],
    "12": [(expanded_left, rulebook_left), (expanded_bottom, strike_zone_middle_y)],
    "13": [(rulebook_right, expanded_right), (expanded_bottom, strike_zone_middle_y)]
}

# Define the file paths
sec_csv_path = "SEC_Pitching_pbp_cleaned_for_catchers.csv"
fawley_csv_path = "Spring Intrasquads MASTER.csv"

# Load datasets with only necessary columns
columns_needed = ['Batter', 'BatterSide', 'Pitcher', 'PitcherThrows', 'Catcher', 'PitchCall', 
                  'TaggedPitchType', 'PlateLocSide', 'PlateLocHeight', 'Date']
df_sec = pd.read_csv(sec_csv_path, usecols=columns_needed)
df_fawley = pd.read_csv(fawley_csv_path, usecols=columns_needed)

# Convert Date column to datetime format
df_sec['Date'] = pd.to_datetime(df_sec['Date'])
df_fawley['Date'] = pd.to_datetime(df_fawley['Date'])

# Streamlit App Title
st.title("Strike Zone Comparison: Catcher vs SEC")

# Sidebar for selecting Catcher and Date Range
catcher_list = ['All'] + sorted(df_fawley['Catcher'].unique().tolist())
selected_catcher = st.sidebar.selectbox("Select Catcher", catcher_list, index=0)

date_selection = st.sidebar.radio("Select Date Filter", ["All", "Single Date", "Date Range"])
if date_selection == "Single Date":
    selected_date = st.sidebar.date_input("Select Date")
    df_fawley = df_fawley[df_fawley['Date'] == pd.to_datetime(selected_date)]
elif date_selection == "Date Range":
    date_range = st.sidebar.date_input("Select Date Range", [])
    if len(date_range) == 2:
        df_fawley = df_fawley[(df_fawley['Date'] >= pd.to_datetime(date_range[0])) & 
                              (df_fawley['Date'] <= pd.to_datetime(date_range[1]))]

# Filter dataset based on Catcher selection
if selected_catcher != 'All':
    df_fawley = df_fawley[df_fawley['Catcher'] == selected_catcher]

# Filter the datasets to include only StrikeCalled or BallCalled
df_sec = df_sec[df_sec['PitchCall'].isin(['StrikeCalled', 'BallCalled'])]
df_fawley = df_fawley[df_fawley['PitchCall'].isin(['StrikeCalled', 'BallCalled'])]

# Combine 9 strike zone sections and 4 shadow zones
zones = {}
zone_id = 1
for i in range(3):
    for j in range(3):
        zones[str(zone_id)] = [(x_splits[j], x_splits[j+1]), (y_splits[i], y_splits[i+1])]
        zone_id += 1
zones.update(shadow_zones)

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
sec_strike_ratios = calculate_strike_ratios(df_sec)
fawley_strike_ratios = calculate_strike_ratios(df_fawley)

# Create a two-plot figure
fig, axs = plt.subplots(1, 2, figsize=(14, 7))

for ax in axs:
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 5)
    ax.set_xticks([])
    ax.set_yticks([])

# Left Plot - Strike Zone with Percentages
axs[0].set_title(f"{selected_catcher} Strike Zone Reports")
for x in x_splits:
    axs[0].plot([x, x], [rulebook_bottom, rulebook_top], 'k-', linewidth=1)
for y in y_splits:
    axs[0].plot([rulebook_left, rulebook_right], [y, y], 'k-', linewidth=1)
for zone, ((x_min, x_max), (y_min, y_max)) in zones.items():
    text_x = (x_min + x_max) / 2
    text_y = (y_min + y_max) / 2
    axs[0].text(text_x, text_y, f"{fawley_strike_ratios[zone]:.2f}", ha='center', va='center', fontsize=12, color='red')

# Right Plot - Pitch Scatter Plot + Zone
axs[1].set_title("Pitch Call Breakdown")
for x in x_splits:
    axs[1].plot([x, x], [rulebook_bottom, rulebook_top], 'k-', linewidth=1)
for y in y_splits:
    axs[1].plot([rulebook_left, rulebook_right], [y, y], 'k-', linewidth=1)
for _, row in df_fawley.iterrows():
    x, y = row['PlateLocSide'], row['PlateLocHeight']
    pitch_call = row['PitchCall']
    inside_zone = any((x_min <= x <= x_max and y_min <= y <= y_max) for (x_min, x_max), (y_min, y_max) in zones.values())
    color, marker = ('green', 'o') if pitch_call == "StrikeCalled" else ('red', 's')
    axs[1].scatter(x, y, color=color, marker=marker)

st.pyplot(fig)
