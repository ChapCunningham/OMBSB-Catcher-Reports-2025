import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Define constants for the strike zone
rulebook_left = -0.83083
rulebook_right = 0.83083
rulebook_bottom = 1.5
rulebook_top = 3.3775

# Define the new shadow zone boundaries
shadow_left = -0.99750
shadow_right = 0.99750
shadow_bottom = 1.377
shadow_top = 3.5

# Define middle of the strike zone
strike_zone_middle_x = (rulebook_left + rulebook_right) / 2
strike_zone_middle_y = (rulebook_bottom + rulebook_top) / 2

# Define 9 even zones inside the original strike zone
x_splits = np.linspace(rulebook_left, rulebook_right, 4)
y_splits = np.linspace(rulebook_bottom, rulebook_top, 4)

# Define shadow zones based on the new dimensions
shadow_zones = {
    "10": [(shadow_left, rulebook_left), (strike_zone_middle_y, shadow_top)],  # Upper Left Shadow
    "11": [(rulebook_right, shadow_right), (strike_zone_middle_y, shadow_top)],  # Upper Right Shadow
    "12": [(shadow_left, rulebook_left), (shadow_bottom, strike_zone_middle_y)],  # Lower Left Shadow
    "13": [(rulebook_right, shadow_right), (shadow_bottom, strike_zone_middle_y)]  # Lower Right Shadow
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

# Function to create a scatter plot with the correctly drawn shadow zones
def create_zone_scatter(title, pitch_df):
    fig = go.Figure()

    # Add scatter plot for pitches
    for index, row in pitch_df.iterrows():
        color = "green" if row["PitchCall"] == "StrikeCalled" else "red"
        fig.add_trace(go.Scatter(
            x=[row["PlateLocSide"]],
            y=[row["PlateLocHeight"]],
            mode="markers",
            marker=dict(color=color, size=8),
            showlegend=False
        ))

    # Draw main strike zone
    for i in range(4):
        fig.add_shape(type="line", x0=x_splits[i], x1=x_splits[i], y0=rulebook_bottom, y1=rulebook_top, line=dict(color="black", width=1))
        fig.add_shape(type="line", x0=rulebook_left, x1=rulebook_right, y0=y_splits[i], y1=y_splits[i], line=dict(color="black", width=1))

    # Draw shadow zones
    for zone, ((x_min, x_max), (y_min, y_max)) in shadow_zones.items():
        fig.add_shape(type="rect", x0=x_min, x1=x_max, y0=y_min, y1=y_max,
                      line=dict(color="blue", width=2, dash="dash"))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis=dict(range=[-2.5, 2.5], title="PlateLocSide"),
        yaxis=dict(range=[0.5, 4.5], title="PlateLocHeight"),
        showlegend=False,
        width=400, height=400
    )

    return fig

# Create individual plots
fig1 = create_zone_scatter("StrikeCalled Pitches", filtered_fawley[filtered_fawley["PitchCall"] == "StrikeCalled"])
fig2 = create_zone_scatter("BallCalled Pitches", filtered_fawley[filtered_fawley["PitchCall"] == "BallCalled"])
fig3 = create_zone_scatter("All Pitches", filtered_fawley)
fig4 = create_zone_scatter("Shadow Zone Pitches", filtered_fawley[
    ((filtered_fawley["PlateLocSide"] < rulebook_left) | (filtered_fawley["PlateLocSide"] > rulebook_right)) |
    ((filtered_fawley["PlateLocHeight"] < rulebook_bottom) | (filtered_fawley["PlateLocHeight"] > rulebook_top))
])

# Streamlit layout
st.write("### Updated Strike Zone Breakdown")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

col1.plotly_chart(fig1, use_container_width=True)  # StrikeCalled Pitches
col2.plotly_chart(fig2, use_container_width=True)  # BallCalled Pitches
col3.plotly_chart(fig3, use_container_width=True)  # All Pitches
col4.plotly_chart(fig4, use_container_width=True)  # Shadow Zone Pitches
