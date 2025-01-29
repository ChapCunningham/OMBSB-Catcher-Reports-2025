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

# Define middle of the strike zone
strike_zone_middle_x = (rulebook_left + rulebook_right) / 2
strike_zone_middle_y = (rulebook_bottom + rulebook_top) / 2

# Define 9 even zones inside the original strike zone
x_splits = np.linspace(rulebook_left, rulebook_right, 4)
y_splits = np.linspace(rulebook_bottom, rulebook_top, 4)
# Define shadow zones
shadow_zones = {
    "10": [(expanded_left, rulebook_left), (strike_zone_middle_y, rulebook_top)],  # Upper Left Shadow
    "11": [(rulebook_right, expanded_right), (strike_zone_middle_y, rulebook_top)],  # Upper Right Shadow
    "12": [(expanded_left, rulebook_left), (expanded_bottom, strike_zone_middle_y)],  # Lower Left Shadow
    "13": [(rulebook_right, expanded_right), (expanded_bottom, strike_zone_middle_y)]  # Lower Right Shadow
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

# Label strike differences with percentage and color coding
for zone, ((x_min, x_max), (y_min, y_max)) in zones.items():
    text_x = (x_min + x_max) / 2
    text_y = (y_min + y_max) / 2
    diff_percentage = strike_diff[zone] * 100  # Convert to percentage
    
    # Determine text color based on the difference
    if -5 <= diff_percentage <= 5:
        color = 'black'
    elif diff_percentage < -5:
        color = 'red'
    else:
        color = 'darkgreen'
    
    # Display the percentage difference in the plot
    ax.text(text_x, text_y, f"{diff_percentage:.1f}%", 
            ha='center', va='center', fontsize=10, color=color)


# Customize plot
title = f"Strike Zone Comparison: {selected_catcher} vs SEC Averages\n({date_range[0]} to {date_range[1]})"
ax.set_title(title)
ax.set_xlabel("Horizontal Location (PlateLocSide)")
ax.set_ylabel("Vertical Location (PlateLocHeight)")

# Show the plot in Streamlit
st.pyplot(fig)





# Prepare table data
table_data = []
for zone, ((x_min, x_max), (y_min, y_max)) in zones.items():
    # Get SEC averages (unchanging)
    sec_avg = sec_strike_ratios[zone] * 100  # Convert to percentage

    # Get selected catcher's data
    fawley_avg = fawley_strike_ratios.get(zone, 0) * 100  # Convert to percentage
    num_pitches = len(filtered_fawley[
        (filtered_fawley['PlateLocSide'] >= x_min) & (filtered_fawley['PlateLocSide'] < x_max) &
        (filtered_fawley['PlateLocHeight'] >= y_min) & (filtered_fawley['PlateLocHeight'] < y_max)
    ])

    # Calculate the difference
    difference = fawley_avg - sec_avg

    # Append to table data
    table_data.append([zone, f"{sec_avg:.1f}%", f"{fawley_avg:.1f}%", num_pitches, f"{difference:.1f}%"])

# Convert to DataFrame for display
table_df = pd.DataFrame(table_data, columns=["Zone", "SEC Avg (%)", f"{selected_catcher} Avg (%)", "Pitches Seen", "Difference (%)"])

# Display in Streamlit
st.write("### Zone Comparison Table")
st.dataframe(table_df, hide_index=True)


import plotly.graph_objects as go
import streamlit as st

# Set x and y limits for all plots
x_limits = [-2.5, 2.5]
y_limits = [0.5, 4.5]

# Define pitch type to marker mapping
pitch_marker_map = {
    "Fastball": "circle",
    "Sinker": "circle",
    "Cutter": "triangle-up",
    "Slider": "triangle-up",
    "Curveball": "triangle-up",
    "Sweeper": "triangle-up",
    "Splitter": "square",
    "ChangeUp": "square"
}

# Function to get marker shape based on pitch type
def get_marker_shape(pitch_type):
    return pitch_marker_map.get(pitch_type, "diamond")  # Default to rhombus (diamond) for "Other"

# Function to calculate Strike% for a given dataset
def calculate_strike_percentage(df):
    if len(df) == 0:
        return 0.0  # Avoid division by zero
    return (len(df[df["PitchCall"] == "StrikeCalled"]) / len(df)) * 100

# Prepare datasets for each plot
strike_pitches_df = filtered_fawley[filtered_fawley["PitchCall"] == "StrikeCalled"]
ball_pitches_df = filtered_fawley[filtered_fawley["PitchCall"] == "BallCalled"]
all_pitches_df = filtered_fawley.copy()
shadow_pitches_df = filtered_fawley[
    ((filtered_fawley["PlateLocSide"] < rulebook_left) | (filtered_fawley["PlateLocSide"] > rulebook_right)) |
    ((filtered_fawley["PlateLocHeight"] < rulebook_bottom) | (filtered_fawley["PlateLocHeight"] > rulebook_top))
]

# Calculate Strike% for each category
strike_percentage_all = calculate_strike_percentage(all_pitches_df)
strike_percentage_shadow = calculate_strike_percentage(shadow_pitches_df)
strike_percentage_strike = 100.0  # Since this plot only contains StrikeCalled pitches
strike_percentage_ball = 0.0  # Since this plot only contains BallCalled pitches

# Function to create a scatter plot with the correctly drawn shadow zones
def create_zone_scatter(title, pitch_df):
    fig = go.Figure()

    # Add scatter plot for pitches with different shapes
    for index, row in pitch_df.iterrows():
        color = "green" if row["PitchCall"] == "StrikeCalled" else "red"
        marker_shape = get_marker_shape(row["TaggedPitchType"])

        fig.add_trace(go.Scatter(
            x=[row["PlateLocSide"]],
            y=[row["PlateLocHeight"]],
            mode="markers",
            marker=dict(symbol=marker_shape, color=color, size=8),
            showlegend=False
        ))

    # Draw main strike zone
    for i in range(4):
        fig.add_shape(type="line", x0=x_splits[i], x1=x_splits[i], y0=rulebook_bottom, y1=rulebook_top, line=dict(color="black", width=1))
        fig.add_shape(type="line", x0=rulebook_left, x1=rulebook_right, y0=y_splits[i], y1=y_splits[i], line=dict(color="black", width=1))

    # Draw expanded strike zone outline (25% larger box)
    fig.add_shape(type="rect", x0=expanded_left, x1=expanded_right, y0=expanded_bottom, y1=expanded_top,
                  line=dict(color="blue", width=2, dash="dash"))

    # Add connecting lines to properly separate shadow zones
    fig.add_shape(type="line", x0=strike_zone_middle_x, x1=strike_zone_middle_x, y0=rulebook_top, y1=expanded_top, 
                  line=dict(color="blue", width=2))  # Top middle connector

    fig.add_shape(type="line", x0=strike_zone_middle_x, x1=strike_zone_middle_x, y0=expanded_bottom, y1=rulebook_bottom, 
                  line=dict(color="blue", width=2))  # Bottom middle connector

    fig.add_shape(type="line", x0=expanded_left, x1=rulebook_left, y0=strike_zone_middle_y, y1=strike_zone_middle_y, 
                  line=dict(color="blue", width=2))  # Left middle connector

    fig.add_shape(type="line", x0=rulebook_right, x1=expanded_right, y0=strike_zone_middle_y, y1=strike_zone_middle_y, 
                  line=dict(color="blue", width=2))  # Right middle connector

    # Add thin black border around the entire plot
    fig.update_layout(
        title=title,
        xaxis=dict(range=x_limits, title="PlateLocSide"),
        yaxis=dict(range=y_limits, title="PlateLocHeight"),
        plot_bgcolor='white',
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=False,
        width=400, height=400
    )

    return fig

# Create individual plots with Strike% in titles
fig1 = create_zone_scatter(f"StrikeCalled Pitches (Strike%: {strike_percentage_strike:.1f}%)", strike_pitches_df)
fig2 = create_zone_scatter(f"BallCalled Pitches (Strike%: {strike_percentage_ball:.1f}%)", ball_pitches_df)
fig3 = create_zone_scatter(f"All Pitches (Strike%: {strike_percentage_all:.1f}%)", all_pitches_df)
fig4 = create_zone_scatter(f"Shadow Zone Pitches (Strike%: {strike_percentage_shadow:.1f}%)", shadow_pitches_df)

# Create the legend for pitch type symbols
legend_fig = go.Figure()

pitch_types = ["Fastball", "Sinker", "Cutter", "Slider", "Curveball", "Sweeper", "Splitter", "ChangeUp", "Other"]
symbols = ["circle", "circle", "triangle-up", "triangle-up", "triangle-up", "triangle-up", "square", "square", "diamond"]

for pitch, symbol in zip(pitch_types, symbols):
    legend_fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(symbol=symbol, color="black", size=10),
        name=pitch if pitch != "Other" else "Rhombus is Other"
    ))

legend_fig.update_layout(
    title="Pitch Type Key",
    showlegend=True,
    width=400, height=200,
    margin=dict(l=50, r=50, t=50, b=50),
    xaxis=dict(visible=False),
    yaxis=dict(visible=False)
)

# Streamlit layout
st.write("### Updated Strike Zone Breakdown with Strike%")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

col1.plotly_chart(fig1, use_container_width=True)  # StrikeCalled Pitches
col2.plotly_chart(fig2, use_container_width=True)  # BallCalled Pitches
col3.plotly_chart(fig3, use_container_width=True)  # All Pitches
col4.plotly_chart(fig4, use_container_width=True)  # Shadow Zone Pitches

# Display pitch type legend
st.plotly_chart(legend_fig, use_container_width=True)


