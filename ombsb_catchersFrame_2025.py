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

# Load CSVs
sec_csv_path = "SEC_Pitching_pbp_cleaned_for_catchers.csv"
fawley_csv_path = "Spring Intrasquads MASTER.csv"
df = "Spring Intrasquads MASTER.csv"

# Load datasets
columns_needed = ['Batter', 'BatterSide', 'Pitcher', 'PitcherThrows',
                  'Catcher', 'PitchCall', 'TaggedPitchType',
                  'PlateLocSide', 'PlateLocHeight', 'Date']

rebs_columns_needed = ['Batter', 'BatterSide', 'Pitcher', 'PitcherThrows',
                  'Catcher', 'PitchCall', 'TaggedPitchType',
                  'PlateLocSide', 'PlateLocHeight', 'Date','Inning','Balls','Strikes']
df_sec = pd.read_csv(sec_csv_path, usecols=columns_needed)
df_fawley = pd.read_csv(fawley_csv_path, usecols=rebs_columns_needed)
df = pd.read_csv(fawley_csv_path, usecols = rebs_columns_needed)

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

# Define pitch categories mapping
pitch_categories = {
    "All Pitches" : ["Curveball", "Cutter", "Slider", "Sweeper","Fastball", "Sinker","ChangeUp", "Splitter"],
    "Fast/Sink": ["Fastball", "Sinker"],
    "Breaking Ball": ["Curveball", "Cutter", "Slider", "Sweeper"],
    "Change/Split": ["ChangeUp", "Splitter"]
}

# Dropdown filter for pitch category
selected_pitch_category = st.selectbox("Select a Pitch Type Category:", options=pitch_categories.keys())

# Filter data based on selected pitch category
if selected_pitch_category:
    valid_pitch_types = pitch_categories[selected_pitch_category]
    filtered_fawley = filtered_fawley[filtered_fawley['TaggedPitchType'].isin(valid_pitch_types)]


# Prepare datasets for each plot
strike_pitches_df = filtered_fawley[filtered_fawley["PitchCall"] == "StrikeCalled"]
ball_pitches_df = filtered_fawley[filtered_fawley["PitchCall"] == "BallCalled"]
all_pitches_df = filtered_fawley.copy()
shadow_pitches_df = filtered_fawley[
    ((filtered_fawley["PlateLocSide"] < rulebook_left) | (filtered_fawley["PlateLocSide"] > rulebook_right)) |
    ((filtered_fawley["PlateLocHeight"] < rulebook_bottom) | (filtered_fawley["PlateLocHeight"] > rulebook_top))
]

# Function to calculate Strike% for a given dataset
def calculate_strike_percentage(df):
    if len(df) == 0:
        return 0.0  # Avoid division by zero
    return (len(df[df["PitchCall"] == "StrikeCalled"]) / len(df)) * 100

# Compute Strike% BEFORE using them in titles
strike_percentage_strike = 100.0  # Since this plot only contains StrikeCalled pitches
strike_percentage_ball = 0.0  # Since this plot only contains BallCalled pitches
strike_percentage_all = calculate_strike_percentage(all_pitches_df)
strike_percentage_shadow = calculate_strike_percentage(shadow_pitches_df)

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

def create_zone_scatter(title, pitch_df):
    fig = go.Figure()

    # Add scatter plot for pitches with hover tooltips
    for index, row in pitch_df.iterrows():
        color = "green" if row["PitchCall"] == "StrikeCalled" else "red"
        marker_shape = get_marker_shape(row["TaggedPitchType"])

        fig.add_trace(go.Scatter(
            x=[row["PlateLocSide"]],
            y=[row["PlateLocHeight"]],
            mode="markers",
            marker=dict(symbol=marker_shape, color=color, size=8, line=dict(color='black',width = 1.5)),
            showlegend=False,
            hoverinfo="text",
            text=f"Inning: {row['Inning']}<br>"
                 f"Balls: {row['Balls']}<br>"
                 f"Strikes: {row['Strikes']}<br>"
                 f"Pitcher: {row['Pitcher']}<br>"
                 f"Pitch Type: {row['TaggedPitchType']}<br>"
                 f"Batter: {row['Batter']}<br>"
                 f"BatterSide: {row['BatterSide']}"
        ))

    # Draw main strike zone
    for i in range(4):
        fig.add_shape(type="line", x0=x_splits[i], x1=x_splits[i], y0=rulebook_bottom, y1=rulebook_top, line=dict(color="black", width=1))
        fig.add_shape(type="line", x0=rulebook_left, x1=rulebook_right, y0=y_splits[i], y1=y_splits[i], line=dict(color="black", width=1))

    # Draw shadow zone outlines (ensuring full enclosure)
    fig.add_shape(type="rect", x0=shadow_left, x1=shadow_right, y0=shadow_bottom, y1=shadow_top,
                  line=dict(color="blue", width=2, dash="dash"))

    # Ensure Horizontal and Vertical Dashed Lines Stop at the Strike Zone Edge
    fig.add_shape(type="line", x0=strike_zone_middle_x, x1=strike_zone_middle_x, y0=shadow_bottom, y1=rulebook_bottom,
                  line=dict(color="blue", width=2, dash="dash"))  # Bottom vertical line stops at strike zone
    fig.add_shape(type="line", x0=strike_zone_middle_x, x1=strike_zone_middle_x, y0=rulebook_top, y1=shadow_top,
                  line=dict(color="blue", width=2, dash="dash"))  # Top vertical line stops at strike zone

    fig.add_shape(type="line", x0=shadow_left, x1=rulebook_left, y0=strike_zone_middle_y, y1=strike_zone_middle_y,
                  line=dict(color="blue", width=2, dash="dash"))  # Left horizontal stops at strike zone
    fig.add_shape(type="line", x0=rulebook_right, x1=shadow_right, y0=strike_zone_middle_y, y1=strike_zone_middle_y,
                  line=dict(color="blue", width=2, dash="dash"))  # Right horizontal stops at strike zone

    # Ensure Full Enclosure with Horizontal Lines
    fig.add_shape(type="line", x0=shadow_left, x1=shadow_right, y0=shadow_top, y1=shadow_top,
                  line=dict(color="blue", width=2, dash="dash"))  # Top boundary of shadow zone
    fig.add_shape(type="line", x0=shadow_left, x1=shadow_right, y0=shadow_bottom, y1=shadow_bottom,
                  line=dict(color="blue", width=2, dash="dash"))  # Bottom boundary of shadow zone

    # Update layout
    fig.update_layout(
        title=title,
        xaxis=dict(range=[-2.5, 2.5], title="PlateLocSide"),
        yaxis=dict(range=[0.5, 4.5], title="PlateLocHeight"),
        showlegend=False,
        width=400, height=400
    )

    return fig


# Create individual plots with correct Strike% values in titles
fig1 = create_zone_scatter(f"StrikeCalled Pitches (Strike%: {strike_percentage_strike:.1f}%)", strike_pitches_df)
fig2 = create_zone_scatter(f"BallCalled Pitches (Strike%: {strike_percentage_ball:.1f}%)", ball_pitches_df)
fig3 = create_zone_scatter(f"All Pitches (Strike%: {strike_percentage_all:.1f}%)", all_pitches_df)
fig4 = create_zone_scatter(f"Shadow Zone Pitches (Strike%: {strike_percentage_shadow:.1f}%)", shadow_pitches_df)

# Streamlit layout
st.write(f"### {selected_catcher} Framing Breakdown:")


col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

col1.plotly_chart(fig1, use_container_width=True)
col2.plotly_chart(fig2, use_container_width=True)
col3.plotly_chart(fig3, use_container_width=True)
col4.plotly_chart(fig4, use_container_width=True)



# Function to calculate framing metrics
# Function to calculate framing metrics based on user selection
def calculate_framing_metrics(df):
    # Balls Called Strikes (Pitches outside the rulebook zone but called strikes)
    balls_called_strikes = df[
        ((df['PlateLocSide'] < rulebook_left) | (df['PlateLocSide'] > rulebook_right) |
         (df['PlateLocHeight'] < rulebook_bottom) | (df['PlateLocHeight'] > rulebook_top))
        & (df['PitchCall'] == 'StrikeCalled')
    ].shape[0]

    # Strikes Called Balls (Pitches inside the rulebook zone but called balls)
    strikes_called_balls = df[
        ((df['PlateLocSide'] >= rulebook_left) & (df['PlateLocSide'] <= rulebook_right) &
         (df['PlateLocHeight'] >= rulebook_bottom) & (df['PlateLocHeight'] <= rulebook_top))
        & (df['PitchCall'] == 'BallCalled')
    ].shape[0]

    # 50/50 Pitches: Between rulebook and shadow zone
    shadow_pitches_df = filtered_fawley[
    (((filtered_fawley["PlateLocSide"] >= shadow_left) & (filtered_fawley["PlateLocSide"] < rulebook_left)) |  
     ((filtered_fawley["PlateLocSide"] > rulebook_right) & (filtered_fawley["PlateLocSide"] <= shadow_right))) &  
    (((filtered_fawley["PlateLocHeight"] >= shadow_bottom) & (filtered_fawley["PlateLocHeight"] < rulebook_bottom)) |  
     ((filtered_fawley["PlateLocHeight"] > rulebook_top) & (filtered_fawley["PlateLocHeight"] <= shadow_top)))
]

    
    total_fifty_fifty_pitches = shadow_pitches_df.shape[0]
    total_fifty_fifty_strikes = fifty_fifty_pitches[fifty_fifty_pitches['PitchCall'] == 'StrikeCalled'].shape[0]

    # Format as "x / y"
    fifty_fifty_display = f"{total_fifty_fifty_strikes} / {total_fifty_fifty_pitches}"

    return [
        ["Balls Called Strikes", balls_called_strikes],
        ["Strikes Called Balls", strikes_called_balls],
        ["50/50 Pitches", fifty_fifty_display]
    ]

# Calculate framing performance using filtered dataset
framing_table = calculate_framing_metrics(filtered_fawley)

# Display table
st.write("### Framing Performance")
st.table(framing_table)

