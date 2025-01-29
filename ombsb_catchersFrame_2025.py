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

# Function to create a scatter plot with the correctly drawn shadow zones
def create_zone_scatter(title, pitch_df):
    fig = go.Figure()

    # Add scatter plot for pitches
    for index, row in pitch_df.iterrows():
        color = "green" if row["PitchCall"] == "StrikeCalled" else "red"
        marker_shape = pitch_marker_map.get(row["TaggedPitchType"], "diamond")  # Default to rhombus

        fig.add_trace(go.Scatter(
            x=[row["PlateLocSide"]],
            y=[row["PlateLocHeight"]],
            mode="markers",
            marker=dict(symbol=marker_shape, color=color, size=8),
            showlegend=False
        ))
# Compute Strike% BEFORE using them in titles
strike_percentage_strike = 100.0  # Since this plot only contains StrikeCalled pitches
strike_percentage_ball = 0.0  # Since this plot only contains BallCalled pitches
strike_percentage_all = calculate_strike_percentage(all_pitches_df)
strike_percentage_shadow = calculate_strike_percentage(shadow_pitches_df)

    # Draw main strike zone
    for i in range(4):
        fig.add_shape(type="line", x0=x_splits[i], x1=x_splits[i], y0=rulebook_bottom, y1=rulebook_top, line=dict(color="black", width=1))
        fig.add_shape(type="line", x0=rulebook_left, x1=rulebook_right, y0=y_splits[i], y1=y_splits[i], line=dict(color="black", width=1))

    # Draw shadow zone outlines (ensuring full enclosure)
    fig.add_shape(type="rect", x0=shadow_left, x1=shadow_right, y0=shadow_bottom, y1=shadow_top,
                  line=dict(color="blue", width=2, dash="dash"))

    # Add missing horizontal lines at top and bottom to fully enclose shadow zone
    fig.add_shape(type="line", x0=shadow_left, x1=shadow_right, y0=shadow_top, y1=shadow_top,
                  line=dict(color="blue", width=2, dash="dash"))  # Top boundary
    fig.add_shape(type="line", x0=shadow_left, x1=shadow_right, y0=shadow_bottom, y1=shadow_bottom,
                  line=dict(color="blue", width=2, dash="dash"))  # Bottom boundary

    # Add vertical connectors to fully split shadow zones
    fig.add_shape(type="line", x0=strike_zone_middle_x, x1=strike_zone_middle_x, y0=shadow_bottom, y1=rulebook_bottom,
                  line=dict(color="blue", width=2, dash="dash"))  # Bottom middle connector
    fig.add_shape(type="line", x0=strike_zone_middle_x, x1=strike_zone_middle_x, y0=rulebook_top, y1=shadow_top,
                  line=dict(color="blue", width=2, dash="dash"))  # Top middle connector

    fig.add_shape(type="line", x0=shadow_left, x1=rulebook_left, y0=strike_zone_middle_y, y1=strike_zone_middle_y,
                  line=dict(color="blue", width=2, dash="dash"))  # Left middle connector
    fig.add_shape(type="line", x0=rulebook_right, x1=shadow_right, y0=strike_zone_middle_y, y1=strike_zone_middle_y,
                  line=dict(color="blue", width=2, dash="dash"))  # Right middle connector

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
st.write("### Updated Strike Zone Breakdown with Strike%")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

col1.plotly_chart(fig1, use_container_width=True)
col2.plotly_chart(fig2, use_container_width=True)
col3.plotly_chart(fig3, use_container_width=True)
col4.plotly_chart(fig4, use_container_width=True)
