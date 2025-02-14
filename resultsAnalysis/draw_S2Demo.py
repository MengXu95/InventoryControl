import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots

# Data for consider rental
data = {
    'inventory_level_0': [10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'replenishment_decision_0': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'inventory_level_1': [10, 8, 7, 0, 0, 0, 0, 0, 0, 0],
    'replenishment_decision_1': [0, 0, 2, 2, 2, 2, 2, 2, 2, 2],
    'rental_decision': [0, 0, 100, 100, 100, 100, 200, 200, 100, 100],
    'total_cost': [46, 59, 74, 20, 20, 20, 206, 20, 20, 20],
}

# # Data for no rental
# data = {
#     'inventory_level_0': [10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     'replenishment_decision_0': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     'inventory_level_1': [10, 8, 7, 0, 0, 0, 0, 0, 0, 0],
#     'replenishment_decision_1': [0, 0, 2, 2, 2, 2, 2, 2, 2, 2],
#     'rental_decision': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     'total_cost': [46, 59, 54, 186, 160, 178, 186, 176, 110, 190],
# }

# Convert to DataFrame
df = pd.DataFrame(data)
x_values = list(df.index)  # Decision points

# Create subplots with reduced vertical spacing
fig = make_subplots(
    rows=6, cols=1,
    subplot_titles=[
        "Inventory Level (Site 0)",
        "Replenishment Decision (Site 0)",
        "Inventory Level (Site 1)",
        "Replenishment Decision (Site 1)",
        "Rental Decision",
        "Total Cost"
    ],
    vertical_spacing=0.05  # Reduced spacing between subplots
)

# Add traces
fig.add_trace(go.Scatter(x=x_values, y=df['inventory_level_0'], mode='lines+markers', name="Inventory Level 0", marker_color='blue'), row=1, col=1)
fig.add_trace(go.Scatter(x=x_values, y=df['replenishment_decision_0'], mode='lines+markers', name="Replenishment Decision 0", marker_color='green'), row=2, col=1)
fig.add_trace(go.Scatter(x=x_values, y=df['inventory_level_1'], mode='lines+markers', name="Inventory Level 1", marker_color='orange'), row=3, col=1)
fig.add_trace(go.Scatter(x=x_values, y=df['replenishment_decision_1'], mode='lines+markers', name="Replenishment Decision 1", marker_color='red'), row=4, col=1)
fig.add_trace(go.Scatter(x=x_values, y=df['rental_decision'], mode='lines+markers', name="Rental Decision", marker_color='purple'), row=5, col=1)
fig.add_trace(go.Scatter(x=x_values, y=df['total_cost'], mode='lines+markers', name="Total Cost", marker_color='black'), row=6, col=1)

# Update layout for compact visualization
fig.update_layout(
    # title_text="Inventory System Analysis",
    height=900,  # Adjusted height for compactness
    width=900,
    showlegend=True,
)

# Move x-axis title to the bottom of all subplots
fig.update_xaxes(title_text="Decision Point", row=6, col=1)

# Save as PNG
fig.write_image("inventory_analysis.png")

# Show the figure
fig.show()
