import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Prepare the data
# Data extracted precisely from the image, ensuring correct pairing
data_combined = {
    'Model_Type': ['router', 'router', 'max', 'max'],
    'Configuration': ['QB, MINI, PALI', 'QB, MINI', 'QB, MINI, PALI', 'QB, MINI'],
    'Score': [0.2833, 0.2766, 0.3368, 0.3124]
}

df_combined = pd.DataFrame(data_combined)

# Define the explicit order for Model_Type on the x-axis
model_type_order = ['router', 'max']

# Define the explicit order for 'Configuration' (hue) to match the legend and desired colors
# This means 'QB, MINI, PALI' will get the first color, 'QB, MINI' the second.
hue_order = ['QB, MINI, PALI', 'QB, MINI']

# Define custom colors to match the image precisely
# These hex codes are visually matched to the image's colors
custom_palette = {
    'QB, MINI, PALI': '#5e3c73', # Dark purple/violet from the image
    'QB, MINI': '#4c6e8b'        # Muted blue-grey from the image
}

# --- 2. Set up the plot aesthetics ---
sns.set_theme(
    style='white',  # Use 'white' style for a plain white background
    # palette=custom_palette, # <-- REMOVED THIS LINE! Pass palette directly to barplot.
    rc={
        'axes.facecolor': 'white',    # Set plot area background to white
        'figure.facecolor': 'white',  # Set entire figure background to white
        'axes.edgecolor': '#333333',  # Darker border for the plot area (matches image bottom line)
        'grid.color': '#cccccc',     # Lighter grid lines (though we won't show many)
        'axes.spines.top': False,    # Remove top spine
        'axes.spines.right': False,  # Remove right spine
        'axes.spines.bottom': True,  # Keep bottom spine to delineate bars from axis
        'axes.spines.left': False,   # Remove left spine for cleaner Y-axis
        'xtick.bottom': False,       # Remove x-axis tick marks
        'ytick.left': False,         # Remove y-axis tick marks
        'axes.labelcolor': '#333333',# Darker label color
        'text.color': '#333333',     # Darker text color
        'font.size': 10              # Base font size
    }
)

# 3. Create the bar plot
plt.figure(figsize=(10, 6))

ax = sns.barplot(
    x='Model_Type',
    y='Score',
    hue='Configuration',
    data=df_combined,
    order=model_type_order,
    hue_order=hue_order, # Ensure order of hue categories
    palette=custom_palette # <-- ADDED palette HERE!
)

# 4. Add titles and labels for clarity
plt.title('Partial F1 Score', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Model Type', fontsize=14, labelpad=15)
plt.ylabel('Partial F1 Score', fontsize=14, labelpad=15)

# 5. Add the score values on top of each bar using ax.bar_label (Matplotlib 3.4+)
# This is the most robust way to annotate grouped bars and avoids the 0.0000 issue.
for container in ax.containers:
    ax.bar_label(container, fmt='%.4f', fontsize=11, fontweight='bold', color='#333333', padding=5)

# 6. Adjust y-axis limits and ticks
plt.ylim(0, df_combined['Score'].max() * 1.1) # Extend slightly above max score for annotation clarity
ax.set_yticks([]) # Hide y-axis ticks as values are annotated on bars
ax.tick_params(axis='x', length=0) # Remove default x-axis tick marks

# Remove any default grid lines from 'white' style
ax.grid(False)

# 7. Customize the legend to match the image's appearance
ax.legend(
    title='Configuration',
    frameon=True,          # Show frame
    edgecolor='#333333',   # Dark border color
    bbox_to_anchor=(1.0, 1.0), # Position top right
    loc='upper left',      # Anchor point on the legend
    borderaxespad=0.2,     # Padding from the axes border
    fontsize=10,           # Font size for legend entries
    title_fontsize=11      # Font size for legend title
)

# 8. Ensure all labels and titles fit within the figure area, making space for legend
plt.tight_layout(rect=[0, 0, 0.95, 1]) # Adjust rect to ensure legend is not cut off

# 9. Save the figure with high resolution
plt.savefig('model_performance_grouped_scores_matched.png', dpi=300, bbox_inches='tight')

# 10. Display the plot
plt.show()