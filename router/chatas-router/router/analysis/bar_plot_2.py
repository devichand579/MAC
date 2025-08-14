import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Prepare the data for Feature Importances
feature_data = {
    'Feature': ['qb_nll', 'prefix_chars', 'num_previous_utterance',
                'num_utterance_after_img', 'qb_pred_subword_len'],
    'Importance': [0.413404, 0.233380, 0.130828, 0.127214, 0.095174]
}

df_features = pd.DataFrame(feature_data)

# Sort the DataFrame by Importance in DESCENDING order
df_features_sorted = df_features.sort_values(by='Importance', ascending=False)

# --- 2. Set up the plot aesthetics ---
# Use Seaborn's set_theme for a modern look
sns.set_theme(
    style='whitegrid',  # A clean white background with subtle grid lines
    palette='rocket_r', # --- KEPT THIS SAME AS PREVIOUS PLOT ---
    rc={
        'axes.facecolor': 'white',    # Set plot area background to white
        'figure.facecolor': 'white',  # Set figure background to white
        'axes.edgecolor': '#333333',  # Darker border for the plot area
        'grid.color': '#cccccc',     # Lighter grid lines
        'axes.spines.top': False,    # Remove top spine
        'axes.spines.right': False,  # Remove right spine
        'axes.spines.left': False,   # Remove left spine for cleaner Y-axis
        'xtick.bottom': False,       # Remove x-axis tick marks
        'ytick.left': False,         # Remove y-axis tick marks
        'axes.labelcolor': '#333333',# Darker label color
        'text.color': '#333333',     # Darker text color
        'font.size': 10
    }
)

# 3. Create the bar plot
plt.figure(figsize=(10, 6)) # Set the figure size

ax = sns.barplot(x='Feature', y='Importance', data=df_features_sorted)

# 4. Add titles and labels for clarity
plt.title('Feature Importance Scores', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Feature', fontsize=14, labelpad=15)
plt.ylabel('Importance Score', fontsize=14, labelpad=15)

# 5. Add the score values on top of each bar
for p in ax.patches:
    ax.annotate(f'{p.get_height():.6f}', # Importance scores have more decimal places, so .6f
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom',
                xytext=(0, 5), # Offset text slightly above the bar
                textcoords='offset points',
                fontsize=11,
                color='#333333',
                fontweight='bold')

# 6. Adjust y-axis limits and ticks
plt.ylim(0, df_features_sorted['Importance'].max() * 1.15) # Extend slightly more for annotation clarity
ax.set_yticks([]) # Hide y-axis ticks as values are annotated on bars
ax.tick_params(axis='x', length=0) # Remove default x-axis tick marks

# Rotate x-axis labels as feature names can be long
plt.xticks(rotation=45, ha='right')

# Add a subtle grid only for the y-axis (horizontal lines)
ax.grid(axis='y', linestyle='--', alpha=0.6)

# 7. Ensure all labels and titles fit within the figure area
plt.tight_layout()

# 8. Save the figure with high resolution
plt.savefig('feature_importance_scores.png', dpi=300, bbox_inches='tight')

# 9. Display the plot
plt.show()