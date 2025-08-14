import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Prepare the data
data = {
    'Model': ['xgboost', 'rnd forest', 'knn', 'max', 'lightgbm', 'log reg'],
    'Score': [0.2416, 0.2833, 0.2738, 0.3368, 0.2377, 0.2194]
}

df = pd.DataFrame(data)

# Sort the DataFrame by Score in DESCENDING order for better visual comparison
df_sorted = df.sort_values(by='Score', ascending=False)

# --- 2. Set up the plot aesthetics ---
# Use Seaborn's set_theme for a modern look
sns.set_theme(
    style='whitegrid',  # A clean white background with subtle grid lines
    palette='rocket_r', # A visually appealing sequential palette (reversed for darker on higher values)
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
        # Removed 'font.family': ['Arial', 'sans-serif'] to avoid font not found errors
        # Matplotlib will default to a robust sans-serif font (e.g., DejaVu Sans)
        'font.size': 10
    }
)

# 3. Create the bar plot
plt.figure(figsize=(10, 6)) # Set the figure size

ax = sns.barplot(x='Model', y='Score', data=df_sorted)

# 4. Add titles and labels for clarity
plt.title('Partial F1 Score by Model', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Model Used', fontsize=14, labelpad=15)
plt.ylabel('Partial F1 Score', fontsize=14, labelpad=15)

# 5. Add the score values on top of each bar
for p in ax.patches:
    ax.annotate(f'{p.get_height():.4f}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom',
                xytext=(0, 5), # Offset text slightly above the bar
                textcoords='offset points',
                fontsize=11,
                color='#333333',
                fontweight='bold')

# 6. Adjust y-axis limits and ticks
plt.ylim(0, df_sorted['Score'].max() * 1.15) # Extend slightly more for annotation clarity
ax.set_yticks([]) # Hide y-axis ticks as values are annotated on bars
ax.tick_params(axis='x', length=0) # Remove default x-axis tick marks

# Add a subtle grid only for the y-axis (horizontal lines)
ax.grid(axis='y', linestyle='--', alpha=0.6)

# 7. Ensure all labels and titles fit within the figure area
plt.tight_layout()

# 8. Save the figure with high resolution
# To save with a completely white background:
plt.savefig('model_performance_scores.png', dpi=300, bbox_inches='tight')

# To save with a transparent background (useful for overlaying on other designs):
# plt.savefig('model_performance_scores_modern_transparent_bg.png', dpi=300, bbox_inches='tight', transparent=True)

# 9. Display the plot
plt.show()