import matplotlib.pyplot as plt
import numpy as np

# Data from results directory
models = ['k-NN', 'QSVM', 'ISDO (Ours)']
accuracies = [92.6, 90.9, 88.8]  # percentages

# Setting up the plot
plt.figure(figsize=(10, 6))
colors = ['#3498db', '#95a5a6', '#e74c3c']  # Blue, Grey, Red (ISDO highlighted)

bars = plt.bar(models, accuracies, color=colors, width=0.6)

# Add titles and labels
plt.title('Classification Accuracy Comparison', fontsize=16, pad=20)
plt.ylabel('Test Accuracy (%)', fontsize=14)
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add accuracy values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Save the figure
plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=300)
print("Graph saved as accuracy_comparison.png")
