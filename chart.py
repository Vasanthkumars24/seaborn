import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)
hours = [f"{h:02d}:00" for h in range(24)]
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

base = [[0.2, 0.25, 0.3, 0.35, 0.5, 0.6, 0.55], [0.15, 0.18, 0.2, 0.25, 0.35, 0.45, 0.4], [0.1, 0.12, 0.15, 0.18, 0.25, 0.3, 0.28], [0.08, 0.09, 0.12, 0.14, 0.18, 0.22, 0.2], [0.07, 0.08, 0.09, 0.11, 0.14, 0.18, 0.16], [0.06, 0.07, 0.08, 0.09, 0.12, 0.15, 0.13], [0.08, 0.1, 0.12, 0.15, 0.2, 0.28, 0.25], [0.12, 0.15, 0.18, 0.22, 0.32, 0.42, 0.38], [0.2, 0.25, 0.35, 0.45, 0.6, 0.7, 0.62], [0.3, 0.35, 0.45, 0.6, 0.8, 0.85, 0.75], [0.4, 0.45, 0.55, 0.7, 0.9, 0.95, 0.85], [0.5, 0.55, 0.65, 0.8, 1.0, 1.05, 0.95], [0.55, 0.6, 0.7, 0.85, 1.1, 1.15, 1.0], [0.5, 0.55, 0.65, 0.8, 1.05, 1.1, 0.95], [0.45, 0.5, 0.6, 0.75, 0.95, 1.0, 0.85], [0.4, 0.45, 0.55, 0.7, 0.85, 0.9, 0.78], [0.38, 0.42, 0.5, 0.62, 0.78, 0.82, 0.7], [0.35, 0.4, 0.48, 0.6, 0.75, 0.8, 0.68], [0.33, 0.38, 0.45, 0.55, 0.7, 0.76, 0.65], [0.28, 0.32, 0.38, 0.45, 0.6, 0.68, 0.58], [0.22, 0.25, 0.3, 0.35, 0.45, 0.52, 0.45], [0.18, 0.2, 0.22, 0.28, 0.35, 0.42, 0.35], [0.15, 0.17, 0.18, 0.2, 0.25, 0.3, 0.27], [0.12, 0.14, 0.15, 0.16, 0.2, 0.24, 0.22]]

noise = np.random.normal(loc=0, scale=0.03, size=np.array(base).shape)
engagement = np.clip(np.array(base) + noise, 0, None) * 1200
df = pd.DataFrame(engagement, index=hours, columns=days)

sns.set_style("whitegrid")
sns.set_context("talk", font_scale=0.9)
plt.figure(figsize=(8,8))
ax = sns.heatmap(df.T, cmap="rocket_r", cbar_kws={"label":"Engagement (visits per 1000 users)"}, linewidths=0.5, linecolor="white")

ax.set_xticks(np.arange(len(hours))+0.5)
ax.set_xticklabels(hours, rotation=45, ha="right", fontsize=8)
ax.set_yticks(np.arange(len(days))+0.5)
ax.set_yticklabels(days, rotation=0, fontsize=10)

ax.set_title("Hourly Customer Engagement by Day (Synthetic Data)", pad=16, fontsize=16)
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Day of Week")
plt.tight_layout()
plt.savefig("chart.png", dpi=64, bbox_inches="tight")
print('Saved chart.png')