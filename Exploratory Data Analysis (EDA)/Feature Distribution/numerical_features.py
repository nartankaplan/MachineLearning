import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("secondary_data.csv", sep=';')

numeric_columns = ['cap-diameter', 'stem-height', 'stem-width']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

fig, axes = plt.subplots(2, 3, figsize=(18, 8))

for idx, col in enumerate(numeric_columns):
    # Histogram 
    sns.histplot(df[col], kde=True, bins=30, ax=axes[0, idx], color='skyblue')
    axes[0, idx].set_title(f"{col} - Histogram")

    # Boxplot 
    sns.boxplot(x=df[col], ax=axes[1, idx], color='salmon')
    axes[1, idx].set_title(f"{col} - Boxplot")

plt.tight_layout()
plt.show()
