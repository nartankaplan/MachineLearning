import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("secondary_data.csv", sep=';')

numeric_columns_to_convert = ['cap-diameter', 'stem-height', 'stem-width']
for col in numeric_columns_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')

numerical_df = df.select_dtypes(include='number')

corr_matrix = numerical_df.corr()

print("📈 Correlation  Matrix:")
print(corr_matrix)

plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Numerical Feature Correlation Heatmap")
plt.tight_layout()
plt.show()
