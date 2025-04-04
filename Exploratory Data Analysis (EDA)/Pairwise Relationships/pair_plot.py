import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("secondary_data.csv", sep=';')

numeric_columns = ['cap-diameter', 'stem-height', 'stem-width']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')


sns.pairplot(df[numeric_columns + ['class']], hue='class', palette='Set2')
plt.suptitle("Pairwise Relationships (Colored by Class)", y=1.02)
plt.show()
