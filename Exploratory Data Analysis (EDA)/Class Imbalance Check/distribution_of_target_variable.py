import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("secondary_data.csv", sep=';')

class_counts = df['class'].value_counts()
print("🎯 Target Variable Distribution:\n")
print(class_counts)
print("\n🔢 Oranlar (%):")
print(class_counts / class_counts.sum() * 100)

plt.figure(figsize=(6, 4))
sns.countplot(x='class', data=df, palette='Set2')
plt.title("Class Distribution (e: Edible, p: Poisonous)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()
