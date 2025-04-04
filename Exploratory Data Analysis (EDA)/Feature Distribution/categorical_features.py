import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("secondary_data.csv", sep=';')

categorical_columns = df.select_dtypes(include='object').columns

for col in categorical_columns[:3]:
    value_counts = df[col].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[0], palette='pastel')
    axes[0].set_title(f"{col} - Bar Chart")
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].set_ylabel("Count")
    axes[0].set_xlabel("Category")

    # Pie chart
    axes[1].pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', startangle=140)
    axes[1].set_title(f"{col} - Pie Chart")

    plt.tight_layout()
    plt.show()
